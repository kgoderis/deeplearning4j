/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.mdp.gym;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Parameter;
import java.util.HashMap;
import java.util.Map;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.jodah.typetools.TypeResolver;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.action.space.ActionSpace;
import org.deeplearning4j.rl4j.environment.action.space.DiscreteActionSpace;
import org.deeplearning4j.rl4j.environment.action.space.TransformActionSpace;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.transform.FilterOperation;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess.Builder;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.collect.Maps;
import org.bytedeco.cpython.*;
import org.bytedeco.numpy.*;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

@Slf4j
public class Gym<OBSERVATION extends Observation, ACTION extends DiscreteAction, ACTIONSPACE extends DiscreteActionSpace<ACTION>>
		implements Environment<ACTION> {

	public static final String GYM_MONITOR_DIR = "/tmp/gym-dqn";

	private static void checkPythonError() {
		if (PyErr_Occurred() != null) {
			PyErr_Print();
			throw new RuntimeException("Python error occurred");
		}
	}

	private static Pointer program;
	private static PyObject globals;
	static {
		try {
			Py_AddPath(org.bytedeco.gym.presets.gym.cachePackages());
			program = Py_DecodeLocale(Gym.class.getSimpleName(), null);
			Py_SetProgramName(program);
			Py_Initialize();
			PyEval_InitThreads();
			PySys_SetArgvEx(1, program, 0);
			if (_import_array() < 0) {
				PyErr_Print();
				throw new RuntimeException("numpy.core.multiarray failed to import");
			}
			globals = PyModule_GetDict(PyImport_AddModule("__main__"));
			PyEval_SaveThread(); // just to release the GIL
		} catch (IOException e) {
			PyMem_RawFree(program);
			throw new RuntimeException(e);
		}
	}
	private PyObject locals;

	final protected ACTIONSPACE actionSpace;
	@Getter
	final private String envId;
	@Getter
	final private boolean render;
	@Getter
	final private boolean monitor;
	private TransformActionSpace transformActionSpace = null;
	private boolean done = false;
	
	private Gym(Builder builder) {
		this(builder.envId, builder.render, builder.monitor, builder.seed, builder.actions);
	}

	public Gym(String envId, boolean render, boolean monitor) {
		this(envId, render, monitor, (Integer) null);
	}

	public Gym(String envId, boolean render, boolean monitor, Integer seed) {
		this.envId = envId;
		this.render = render;
		this.monitor = monitor;

		int gstate = PyGILState_Ensure();
		try {
			locals = PyDict_New();

			Py_DecRef(PyRun_StringFlags("import gym; env = gym.make('" + envId + "')", Py_single_input, globals, locals,
					null));
			checkPythonError();
			if (monitor) {
				Py_DecRef(PyRun_StringFlags("env = gym.wrappers.Monitor(env, '" + GYM_MONITOR_DIR + "')",
						Py_single_input, globals, locals, null));
				checkPythonError();
			}
			if (seed != null) {
				Py_DecRef(PyRun_StringFlags("env.seed(" + seed + ")", Py_single_input, globals, locals, null));
				checkPythonError();
			}
			PyObject shapeTuple = PyRun_StringFlags("env.observation_space.shape", Py_eval_input, globals, locals,
					null);
			int[] shape = new int[(int) PyTuple_Size(shapeTuple)];
			for (int i = 0; i < shape.length; i++) {
				shape[i] = (int) PyLong_AsLong(PyTuple_GetItem(shapeTuple, i));
			}
			Py_DecRef(shapeTuple);

			PyObject n = PyRun_StringFlags("env.action_space.n", Py_eval_input, globals, locals, null);

			try {
				Class<?>[] typeArguments = TypeResolver.resolveRawArguments(Gym.class, getClass());
				Class<ACTION> actionType = (Class<ACTION>) typeArguments[1];
				Class<ACTIONSPACE> actionSpaceType = (Class<ACTIONSPACE>) typeArguments[2];
				
				ACTION noOpAction = actionType.getDeclaredConstructor(int.class).newInstance(0);
				actionSpace =  (ACTIONSPACE) actionSpaceType.getDeclaredConstructor(int.class,actionType).newInstance((int)PyLong_AsLong(n),noOpAction);
			} catch (Exception e) {
				// Oops, no appropriate constructor
				throw new RuntimeException(e);
			}

			Py_DecRef(n);
			checkPythonError();
		} finally {
			PyGILState_Release(gstate);
		}
	}

	public Gym(String envId, boolean render, boolean monitor, int[] actions) {
		this(envId, render, monitor, null, actions);
	}

	public Gym(String envId, boolean render, boolean monitor, Integer seed, int[] actions) {
		this(envId, render, monitor, seed);
		if (actions != null) {
			transformActionSpace = new TransformActionSpace<ACTION>(getActionSpace(), actions);
		}
	}

	@Override
	public ACTIONSPACE getActionSpace() {
		if (transformActionSpace == null)
			return (ACTIONSPACE) actionSpace;
		else
			return (ACTIONSPACE) transformActionSpace;
	}

	@Override
	public StepResult step(ACTION action) {
		int gstate = PyGILState_Ensure();
		try {
			if (render) {
				Py_DecRef(PyRun_StringFlags("env.render()", Py_single_input, globals, locals, null));
				checkPythonError();
			}
			Py_DecRef(PyRun_StringFlags(
					"state, reward, done, info = env.step(" + (Integer) actionSpace.encode(action) + ")",
					Py_single_input, globals, locals, null));
			checkPythonError();

			PyArrayObject state = new PyArrayObject(PyDict_GetItemString(locals, "state"));
			DoublePointer stateData = new DoublePointer(PyArray_BYTES(state)).capacity(PyArray_Size(state));
			SizeTPointer stateDims = PyArray_DIMS(state).capacity(PyArray_NDIM(state));

			double reward = PyFloat_AsDouble(PyDict_GetItemString(locals, "reward"));
			done = PyLong_AsLong(PyDict_GetItemString(locals, "done")) != 0;
			checkPythonError();

			double[] data = new double[(int) stateData.capacity()];
			stateData.get(data);

			Map<String, Object> channelsData = new HashMap<String, Object>() {
				{
					put("data", data);
				}
			};

			return new StepResult(channelsData, reward, done);
		} finally {
			PyGILState_Release(gstate);
		}
	}

	@Override
	public boolean isEpisodeFinished() {
		return done;
	}

	@Override
	public Map<String, Object> reset() {
		int gstate = PyGILState_Ensure();
		try {
			Py_DecRef(PyRun_StringFlags("state = env.reset()", Py_single_input, globals, locals, null));
			checkPythonError();

			PyArrayObject state = new PyArrayObject(PyDict_GetItemString(locals, "state"));
			DoublePointer stateData = new DoublePointer(PyArray_BYTES(state)).capacity(PyArray_Size(state));
			SizeTPointer stateDims = PyArray_DIMS(state).capacity(PyArray_NDIM(state));
			checkPythonError();

			done = false;

			double[] data = new double[(int) stateData.capacity()];
			stateData.get(data);

			Map<String, Object> channelsData = new HashMap<String, Object>() {
				{
					put("data", data);
				}
			};

			return channelsData;
		} finally {
			PyGILState_Release(gstate);
		}
	}

	@Override
	public void close() {
		int gstate = PyGILState_Ensure();
		try {
			Py_DecRef(PyRun_StringFlags("env.close()", Py_single_input, globals, locals, null));
			checkPythonError();
			Py_DecRef(locals);
		} finally {
			PyGILState_Release(gstate);
		}
	}

	public static Builder builder() {
		return new Builder();
	}

	public static class Builder<OBSERVATION extends Observation, ACTION extends DiscreteAction, ACTIONSPACE extends DiscreteActionSpace<ACTION>> {
		private String envId;
		private boolean render;
		private boolean monitor;
		private int seed;
		private int[] actions;

		public Builder environment(String envId) {
			Preconditions.checkNotNull(envId, "The envId must not be null");

			this.envId = envId;
			return this;
		}
		
		public Builder render(boolean render) {
			Preconditions.checkNotNull(render, "The render must not be null");

			this.render = render;
			return this;
		}
		
		public Builder monitor(boolean monitor) {
			Preconditions.checkNotNull(monitor, "The monitor must not be null");

			this.monitor = monitor;
			return this;
		}
		
		public Builder seed(int seed) {
			Preconditions.checkNotNull(seed, "The seed must not be null");

			this.seed = seed;
			return this;
		}
		
		public Builder actions(int[] actions) {
			Preconditions.checkNotNull(seed, "The actions must not be null");

			this.actions = actions;
			return this;
		}

		public Gym build() {
			return new Gym<OBSERVATION , ACTION , ACTIONSPACE>(envId, render, monitor, seed, actions);
		}
	}

}
