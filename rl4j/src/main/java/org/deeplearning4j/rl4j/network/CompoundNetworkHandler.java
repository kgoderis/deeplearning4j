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
package org.deeplearning4j.rl4j.network;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.network.MultiLayerNetworkHandler.Configuration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class CompoundNetworkHandler extends PersistentNetworkHandler {

	public static final String CONFIGURATION_JSON = "compoundNetworkHandler.json";

	private final NetworkHandler[] networkHandlers;
	@Getter
	private boolean recurrent;

	/**
	 * @param networkHandlers All networks to be used in this instance.
	 */
	public CompoundNetworkHandler(NetworkHandler... networkHandlers) {
		this.networkHandlers = networkHandlers;

		for (NetworkHandler handler : networkHandlers) {
			recurrent |= handler.isRecurrent();
		}
	}

	@Override
	public void notifyGradientCalculation() {
		for (NetworkHandler handler : networkHandlers) {
			handler.notifyGradientCalculation();
		}
	}

	@Override
	public void notifyIterationDone() {
		for (NetworkHandler handler : networkHandlers) {
			handler.notifyIterationDone();
		}
	}

	@Override
	public void performFit(FeaturesLabels featuresLabels) {
		for (NetworkHandler handler : networkHandlers) {
			handler.performFit(featuresLabels);
		}
	}

	@Override
	public void performGradientsComputation(FeaturesLabels featuresLabels) {
		for (NetworkHandler handler : networkHandlers) {
			handler.performGradientsComputation(featuresLabels);
		}
	}

	@Override
	public void fillGradientsResponse(Gradients gradients) {
		for (NetworkHandler handler : networkHandlers) {
			handler.fillGradientsResponse(gradients);
		}
	}

	@Override
	public void applyGradient(Gradients gradients, long batchSize) {
		for (NetworkHandler handler : networkHandlers) {
			handler.applyGradient(gradients, batchSize);
		}
	}

	@Override
	public INDArray[] recurrentStepOutput(Observation observation) {
		List<INDArray> outputs = new ArrayList<INDArray>();
		for (NetworkHandler handler : networkHandlers) {
			Collections.addAll(outputs, handler.recurrentStepOutput(observation));
		}

		return outputs.toArray(new INDArray[0]);
	}

	@Override
	public INDArray[] stepOutput(Observation observation) {
		List<INDArray> outputs = new ArrayList<INDArray>();
		for (NetworkHandler handler : networkHandlers) {
			Collections.addAll(outputs, handler.stepOutput(observation));
		}

		return outputs.toArray(new INDArray[0]);
	}

	@Override
	public INDArray[] batchOutput(Features features) {
		List<INDArray> outputs = new ArrayList<INDArray>();
		for (NetworkHandler handler : networkHandlers) {
			Collections.addAll(outputs, handler.batchOutput(features));
		}

		return outputs.toArray(new INDArray[0]);
	}

	@Override
	public void resetState() {
		for (NetworkHandler handler : networkHandlers) {
			if (handler.isRecurrent()) {
				handler.resetState();
			}
		}
	}

	@Override
	public NetworkHandler clone() {
		NetworkHandler[] clonedHandlers = new NetworkHandler[networkHandlers.length];
		for (int i = 0; i < networkHandlers.length; ++i) {
			clonedHandlers[i] = networkHandlers[i].clone();
		}

		return new CompoundNetworkHandler(clonedHandlers);
	}

	@Override
	public void copyFrom(NetworkHandler from) {
		for (int i = 0; i < networkHandlers.length; ++i) {
			networkHandlers[i].copyFrom(((CompoundNetworkHandler) from).networkHandlers[i]);
		}
	}

	@Override
	public void saveTo(File file, boolean saveUpdater) throws IOException {

		if (!file.exists()) {
			file.getParentFile().mkdirs();
		}
		else {
			file.delete();
		}
		boolean success = file.createNewFile();
		if (success) {

			String tempDirectoryName = "";
			File tempDirectory = null;
			try {
				tempDirectory = Files.createTempDirectory("").toFile();
				tempDirectoryName = tempDirectory.getAbsolutePath();
			} catch (IOException e) {
				e.printStackTrace();
			}

			int counter = 1;
			for (NetworkHandler handler : networkHandlers) {
				File networkHandlerFile = new File(tempDirectoryName + "/" + String.valueOf(counter) + ".zip");
				handler.saveTo(networkHandlerFile, saveUpdater);
				this.saveToZip(file, networkHandlerFile);
				counter++;
			}

			saveConfigurationTo(file, new Configuration(recurrent));

			Files.walk(tempDirectory.toPath()).sorted(Comparator.reverseOrder()).map(Path::toFile)
					.forEach(File::delete);
		}
	}

	@Override
	public void loadFrom(File file, boolean loadUpdater) throws IOException {
		Configuration conf = (Configuration) loadConfigurationFrom(file, Configuration.class);

		if (conf != null) {

			String tempDirectoryName = "";
			File tempDirectory = null;
			try {
				tempDirectory = Files.createTempDirectory("").toFile();
				tempDirectoryName = tempDirectory.getAbsolutePath();
			} catch (IOException e) {
				e.printStackTrace();
			}

			int counter = 1;
			for (NetworkHandler handler : networkHandlers) {
				File networkHandlerFile = new File(tempDirectoryName + "/" + String.valueOf(counter) + ".zip");
				Files.write(networkHandlerFile.toPath(), loadFromZip(file, String.valueOf(counter) + ".zip"));
				handler.loadFrom(networkHandlerFile, loadUpdater);
				counter++;
			}

			this.recurrent = conf.recurrent;

			Files.walk(tempDirectory.toPath()).sorted(Comparator.reverseOrder()).map(Path::toFile)
					.forEach(File::delete);
		}
	}

	@SuppressWarnings("serial")
	@EqualsAndHashCode(callSuper = false)
	@Data
	public class Configuration extends PersistentNetworkHandler.Configuration {
		final boolean recurrent;
	}

	@Override
	String getConfigurationName() {
		return CONFIGURATION_JSON;
	}
}
