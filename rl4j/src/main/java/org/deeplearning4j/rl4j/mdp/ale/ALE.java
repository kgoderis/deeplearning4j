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

package org.deeplearning4j.rl4j.mdp.ale;

import lombok.Getter;
import lombok.Setter;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;

import java.util.HashMap;
import java.util.Map;

import org.bytedeco.ale.ALEInterface;
import org.bytedeco.javacpp.IntPointer;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.deeplearning4j.rl4j.environment.action.space.ActionSpace;
import org.deeplearning4j.rl4j.environment.action.space.DiscreteActionSpace;
import org.deeplearning4j.rl4j.environment.action.space.IntegerActionSpace;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.mdp.gym.Gym;
import org.deeplearning4j.rl4j.mdp.gym.Gym.Builder;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author saudet
 */
@Slf4j
public class ALE implements Environment<IntegerAction> {
	protected ALEInterface ale;
	final protected int[] actions;
	final protected IntegerActionSpace actionSpace;
	final protected long[] observationShape;
	@Getter
	final protected String romFile;
	@Getter
	final protected boolean render;
	@Getter
	final protected Configuration configuration;
	@Setter
	protected double scaleFactor = 1;

	private byte[] screenBuffer;

	public ALE(Builder builder) {
		this(builder.romFile, builder.render, new Configuration(builder.randomSeed, builder.repeatActionProbability,
				builder.maxNumFrames, builder.maxNumFramesPerEpisode, builder.minimalActionSet));
	}

	public ALE(String romFile) {
		this(romFile, false);
	}

	public ALE(String romFile, boolean render) {
		this(romFile, render, new Configuration(123, 0, 0, 0, true));
	}

	public ALE(String romFile, boolean render, Configuration configuration) {
		this.romFile = romFile;
		this.configuration = configuration;
		this.render = render;
		ale = new ALEInterface();
		setupGame();

		// Get the vector of minimal or legal actions
		IntPointer a = (getConfiguration().minimalActionSet ? ale.getMinimalActionSet() : ale.getLegalActionSet());
		actions = new int[(int) a.limit()];
		a.get(actions);

		int height = (int) ale.getScreen().height();
		int width = (int) (int) ale.getScreen().width();

		actionSpace = new IntegerActionSpace(actions.length ,actions[0]);
		observationShape = new long[] { 3, height, width };
		screenBuffer = new byte[(int) (observationShape[0] * observationShape[1] * observationShape[2])];
	}

	public void setupGame() {
		Configuration conf = getConfiguration();

		// Get & Set the desired settings
		ale.setInt("random_seed", conf.randomSeed);
		ale.setFloat("repeat_action_probability", conf.repeatActionProbability);

		ale.setBool("display_screen", render);
		ale.setBool("sound", render);

		// Causes episodes to finish after timeout tics
		ale.setInt("max_num_frames", conf.maxNumFrames);
		ale.setInt("max_num_frames_per_episode", conf.maxNumFramesPerEpisode);

		// Load the ROM file. (Also resets the system for new settings to
		// take effect.)
		ale.loadROM(romFile);
	}

	@Override
	public IntegerActionSpace getActionSpace() {
		return actionSpace;
	}
	
	public long[] getObservationShape() {
		return observationShape;
	}

	@Override
	public boolean isEpisodeFinished() {
		return ale.game_over();
	}

	@Override
	public Map<String, Object> reset() {
		ale.reset_game();
		ale.getScreenRGB(screenBuffer);

		Map<String, Object> channelsData = new HashMap<String, Object>() {
			{
				put("data", screenBuffer);
			}
		};

		return channelsData;
	}

	public void close() {
		ale.deallocate();
	}

	@Override
	public StepResult step(IntegerAction action) {
		double r = ale.act(actions[(int) actionSpace.encode(action)]) * scaleFactor;
		log.info(ale.getEpisodeFrameNumber() + " " + r + " " + action + " ");
		ale.getScreenRGB(screenBuffer);

		Map<String, Object> channelsData = new HashMap<String, Object>() {
			{
				put("data", screenBuffer);
			}
		};

		return new StepResult(channelsData, r, ale.game_over());
	}

	@Value
	public static class Configuration {
		int randomSeed;
		float repeatActionProbability;
		int maxNumFrames;
		int maxNumFramesPerEpisode;
		boolean minimalActionSet;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static class Builder {

		private String romFile;
		private boolean render;
		private int randomSeed = 123;
		private float repeatActionProbability = 0;
		private int maxNumFrames = 0;
		private int maxNumFramesPerEpisode = 0;
		private boolean minimalActionSet = false;

		public Builder rom(String romFile) {
			Preconditions.checkNotNull(romFile, "The romFile must not be null");

			this.romFile = romFile;
			return this;
		}

		public Builder render(boolean render) {
			Preconditions.checkNotNull(render, "The render must not be null");

			this.render = render;
			return this;
		}

		public Builder seed(int seed) {
			Preconditions.checkNotNull(seed, "The seed must not be null");

			this.randomSeed = seed;
			return this;
		}

		public Builder repeatProbability(float repeatProbability) {
			Preconditions.checkNotNull(repeatProbability, "The repeatProbability must not be null");

			this.repeatActionProbability = repeatProbability;
			return this;
		}

		public Builder maxFrames(int maxNumFrames) {
			Preconditions.checkNotNull(maxNumFrames, "The maxNumFrames must not be null");

			this.maxNumFrames = maxNumFrames;
			return this;
		}

		public Builder maxFramesPerPeriod(int maxNumFramesPerEpisode) {
			Preconditions.checkNotNull(maxNumFrames, "The maxNumFramesPerEpisode must not be null");

			this.maxNumFramesPerEpisode = maxNumFramesPerEpisode;
			return this;
		}

		public Builder minimalActionSet(boolean minimalActionSet) {
			Preconditions.checkNotNull(minimalActionSet, "The minimalActionSet must not be null");

			this.minimalActionSet = minimalActionSet;
			return this;
		}

		public ALE build() {
			return new ALE(romFile, render, new Configuration(randomSeed, repeatActionProbability, maxNumFrames,
					maxNumFramesPerEpisode, minimalActionSet));
		}
	}
}
