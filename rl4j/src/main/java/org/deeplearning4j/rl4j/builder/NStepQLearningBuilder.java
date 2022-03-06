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
package org.deeplearning4j.rl4j.builder;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;
import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.LearningAgent;
import org.deeplearning4j.rl4j.agent.learning.algorithm.UpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning.NStepQLearning;
import org.deeplearning4j.rl4j.agent.learning.history.HistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.async.AsyncSharedNetworksUpdateHandler;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.environment.action.space.ActionSpace;
import org.deeplearning4j.rl4j.experience.ObservationActionReward;
import org.deeplearning4j.rl4j.network.TrainableNeuralNet;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.NeuralNetPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.rng.Random;

public class NStepQLearningBuilder<OBSERVATION extends Observation, ACTION extends DiscreteAction> extends
		BaseAsyncAgentLearnerBuilder<OBSERVATION, ACTION, NStepQLearningBuilder.Configuration<OBSERVATION, ACTION>> {

	private final Random rnd;

	public NStepQLearningBuilder(Configuration<OBSERVATION, ACTION> configuration, TrainableNeuralNet neuralNet,
			Builder<Environment<ACTION>> environmentBuilder,
			Builder<TransformProcess<OBSERVATION>> transformProcessBuilder,
			Builder<HistoryProcessor<OBSERVATION>> historyProcessorBuilder, Random rnd) {
		super(configuration, neuralNet, environmentBuilder, transformProcessBuilder, historyProcessorBuilder);

		// TODO: remove once RNN networks states are stored in the experience elements
		Preconditions.checkArgument(
				!neuralNet.isRecurrent()
						|| configuration.getExperienceHandlerConfiguration().getBatchSize() == Integer.MAX_VALUE,
				"RL with a recurrent network currently only works with whole-trajectory updates. Until RNN are fully supported, please set the batch size of your experience handler to Integer.MAX_VALUE");

		this.rnd = rnd;
	}

	@Override
	protected Policy<OBSERVATION, ACTION> buildPolicy() {
		NeuralNetPolicy<OBSERVATION, ACTION> greedyPolicy = new DQNPolicy<OBSERVATION, ACTION>(
				networks.getThreadCurrentNetwork(), getEnvironment().getActionSpace());
		return new EpsGreedy(greedyPolicy, getEnvironment().getActionSpace(), configuration.getPolicyConfiguration(),
				rnd);
	}

	@Override
	protected UpdateAlgorithm<Gradients, ObservationActionReward<OBSERVATION, ACTION>> buildUpdateAlgorithm() {
		return new NStepQLearning(networks.getThreadCurrentNetwork(), networks.getTargetNetwork(),
				getEnvironment().getActionSpace(), configuration.getNstepQLearningConfiguration());
	}

	@Override
	protected AsyncSharedNetworksUpdateHandler buildAsyncSharedNetworksUpdateHandler() {
		return new AsyncSharedNetworksUpdateHandler(networks.getGlobalCurrentNetwork(), networks.getTargetNetwork(),
				configuration.getNeuralNetUpdaterConfiguration());
	}

	@EqualsAndHashCode(callSuper = true)
	@SuperBuilder
	@Data
	public static class Configuration<OBSERVATION extends Observation, ACTION extends Action>
			extends BaseAsyncAgentLearnerBuilder.Configuration<OBSERVATION, ACTION> {
		NStepQLearning.Configuration nstepQLearningConfiguration;
	}
}
