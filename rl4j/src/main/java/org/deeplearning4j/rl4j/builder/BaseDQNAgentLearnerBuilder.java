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

import java.util.List;

import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.LearningAgent;
import org.deeplearning4j.rl4j.agent.LearningAgent;
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.BaseTransitionTDAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.history.HistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdater;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.agent.learning.update.updater.sync.SyncLabelsNeuralNetUpdater;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.environment.action.space.ActionSpace;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.experience.ReplayMemoryExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionRewardState;
import org.deeplearning4j.rl4j.network.TrainableNeuralNet;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.NeuralNetPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.rng.Random;

public abstract class BaseDQNAgentLearnerBuilder<OBSERVATION extends Observation, ACTION extends DiscreteAction, CONFIGURATION extends BaseDQNAgentLearnerBuilder.Configuration<OBSERVATION, ACTION>>
		extends
		BaseAgentLearnerBuilder<OBSERVATION, ACTION, StateActionRewardState<ACTION>, FeaturesLabels, CONFIGURATION> {

	private final Random rnd;

	public BaseDQNAgentLearnerBuilder(CONFIGURATION configuration, TrainableNeuralNet neuralNet,
			Builder<Environment<ACTION>> environmentBuilder,
			Builder<TransformProcess<OBSERVATION>> transformProcessBuilder, Builder<HistoryProcessor<OBSERVATION>> historyProcessorBuilder, Random rnd) {
		super(configuration, neuralNet, environmentBuilder, transformProcessBuilder,historyProcessorBuilder);

		// TODO: remove once RNN networks states are supported with BaseDQN
		Preconditions.checkArgument(!neuralNet.isRecurrent(), "Recurrent networks are not yet supported with BaseDQN.");
		this.rnd = rnd;
	}

	@Override
	protected Policy<OBSERVATION,ACTION> buildPolicy() {
		NeuralNetPolicy<OBSERVATION,ACTION> greedyPolicy = new DQNPolicy<OBSERVATION,ACTION>(networks.getThreadCurrentNetwork(),getEnvironment().getActionSpace());
		return new EpsGreedy(greedyPolicy, getEnvironment().getActionSpace(), configuration.getPolicyConfiguration(), rnd);
	}

	@Override
	protected ExperienceHandler<OBSERVATION,ACTION, StateActionRewardState<ACTION>> buildExperienceHandler() {
		return new ReplayMemoryExperienceHandler(configuration.getExperienceHandlerConfiguration(), rnd);
	}

	@Override
	protected NeuralNetUpdater<FeaturesLabels> buildNeuralNetUpdater() {
		if (configuration.isAsynchronous()) {
			throw new UnsupportedOperationException("Only synchronized use is currently supported");
		}

		return new SyncLabelsNeuralNetUpdater(networks.getThreadCurrentNetwork(), networks.getTargetNetwork(),
				configuration.getNeuralNetUpdaterConfiguration());
	}

	@EqualsAndHashCode(callSuper = true)
	@SuperBuilder
	@Data
	public static class Configuration<OBSERVATION extends Observation, ACTION extends Action>
			extends BaseAgentLearnerBuilder.Configuration<OBSERVATION, ACTION> {
		EpsGreedy.Configuration policyConfiguration;
		ReplayMemoryExperienceHandler.Configuration experienceHandlerConfiguration;
		NeuralNetUpdaterConfiguration neuralNetUpdaterConfiguration;
		BaseTransitionTDAlgorithm.Configuration updateAlgorithmConfiguration;
	}
}
