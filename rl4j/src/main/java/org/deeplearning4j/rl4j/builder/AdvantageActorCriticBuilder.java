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
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.LearningAgent;
import org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic.AdvantageActorCritic;
import org.deeplearning4j.rl4j.agent.learning.history.HistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.algorithm.UpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.async.AsyncSharedNetworksUpdateHandler;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.environment.action.space.ActionSpace;
import org.deeplearning4j.rl4j.experience.StateActionReward;
import org.deeplearning4j.rl4j.network.TrainableNeuralNet;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.nd4j.linalg.api.rng.Random;

public class AdvantageActorCriticBuilder<OBSERVATION extends Observation, ACTION extends Action> extends
		BaseAsyncAgentLearnerBuilder<OBSERVATION, ACTION, AdvantageActorCriticBuilder.Configuration<OBSERVATION, ACTION>> {

	private final Random rnd;

	public AdvantageActorCriticBuilder(@NonNull Configuration<OBSERVATION, ACTION> configuration,
			@NonNull TrainableNeuralNet neuralNet, @NonNull Builder<Environment<ACTION>> environmentBuilder,
			@NonNull Builder<TransformProcess<OBSERVATION>> transformProcessBuilder,
			Builder<HistoryProcessor<OBSERVATION>> historyProcessorBuilder, Random rnd) {
		super(configuration, neuralNet, environmentBuilder, transformProcessBuilder, historyProcessorBuilder);
		this.rnd = rnd;
	}

	@SuppressWarnings("unchecked")
	@Override
	protected Policy<OBSERVATION, ACTION> buildPolicy() {
		return (Policy<OBSERVATION, ACTION>) ACPolicy.builder().actionSpace(getEnvironment().getActionSpace())
				.neuralNet(networks.getThreadCurrentNetwork()).isTraining(true).rnd(rnd).build();
	}

	@SuppressWarnings("rawtypes")
	@Override
	protected UpdateAlgorithm<Gradients, StateActionReward<ACTION>> buildUpdateAlgorithm() {
		return new AdvantageActorCritic<ACTION>(networks.getThreadCurrentNetwork(), getEnvironment().getActionSpace(),
				configuration.getAdvantageActorCriticConfiguration());
	}

	@Override
	protected AsyncSharedNetworksUpdateHandler buildAsyncSharedNetworksUpdateHandler() {
		return new AsyncSharedNetworksUpdateHandler(networks.getGlobalCurrentNetwork(),
				configuration.getNeuralNetUpdaterConfiguration());
	}

	@EqualsAndHashCode(callSuper = true)
	@SuperBuilder
	@Data
	public static class Configuration<OBSERVATION extends Observation, ACTION extends Action>
			extends BaseAsyncAgentLearnerBuilder.Configuration<OBSERVATION, ACTION> {
		AdvantageActorCritic.Configuration advantageActorCriticConfiguration;
	}

}
