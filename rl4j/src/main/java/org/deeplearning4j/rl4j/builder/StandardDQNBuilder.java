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
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.StandardDQN;
import org.deeplearning4j.rl4j.agent.learning.history.HistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.experience.ObservationActionRewardObservation;
import org.deeplearning4j.rl4j.network.TrainableNeuralNet;
import org.nd4j.linalg.api.rng.Random;

/**
 * A {@link LearningAgent} builder that will setup a {@link StandardDQN standard
 * BaseDQN} algorithm in addition to the setup done by
 * {@link BaseDQNAgentLearnerBuilder}.
 */
public class StandardDQNBuilder<OBSERVATION extends Observation, ACTION extends DiscreteAction>
		extends BaseDQNAgentLearnerBuilder<OBSERVATION, ACTION, StandardDQNBuilder.Configuration<OBSERVATION, ACTION>> {

	public StandardDQNBuilder(Configuration<OBSERVATION, ACTION> configuration, TrainableNeuralNet neuralNet,
			Builder<Environment<ACTION>> environmentBuilder,
			Builder<TransformProcess<OBSERVATION>> transformProcessBuilder,
			Builder<HistoryProcessor<OBSERVATION>> historyProcessorBuilder, Random rnd) {
		super(configuration, neuralNet, environmentBuilder, transformProcessBuilder, historyProcessorBuilder, rnd);
	}

	@Override
	protected UpdateAlgorithm<FeaturesLabels, ObservationActionRewardObservation<ACTION>> buildUpdateAlgorithm() {
		return new StandardDQN(networks.getThreadCurrentNetwork(), networks.getTargetNetwork(),
				getEnvironment().getActionSpace(), configuration.getUpdateAlgorithmConfiguration());
	}

	@EqualsAndHashCode(callSuper = true)
	@SuperBuilder
	@Data
	public static class Configuration<OBSERVATION extends Observation, ACTION extends Action>
			extends BaseDQNAgentLearnerBuilder.Configuration<OBSERVATION, ACTION> {
	}
}
