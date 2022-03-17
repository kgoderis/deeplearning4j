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
import org.deeplearning4j.rl4j.agent.learning.history.HistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdater;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.agent.learning.update.updater.async.AsyncGradientsNeuralNetUpdater;
import org.deeplearning4j.rl4j.agent.learning.update.updater.async.AsyncSharedNetworksUpdateHandler;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.experience.ObservationActionExperienceHandler;
import org.deeplearning4j.rl4j.experience.ObservationActionReward;
import org.deeplearning4j.rl4j.network.TrainableNeuralNet;
import org.deeplearning4j.rl4j.policy.EpsGreedy;

public abstract class BaseAsyncAgentLearnerBuilder<OBSERVATION extends Observation, ACTION extends Action, CONFIGURATION extends BaseAsyncAgentLearnerBuilder.Configuration<OBSERVATION, ACTION>>
		extends BaseAgentLearnerBuilder<OBSERVATION, ACTION, ObservationActionReward<OBSERVATION, ACTION>, Gradients, CONFIGURATION> {

	private final AsyncSharedNetworksUpdateHandler asyncSharedNetworksUpdateHandler;

	public BaseAsyncAgentLearnerBuilder(CONFIGURATION configuration, TrainableNeuralNet neuralNet,
			Builder<Environment<ACTION>> environmentBuilder, Builder<TransformProcess<OBSERVATION>> transformProcessBuilder,Builder<HistoryProcessor<OBSERVATION>> historyProcessorBuilder) {
		super(configuration, neuralNet, environmentBuilder, transformProcessBuilder,historyProcessorBuilder);

		asyncSharedNetworksUpdateHandler = buildAsyncSharedNetworksUpdateHandler();
	}

	@Override
	protected ExperienceHandler<OBSERVATION,ACTION, ObservationActionReward<OBSERVATION,ACTION>> buildExperienceHandler() {
		return new ObservationActionExperienceHandler<OBSERVATION,ACTION>(configuration.getExperienceHandlerConfiguration());
	}

	@Override
	protected NeuralNetUpdater<Gradients> buildNeuralNetUpdater() {
		return new AsyncGradientsNeuralNetUpdater(neuralNetHandler.getThreadCurrentNetwork(), asyncSharedNetworksUpdateHandler);
	}

	protected abstract AsyncSharedNetworksUpdateHandler buildAsyncSharedNetworksUpdateHandler();

	@EqualsAndHashCode(callSuper = true)
	@SuperBuilder
	@Data
	public static class Configuration<OBSERVATION extends Observation, ACTION extends Action>
			extends BaseAgentLearnerBuilder.Configuration<OBSERVATION, ACTION> {
		EpsGreedy.Configuration policyConfiguration;
		NeuralNetUpdaterConfiguration neuralNetUpdaterConfiguration;
		ObservationActionExperienceHandler.Configuration experienceHandlerConfiguration;
	}
}
