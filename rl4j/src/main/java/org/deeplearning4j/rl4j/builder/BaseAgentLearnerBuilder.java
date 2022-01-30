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

import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.LearningAgent;
import org.deeplearning4j.rl4j.agent.LearningAgent;
import org.deeplearning4j.rl4j.agent.learning.algorithm.UpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.behavior.LearningBehavior;
import org.deeplearning4j.rl4j.agent.learning.history.HistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.behavior.DefaultLearningBehavior;
import org.deeplearning4j.rl4j.agent.learning.update.UpdateRule;
import org.deeplearning4j.rl4j.agent.learning.update.DefaultUpdateRule;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdater;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.network.TrainableNeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;

import java.util.List;

public abstract class BaseAgentLearnerBuilder<OBSERVATION extends Observation, ACTION extends Action, EXPERIENCE, ALGORITHM_RESULT, CONFIGURATION extends BaseAgentLearnerBuilder.Configuration<OBSERVATION, ACTION>> implements Builder<LearningAgent<OBSERVATION, ACTION>> {

    protected final CONFIGURATION configuration;
    private final Builder<Environment<ACTION>> environmentBuilder;
    private final Builder<TransformProcess<OBSERVATION>> transformProcessBuilder;
    private final Builder<HistoryProcessor<OBSERVATION>> historyProcessorBuilder;
    protected final NetworkHandler networks;

    protected int createdAgentLearnerCount;

    public BaseAgentLearnerBuilder(@NonNull CONFIGURATION configuration,
                                   @NonNull TrainableNeuralNet neuralNet,
                                   @NonNull Builder<Environment<ACTION>> environmentBuilder,
                                   @NonNull Builder<TransformProcess<OBSERVATION>> transformProcessBuilder,
                                   @NonNull Builder<HistoryProcessor<OBSERVATION>> historyProcessorBuilder) {
        this.configuration = configuration;
        this.environmentBuilder = environmentBuilder;
        this.transformProcessBuilder = transformProcessBuilder;
        this.historyProcessorBuilder = historyProcessorBuilder;

        this.networks = configuration.isAsynchronous()
                ? new AsyncNetworkHandler(neuralNet)
                : new SyncNetworkHandler(neuralNet);
    }

	@Getter(AccessLevel.PROTECTED)
    private Environment<ACTION> environment;

    @Getter(AccessLevel.PROTECTED)
    private TransformProcess<OBSERVATION> transformProcess;
    
    @Getter(AccessLevel.PROTECTED)
    private HistoryProcessor<OBSERVATION> historyProcessor;

    @Getter(AccessLevel.PROTECTED)
    private Policy<OBSERVATION, ACTION> policy;

    @Getter(AccessLevel.PROTECTED)
    private ExperienceHandler<OBSERVATION, ACTION, EXPERIENCE> experienceHandler;

    @Getter(AccessLevel.PROTECTED)
    private UpdateAlgorithm<ALGORITHM_RESULT, EXPERIENCE> updateAlgorithm;

    @Getter(AccessLevel.PROTECTED)
    private NeuralNetUpdater<ALGORITHM_RESULT> neuralNetUpdater;

    @Getter(AccessLevel.PROTECTED)
    private UpdateRule<EXPERIENCE> updateRule;

    @Getter(AccessLevel.PROTECTED)
    private LearningBehavior<OBSERVATION,ACTION> learningBehavior;

    protected abstract Policy<OBSERVATION,ACTION> buildPolicy();
    protected abstract ExperienceHandler<OBSERVATION, ACTION, EXPERIENCE> buildExperienceHandler();
    protected abstract UpdateAlgorithm<ALGORITHM_RESULT, EXPERIENCE> buildUpdateAlgorithm();
    protected abstract NeuralNetUpdater<ALGORITHM_RESULT> buildNeuralNetUpdater();
    protected UpdateRule<EXPERIENCE> buildUpdateRule() {
        return new DefaultUpdateRule<ALGORITHM_RESULT, EXPERIENCE>(getUpdateAlgorithm(), getNeuralNetUpdater());
    }
    protected LearningBehavior<OBSERVATION,ACTION> buildLearningBehavior() {
        return DefaultLearningBehavior.<OBSERVATION,ACTION, EXPERIENCE>builder()
                .experienceHandler(getExperienceHandler())
                .updateRule(getUpdateRule())
                .build();
    }

    protected void resetForNewBuild() {
        networks.resetForNewBuild();
        environment = environmentBuilder.build();
        transformProcess = transformProcessBuilder.build();
        historyProcessor = historyProcessorBuilder.build();
        policy = buildPolicy();
        experienceHandler = buildExperienceHandler();
        updateAlgorithm = buildUpdateAlgorithm();
        neuralNetUpdater = buildNeuralNetUpdater();
        updateRule = buildUpdateRule();
        learningBehavior = buildLearningBehavior();

        ++createdAgentLearnerCount;
    }

    protected String getThreadId() {
        return "LearningAgent-" + createdAgentLearnerCount;
    }

    protected LearningAgent<OBSERVATION,ACTION> buildAgentLearner() {
        LearningAgent<OBSERVATION,ACTION> result = new LearningAgent<OBSERVATION, ACTION>(getEnvironment(), getTransformProcess(), getPolicy(), getHistoryProcessor(), configuration.getAgentLearnerConfiguration(), getThreadId(), getLearningBehavior());
        if(configuration.getAgentLearnerListeners() != null) {
            for (AgentListener<OBSERVATION,ACTION> listener : configuration.getAgentLearnerListeners()) {
                result.addListener(listener);
            }
        }

        return result;
    }

    /**
     * Build a properly assembled / configured LearningAgent.
     * @return a {@link LearningAgent}
     */
    @Override
    public LearningAgent<OBSERVATION,ACTION> build() {
        resetForNewBuild();
        return buildAgentLearner();
    }

    @SuperBuilder
    @Data
    public static class Configuration<OBSERVATION extends Observation,ACTION extends Action> {
        /**
         * The configuration that will be used to build the {@link LearningAgent}
         */
        LearningAgent.Configuration agentLearnerConfiguration;

        /**
         * A list of {@link AgentListener AgentListeners} that will be added to the LearningAgent. (default = null; no listeners)
         */
        List<AgentListener<OBSERVATION,ACTION>> agentLearnerListeners;

        /**
         * Tell the builder that the AgentLearners will be used in an asynchronous setup
         */
        boolean asynchronous;
    }
}
