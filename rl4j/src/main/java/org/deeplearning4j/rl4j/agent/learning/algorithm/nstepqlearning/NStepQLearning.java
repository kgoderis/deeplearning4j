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
package org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import org.deeplearning4j.rl4j.agent.learning.algorithm.UpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesBuilder;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.action.space.ActionSpace;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.experience.ObservationActionReward;
import org.deeplearning4j.rl4j.network.CommonLabelNames;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.OutputNeuralNet;
import org.deeplearning4j.rl4j.network.TrainableNeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class NStepQLearning<OBSERVATION extends Observation, ACTION extends DiscreteAction> implements UpdateAlgorithm<Gradients, ObservationActionReward<OBSERVATION, ACTION>> {

    private final TrainableNeuralNet threadCurrent;
    private final OutputNeuralNet target;
    private final double gamma;
    private final NStepQLearningHelper<OBSERVATION,ACTION> algorithmHelper;
    private final FeaturesBuilder featuresBuilder;
    private final ActionSpace<ACTION> actionSpace;

    /**
     * @param threadCurrent The &theta;' parameters (the thread-specific network)
     * @param target The &theta;<sup>&ndash;</sup> parameters (the global target network)
     * @param actionSpaceSize The numbers of possible actions that can be taken on the environment
     */
    public NStepQLearning(@NonNull TrainableNeuralNet threadCurrent,
                          @NonNull OutputNeuralNet target,
                          @NonNull ActionSpace<ACTION> actionSpace,
                          @NonNull Configuration configuration) {
        this.threadCurrent = threadCurrent;
        this.target = target;
        this.gamma = configuration.getGamma();
        this.actionSpace = actionSpace;

        algorithmHelper = threadCurrent.isRecurrent()
                ? new RecurrentNStepQLearningHelper<OBSERVATION,ACTION>(actionSpace.getActionSpaceSize())
                : new NonRecurrentNStepQLearningHelper<OBSERVATION, ACTION>(actionSpace.getActionSpaceSize());

        featuresBuilder = new FeaturesBuilder(threadCurrent.isRecurrent());
    }

    @Override
    public Gradients compute(List<ObservationActionReward<OBSERVATION,ACTION>> trainingBatch) {
        int size = trainingBatch.size();

        ObservationActionReward<OBSERVATION,ACTION> observationActionReward = trainingBatch.get(size - 1);
        Features features = featuresBuilder.build(trainingBatch);
        INDArray labels = algorithmHelper.createLabels(size);

        double r;
        if (observationActionReward.isTerminal()) {
            r = 0;
        } else {
            INDArray expectedValuesOfLast = algorithmHelper.getTargetExpectedQValuesOfLast(target, trainingBatch, features);
            r = Nd4j.max(expectedValuesOfLast).getDouble(0);
        }

        for (int i = size - 1; i >= 0; --i) {
            observationActionReward = trainingBatch.get(i);

            r = observationActionReward.getReward() + gamma * r;
            
            INDArray expectedQValues = threadCurrent.output(observationActionReward.getObservation()).get(CommonOutputNames.QValues);
            expectedQValues = expectedQValues.putScalar(actionSpace.getIndex(observationActionReward.getAction()), r);
            algorithmHelper.setLabels(labels, i, expectedQValues);
        }

        FeaturesLabels featuresLabels = new FeaturesLabels(features);
        featuresLabels.putLabels(CommonLabelNames.QValues, labels);
        
        return threadCurrent.computeGradients(featuresLabels);
    }

    @SuperBuilder
    @Data
    public static class Configuration {
        /**
         * The discount factor (default is 0.99)
         */
        @Builder.Default
        double gamma = 0.99;
    }
}
