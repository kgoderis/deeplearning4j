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

package org.deeplearning4j.rl4j.agent.learning.algorithm.dqn;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import org.deeplearning4j.rl4j.agent.learning.algorithm.UpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesBuilder;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.action.space.ActionSpace;
import org.deeplearning4j.rl4j.experience.StateActionRewardState;
import org.deeplearning4j.rl4j.network.CommonLabelNames;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.OutputNeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public abstract class BaseTransitionTDAlgorithm<ACTION extends DiscreteAction> implements UpdateAlgorithm<FeaturesLabels, StateActionRewardState<ACTION>> {

    protected final OutputNeuralNet qNetwork;
    protected final ActionSpace<ACTION> actionSpace;
    protected final double gamma;

    private final double errorClamp;
    private final boolean isClamped;

    private final FeaturesBuilder featuresBuilder;
    /**
     *
     * @param qNetwork The Q-Network
     * @param configuration The {@link Configuration} to use
     */
    protected BaseTransitionTDAlgorithm(@NonNull OutputNeuralNet qNetwork, @NonNull ActionSpace<ACTION> actionSpace, @NonNull Configuration configuration) {
        this.qNetwork = qNetwork;
        this.actionSpace = actionSpace;
        this.gamma = configuration.getGamma();

        this.errorClamp = configuration.getErrorClamp();
        isClamped = !Double.isNaN(errorClamp);

        featuresBuilder = new FeaturesBuilder(qNetwork.isRecurrent());
    }

    /**
     * Called just before the calculation starts
     * @param features A {@link Features} instance of all observations in the batch
     * @param nextFeatures A {@link Features} instance of all next observations in the batch
     */
    protected void initComputation(Features features, Features nextFeatures) {
        // Do nothing
    }

    /**
     * Compute the new estimated Q-Value for every transition in the batch
     *
     * @param batchIdx The index in the batch of the current transition
     * @param reward The reward of the current transition
     * @param isTerminal True if it's the last transition of the "game"
     * @return The estimated Q-Value
     */
    protected abstract double computeTarget(int batchIdx, double reward, boolean isTerminal);

    @Override
    public FeaturesLabels compute(List<StateActionRewardState<ACTION>> stateActionRewardStates) {

        int size = stateActionRewardStates.size();

        Features features = featuresBuilder.build(stateActionRewardStates);
        Features nextFeatures = featuresBuilder.build(stateActionRewardStates.stream().map(e -> e.getNextObservation()), stateActionRewardStates.size());

        initComputation(features, nextFeatures);

        INDArray updatedQValues = qNetwork.output(features).get(CommonOutputNames.QValues);
        for (int i = 0; i < size; ++i) {
            StateActionRewardState<ACTION> stateActionRewardState = stateActionRewardStates.get(i);
            double yTarget = computeTarget(i, stateActionRewardState.getReward(), stateActionRewardState.isTerminal());

            if(isClamped) {
                double previousQValue = updatedQValues.getDouble(i, actionSpace.getIndex(stateActionRewardState.getAction()));
                double lowBound = previousQValue - errorClamp;
                double highBound = previousQValue + errorClamp;
                yTarget = Math.min(highBound, Math.max(yTarget, lowBound));
            }
            updatedQValues.putScalar(i, actionSpace.getIndex(stateActionRewardState.getAction()), yTarget);
        }

        FeaturesLabels featuresLabels = new FeaturesLabels(features);
        featuresLabels.putLabels(CommonLabelNames.QValues, updatedQValues);

        return featuresLabels;
    }

    @SuperBuilder
    @Data
    public static class Configuration {
        /**
         * The discount factor (default is 0.99)
         */
        @Builder.Default
        double gamma = 0.99;

        /**
         * Will prevent the new Q-Value from being farther than <i>errorClamp</i> away from the previous value. Double.NaN will disable the clamping (default).
         */
        @Builder.Default
        double errorClamp = Double.NaN;
    }
}
