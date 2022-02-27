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

package org.deeplearning4j.rl4j.policy;

import lombok.Builder;
import lombok.NonNull;

import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.ObservationSource;
import org.deeplearning4j.rl4j.environment.action.space.ActionSpace;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.OutputNeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class ACPolicy<OBSERVATION extends Observation, ACTION extends Action> extends BasePolicy<OBSERVATION,ACTION> {

    private final boolean isTraining;
    private final Random rnd;

    @Builder
    public ACPolicy(@NonNull OutputNeuralNet neuralNet, @NonNull ActionSpace actionSpace, boolean isTraining, Random rnd) {
    	super(neuralNet, actionSpace);
        this.isTraining = isTraining;
        this.rnd = !isTraining || rnd != null ? rnd : Nd4j.getRandom();
    }

    @SuppressWarnings("unchecked")
	@Override
    public ACTION nextAction(OBSERVATION obs) {
        INDArray output = neuralNet.output(obs).get(CommonOutputNames.ActorCritic.Policy);
        if (!isTraining) {
            return (ACTION) actionSpace.fromArray(output);
        }

        float rVal = rnd.nextFloat();
        for (int i = 0; i < output.length(); i++) {
            if (rVal < output.getFloat(i)) {
                return (ACTION) actionSpace.fromInteger(i);
            } else
                rVal -= output.getFloat(i);
        }

        throw new RuntimeException("Output from network is not a probability distribution: " + output);
    }
}
