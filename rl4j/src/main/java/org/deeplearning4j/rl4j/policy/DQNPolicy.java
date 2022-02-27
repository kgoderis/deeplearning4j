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

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NonNull;

import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.action.space.ActionSpace;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.OutputNeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class DQNPolicy<OBSERVATION extends Observation, ACTION extends DiscreteAction> extends BasePolicy<OBSERVATION,ACTION> {

    @Builder
    public DQNPolicy(@NonNull OutputNeuralNet neuralNet, @NonNull ActionSpace<ACTION> actionSpace) {
    	super(neuralNet, actionSpace);
    }

    @SuppressWarnings("unchecked")
	@Override
    public ACTION nextAction(OBSERVATION obs) {
        INDArray output = neuralNet.output(obs).get(CommonOutputNames.QValues);
        return (ACTION) actionSpace.fromArray(output);
    }
}
