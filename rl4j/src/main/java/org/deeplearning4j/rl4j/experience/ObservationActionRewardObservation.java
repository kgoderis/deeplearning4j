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

package org.deeplearning4j.rl4j.experience;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;

import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.ObservationSource;

@Data
public class ObservationActionRewardObservation<OBSERVATION extends Observation, ACTION extends Action> implements ObservationSource {

    @Getter
    private final OBSERVATION observation;

    @Getter
    private final ACTION action;
    
    @Getter
    private final double reward;
    
    @Getter
    private final boolean isTerminal;

    @Getter @Setter
    Observation nextObservation;

    public ObservationActionRewardObservation(OBSERVATION observation, ACTION action, double reward, boolean isTerminal) {
        this.observation = observation;
        this.action = action;
        this.reward = reward;
        this.isTerminal = isTerminal;
        this.nextObservation = null;
    }

    private ObservationActionRewardObservation(OBSERVATION observation, ACTION action, double reward, boolean isTerminal, Observation nextObservation) {
        this.observation = observation;
        this.action = action;
        this.reward = reward;
        this.isTerminal = isTerminal;
        this.nextObservation = nextObservation;
    }

    /**
     * @return a duplicate of this instance
     */
    public ObservationActionRewardObservation<OBSERVATION,ACTION> dup() {
        OBSERVATION dupObservation = (OBSERVATION) observation.dup();
        OBSERVATION nextObs = (OBSERVATION) nextObservation.dup();

        return new ObservationActionRewardObservation<OBSERVATION,ACTION>(dupObservation, action, reward, isTerminal, nextObs);
    }
}
