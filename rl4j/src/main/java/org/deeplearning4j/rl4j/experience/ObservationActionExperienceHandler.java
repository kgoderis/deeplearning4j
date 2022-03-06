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

import lombok.Builder;
import lombok.Data;
import lombok.experimental.SuperBuilder;

import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.observation.Observation;

import java.util.ArrayList;
import java.util.List;

public class ObservationActionExperienceHandler<OBSERVATION extends Observation, ACTION extends Action> implements ExperienceHandler<OBSERVATION, ACTION, ObservationActionReward<OBSERVATION,ACTION>> {
    private static final int DEFAULT_BATCH_SIZE = 8;

    private final int batchSize;

    private boolean isFinalObservationSet;

    public ObservationActionExperienceHandler(Configuration configuration) {
        this.batchSize = configuration.getBatchSize();
    }

    private List<ObservationActionReward<OBSERVATION,ACTION>> observationActionRewards = new ArrayList<>();

    public void setFinalObservation(OBSERVATION observation) {
        isFinalObservationSet = true;
    }

    public void addExperience(OBSERVATION observation, ACTION action, double reward, boolean isTerminal) {
        observationActionRewards.add(new ObservationActionReward<OBSERVATION,ACTION>(observation, action, reward, isTerminal));
    }

    @Override
    public int getTrainingBatchSize() {
        return observationActionRewards.size();
    }

    @Override
    public boolean isTrainingBatchReady() {
        return observationActionRewards.size() >= batchSize
                || (isFinalObservationSet && observationActionRewards.size() > 0);
    }

    /**
     * The elements are returned in the historical order (i.e. in the order they happened)
     * Note: the experience store is cleared after calling this method.
     *
     * @return The list of experience elements
     */
    @Override
    public List<ObservationActionReward<OBSERVATION,ACTION>> generateTrainingBatch() {
        List<ObservationActionReward<OBSERVATION,ACTION>> result = observationActionRewards;
        observationActionRewards = new ArrayList<>();

        return result;
    }

    @Override
    public void reset() {
        observationActionRewards = new ArrayList<>();
        isFinalObservationSet = false;
    }

    @SuperBuilder
    @Data
    public static class Configuration {
        /**
         * The default training batch size. Default is 8.
         */
        @Builder.Default
        private int batchSize = DEFAULT_BATCH_SIZE;
    }
}
