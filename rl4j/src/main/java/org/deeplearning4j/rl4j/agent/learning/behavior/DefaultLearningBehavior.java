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
package org.deeplearning4j.rl4j.agent.learning.behavior;

import lombok.Builder;
import lombok.NonNull;
import org.deeplearning4j.rl4j.agent.learning.update.UpdateRule;
import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;

@Builder
public class DefaultLearningBehavior<OBSERVATION extends Observation, ACTION extends Action, EXPERIENCE> implements LearningBehavior<OBSERVATION, ACTION> {

    private boolean hasBatchChanged = false;

    @NonNull
    private final ExperienceHandler<OBSERVATION, ACTION, EXPERIENCE> experienceHandler;

    @NonNull
    private final UpdateRule<EXPERIENCE> updateRule;

    @Override
    public void handleEpisodeStart() {
        experienceHandler.reset();
    }

    @Override
    public void handleNewExperience(OBSERVATION observation, ACTION action, double reward, boolean isTerminal) {
        experienceHandler.addExperience(observation, action, reward, isTerminal);
        if(experienceHandler.isTrainingBatchReady()) {
            handleBatch();
        }
    }

    @Override
    public void handleEpisodeEnd(OBSERVATION finalObservation) {
        experienceHandler.setFinalObservation(finalObservation);
        if(experienceHandler.isTrainingBatchReady()) {
            handleBatch();
        }
    }

    private void handleBatch() {
        updateRule.update(experienceHandler.generateTrainingBatch());
        hasBatchChanged = true;
    }

    /**
     * Will notify the update rule if a new training batch has been started
     */
    public void notifyBeforeStep() {
        if(hasBatchChanged) {
            updateRule.notifyNewBatchStarted();
            hasBatchChanged = false;
        }
    }
}
