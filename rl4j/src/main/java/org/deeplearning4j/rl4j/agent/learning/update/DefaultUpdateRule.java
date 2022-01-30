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
package org.deeplearning4j.rl4j.agent.learning.update;

import lombok.Getter;
import lombok.NonNull;
import org.deeplearning4j.rl4j.agent.learning.algorithm.UpdateAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdater;

import java.util.List;

public class DefaultUpdateRule<RESULT, EXPERIENCE> implements UpdateRule<EXPERIENCE> {

    private final NeuralNetUpdater<RESULT> updater;

    private final UpdateAlgorithm<RESULT, EXPERIENCE> updateAlgorithm;

    @Getter
    private int updateCount = 0;

    public DefaultUpdateRule(@NonNull UpdateAlgorithm<RESULT, EXPERIENCE> updateAlgorithm,
                      @NonNull NeuralNetUpdater<RESULT> updater) {
        this.updateAlgorithm = updateAlgorithm;
        this.updater = updater;
    }

    @Override
    public void update(List<EXPERIENCE> trainingBatch) {
    	RESULT featuresLabels = updateAlgorithm.compute(trainingBatch);
        updater.update(featuresLabels);
        ++updateCount;
    }

    @Override
    public void notifyNewBatchStarted() {
        updater.synchronizeCurrent();
    }

}
