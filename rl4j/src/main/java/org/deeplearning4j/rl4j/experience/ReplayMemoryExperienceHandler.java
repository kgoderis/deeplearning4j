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
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;

import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.nd4j.linalg.api.rng.Random;

import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;

import java.util.ArrayList;
import java.util.List;

@EqualsAndHashCode
public class ReplayMemoryExperienceHandler<OBSERVATION extends Observation, ACTION extends Action>
		implements ExperienceHandler<OBSERVATION, ACTION, ObservationActionRewardObservation<OBSERVATION,ACTION>> {
	private static final int DEFAULT_MAX_REPLAY_MEMORY_SIZE = 150000;
	private static final int DEFAULT_BATCH_SIZE = 32;
	private final int batchSize;
    final private Random random;
    private CircularFifoQueue<ObservationActionRewardObservation<OBSERVATION,ACTION>> storage;
	private ObservationActionRewardObservation<OBSERVATION, ACTION> pendingObservationActionRewardObservation;

	public ReplayMemoryExperienceHandler(Configuration configuration, Random random) {
        this.batchSize = configuration.batchSize;
        this.storage = new CircularFifoQueue<>(configuration.maxReplayMemorySize);
        this.random = random;
	}

	public void addExperience(OBSERVATION observation, ACTION action, double reward, boolean isTerminal) {
		setNextObservationOnPending(observation);
		pendingObservationActionRewardObservation = new ObservationActionRewardObservation<>(observation, action, reward, isTerminal);
	}

	public void setFinalObservation(OBSERVATION observation) {
		setNextObservationOnPending(observation);
		pendingObservationActionRewardObservation = null;
	}

	@Override
	public int getTrainingBatchSize() {
        int storageSize = storage.size();
        return Math.min(storageSize, batchSize);
   }

	@Override
	public boolean isTrainingBatchReady() {
		return getTrainingBatchSize() >= batchSize;
	}

	/**
	 * @return A batch of experience selected from the replay memory. The replay
	 *         memory is unchanged after the call.
	 */
	@Override
	public List<ObservationActionRewardObservation<OBSERVATION,ACTION>> generateTrainingBatch() {
		return generateTrainingBatch(batchSize);
	}
	
	public List<ObservationActionRewardObservation<OBSERVATION,ACTION>> generateTrainingBatch(int size) {
        ArrayList<ObservationActionRewardObservation<OBSERVATION,ACTION>> batch = new ArrayList<>(size);
        int storageSize = storage.size();
        int actualBatchSize = Math.min(storageSize, size);

        int[] actualIndex = new int[actualBatchSize];
        IntSet set = new IntOpenHashSet();
        for( int i=0; i<actualBatchSize; i++ ){
            int next = random.nextInt(storageSize);
            while(set.contains(next)){
                next = random.nextInt(storageSize);
            }
            set.add(next);
            actualIndex[i] = next;
        }

        for (int i = 0; i < actualBatchSize; i ++) {
            ObservationActionRewardObservation<OBSERVATION,ACTION> trans = storage.get(actualIndex[i]);
            batch.add(trans.dup());
        }

        return batch;	}

	@Override
	public void reset() {
		pendingObservationActionRewardObservation = null;
	}

	private void setNextObservationOnPending(Observation observation) {
		if (pendingObservationActionRewardObservation != null) {
			pendingObservationActionRewardObservation.setNextObservation(observation);
	        storage.add(pendingObservationActionRewardObservation);
		}
	}

	@SuperBuilder
	@Data
	public static class Configuration {
		/**
		 * The maximum replay memory size. Default is 150000
		 */
		@Builder.Default
		private int maxReplayMemorySize = DEFAULT_MAX_REPLAY_MEMORY_SIZE;

		/**
		 * The size of training batches. Default is 32.
		 */
		@Builder.Default
		private int batchSize = DEFAULT_BATCH_SIZE;
	}
}
