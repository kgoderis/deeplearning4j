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

import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.ObservationSource;
import org.deeplearning4j.rl4j.util.INDArrayHelper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

public class FeaturesBuilder<OBSERVATION extends Observation> {
    private final boolean isRecurrent;
    private int numChannels;
    private long[][] shapeByChannel;

    /**
     * @param isRecurrent True if the network is a recurrent one.
     */
    public FeaturesBuilder(boolean isRecurrent) {
        this.isRecurrent = isRecurrent;
    }

    /**
     * Build a {@link Features} instance
     * @param trainingBatch A container of observation list (see {@link ObservationSource})
     * @return
     */
    public Features build(List<? extends ObservationSource> trainingBatch) {
        return new Features(createFeatures(trainingBatch));
    }

    /**
     * Build a {@link Features} instance
     * @param trainingBatch An observation stream
     * @param size The total number of observations
     * @return
     */
    public Features build(Stream<OBSERVATION> trainingBatch, int size) {
        return new Features(createFeatures(trainingBatch, size));
    }

    private INDArray[] createFeatures(List<? extends ObservationSource> trainingBatch) {
        int size = trainingBatch.size();

        if(shapeByChannel == null) {
            setMetadata((OBSERVATION)trainingBatch.get(0).getObservation());
        }

        INDArray[] features;
        if(isRecurrent) {
            features = recurrentCreateFeaturesArray(size);
            INDArrayIndex[][] arrayIndicesByChannel = createChannelsArrayIndices((OBSERVATION)trainingBatch.get(0).getObservation());
            for(int observationIdx = 0; observationIdx < size; ++observationIdx) {
            	OBSERVATION observation = (OBSERVATION) trainingBatch.get(observationIdx).getObservation();
                recurrentAddObservation(features, observationIdx, observation, arrayIndicesByChannel);
            }
        } else {
            features = nonRecurrentCreateFeaturesArray(size);
            for(int observationIdx = 0; observationIdx < size; ++observationIdx) {
            	OBSERVATION observation = (OBSERVATION) trainingBatch.get(observationIdx).getObservation();
                nonRecurrentAddObservation(features, observationIdx, observation);
            }
        }

        return features;
    }

    private INDArray[] createFeatures(Stream<OBSERVATION> trainingBatch, int size) {
        INDArray[] features = null;
        if(isRecurrent) {
            Iterator<OBSERVATION> it = trainingBatch.iterator();
            int observationIdx = 0;
            INDArrayIndex[][] arrayIndicesByChannel = null;
            while (it.hasNext()) {
            	OBSERVATION observation = it.next();

                if(shapeByChannel == null) {
                    setMetadata(observation);
                }

                if(features == null) {
                    features = recurrentCreateFeaturesArray(size);
                    arrayIndicesByChannel = createChannelsArrayIndices(observation);
                }

                recurrentAddObservation(features, observationIdx++, observation, arrayIndicesByChannel);
            }
        } else {
            Iterator<OBSERVATION> it = trainingBatch.iterator();
            int observationIdx = 0;
            while (it.hasNext()) {
            	OBSERVATION observation = it.next();

                if(shapeByChannel == null) {
                    setMetadata(observation);
                }

                if(features == null) {
                    features = nonRecurrentCreateFeaturesArray(size);
                }

                nonRecurrentAddObservation(features, observationIdx++, observation);
            }
        }

        return features;
    }

    private void nonRecurrentAddObservation(INDArray[] features, int observationIdx, OBSERVATION observation) {
        for(int channelIdx = 0; channelIdx < numChannels; ++channelIdx) {
            features[channelIdx].putRow(observationIdx, observation.getChannelData(channelIdx));
        }
    }

    private void recurrentAddObservation(INDArray[] features, int observationIdx, OBSERVATION observation, INDArrayIndex[][] arrayIndicesByChannel) {
        INDArrayIndex[] arrayIndices;

        for (int channelIdx = 0; channelIdx < numChannels; channelIdx++) {
            INDArray channelData = observation.getChannelData(channelIdx);
            arrayIndices = arrayIndicesByChannel[channelIdx];
            arrayIndices[arrayIndices.length - 1] = NDArrayIndex.point(observationIdx);

            features[channelIdx].get(arrayIndices).assign(channelData);
        }
    }

    private INDArrayIndex[][] createChannelsArrayIndices(OBSERVATION observation) {
        INDArrayIndex[][] result = new INDArrayIndex[numChannels][];
        for (int channelIdx = 0; channelIdx < numChannels; channelIdx++) {
            INDArray channelData = observation.getChannelData(channelIdx);

            INDArrayIndex[] arrayIndices = new INDArrayIndex[channelData.shape().length];
            arrayIndices[0] = NDArrayIndex.point(0);
            for(int i = 1; i < arrayIndices.length - 1; ++i) {
                arrayIndices[i] = NDArrayIndex.all();
            }

            result[channelIdx] = arrayIndices;
        }

        return result;
    }

    private void setMetadata(OBSERVATION observation) {
        INDArray[] featuresData = observation.getChannelsData();
        numChannels = observation.numChannels();
        shapeByChannel = new long[numChannels][];
        for (int channelIdx = 0; channelIdx < featuresData.length; ++channelIdx) {
            shapeByChannel[channelIdx] = featuresData[channelIdx].shape();
        }
    }

    private INDArray[] nonRecurrentCreateFeaturesArray(int size) {
        INDArray[] features = new INDArray[numChannels];
        for (int channelIdx = 0; channelIdx < numChannels; ++channelIdx) {
            long[] observationShape = shapeByChannel[channelIdx];
            features[channelIdx] = nonRecurrentCreateFeatureArray(size, observationShape);
        }

        return features;
    }
    protected INDArray nonRecurrentCreateFeatureArray(int size, long[] observationShape) {
        return INDArrayHelper.createBatchForShape(size, observationShape);
    }

    private INDArray[] recurrentCreateFeaturesArray(int size) {
        INDArray[] features = new INDArray[numChannels];
        for (int channelIdx = 0; channelIdx < numChannels; ++channelIdx) {
            long[] observationShape = shapeByChannel[channelIdx];
            features[channelIdx] = INDArrayHelper.createRnnBatchForShape(size, observationShape);
        }

        return features;
    }
}
