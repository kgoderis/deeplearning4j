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

package org.deeplearning4j.rl4j.agent.learning.history;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.experimental.SuperBuilder;

import java.io.File;
import java.util.List;

import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.environment.observation.Observation;

public interface HistoryProcessor<OBSERVATION extends Observation> {
//
//    Configuration getConf();

    /** Returns compressed arrays, which must be rescaled based
     *  on the value returned by {@link #getScale()}. */
//    INDArray[] getHistory();
    
    List<OBSERVATION> getHistory();
    
    void record(OBSERVATION Observation);
    
    void add(OBSERVATION Observation);

//    void record(INDArray image);
//
//    void add(INDArray image);

    void startMonitor(File file);

    void stopMonitor();

    boolean isMonitoring();

//    /** Returns the scale of the arrays returned by {@link #getHistory()}, typically 255. */
//    double getScale();
//
//    @AllArgsConstructor
//    @SuperBuilder
//    @Data
//    public static class Configuration {
//    }
    
    
}
