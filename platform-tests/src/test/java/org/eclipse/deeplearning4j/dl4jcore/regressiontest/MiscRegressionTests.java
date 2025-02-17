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

package org.eclipse.deeplearning4j.dl4jcore.regressiontest;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
public class MiscRegressionTests extends BaseDL4JTest {

    @Test
    public void testFrozen() throws Exception {
        File f = new ClassPathResource("regression_testing/misc/legacy_frozen/configuration.json").getFile();
        String json = FileUtils.readFileToString(f, StandardCharsets.UTF_8.name());
        ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(json);

        int countFrozen = 0;
        for(Map.Entry<String,GraphVertex> e : conf.getVertices().entrySet()){
            GraphVertex gv = e.getValue();
            assertNotNull(gv);
            if(gv instanceof LayerVertex){
                LayerVertex lv = (LayerVertex)gv;
                Layer layer = lv.getLayerConf().getLayer();
                if(layer instanceof FrozenLayer)
                    countFrozen++;
            }
        }

        assertTrue(countFrozen > 0);
    }

    @Test
    public void testFrozenNewFormat(){
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new FrozenLayer(new DenseLayer.Builder().nIn(10).nOut(10).build()))
                .build();

        String json = configuration.toJson();
        MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(json);
        assertEquals(configuration, fromJson);
    }
}
