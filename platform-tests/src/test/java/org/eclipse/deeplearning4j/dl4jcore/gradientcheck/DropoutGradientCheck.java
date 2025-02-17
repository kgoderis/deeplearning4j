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

package org.eclipse.deeplearning4j.dl4jcore.gradientcheck;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.eclipse.deeplearning4j.dl4jcore.TestUtils;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.dropout.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.function.Consumer;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Slf4j
@Tag(TagNames.NDARRAY_ETL)
@Tag(TagNames.TRAINING)
@Tag(TagNames.DL4J_OLD_API)
@NativeTag
public class DropoutGradientCheck extends BaseDL4JTest {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.setDataType(DataType.DOUBLE);
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    @Test
    public void testDropoutGradient() {
        int minibatch = 3;

        for(boolean cnn : new boolean[]{false, true}) {
            for (int i = 0; i < 5; i++) {

                IDropout dropout;
                switch (i){
                    case 0:
                        dropout = new Dropout(0.6);
                        break;
                    case 1:
                        dropout = new AlphaDropout(0.6);
                        break;
                    case 2:
                        dropout = new GaussianDropout(0.1);    //0.01 rate -> stdev 0.1; 0.1 rate -> stdev 0.333
                        break;
                    case 3:
                        dropout = new GaussianNoise(0.3);
                        break;
                    case 4:
                        dropout = new SpatialDropout(0.6);
                        break;
                    default:
                        throw new RuntimeException();
                }

                if(!cnn && i == 4){
                    //Skip spatial dropout for dense layer (not applicable)
                    continue;
                }

                NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                        .dataType(DataType.DOUBLE)
                        .dist(new NormalDistribution(0,1))
                        .convolutionMode(ConvolutionMode.Same)
                        .dropOut(dropout)
                        .activation(Activation.TANH)
                        .updater(new NoOp())
                        .list();

                if(cnn){
                    builder.layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(2,2).nOut(2).build());
                    builder.layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(2,2).nOut(2).build());
                    builder.setInputType(InputType.convolutional(6,6,2));
                } else {
                    builder.layer(new DenseLayer.Builder().nOut(3).build());
                    builder.layer(new DenseLayer.Builder().nOut(3).build());
                    builder.setInputType(InputType.feedForward(6));
                }
                builder.layer(new OutputLayer.Builder().nOut(3).activation(Activation.SOFTMAX).lossFunction(LossFunction.MCXENT).build());

                MultiLayerConfiguration conf = builder.build();
                //Remove spatial dropout from output layer - can't be used for 2d input
                if(i == 4){
                   conf.getConf(2).getLayer().setIDropout(null);
                }

                MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                mln.init();

                String msg = (cnn ? "CNN" : "Dense") + ": " + dropout.getClass().getSimpleName();

                INDArray f;
                if(cnn){
                    f = Nd4j.rand(new int[]{minibatch, 2, 6, 6}).muli(10).subi(5);
                } else {
                    f = Nd4j.rand(minibatch, 6).muli(10).subi(5);
                }
                INDArray l = TestUtils.randomOneHot(minibatch, 3);

                log.info("*** Starting test: " + msg + " ***");
                boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, f, l, null, null,
                        false, -1, null, 12345);    //Last arg: ensures RNG is reset at each iter... otherwise will fail due to randomness!

                assertTrue(gradOK, msg);
                TestUtils.testModelSerialization(mln);
            }
        }
    }


    @Test
    public void testCompGraphMultiInput(){
        //Validate nets where the one output array is used as the input to multiple layers...
        Nd4j.getRandom().setSeed(12345);
        int mb = 3;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .dataType(DataType.DOUBLE)
                .dist(new NormalDistribution(0,1))
                .convolutionMode(ConvolutionMode.Same)
                .dropOut(new GaussianDropout(0.1))  //0.33 stdev. Gaussian dropout: out = in * N(1,stdev)
                .activation(Activation.TANH)
                .updater(new NoOp())
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(5).nOut(5).build(), "in")
                .addLayer("1", new DenseLayer.Builder().nIn(5).nOut(5).build(), "0")
                .addLayer("2", new DenseLayer.Builder().nIn(5).nOut(5).build(), "0")
                .addLayer("3", new DenseLayer.Builder().nIn(5).nOut(5).build(), "0")
                .addLayer("out", new OutputLayer.Builder().nIn(15).nOut(5).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunction.MCXENT).build(), "1", "2", "3")
                .setOutputs("out")
                .build();

        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        INDArray[] in = new INDArray[]{Nd4j.rand(mb, 5)};
        INDArray[] l = new INDArray[]{TestUtils.randomOneHot(mb, 5)};

        boolean gradOK = GradientCheckUtil.checkGradients(new GradientCheckUtil.GraphConfig().net(cg).inputs(in)
                .labels(l).callEachIter(new Consumer<ComputationGraph>() {
                    @Override
                    public void accept(ComputationGraph net) {
                        Nd4j.getRandom().setSeed(12345);
                    }
                }));

        assertTrue(gradOK);
    }

}
