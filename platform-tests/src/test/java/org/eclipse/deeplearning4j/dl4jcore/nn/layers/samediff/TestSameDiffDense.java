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

package org.eclipse.deeplearning4j.dl4jcore.nn.layers.samediff;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.eclipse.deeplearning4j.dl4jcore.TestUtils;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.eclipse.deeplearning4j.dl4jcore.nn.layers.samediff.testlayers.SameDiffDense;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@Tag(TagNames.DL4J_OLD_API)
public class TestSameDiffDense extends BaseDL4JTest {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    @Test
    public void testSameDiffDenseBasic() {
        int nIn = 3;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Map<String, INDArray> pt1 = net.getLayer(0).paramTable();
        assertNotNull(pt1);
        assertEquals(2, pt1.size());
        assertNotNull(pt1.get(DefaultParamInitializer.WEIGHT_KEY));
        assertNotNull(pt1.get(DefaultParamInitializer.BIAS_KEY));

        assertArrayEquals(new long[]{nIn, nOut}, pt1.get(DefaultParamInitializer.WEIGHT_KEY).shape());
        assertArrayEquals(new long[]{1, nOut}, pt1.get(DefaultParamInitializer.BIAS_KEY).shape());
    }

    @Test
    public void testSameDiffDenseForward() {
        for(WorkspaceMode wsm : new WorkspaceMode[]{WorkspaceMode.ENABLED, WorkspaceMode.NONE}) {
            for (int minibatch : new int[]{5, 1}) {
                int nIn = 3;
                int nOut = 4;

                Activation[] afns = new Activation[]{
                        Activation.TANH,
                        Activation.SIGMOID,
                        Activation.ELU,
                        Activation.IDENTITY,
                        Activation.SOFTPLUS,
                        Activation.SOFTSIGN,
                        Activation.CUBE,
                        Activation.HARDTANH,
                        Activation.RELU
                };

                for (Activation a : afns) {
                    log.info("Starting test - " + a + ", workspace = " + wsm);
                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .inferenceWorkspaceMode(wsm)
                            .trainingWorkspaceMode(wsm)
                            .list()
                            .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut)
                                    .activation(a)
                                    .build())
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    assertNotNull(net.paramTable());

                    MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                            .list()
                            .layer(new DenseLayer.Builder().activation(a).nIn(nIn).nOut(nOut).build())
                            .build();

                    MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
                    net2.init();

                    net.params().assign(net2.params());

                    //Check params:
                    assertEquals(net2.params(), net.params());
                    Map<String, INDArray> params1 = net.paramTable();
                    Map<String, INDArray> params2 = net2.paramTable();
                    assertEquals(params2, params1);

                    INDArray in = Nd4j.rand(minibatch, nIn);
                    INDArray out = net.output(in);
                    INDArray outExp = net2.output(in);

                    assertEquals(outExp, out);

                    //Also check serialization:
                    MultiLayerNetwork netLoaded = TestUtils.testModelSerialization(net);
                    INDArray outLoaded = netLoaded.output(in);

                    assertEquals(outExp, outLoaded);

                    //Sanity check on different minibatch sizes:
                    INDArray newIn = Nd4j.vstack(in, in);
                    INDArray outMbsd = net.output(newIn);
                    INDArray outMb = net2.output(newIn);
                    assertEquals(outMb, outMbsd);
                }
            }
        }
    }

    @Test
    public void testSameDiffDenseForwardMultiLayer() {
        for(WorkspaceMode wsm : new WorkspaceMode[]{WorkspaceMode.ENABLED, WorkspaceMode.NONE}) {
            for (int minibatch : new int[]{5, 1}) {
                int nIn = 3;
                int nOut = 4;

                Activation[] afns = new Activation[]{
                        Activation.TANH,
                        Activation.SIGMOID,
                        Activation.ELU,
                        Activation.IDENTITY,
                        Activation.SOFTPLUS,
                        Activation.SOFTSIGN,
                        Activation.CUBE,    //https://github.com/deeplearning4j/nd4j/issues/2426
                        Activation.HARDTANH,
                        Activation.RELU      //JVM crash
                };

                for (Activation a : afns) {
                    log.info("Starting test - " + a + " - workspace=" + wsm);
                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .seed(12345)
                            .list()
                            .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut)
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(a).build())
                            .layer(new SameDiffDense.Builder().nIn(nOut).nOut(nOut)
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(a).build())
                            .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut)
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(a).build())
                            .validateOutputLayerConfig(false)
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    assertNotNull(net.paramTable());

                    MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                            .seed(12345)
                            .weightInit(WeightInit.XAVIER)
                            .list()
                            .layer(new DenseLayer.Builder().activation(a).nIn(nIn).nOut(nOut).build())
                            .layer(new DenseLayer.Builder().activation(a).nIn(nOut).nOut(nOut).build())
                            .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut)
                                    .activation(a).build())
                            .validateOutputLayerConfig(false)
                            .build();

                    MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
                    net2.init();

                    assertEquals(net2.params(), net.params());

                    //Check params:
                    assertEquals(net2.params(), net.params());
                    Map<String, INDArray> params1 = net.paramTable();
                    Map<String, INDArray> params2 = net2.paramTable();
                    assertEquals(params2, params1);

                    INDArray in = Nd4j.rand(minibatch, nIn);
                    INDArray out = net.output(in);
                    INDArray outExp = net2.output(in);

                    assertEquals(outExp, out);

                    //Also check serialization:
                    MultiLayerNetwork netLoaded = TestUtils.testModelSerialization(net);
                    INDArray outLoaded = netLoaded.output(in);

                    assertEquals(outExp, outLoaded);


                    //Sanity check different minibatch sizes
                    in = Nd4j.rand(2 * minibatch, nIn);
                    out = net.output(in);
                    outExp = net2.output(in);
                    assertEquals(outExp, out);
                }
            }
        }
    }

    @Test
    public void testSameDiffDenseBackward() {
        int nIn = 3;
        int nOut = 4;

        for (boolean workspaces : new boolean[]{false, true}) {

            for (int minibatch : new int[]{5, 1}) {

                Activation[] afns = new Activation[]{
                        Activation.TANH,
                        Activation.SIGMOID,
                        Activation.ELU,
                        Activation.IDENTITY,
                        Activation.SOFTPLUS,
                        Activation.SOFTSIGN,
                        Activation.HARDTANH,
                        Activation.CUBE,
                        Activation.RELU
                };

                for (Activation a : afns) {
                    log.info("Starting test - " + a + " - minibatch " + minibatch + ", workspaces: " + workspaces);
                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                            .trainingWorkspaceMode(workspaces ? WorkspaceMode.ENABLED : WorkspaceMode.NONE)
                            .inferenceWorkspaceMode(workspaces ? WorkspaceMode.ENABLED : WorkspaceMode.NONE)
                            .list()
                            .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut)
                                    .activation(a)
                                    .build())
                            .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut).activation(Activation.SOFTMAX)
                                    .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                            .build();

                    MultiLayerNetwork netSD = new MultiLayerNetwork(conf);
                    netSD.init();

                    MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                            .list()
                            .layer(new DenseLayer.Builder().activation(a).nIn(nIn).nOut(nOut).build())
                            .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut).activation(Activation.SOFTMAX)
                                    .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                            .build();

                    MultiLayerNetwork netStandard = new MultiLayerNetwork(conf2);
                    netStandard.init();

                    netSD.params().assign(netStandard.params());

                    //Check params:
                    assertEquals(netStandard.params(), netSD.params());
                    assertEquals(netStandard.paramTable(), netSD.paramTable());

                    INDArray in = Nd4j.rand(minibatch, nIn);
                    INDArray l = TestUtils.randomOneHot(minibatch, nOut, 12345);
                    netSD.setInput(in);
                    netStandard.setInput(in);
                    netSD.setLabels(l);
                    netStandard.setLabels(l);

                    netSD.computeGradientAndScore();
                    netStandard.computeGradientAndScore();

                    Gradient gSD = netSD.gradient();
                    Gradient gStd = netStandard.gradient();

                    Map<String, INDArray> m1 = gSD.gradientForVariable();
                    Map<String, INDArray> m2 = gStd.gradientForVariable();

                    assertEquals(m2.keySet(), m1.keySet());

                    for (String s : m1.keySet()) {
                        INDArray i1 = m1.get(s);
                        INDArray i2 = m2.get(s);

                        assertEquals(i2, i1, s);
                    }

                    assertEquals(gStd.gradient(), gSD.gradient());

                    //Sanity check: different minibatch size
                    in = Nd4j.rand(2 * minibatch, nIn);
                    l = TestUtils.randomOneHot(2 * minibatch, nOut, 12345);
                    netSD.setInput(in);
                    netStandard.setInput(in);
                    netSD.setLabels(l);
                    netStandard.setLabels(l);

                    netSD.computeGradientAndScore();
//                    netStandard.computeGradientAndScore();
//                    assertEquals(netStandard.gradient().gradient(), netSD.gradient().gradient());

                    //Sanity check on different minibatch sizes:
                    INDArray newIn = Nd4j.vstack(in, in);
                    INDArray outMbsd = netSD.output(newIn);
                    INDArray outMb = netStandard.output(newIn);
                    assertEquals(outMb, outMbsd);
                }
            }
        }
    }

    @Test
    public void testSameDiffDenseTraining() {
        Nd4j.getRandom().setSeed(12345);
        Nd4j.EPS_THRESHOLD = 1e-4;
        int nIn = 4;
        int nOut = 3;

        for(WorkspaceMode wsm : new WorkspaceMode[]{WorkspaceMode.ENABLED, WorkspaceMode.NONE}) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .dataType(DataType.DOUBLE)
                    .trainingWorkspaceMode(wsm)
                    .inferenceWorkspaceMode(wsm)
                    .updater(new Adam(1e-3))
                    .list()
                    .layer(new SameDiffDense.Builder().nIn(nIn).nOut(5).activation(Activation.TANH).build())
                    .layer(new SameDiffDense.Builder().nIn(5).nOut(5).activation(Activation.TANH).build())
                    .layer(new OutputLayer.Builder().nIn(5).nOut(nOut).activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                    .build();

            MultiLayerNetwork netSD = new MultiLayerNetwork(conf);
            netSD.init();

            MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .dataType(DataType.DOUBLE)
                    .updater(new Adam(1e-3))
                    .list()
                    .layer(new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(5).build())
                    .layer(new DenseLayer.Builder().activation(Activation.TANH).nIn(5).nOut(5).build())
                    .layer(new OutputLayer.Builder().nIn(5).nOut(nOut).activation(Activation.SOFTMAX)
                            .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                    .build();

            MultiLayerNetwork netStandard = new MultiLayerNetwork(conf2);
            netStandard.init();

            netSD.params().assign(netStandard.params());

            //Check params:
            assertEquals(netStandard.params(), netSD.params());
            assertEquals(netStandard.paramTable(), netSD.paramTable());

            DataSetIterator iter = new IrisDataSetIterator(150, 150);
            DataSet ds = iter.next();

            INDArray outSD = netSD.output(ds.getFeatures());
            INDArray outStd = netStandard.output(ds.getFeatures());

            assertEquals(outStd, outSD);

            for (int i = 0; i < 3; i++) {
                netSD.fit(ds);
                netStandard.fit(ds);
                String s = String.valueOf(i);
                INDArray netStandardGrad = netStandard.getFlattenedGradients();
                INDArray netSDGrad = netSD.getFlattenedGradients();
                assertEquals(netStandardGrad, netSDGrad, s);
                assertEquals(netStandard.params(), netSD.params(), s);
                assertEquals(netStandard.getUpdater().getStateViewArray(), netSD.getUpdater().getStateViewArray(), s);
            }

            //Sanity check on different minibatch sizes:
            INDArray newIn = Nd4j.vstack(ds.getFeatures(), ds.getFeatures()).castTo(DataType.DOUBLE);
            INDArray outMbsd = netSD.output(newIn).castTo(DataType.DOUBLE);
            INDArray outMb = netStandard.output(newIn).castTo(DataType.DOUBLE);
            assertEquals(outMb, outMbsd);
        }
    }

    @Test
    public void gradientCheck() {
        int nIn = 4;
        int nOut = 4;

        for (boolean workspaces : new boolean[]{true, false}) {
            for (Activation a : new Activation[]{Activation.TANH, Activation.IDENTITY}) {

                String msg = "workspaces: " + workspaces + ", " + a;
                Nd4j.getRandom().setSeed(12345);

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .dataType(DataType.DOUBLE)
                        .seed(12345)
                        .updater(new NoOp())
                        .trainingWorkspaceMode(workspaces ? WorkspaceMode.ENABLED : WorkspaceMode.NONE)
                        .inferenceWorkspaceMode(workspaces ? WorkspaceMode.ENABLED : WorkspaceMode.NONE)
                        .list()
                        .layer(new SameDiffDense.Builder().nIn(nIn).nOut(nOut).activation(a).build())
                        .layer(new SameDiffDense.Builder().nIn(nOut).nOut(nOut).activation(a).build())
                        .layer(new OutputLayer.Builder().nIn(nOut).nOut(nOut).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                INDArray f = Nd4j.rand(3, nIn);
                INDArray l = TestUtils.randomOneHot(3, nOut);

                log.info("Starting: " + msg);
                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, f, l);

                assertTrue(gradOK, msg);

                TestUtils.testModelSerialization(net);

                //Sanity check on different minibatch sizes:
                INDArray newIn = Nd4j.vstack(f, f);
                net.output(newIn);
            }
        }
    }
}
