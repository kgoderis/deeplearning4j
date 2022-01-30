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
package org.deeplearning4j.rl4j.network;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.network.MultiLayerNetworkHandler.Configuration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonIgnore;

public class ComputationGraphHandler extends PersistentNetworkHandler {

	public static final String CONFIGURATION_JSON = "computationGraphHandler.json";

	private ComputationGraph model;

	@Getter
	private boolean recurrent;
	private ComputationGraphConfiguration configuration;
	private String[] labelNames;
	private String gradientName;
	private int inputFeatureIdx;
	private ChannelToNetworkInputMapper channelToNetworkInputMapper;

	/**
	 * @param model                       The {@link ComputationGraph} to use
	 *                                    internally.
	 * @param labelNames                  An array of the labels (in
	 *                                    {@link FeaturesLabels}) to use as the
	 *                                    network's input.
	 * @param gradientName                The name of the gradient (in
	 *                                    {@link Gradients}) to use as the network's
	 *                                    output.
	 * @param channelToNetworkInputMapper a {@link ChannelToNetworkInputMapper}
	 *                                    instance that map the network inputs to
	 *                                    the feature channels
	 */
	public ComputationGraphHandler(ComputationGraph model, String[] labelNames, String gradientName,
			ChannelToNetworkInputMapper channelToNetworkInputMapper) {
		this.model = model;
		recurrent = model.getOutputLayer(0) instanceof RnnOutputLayer;
		configuration = model.getConfiguration();
		this.labelNames = labelNames;
		this.gradientName = gradientName;

		this.inputFeatureIdx = 0;
		this.channelToNetworkInputMapper = channelToNetworkInputMapper;
	}

	/**
	 * @param model           The {@link ComputationGraph} to use internally.
	 * @param labelNames      An array of the labels (in {@link FeaturesLabels}) to
	 *                        use as the network's input.
	 * @param gradientName    The name of the gradient (in {@link Gradients}) to use
	 *                        as the network's output.
	 * @param inputFeatureIdx The channel index to use as the input of the model
	 */
	public ComputationGraphHandler(ComputationGraph model, String[] labelNames, String gradientName,
			int inputFeatureIdx) {
		this.model = model;
		recurrent = model.getOutputLayer(0) instanceof RnnOutputLayer;
		configuration = model.getConfiguration();
		this.labelNames = labelNames;
		this.gradientName = gradientName;

		this.inputFeatureIdx = inputFeatureIdx;
		this.channelToNetworkInputMapper = null;
	}

	@Override
	public void notifyGradientCalculation() {
		Iterable<TrainingListener> listeners = model.getListeners();

		if (listeners != null) {
			for (TrainingListener l : listeners) {
				l.onGradientCalculation(model);
			}
		}
	}

	@Override
	public void notifyIterationDone() {
		BaseNetwork.ModelCounters modelCounters = getModelCounters();
		Iterable<TrainingListener> listeners = model.getListeners();
		if (listeners != null) {
			for (TrainingListener l : listeners) {
				l.iterationDone(model, modelCounters.getIterationCount(), modelCounters.getEpochCount());
			}
		}
	}

	@Override
	public void performFit(FeaturesLabels featuresLabels) {
		model.fit(buildInputs(featuresLabels.getFeatures()), buildLabels(featuresLabels));
	}

	@Override
	public void performGradientsComputation(FeaturesLabels featuresLabels) {
		model.setInputs(buildInputs(featuresLabels.getFeatures()));
		model.setLabels(buildLabels(featuresLabels));
		model.computeGradientAndScore();
	}

	@Override
	public void fillGradientsResponse(Gradients gradients) {
		gradients.putGradient(gradientName, model.gradient());
	}

	private INDArray[] buildLabels(FeaturesLabels featuresLabels) {
		int numLabels = labelNames.length;
		INDArray[] result = new INDArray[numLabels];
		for (int i = 0; i < numLabels; ++i) {
			result[i] = featuresLabels.getLabels(labelNames[i]);
		}

		return result;
	}

	private BaseNetwork.ModelCounters getModelCounters() {
		return new BaseNetwork.ModelCounters(configuration.getIterationCount(), configuration.getEpochCount());
	}

	@Override
	public void applyGradient(Gradients gradients, long batchSize) {
		BaseNetwork.ModelCounters modelCounters = getModelCounters();
		int iterationCount = modelCounters.getIterationCount();
		Gradient gradient = gradients.getGradient(gradientName);
		model.getUpdater().update(gradient, iterationCount, modelCounters.getEpochCount(), (int) batchSize,
				LayerWorkspaceMgr.noWorkspaces());
		model.params().subi(gradient.gradient());
		configuration.setIterationCount(iterationCount + 1);
	}

	@Override
	public INDArray[] recurrentStepOutput(Observation observation) {
		return model.rnnTimeStep(buildInputs(observation));
	}

	@Override
	public INDArray[] stepOutput(Observation observation) {
		return model.output(buildInputs(observation));
	}

	@Override
	public INDArray[] batchOutput(Features features) {
		return model.output(buildInputs(features));
	}

	@Override
	public void resetState() {
		model.rnnClearPreviousState();
	}

	@Override
	public NetworkHandler clone() {
		if (channelToNetworkInputMapper != null) {
			return new ComputationGraphHandler(model.clone(), labelNames, gradientName, channelToNetworkInputMapper);
		}
		return new ComputationGraphHandler(model.clone(), labelNames, gradientName, inputFeatureIdx);
	}

	@Override
	public void copyFrom(NetworkHandler from) {
		model.setParams(((ComputationGraphHandler) from).model.params());
	}

	protected INDArray[] buildInputs(Observation observation) {
		return channelToNetworkInputMapper == null ? new INDArray[] { observation.getChannelData(inputFeatureIdx) }
				: channelToNetworkInputMapper.getNetworkInputs(observation);
	}

	protected INDArray[] buildInputs(Features features) {
		return channelToNetworkInputMapper == null ? new INDArray[] { features.get(inputFeatureIdx) }
				: channelToNetworkInputMapper.getNetworkInputs(features);
	}

	@Override
	String getConfigurationName() {
		return CONFIGURATION_JSON;
	}

	@Override
	public void saveTo(File file, boolean saveUpdater) throws IOException {
		if (!file.exists()) {
			file.getParentFile().mkdirs();
		}
		else {
			file.delete();
		}
		boolean success = file.createNewFile();
		if (success) {
			model.save(file, saveUpdater);
			saveConfigurationTo(file, new Configuration(recurrent, labelNames, gradientName, inputFeatureIdx,
					channelToNetworkInputMapper));
		}
	}

	@Override
	public void loadFrom(File file, boolean loadUpdater) throws IOException {

		Configuration conf = (Configuration) loadConfigurationFrom(file, Configuration.class);

		if (conf != null) {
			model = ComputationGraph.load(file, loadUpdater);

			this.recurrent = conf.recurrent;
			this.labelNames = conf.labelNames;
			this.gradientName = conf.gradientName;
			this.inputFeatureIdx = conf.inputFeatureIdx;
			this.channelToNetworkInputMapper = conf.channelToNetworkInputMapper;
		}
	}

	@SuppressWarnings("serial")
	@EqualsAndHashCode(callSuper = false)
	@Data
	public class Configuration extends PersistentNetworkHandler.Configuration {
		private final boolean recurrent;
		private final String[] labelNames;
		private final String gradientName;
		private final int inputFeatureIdx;
		private final ChannelToNetworkInputMapper channelToNetworkInputMapper;
	}
}
