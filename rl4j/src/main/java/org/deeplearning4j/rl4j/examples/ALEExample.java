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

package org.deeplearning4j.rl4j.examples;

import org.apache.commons.lang3.builder.Builder;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.agent.SteppingAgent;
import org.deeplearning4j.rl4j.agent.LearningAgent;
import org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic.AdvantageActorCritic;
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.BaseTransitionTDAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning.NStepQLearning;
import org.deeplearning4j.rl4j.agent.learning.history.VideoHistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.agent.listener.utils.EpisodeScorePrinter;
import org.deeplearning4j.rl4j.builder.AdvantageActorCriticBuilder;
import org.deeplearning4j.rl4j.builder.DoubleDQNBuilder;
import org.deeplearning4j.rl4j.builder.NStepQLearningBuilder;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.deeplearning4j.rl4j.environment.action.space.IntegerActionSpace;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.environment.observation.transform.filter.UniformSkippingFilter;
import org.deeplearning4j.rl4j.environment.observation.transform.legacy.ImageWritableToINDArrayTransform;
import org.deeplearning4j.rl4j.environment.observation.transform.legacy.ObservationToImageWritableTransform;
import org.deeplearning4j.rl4j.environment.observation.transform.operation.ArrayToINDArrayTransform;
import org.deeplearning4j.rl4j.environment.observation.transform.operation.HistoryMergeTransform;
import org.deeplearning4j.rl4j.environment.observation.transform.operation.INDArrayToObservationTransform;
import org.deeplearning4j.rl4j.environment.observation.transform.operation.ScreenToINDArrayTransform;
import org.deeplearning4j.rl4j.environment.observation.transform.operation.SimpleNormalizationTransform;
import org.deeplearning4j.rl4j.experience.ReplayMemoryExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionExperienceHandler;
import org.deeplearning4j.rl4j.mdp.Cartpole;
import org.deeplearning4j.rl4j.mdp.ale.ALE;
import org.deeplearning4j.rl4j.mdp.gym.Gym;
import org.deeplearning4j.rl4j.network.ActorCriticNetwork;
import org.deeplearning4j.rl4j.network.TrainableNeuralNet;
import org.deeplearning4j.rl4j.network.QNetwork;
import org.deeplearning4j.rl4j.network.configuration.ActorCriticDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.network.factory.ActorCriticFactoryCompGraphStdDense;
import org.deeplearning4j.rl4j.network.factory.ActorCriticFactorySeparateStdDense;
import org.deeplearning4j.rl4j.network.factory.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.factory.NetworkFactory;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.trainer.AsyncTrainer;
import org.deeplearning4j.rl4j.trainer.Trainer;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.trainer.SyncTrainer;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import lombok.NonNull;

import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.util.ArrayList;
import java.util.List;

public class ALEExample {

	private static final String romFile = System.getProperty("user.home") + "/" + Constants.DATA_DIR + "/rom/pong.bin";

	private static final boolean IS_ASYNC = false;
	private static final int NUM_THREADS = 8;
	private static final boolean USE_SEPARATE_NETWORKS = false;

	private static final int NUM_EPISODES = 3000;

	public static void main(String[] args) {

		// Build one instance of the environment, just to get config parameters
		ALE tempALE = ALE.builder().rom(romFile).render(false).build();
		long[] observationShape = tempALE.getObservationShape();

		int historyLength = 4;
		int rescaledWidth = 84;
		int rescaledHeight = 110;
		int croppingWidth = 84;
		int croppingHeight = 84;
		int offsetX = 0;
		int offsetY = 0;
		int skipFrame = 4;

		int height = (int) observationShape[1];
		int width = (int) observationShape[2];

		int cropBottom = height - croppingHeight;
		int cropRight = width - croppingWidth;

		Builder<Environment<IntegerAction>> environmentBuilder = () -> ALE.builder().rom(romFile).render(true).build();
		Builder<TransformProcess<Observation>> transformProcessBuilder = () -> TransformProcess.<Observation>builder()
				.filter(new UniformSkippingFilter(skipFrame))
				.transform("data", new ObservationToImageWritableTransform())
				.transform("data",
						new MultiImageTransform(new CropImageTransform(offsetY, offsetX, cropBottom, cropRight),
								new ResizeImageTransform(rescaledWidth, rescaledHeight),
								new ColorConversionTransform(COLOR_BGR2GRAY)
						// new ShowImageTransform("crop + resize + greyscale")
						)).transform("data", new ImageWritableToINDArrayTransform())
				.transform("data", new SimpleNormalizationTransform(0.0, 255.0))
				.transform("data", HistoryMergeTransform.builder().isFirstDimenstionBatch(true).build(historyLength))
				.transform("data", new INDArrayToObservationTransform())
				.build("data");
		Builder<VideoHistoryProcessor> videoHistoryProcessorBuilder = () -> VideoHistoryProcessor.builder()
				.historyLength(historyLength).rescaledHeight(rescaledHeight).rescaledWidth(rescaledWidth)
				.croppingHeight(croppingHeight).croppingWidth(croppingWidth).offsetX(offsetX).offsetY(offsetY)
				.skipFrame(skipFrame).build();

		Random rnd = Nd4j.getRandomFactory().getNewRandomInstance(123);

		List<AgentListener<Observation, IntegerAction>> listeners = new ArrayList<AgentListener<Observation, IntegerAction>>() {
			{
				add(new EpisodeScorePrinter());
			}
		};

        TrainableNeuralNet network = buildDQNNetwork();
        //TrainableNeuralNet network = buildActorCriticNetwork();
        
		// Builder<LearningAgent<Observation,IntegerAction>> builder =
		// setupDoubleDQN(network,environmentBuilder, transformProcessBuilder, videoHistoryProcessorBuilder,listeners, rnd);
		Builder<LearningAgent<Observation, IntegerAction>> builder = setupNStepQLearning(network,environmentBuilder,
				transformProcessBuilder, videoHistoryProcessorBuilder,listeners, rnd);
		// Builder<LearningAgent<Observation,IntegerAction>> builder =
		// setupAdvantageActorCritic(network,environmentBuilder, transformProcessBuilder,videoHistoryProcessorBuilder,
		// listeners, rnd);

		Trainer trainer;
		if (IS_ASYNC) {
			trainer = AsyncTrainer.<Observation, IntegerAction>builder().agentLearnerBuilder(builder)
					.numThreads(NUM_THREADS).stoppingCondition(t -> t.getEpisodeCount() >= NUM_EPISODES).build();
		} else {
			trainer = SyncTrainer.<Observation, IntegerAction>builder().agentLearnerBuilder(builder)
					.stoppingCondition(t -> t.getEpisodeCount() >= NUM_EPISODES).build();
		}

		long before = System.nanoTime();
		trainer.train();
		long after = System.nanoTime();

		System.out.println(String.format("Total time for %d episodes: %fms", NUM_EPISODES, (after - before) / 1e6));
		
        try {
			network.saveTo(new File( System.getProperty("user.home") + "/" + Constants.DATA_DIR + "/examples/ALE.training"), false);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private static Builder<LearningAgent<Observation, IntegerAction>> setupDoubleDQN(TrainableNeuralNet network,
			Builder<Environment<IntegerAction>> environmentBuilder, Builder<TransformProcess> transformProcessBuilder,
			Builder<VideoHistoryProcessor> videoHistoryProcessorBuilder,
			List<AgentListener<Observation, IntegerAction>> listeners, Random rnd) {

		DoubleDQNBuilder.Configuration configuration = DoubleDQNBuilder.Configuration
				.<Observation, IntegerAction>builder()
				.policyConfiguration(EpsGreedy.Configuration.builder().epsilonNbStep(3000).minEpsilon(0.1).build())
				.experienceHandlerConfiguration(ReplayMemoryExperienceHandler.Configuration.builder()
						.maxReplayMemorySize(10000).batchSize(64).build())
				.neuralNetUpdaterConfiguration(
						NeuralNetUpdaterConfiguration.builder().targetUpdateFrequency(50).build())
				.updateAlgorithmConfiguration(BaseTransitionTDAlgorithm.Configuration.builder().gamma(0.99).build())
				.agentLearnerConfiguration(LearningAgent.Configuration.builder().maxEpisodeSteps(200).build())
				.agentLearnerListeners(listeners).asynchronous(IS_ASYNC).build();
		return new DoubleDQNBuilder(configuration, network, environmentBuilder, transformProcessBuilder,
				videoHistoryProcessorBuilder, rnd);
	}

	private static Builder<LearningAgent<Observation, IntegerAction>> setupNStepQLearning(TrainableNeuralNet network,
			Builder<Environment<IntegerAction>> environmentBuilder,
			Builder<TransformProcess<Observation>> transformProcessBuilder,
			Builder<VideoHistoryProcessor> videoHistoryProcessorBuilder,
			List<AgentListener<Observation, IntegerAction>> listeners, Random rnd) {

		NStepQLearningBuilder.Configuration configuration = NStepQLearningBuilder.Configuration
				.<Observation, IntegerAction>builder()
				.policyConfiguration(EpsGreedy.Configuration.builder()
						.epsilonNbStep(75000 / (IS_ASYNC ? NUM_THREADS : 1)).minEpsilon(0.1).build())
				.neuralNetUpdaterConfiguration(
						NeuralNetUpdaterConfiguration.builder().targetUpdateFrequency(50).build())
				.nstepQLearningConfiguration(NStepQLearning.Configuration.builder().build())
				.experienceHandlerConfiguration(
						StateActionExperienceHandler.Configuration.builder().batchSize(5).build())
				.agentLearnerConfiguration(LearningAgent.Configuration.builder().maxEpisodeSteps(10000).build())
				.agentLearnerListeners(listeners).asynchronous(IS_ASYNC).build();
		return new NStepQLearningBuilder(configuration, network, environmentBuilder, transformProcessBuilder,
				videoHistoryProcessorBuilder, rnd);
	}

	private static Builder<LearningAgent<Observation, IntegerAction>> setupAdvantageActorCritic(TrainableNeuralNet network,
			Builder<Environment<IntegerAction>> environmentBuilder, Builder<TransformProcess> transformProcessBuilder,
			Builder<VideoHistoryProcessor> videoHistoryProcessorBuilder,
			List<AgentListener<Observation, IntegerAction>> listeners, Random rnd) {

		AdvantageActorCriticBuilder.Configuration<Observation, IntegerAction> configuration = AdvantageActorCriticBuilder.Configuration
				.<Observation, IntegerAction>builder()
				.neuralNetUpdaterConfiguration(NeuralNetUpdaterConfiguration.builder().build())
				.advantageActorCriticConfiguration(AdvantageActorCritic.Configuration.builder().gamma(0.99).build())
				.experienceHandlerConfiguration(
						StateActionExperienceHandler.Configuration.builder().batchSize(5).build())
				.agentLearnerConfiguration(LearningAgent.Configuration.builder().maxEpisodeSteps(10000).build())
				.agentLearnerListeners(listeners).asynchronous(IS_ASYNC).build();
		return new AdvantageActorCriticBuilder(configuration, network, environmentBuilder, transformProcessBuilder,
				videoHistoryProcessorBuilder, rnd);
	}

	private static TrainableNeuralNet buildDQNNetwork() {
		DQNDenseNetworkConfiguration netConf = DQNDenseNetworkConfiguration.builder().updater(new Adam())
				.numHiddenNodes(40).numLayers(2).build();
		NetworkFactory factory = new DQNFactoryStdDense(netConf);
		NeuralNetwork[] dqnNetwork = factory.build(new int[] { 4 }, 2);
		return QNetwork.builder().withNetwork((@NonNull MultiLayerNetwork) dqnNetwork[0]).build();
	}

	private static TrainableNeuralNet buildActorCriticNetwork() {
		ActorCriticDenseNetworkConfiguration netConf = ActorCriticDenseNetworkConfiguration.builder()
				.updater(new Adam(0.00025)).l2(0).build();

		if (USE_SEPARATE_NETWORKS) {
			ActorCriticFactorySeparateStdDense factory = new ActorCriticFactorySeparateStdDense(netConf);
			NeuralNetwork[] networks = factory.build(new int[] { 4 }, 2);
			return ActorCriticNetwork.builder()
					.withSeparateNetworks((MultiLayerNetwork) networks[0], (MultiLayerNetwork) networks[1]).build();
		}

		ActorCriticFactoryCompGraphStdDense factory = new ActorCriticFactoryCompGraphStdDense(netConf);
		ComputationGraph[] networks = factory.build(new int[] { 4 }, 2);
		return ActorCriticNetwork.builder().withCombinedNetwork((ComputationGraph) networks[0]).build();
	}
}