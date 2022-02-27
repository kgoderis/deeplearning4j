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
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.agent.SteppingAgent;
import org.deeplearning4j.rl4j.agent.LearningAgent;
import org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic.AdvantageActorCritic;
import org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.BaseTransitionTDAlgorithm;
import org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning.NStepQLearning;
import org.deeplearning4j.rl4j.agent.learning.history.DefaultHistoryProcessor;
import org.deeplearning4j.rl4j.agent.learning.history.HistoryProcessor;
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
import org.deeplearning4j.rl4j.environment.observation.transform.operation.ArrayToINDArrayTransform;
import org.deeplearning4j.rl4j.experience.ReplayMemoryExperienceHandler;
import org.deeplearning4j.rl4j.experience.ObservationActionExperienceHandler;
import org.deeplearning4j.rl4j.mdp.gym.Gym;
import org.deeplearning4j.rl4j.mdp.gym.IntegerGym;
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
import org.nd4j.linalg.learning.config.RmsProp;

import lombok.NonNull;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NativeCartpoleExample {

    private static final String envID = "CartPole-v1";
	
    private static final boolean IS_ASYNC = false;
    private static final int NUM_THREADS = 8;
    private static final boolean USE_SEPARATE_NETWORKS = false;

    private static final int NUM_EPISODES = 3000;

    public static void main(String[] args) {

        Builder<Environment<IntegerAction>> environmentBuilder = () -> IntegerGym.integerBuilder().environment(envID).render(false).monitor(false).build();
        Builder<TransformProcess<Observation>> transformProcessBuilder = () -> TransformProcess.<Observation>builder()
                .transform("data", new ArrayToINDArrayTransform())
                .build("data");
        Builder<HistoryProcessor<Observation>> defaultHistoryProcessorBuilder = DefaultHistoryProcessor::new;

                Random rnd = Nd4j.getRandomFactory().getNewRandomInstance(123);

        List<AgentListener<Observation,IntegerAction>> listeners = new ArrayList<AgentListener<Observation,IntegerAction>>() {
            {
                add(new EpisodeScorePrinter());
            }
        };

        TrainableNeuralNet network = buildDQNNetwork();
        //TrainableNeuralNet network = buildActorCriticNetwork();

        //Builder<LearningAgent<Observation,IntegerAction>> builder = setupDoubleDQN(network, environmentBuilder, transformProcessBuilder, listeners, rnd);
        Builder<LearningAgent<Observation,IntegerAction>> builder = setupNStepQLearning(network, environmentBuilder, transformProcessBuilder, defaultHistoryProcessorBuilder,listeners, rnd);
        //Builder<LearningAgent<Observation,IntegerAction>> builder = setupAdvantageActorCritic(network, environmentBuilder, transformProcessBuilder, listeners, rnd);

        Trainer trainer;
        if(IS_ASYNC) {
            trainer = AsyncTrainer.<Observation,IntegerAction>builder()
                    .agentLearnerBuilder(builder)
                    .numThreads(NUM_THREADS)
                    .stoppingCondition(t -> t.getEpisodeCount() >= NUM_EPISODES)
                    .build();
        } else {
            trainer = SyncTrainer.<Observation,IntegerAction>builder()
                    .agentLearnerBuilder(builder)
                    .stoppingCondition(t -> t.getEpisodeCount() >= NUM_EPISODES)
                    .build();
        }

        long before = System.nanoTime();
        trainer.train();
        long after = System.nanoTime();

        System.out.println(String.format("Total time for %d episodes: %fms", NUM_EPISODES, (after - before) / 1e6));
        
        try {
			network.saveTo(new File( System.getProperty("user.home") + "/" + Constants.DATA_DIR + "/examples/nativecartpole.training"), false);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        //TODO : Evaluate the trained network
    }

    private static Builder<LearningAgent<Observation,IntegerAction>> setupDoubleDQN(TrainableNeuralNet network,Builder<Environment<IntegerAction>> environmentBuilder,
                                                                  Builder<TransformProcess> transformProcessBuilder,Builder<HistoryProcessor<Observation>> defaultHistoryProcessorBuilder,
                                                                  List<AgentListener<Observation,IntegerAction>> listeners,
                                                                  Random rnd) {

        DoubleDQNBuilder.Configuration configuration = DoubleDQNBuilder.Configuration.<Observation,IntegerAction>builder()
                .policyConfiguration(EpsGreedy.Configuration.builder()
                        .epsilonNbStep(3000)
                        .annealingStart(10)
                        .minEpsilon(0.1)
                        .build())
                .experienceHandlerConfiguration(ReplayMemoryExperienceHandler.Configuration.builder()
                        .maxReplayMemorySize(150000)
                        .batchSize(128)
                        .build())
                .neuralNetUpdaterConfiguration(NeuralNetUpdaterConfiguration.builder()
                        .targetUpdateFrequency(500)
                        .build())
                .updateAlgorithmConfiguration(BaseTransitionTDAlgorithm.Configuration.builder()
                        .gamma(0.99)
                        .build())
                .agentLearnerConfiguration(LearningAgent.Configuration.builder()
                        .maxEpisodeSteps(200)
                        .build())
                .agentLearnerListeners(listeners)
                .asynchronous(IS_ASYNC)
                .build();
        return new DoubleDQNBuilder(configuration, network, environmentBuilder, transformProcessBuilder,defaultHistoryProcessorBuilder, rnd);
    }

    private static Builder<LearningAgent<Observation,IntegerAction>> setupNStepQLearning(TrainableNeuralNet network,Builder<Environment<IntegerAction>> environmentBuilder,
                                                                       Builder<TransformProcess<Observation>> transformProcessBuilder,Builder<HistoryProcessor<Observation>> defaultHistoryProcessorBuilder,
                                                                       List<AgentListener<Observation,IntegerAction>> listeners,
                                                                       Random rnd) {

        NStepQLearningBuilder.Configuration configuration = NStepQLearningBuilder.Configuration.<Observation,IntegerAction>builder()
                .policyConfiguration(EpsGreedy.Configuration.builder()
                        .epsilonNbStep(75000  / (IS_ASYNC ? NUM_THREADS : 1))
                        .annealingStart(10)
                        .minEpsilon(0.1)
                        .build())
                .neuralNetUpdaterConfiguration(NeuralNetUpdaterConfiguration.builder()
                        .targetUpdateFrequency(500)
                        .build())
                .nstepQLearningConfiguration(NStepQLearning.Configuration.builder()
                        .build())
                .experienceHandlerConfiguration(ObservationActionExperienceHandler.Configuration.builder()
                        .batchSize(128)
                        .build())
                .agentLearnerConfiguration(LearningAgent.Configuration.builder()
                        .maxEpisodeSteps(200)
                        .build())
                .agentLearnerListeners(listeners)
                .asynchronous(IS_ASYNC)
                .build();
        return new NStepQLearningBuilder(configuration, network, environmentBuilder, transformProcessBuilder, defaultHistoryProcessorBuilder,rnd);
    }

    private static Builder<LearningAgent<Observation,IntegerAction>> setupAdvantageActorCritic(TrainableNeuralNet network,Builder<Environment<IntegerAction>> environmentBuilder,
                                                                             Builder<TransformProcess> transformProcessBuilder,Builder<HistoryProcessor<Observation>> defaultHistoryProcessorBuilder,
                                                                             List<AgentListener<Observation,IntegerAction>> listeners,
                                                                             Random rnd) {

        AdvantageActorCriticBuilder.Configuration<Observation,IntegerAction> configuration = AdvantageActorCriticBuilder.Configuration.<Observation,IntegerAction>builder()
                .neuralNetUpdaterConfiguration(NeuralNetUpdaterConfiguration.builder()
                        .build())
                .advantageActorCriticConfiguration(AdvantageActorCritic.Configuration.builder()
                        .gamma(0.99)
                        .build())
                .experienceHandlerConfiguration(ObservationActionExperienceHandler.Configuration.builder()
                        .batchSize(5)
                        .build())
                .agentLearnerConfiguration(LearningAgent.Configuration.builder()
                        .maxEpisodeSteps(200)
                        .build())
                .agentLearnerListeners(listeners)
                .asynchronous(IS_ASYNC)
                .build();
        return new AdvantageActorCriticBuilder(configuration, network, environmentBuilder, transformProcessBuilder, defaultHistoryProcessorBuilder,rnd);
    }

    private static TrainableNeuralNet buildDQNNetwork() {    	
        DQNDenseNetworkConfiguration netConf = DQNDenseNetworkConfiguration.builder()
                .updater(new RmsProp(0.000025))
                .l2(0)
                .numHiddenNodes(300)
                .numLayers(2)
                .build();
        NetworkFactory factory = new DQNFactoryStdDense(netConf);
        NeuralNetwork[] dqnNetwork = factory.build(new int[] { 4 }, 2);
        return QNetwork.builder().withNetwork((@NonNull MultiLayerNetwork) dqnNetwork[0]).build();
    }
    
    private static TrainableNeuralNet buildActorCriticNetwork() {
        ActorCriticDenseNetworkConfiguration netConf =  ActorCriticDenseNetworkConfiguration.builder()
                .updater(new Adam(1e-2))
                .l2(0)
                .numHiddenNodes(16)
                .numLayers(3)
                .build();

        if(USE_SEPARATE_NETWORKS) {
            ActorCriticFactorySeparateStdDense factory = new ActorCriticFactorySeparateStdDense(netConf);
            NeuralNetwork[] networks =  factory.build(new int[] { 4 }, 2);
            return ActorCriticNetwork.builder()
                    .withSeparateNetworks((MultiLayerNetwork)networks[0], (MultiLayerNetwork)networks[1])
                    .build();
        }

        ActorCriticFactoryCompGraphStdDense factory = new ActorCriticFactoryCompGraphStdDense(netConf);
        ComputationGraph[] networks = factory.build(new int[] { 4 }, 2);
        return ActorCriticNetwork.builder()
                    .withCombinedNetwork((ComputationGraph) networks[0])
                    .build();
    }
}