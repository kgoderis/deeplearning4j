package org.deeplearning4j.rl4j.agent.listener.utils;

import org.deeplearning4j.rl4j.agent.SteppingAgent;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.deeplearning4j.rl4j.environment.observation.Observation;

public class EpisodeScorePrinter implements AgentListener<Observation, IntegerAction> {

	private int episodeCount;

	@Override
	public ListenerResponse onBeforeEpisode(SteppingAgent<Observation, IntegerAction> agent) {
		return ListenerResponse.CONTINUE;
	}

	@Override
	public ListenerResponse onBeforeStep(SteppingAgent<Observation, IntegerAction> agent, Observation observation,
			IntegerAction action) {
		return ListenerResponse.CONTINUE;
	}

	@Override
	public ListenerResponse onAfterStep(SteppingAgent<Observation, IntegerAction> agent, StepResult stepResult) {
		return ListenerResponse.CONTINUE;
	}

	@Override
	public void onAfterEpisode(SteppingAgent<Observation, IntegerAction> agent) {
		System.out.println(
				String.format("[%s] Episode %4d : score = %3d", agent.getId(), episodeCount, (int) agent.getReward()));
		++episodeCount;
	}
}
