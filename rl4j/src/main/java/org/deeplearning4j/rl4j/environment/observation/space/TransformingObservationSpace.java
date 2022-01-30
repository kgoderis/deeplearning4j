package org.deeplearning4j.rl4j.environment.observation.space;

import java.util.Map;

import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;

public class TransformingObservationSpace extends ObservationSpace<Observation> {
	
	private final TransformProcess<Observation> transformProcess;

	public TransformingObservationSpace(TransformProcess<Observation> transformProcess) {
		super();
		this.transformProcess = transformProcess;
	}

	public Observation transform(Map<String, Object>  channelData, int episodeStepNumberOfObs, boolean isFinal) {
		return transformProcess.transform(channelData, episodeStepNumberOfObs, isFinal);
	}

}
