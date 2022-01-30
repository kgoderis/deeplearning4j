package org.deeplearning4j.rl4j.agent.learning.history;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.rl4j.environment.observation.Observation;

public class DefaultHistoryProcessor implements HistoryProcessor<Observation> {

	public DefaultHistoryProcessor() {
	}

	@Override
	public List<Observation> getHistory() {
		return new ArrayList<Observation>();
	}

	@Override
	public void record(Observation Observation) {
	}

	@Override
	public void add(Observation Observation) {
	}

	@Override
	public void startMonitor(File file) {
	}

	@Override
	public void stopMonitor() {
	}

	@Override
	public boolean isMonitoring() {
		return false;
	}

}
