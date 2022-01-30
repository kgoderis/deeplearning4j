package org.deeplearning4j.rl4j.mdp.gym;

import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.deeplearning4j.rl4j.environment.action.space.DiscreteActionSpace;
import org.deeplearning4j.rl4j.environment.action.space.IntegerActionSpace;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.mdp.gym.Gym.Builder;
import org.nd4j.common.base.Preconditions;

public class IntegerGym extends Gym<Observation, IntegerAction, IntegerActionSpace> {

	public IntegerGym(String envId, boolean render, boolean monitor, Integer seed) {
		super(envId, render, monitor, seed);
	}
	
	public static Builder integerBuilder() {
		return new Builder();
	}

	public static class Builder {
		private String envId;
		private boolean render;
		private boolean monitor;
		private int seed;
		private int[] actions;

		public Builder environment(String envId) {
			Preconditions.checkNotNull(envId, "The envId must not be null");

			this.envId = envId;
			return this;
		}
		
		public Builder render(boolean render) {
			Preconditions.checkNotNull(render, "The render must not be null");

			this.render = render;
			return this;
		}
		
		public Builder monitor(boolean monitor) {
			Preconditions.checkNotNull(monitor, "The monitor must not be null");

			this.monitor = monitor;
			return this;
		}
		
		public Builder seed(int seed) {
			Preconditions.checkNotNull(seed, "The seed must not be null");

			this.seed = seed;
			return this;
		}
		
		public Builder actions(int[] actions) {
			Preconditions.checkNotNull(seed, "The actions must not be null");

			this.actions = actions;
			return this;
		}

		public IntegerGym build() {
			return new IntegerGym(envId, render, monitor, seed);
		}
	}

}
