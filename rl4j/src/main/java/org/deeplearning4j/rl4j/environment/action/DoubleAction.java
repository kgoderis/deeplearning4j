package org.deeplearning4j.rl4j.environment.action;

public class DoubleAction extends ContinuousAction {

	private double action;
	
	public DoubleAction() {
		this.action = 0 ;
	}
	
	public DoubleAction(double action) {
		this.action = action;
	}

	@Override
	public Action fromInteger(int action) {
		this.action = action;
		return this;
	}
	
	public double toDouble() {
		return action;
	}

}
