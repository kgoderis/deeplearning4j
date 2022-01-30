package org.deeplearning4j.rl4j.environment.action;

public class IntegerAction extends DiscreteAction {

	private int action;
	
	public IntegerAction() {
		this.action = 0 ;
	}

	public int toInteger() {
		return action;
	}
	
	public IntegerAction(int action) {
		this.action = action;
	}

	@Override
	public Action fromInteger(int action) {
		this.action = action;
		return this;
	}
}
