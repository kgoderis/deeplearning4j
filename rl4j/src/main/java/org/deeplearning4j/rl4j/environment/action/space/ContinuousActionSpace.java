package org.deeplearning4j.rl4j.environment.action.space;

import org.deeplearning4j.rl4j.environment.action.ContinuousAction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import lombok.Getter;

public abstract class ContinuousActionSpace<ACTION extends ContinuousAction> extends ActionSpace<ACTION> {

	protected final INDArray data;

    protected final ACTION noOpAction;
    protected final Random rnd;

    public ContinuousActionSpace(INDArray toDup,ACTION noOpAction) {
        this(toDup, noOpAction, Nd4j.getRandom());
    }

    public ContinuousActionSpace(INDArray toDup, ACTION noOpAction, Random rnd) {
    	this.data = toDup.dup();
        this.noOpAction = noOpAction;
        this.rnd = rnd;
    }
    
	@Override
	public int getActionSpaceSize() {
		return (int) data.length();
	}
	
    @Override
    public ACTION getNoOp() {
        return noOpAction;
    }

    @Override
    abstract public ACTION getRandomAction() ;
    
}
