package org.deeplearning4j.rl4j.environment.action.space;

import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import lombok.Getter;

public abstract class DiscreteActionSpace<ACTION extends DiscreteAction> extends ActionSpace<ACTION> {

    @Getter
    protected final int actionSpaceSize;

    protected final ACTION noOpAction;
    protected final Random rnd;

    public DiscreteActionSpace() {
        this(0, null, Nd4j.getRandom());
    }
    
    public DiscreteActionSpace(int numActions, ACTION noOpAction) {
        this(numActions, noOpAction, Nd4j.getRandom());
    }

    public DiscreteActionSpace(int numActions, ACTION noOpAction, Random rnd) {
        this.actionSpaceSize = numActions;
        this.noOpAction = noOpAction;
        this.rnd = rnd;
    }
    
    @Override
    public int getActionSpaceSize() {
    	return actionSpaceSize;
    }

    @Override
    public ACTION getNoOp() {
        return noOpAction;
    }

    @Override
    abstract public ACTION getRandomAction();
    
    @Override
    public int getIndex(ACTION action) {
    	return action.toInteger();
    }
 
}
