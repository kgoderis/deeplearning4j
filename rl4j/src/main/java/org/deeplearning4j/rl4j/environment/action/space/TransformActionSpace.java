package org.deeplearning4j.rl4j.environment.action.space;

import java.lang.reflect.InvocationTargetException;

import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;

import net.jodah.typetools.TypeResolver;

public class TransformActionSpace<ACTION extends DiscreteAction> extends DiscreteActionSpace<ACTION> {
	
    final private int[] availableAction;
    final private DiscreteActionSpace<ACTION> actionSpace;


	public TransformActionSpace(DiscreteActionSpace<ACTION> actionSpace, int[] availableAction) {
		this.actionSpace = actionSpace;
		this.availableAction = availableAction;
	}

	@Override
	public ACTION getRandomAction() {
		return (ACTION) new IntegerAction(availableAction[rnd.nextInt(availableAction.length)]);
	}

	@Override
	public int getIndex(ACTION action) {
		return availableAction[action.toInteger()];
	}
	
    @Override
    public Object encode(ACTION action) {
    	ACTION theAction = null;
		try {
	        Class<?>[] typeArguments = TypeResolver.resolveRawArguments(ActionSpace.class, getClass());
	        Class<ACTION> actionType = (Class<ACTION>) typeArguments[0];
			theAction = actionType.getDeclaredConstructor().newInstance();
			theAction.fromInteger(availableAction[action.toInteger()]);
		} catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException
				| NoSuchMethodException | SecurityException e) {
			e.printStackTrace();
		}    	
		
        return actionSpace.encode((ACTION) theAction);
    }

}
