package org.deeplearning4j.rl4j.environment.action;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import lombok.Builder;

public abstract class Action {

	public abstract Action fromInteger(int i);

	public Action fromArray(INDArray a) {
		return fromInteger(Nd4j.argMax(a, Integer.MAX_VALUE).getInt(0));
	}	
}
