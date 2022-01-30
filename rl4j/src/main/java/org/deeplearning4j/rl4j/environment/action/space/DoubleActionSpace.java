package org.deeplearning4j.rl4j.environment.action.space;

import org.deeplearning4j.rl4j.environment.action.DoubleAction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DoubleActionSpace extends ContinuousActionSpace<DoubleAction> {

	public DoubleActionSpace(INDArray toDup, DoubleAction noOpAction) {
		super(toDup, noOpAction);
	}

	public DoubleActionSpace(DoubleAction noOpAction, double... arr) {
		this(Nd4j.create(arr), noOpAction);
	}

	public DoubleActionSpace(int[] shape, DoubleAction noOpAction, double... arr) {
		this(Nd4j.create(arr).reshape(shape), noOpAction);
	}

	@Override
	public DoubleAction getRandomAction() {
		return new DoubleAction(rnd.nextDouble());
	}

	public double[] toArray() {
		return data.data().asDouble();
	}

	@Override
	public int getIndex(DoubleAction action) {
		double[] arr = data.data().asDouble();

		int k = 0;
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == action.toDouble()) {
				k = i;
				break;
			}
		}
		
		return k;
		
		//TODO : Improve for DoubleAction that are not part of the space
	}

	@Override
	public Double encode(DoubleAction action) {
		return action.toDouble();
	}
}
