package org.deeplearning4j.rl4j.environment.action.space;

import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.action.DiscreteAction;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;

public abstract class HighLowActionSpace extends DiscreteActionSpace<IntegerAction> {

	// size of the space also defined as the number of different actions
	protected INDArray matrix;

	public HighLowActionSpace(INDArray matrix,IntegerAction noOpAction) {
		super(matrix.rows(),noOpAction);
		this.matrix = matrix;
	}
	
	public HighLowActionSpace(INDArray matrix,IntegerAction noOpAction,Random rnd) {
		super(matrix.rows(),noOpAction,rnd);
		this.matrix = matrix;
	}

	// TODO
//	@Override
//	public ACTION getRandomAction() {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//	@Override
//	public int getIndex(ACTION action) {
//		// TODO Auto-generated method stub
//		return 0;
//	}
	
  
  public INDArray encode(IntegerAction action) {
      INDArray m = matrix.dup();
      m.put(action.toInteger() - 1, 0, matrix.getDouble(action.toInteger() - 1, 1));
      return m;
  }

}
