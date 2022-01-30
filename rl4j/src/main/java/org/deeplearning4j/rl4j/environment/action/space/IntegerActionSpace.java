/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.rl4j.environment.action.space;

import lombok.Getter;

import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

// Work in progress
public class IntegerActionSpace extends DiscreteActionSpace<IntegerAction> {

    public IntegerActionSpace(int numActions, IntegerAction noOpAction) {
        super(numActions, noOpAction, Nd4j.getRandom());
    }
    
    public IntegerActionSpace(int numActions, int noOpAction) {
        this(numActions, noOpAction, Nd4j.getRandom());
    }

    public IntegerActionSpace(int numActions, int noOpAction, Random rnd) {
		super(numActions, new IntegerAction(noOpAction), rnd);
    }

    @Override
    public IntegerAction getRandomAction() {
        return new IntegerAction(rnd.nextInt(actionSpaceSize));
    }

	@Override
	public Object encode(IntegerAction action) {
		return action.toInteger();
	}
}
