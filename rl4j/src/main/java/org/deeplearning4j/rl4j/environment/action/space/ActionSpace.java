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

import java.lang.reflect.ParameterizedType;

import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.observation.transform.TransformProcess;
import org.nd4j.linalg.api.ndarray.INDArray;

import lombok.Value;
import net.jodah.typetools.TypeResolver;

public abstract class ActionSpace<ACTION extends Action> {
	
    private final Class<?> actionType;

    public abstract int getActionSpaceSize();

    public abstract ACTION getNoOp();

    public abstract ACTION getRandomAction();
    
    public ActionSpace() {
        Class<?>[] typeArguments = TypeResolver.resolveRawArguments(ActionSpace.class, getClass());
        this.actionType = (Class<ACTION>) typeArguments[0];
    }
    
    @SuppressWarnings("unchecked")
	public ACTION fromInteger(int i) {
    	Action a = createAction();
    	return (ACTION) a.fromInteger(i);
    }
    
    @SuppressWarnings("unchecked")
	public ACTION fromArray(INDArray array) {
    	Action a = createAction();
    	return (ACTION) a.fromArray(array);
    }
    
    public abstract Object encode(ACTION action);
    
    // Get an "id" that uniquely identifies this Action within its ActionSpace
 	// Implementing classes can use their internal datastructures to derive such an index	
    public abstract int getIndex(ACTION action);
    
    protected ACTION createAction() {
        try
        {
            return (ACTION) actionType.getDeclaredConstructor().newInstance();
        }
        catch (Exception e)
        {
            // Oops, no default constructor
            throw new RuntimeException(e);
        }
    }

}
