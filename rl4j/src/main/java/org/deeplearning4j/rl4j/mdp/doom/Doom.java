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

package org.deeplearning4j.rl4j.mdp.doom;

import lombok.Getter;
import lombok.Setter;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.environment.action.IntegerAction;
import org.deeplearning4j.rl4j.environment.action.space.IntegerActionSpace;
import org.deeplearning4j.rl4j.mdp.doom.vizdoom.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
abstract public class Doom implements Environment<IntegerAction> {

	final public static String DOOM_ROOT = "org.deeplearning4j.rl4j.mdp.doom.vizdoom";

	protected DoomGame game;
	final protected List<double[]> actions;
	final protected IntegerActionSpace actionSpace;
	final protected int[] observationShape;

	@Getter
	final protected boolean render;
	@Setter
	protected double scaleFactor = 1;

	public Doom() {
		this(false);
	}

	public Doom(boolean render) {
		this.render = render;
		actions = new ArrayList<double[]>();
		game = new DoomGame();
		setupGame();
		actionSpace = new IntegerActionSpace(getConfiguration().getButtons().size() + 1,0);
		observationShape = new int[] { game.getScreenHeight(), game.getScreenWidth(), 3 };
	}

	public void setupGame() {

		Configuration conf = getConfiguration();

		game.setViZDoomPath(DOOM_ROOT + "/org.deeplearning4j.rl4j.mdp.doom.vizdoom");
		game.setDoomGamePath(DOOM_ROOT + "/freedoom2.wad");
		game.setDoomScenarioPath(DOOM_ROOT + "/scenarios/" + conf.getScenario() + ".wad");

		game.setDoomMap("map01");

		game.setScreenFormat(ScreenFormat.RGB24);
		game.setScreenResolution(ScreenResolution.RES_800X600);
		// Sets other rendering options
		game.setRenderHud(false);
		game.setRenderCrosshair(false);
		game.setRenderWeapon(true);
		game.setRenderDecals(false);
		game.setRenderParticles(false);

		GameVariable[] gameVar = new GameVariable[] { GameVariable.KILLCOUNT, GameVariable.ITEMCOUNT,
				GameVariable.SECRETCOUNT, GameVariable.FRAGCOUNT, GameVariable.HEALTH, GameVariable.ARMOR,
				GameVariable.DEAD, GameVariable.ON_GROUND, GameVariable.ATTACK_READY, GameVariable.ALTATTACK_READY,
				GameVariable.SELECTED_WEAPON, GameVariable.SELECTED_WEAPON_AMMO, GameVariable.AMMO1, GameVariable.AMMO2,
				GameVariable.AMMO3, GameVariable.AMMO4, GameVariable.AMMO5, GameVariable.AMMO6, GameVariable.AMMO7,
				GameVariable.AMMO8, GameVariable.AMMO9, GameVariable.AMMO0 };
		// Adds game variables that will be included in state.

		for (int i = 0; i < gameVar.length; i++) {
			game.addAvailableGameVariable(gameVar[i]);
		}

		// Causes episodes to finish after timeout tics
		game.setEpisodeTimeout(conf.getTimeout());

		game.setEpisodeStartTime(conf.getStartTime());

		game.setWindowVisible(render);
		game.setSoundEnabled(false);
		game.setMode(Mode.PLAYER);

		game.setLivingReward(conf.getLivingReward());

		// Adds buttons that will be allowed.
		List<Button> buttons = conf.getButtons();
		int size = buttons.size();

		actions.add(new double[size + 1]);
		for (int i = 0; i < size; i++) {
			game.addAvailableButton(buttons.get(i));
			double[] action = new double[size + 1];
			action[i] = 1;
			actions.add(action);
		}

		game.setDeathPenalty(conf.getDeathPenalty());
		game.setDoomSkill(conf.getDoomSkill());

		game.init();
	}

	public boolean isDone() {
		return game.isEpisodeFinished();
	}

	public Map<String, Object> reset() {
		log.info("free Memory: " + Pointer.formatBytes(Pointer.availablePhysicalBytes()) + "/"
				+ Pointer.formatBytes(Pointer.totalPhysicalBytes()));

		game.newEpisode();
		Map<String, Object> channelsData = new HashMap<String, Object>() {
			{
				put("data", game.getState().screenBuffer);
			}
		};

		return channelsData;
	}

	public void close() {
		game.close();
	}

	@Override
	public StepResult step(IntegerAction action) {
		double r = game.makeAction(actions.get((int) actionSpace.encode(action))) * scaleFactor;
		log.info(game.getEpisodeTime() + " " + r + " " + action + " ");

		Map<String, Object> channelsData = new HashMap<String, Object>() {
			{
				put("data", game.isEpisodeFinished() ? new byte[game.getScreenSize()] : game.getState().screenBuffer);
			}
		};

		return new StepResult(channelsData, r, game.isEpisodeFinished());
	}

	public IntegerActionSpace getActionSpace() {
		return actionSpace;
	}
	

	@Override
	public boolean isEpisodeFinished() {
		return game.isEpisodeFinished();
	}

	public abstract Configuration getConfiguration();

	@Value
	public static class Configuration {
		String scenario;
		double livingReward;
		double deathPenalty;
		int doomSkill;
		int timeout;
		int startTime;
		List<Button> buttons;
	}
}
