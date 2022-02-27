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

package org.deeplearning4j.rl4j.agent.listener.utils;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.apache.commons.io.output.CloseShieldOutputStream;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.rl4j.agent.SteppingAgent;
import org.deeplearning4j.rl4j.agent.learning.history.HistoryProcessor;
import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.environment.action.Action;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.policy.NeuralNetPolicy;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.shade.jackson.databind.ObjectMapper;

@Slf4j
public class AgentHistoryListener<OBSERVATION extends Observation, ACTION extends Action>
		implements AgentListener<OBSERVATION, ACTION> {

	public static final String AGENT_CONFIGURATION_JSON = "agentConfiguration.json";
	public static final String AGENT_INFO_JSON = "agentInformation.json";

	private int episodeCount;

	private int lastSave = -Constants.MODEL_SAVE_FREQ;

	private String dataRoot;
	@Getter
	private boolean saveData;

	public AgentHistoryListener() throws IOException {
		this(System.getProperty("user.home") + "/" + Constants.DATA_DIR, false);
	}

	public AgentHistoryListener(boolean saveData) throws IOException {
		this(System.getProperty("user.home") + "/" + Constants.DATA_DIR, saveData);
	}

	public AgentHistoryListener(String dataRoot, boolean saveData) throws IOException {
		this.saveData = saveData;
		this.dataRoot = dataRoot;
	}

	@Override
	public ListenerResponse onBeforeEpisode(SteppingAgent<OBSERVATION, ACTION> agent) {
		HistoryProcessor<OBSERVATION> hp = agent.getHistoryProcessor();
		if (hp != null) {
			String filename = dataRoot + "/" + Constants.VIDEO_DIR + "/video-";
			filename += agent.getId() + "-" + episodeCount + ".mp4";
			
			File monitorFile = new File(filename);
			if (!monitorFile.exists()) monitorFile.getParentFile().mkdirs();
			try {
				boolean success = monitorFile.createNewFile();
				if(success) {
				hp.startMonitor(monitorFile);
				}
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		}

		return ListenerResponse.CONTINUE;
	}

	@Override
	public ListenerResponse onBeforeStep(SteppingAgent<OBSERVATION, ACTION> agent, OBSERVATION observation,
			ACTION action) {
		return ListenerResponse.CONTINUE;
	}

	@Override
	public ListenerResponse onAfterStep(SteppingAgent<OBSERVATION, ACTION> agent, StepResult stepResult) {
		try {
			int stepCounter = agent.getEpisodeStepCount();
			if (stepCounter - lastSave >= Constants.MODEL_SAVE_FREQ) {
				save(agent);
				lastSave = stepCounter;
			}

		} catch (Exception e) {
			log.error("Training failed.", e);
			return ListenerResponse.STOP;
		}

		return ListenerResponse.CONTINUE;
	}

	@Override
	public void onAfterEpisode(SteppingAgent<OBSERVATION, ACTION> agent) {
		HistoryProcessor<OBSERVATION> hp = agent.getHistoryProcessor();
		if (hp != null) {
			hp.stopMonitor();
		}

		++episodeCount;
	}

	public void save(SteppingAgent<OBSERVATION, ACTION> agent) throws IOException {

		if (!saveData)
			return;

		File agentFile = new File(dataRoot + "/" + Constants.MODEL_DIR + "/" + agent.getId()+"-"+agent.getEpisodeStepCount() + ".training");
		if (!agentFile.exists()) agentFile.getParentFile().mkdirs();
		boolean success = agentFile.createNewFile();

		if (success) {

			saveToZip(agentFile, AGENT_CONFIGURATION_JSON, toJson(agent.getConfiguration()).getBytes());
			saveToZip(agentFile, AGENT_INFO_JSON,
					toJson(new Info(agent.getClass().getSimpleName(), agent.getEnvironment().getClass().getSimpleName(),
							agent.getEpisodeStepCount(), System.currentTimeMillis())).getBytes());
			
			if (agent.getPolicy() instanceof NeuralNetPolicy) {
					((NeuralNetPolicy<OBSERVATION, ACTION>) agent.getPolicy()).getNeuralNet().saveTo(agentFile, true);
			}

		}
	}

	protected void saveToZip(File file, String name, byte[] data) throws IOException {
		BufferedOutputStream stream = new BufferedOutputStream(new FileOutputStream(file));
		ZipOutputStream zipfile = new ZipOutputStream(new CloseShieldOutputStream(stream));

		ZipEntry config = new ZipEntry(name);
		zipfile.putNextEntry(config);
		zipfile.write(data);

		zipfile.close();
	}
	
	protected String toJson(Object configuration) {
		ObjectMapper mapper = NeuralNetConfiguration.mapper();
		synchronized (mapper) {
			// JSON mappers are supposed to be thread safe: however, in practice they seem
			// to miss fields occasionally
			// when writeValueAsString is used by multiple threads. This results in invalid
			// JSON. See issue #3243
			try {
				return mapper.writeValueAsString(configuration);
			} catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
				throw new RuntimeException(e);
			}
		}
	}

	@AllArgsConstructor
	@Value
	@Builder
	public static class Info {
		String trainingName;
		String mdpName;
		int stepCounter;
		long millisTime;
	}
}
