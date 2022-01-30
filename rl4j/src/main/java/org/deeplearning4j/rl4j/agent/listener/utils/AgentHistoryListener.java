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
//		createSubdir();
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
//		try {
//			appendStat(statEntry);
//		} catch (Exception e) {
//			log.error("Training failed.", e);
//		}

		++episodeCount;
	}

//	private static void writeEntry(InputStream inputStream, ZipOutputStream zipStream) throws IOException {
//		byte[] bytes = new byte[1024];
//		int bytesRead;
//		while ((bytesRead = inputStream.read(bytes)) != -1) {
//			zipStream.write(bytes, 0, bytesRead);
//		}
//	}

//	public void writeInfo(SteppingAgent<OBSERVATION, ACTION> agent) throws IOException {
//		if (!saveData)
//			return;
//		
//		
//		
//		saveToZip(file, AGENT_CONFIGURATION_JSON, toJson(agent.getConfiguration()).getBytes());
//		saveToZip(file, AGENT_INFO_JSON, toJson(new Info(agent.getClass().getSimpleName(), agent.getEnvironment().getClass().getSimpleName(),
//				agent.getEpisodeStepCount(), System.currentTimeMillis())).getBytes());
//
//		
//
////		Path infoPath = Paths.get(getInfo());
////
////		Info info = new Info(agent.getClass().getSimpleName(), agent.getEnvironment().getClass().getSimpleName(),
////				agent.getConfiguration(), agent.getEpisodeStepCount(), System.currentTimeMillis());
////		String toWrite = toJson(info);
////
////		Files.write(infoPath, toWrite.getBytes(), StandardOpenOption.TRUNCATE_EXISTING);
//	}

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

//		save(getModelDir() + "/" + agent.getEpisodeStepCount() + ".training", agent);
//
//		if (agent.getPolicy() instanceof NeuralNetPolicy) {
//			try {
//				((NeuralNetPolicy<OBSERVATION, ACTION>) agent.getPolicy()).getNeuralNet()
//						.save(getModelDir() + "/" + agent.getEpisodeStepCount() + ".model");
//			} catch (UnsupportedOperationException e) {
//				String path = getModelDir() + "/" + agent.getEpisodeStepCount();
//				((NeuralNetPolicy<OBSERVATION, ACTION>) agent.getPolicy()).getNeuralNet().save(path + "_value.model",
//						path + "_policy.model");
//			}
//		}

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

//	public void save(String path, SteppingAgent<OBSERVATION, ACTION> agent) throws IOException {
//		try (BufferedOutputStream os = new BufferedOutputStream(new FileOutputStream(path))) {
//			try (ZipOutputStream zipfile = new ZipOutputStream(os)) {
//
//				ZipEntry config = new ZipEntry("configuration.json");
//				zipfile.putNextEntry(config);
//				String json = new ObjectMapper().writeValueAsString(agent.getConfiguration());
//				writeEntry(new ByteArrayInputStream(json.getBytes()), zipfile);
//
//				try {
//					ZipEntry dqn = new ZipEntry("dqn.bin");
//					zipfile.putNextEntry(dqn);
//
//					ByteArrayOutputStream bos = new ByteArrayOutputStream();
//					if (agent.getPolicy() instanceof NeuralNetPolicy) {
//						((NeuralNetPolicy<OBSERVATION, ACTION>) agent.getPolicy()).getNeuralNet().save(bos);
//					}
//					bos.flush();
//					bos.close();
//
//					InputStream inputStream = new ByteArrayInputStream(bos.toByteArray());
//					writeEntry(inputStream, zipfile);
//				} catch (UnsupportedOperationException e) {
//					ByteArrayOutputStream bos = new ByteArrayOutputStream();
//					ByteArrayOutputStream bos2 = new ByteArrayOutputStream();
//					((NeuralNetPolicy<OBSERVATION, ACTION>) agent.getPolicy()).getNeuralNet().save(bos, bos2);
//
//					bos.flush();
//					bos.close();
//					InputStream inputStream = new ByteArrayInputStream(bos.toByteArray());
//					ZipEntry value = new ZipEntry("value.bin");
//					zipfile.putNextEntry(value);
//					writeEntry(inputStream, zipfile);
//
//					bos2.flush();
//					bos2.close();
//					InputStream inputStream2 = new ByteArrayInputStream(bos2.toByteArray());
//					ZipEntry policy = new ZipEntry("policy.bin");
//					zipfile.putNextEntry(policy);
//					writeEntry(inputStream2, zipfile);
//				}
//
//				if (agent.getHistoryProcessor() != null) {
//					ZipEntry hpconf = new ZipEntry("hpconf.bin");
//					zipfile.putNextEntry(hpconf);
//
//					ByteArrayOutputStream bos2 = new ByteArrayOutputStream();
//					if (agent.getPolicy() instanceof NeuralNetPolicy) {
//						((NeuralNetPolicy<OBSERVATION, ACTION>) agent.getPolicy()).getNeuralNet().save(bos2);
//					}
//					bos2.flush();
//					bos2.close();
//
//					InputStream inputStream2 = new ByteArrayInputStream(bos2.toByteArray());
//					writeEntry(inputStream2, zipfile);
//				}
//
//				zipfile.flush();
//				zipfile.close();
//
//			}
//		}
//	}

//	public static <C> Pair<DQN, C> load(File file, Class<C> cClass) throws IOException {
//		log.info("Deserializing: " + file.getName());
//
//		C conf = null;
//		DQN dqn = null;
//		try (ZipFile zipFile = new ZipFile(file)) {
//			ZipEntry config = zipFile.getEntry("configuration.json");
//			InputStream stream = zipFile.getInputStream(config);
//			BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
//			String line = "";
//			StringBuilder js = new StringBuilder();
//			while ((line = reader.readLine()) != null) {
//				js.append(line).append("\n");
//			}
//			String json = js.toString();
//
//			reader.close();
//			stream.close();
//
//			conf = new ObjectMapper().readValue(json, cClass);
//
//			ZipEntry dqnzip = zipFile.getEntry("dqn.bin");
//			InputStream dqnstream = zipFile.getInputStream(dqnzip);
//			File tmpFile = File.createTempFile("restore", "dqn");
//			Files.copy(dqnstream, Paths.get(tmpFile.getAbsolutePath()), StandardCopyOption.REPLACE_EXISTING);
//			dqn = new BaseDQN(ModelSerializer.restoreMultiLayerNetwork(tmpFile));
//			dqnstream.close();
//		}
//
//		return new Pair<DQN, C>(dqn, conf);
//	}
//
//	public static <C> Pair<DQN, C> load(String path, Class<C> cClass) throws IOException {
//		return load(new File(path), cClass);
//	}
//
//	public static <C> Pair<DQN, C> load(InputStream is, Class<C> cClass) throws IOException {
//		File tmpFile = File.createTempFile("restore", "learning");
//		Files.copy(is, Paths.get(tmpFile.getAbsolutePath()), StandardCopyOption.REPLACE_EXISTING);
//		return load(tmpFile, cClass);
//	}

//    private void create(String dataRoot, boolean saveData) throws IOException {
//        this.saveData = saveData;
//        this.dataRoot = dataRoot;
//        createSubdir();
//    }

	// FIXME race condition if you create them at the same time where checking if
	// dir exists is not atomic with the creation
//	public String createSubdir() throws IOException {
//
//		if (!saveData)
//			return "";
//
//		File dr = new File(dataRoot);
//		dr.mkdirs();
//		File[] rootChildren = dr.listFiles();
//
//		int i = 1;
//		while (childrenExist(rootChildren, i + ""))
//			i++;
//
//		File f = new File(dataRoot + "/" + i);
//		f.mkdirs();
//
//		currentDir = f.getAbsolutePath();
//		log.info("Created training data directory: " + currentDir);
//
//		File mov = new File(getVideoDir());
//		mov.mkdirs();
//
//		File model = new File(getModelDir());
//		model.mkdirs();
//
//		File stat = new File(getStat());
//		File info = new File(getInfo());
//		stat.createNewFile();
//		info.createNewFile();
//		return f.getAbsolutePath();
//	}
//
//	public String getVideoDir() {
//		return currentDir + "/" + Constants.VIDEO_DIR;
//	}
//
//	public String getModelDir() {
//		return currentDir + "/" + Constants.MODEL_DIR;
//	}
//
//	public String getInfo() {
//		return currentDir + "/" + Constants.INFO_FILENAME;
//	}
//
//	public String getStat() {
//		return currentDir + "/" + Constants.STATISTIC_FILENAME;
//	}
//
//	public void appendStat(StatEntry statEntry) throws IOException {
//
//		if (!saveData)
//			return;
//
//		Path statPath = Paths.get(getStat());
//		String toAppend = toJson(statEntry);
//		Files.write(statPath, toAppend.getBytes(), StandardOpenOption.APPEND);
//
//	}

//	private String toJson(Object object) throws IOException {
//		return mapper.writeValueAsString(object) + "\n";
//	}

//
//
//	private boolean childrenExist(File[] files, String children) {
//		boolean exists = false;
//		for (int i = 0; i < files.length; i++) {
//			if (files[i].getName().equals(children)) {
//				exists = true;
//				break;
//			}
//		}
//		return exists;
//	}

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
