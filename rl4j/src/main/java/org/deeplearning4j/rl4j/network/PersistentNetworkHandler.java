package org.deeplearning4j.rl4j.network;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.output.CloseShieldOutputStream;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import lombok.Data;

public abstract class PersistentNetworkHandler implements NetworkHandler {

	public abstract NetworkHandler clone();

	abstract String getConfigurationName();

	protected void saveConfigurationTo(File file, Configuration configuration) throws IOException {
		saveToZip(file, getConfigurationName(), toJson(configuration).getBytes());
	}

	protected Configuration loadConfigurationFrom(File file, Class<?> configurationClazz) throws IOException {
		byte[] config = loadFromZip(file, getConfigurationName());

		Configuration conf = null;
		if (config != null) {
			InputStream stream = new ByteArrayInputStream(config);
			BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
			String line = "";
			StringBuilder js = new StringBuilder();
			while ((line = reader.readLine()) != null) {
				js.append(line).append("\n");
			}
			reader.close();
			stream.close();

			conf = (Configuration) fromJson(js.toString(), configurationClazz);
		}

		return conf;
	}

	/**
	 * @return JSON representation of the network handler configuration
	 */
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

	protected Object fromJson(String json, Class<?> configurationClazz) {
		ObjectMapper mapper = NeuralNetConfiguration.mapper();
		synchronized (mapper) {
			try {
				return mapper.readValue(json, configurationClazz);
			} catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
				throw new RuntimeException(e);
			}
		}
	}

	protected byte[] loadFromZip(File file, String name) throws IOException {

		FileInputStream inputStream = null;
		try {
			inputStream = new FileInputStream(file);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// available method can return 0 in some cases:
		// https://github.com/eclipse/deeplearning4j/issues/4887
		int available;
		try {
			// InputStream.available(): A subclass' implementation of this method may choose
			// to throw an IOException
			// if this input stream has been closed by invoking the close() method.
			available = inputStream.available();
		} catch (IOException e) {
			throw new IOException(
					"Cannot read from stream: stream may have been closed or is attempting to be read from"
							+ "multiple times?",
					e);
		}
		if (available <= 0) {
			throw new IOException(
					"Cannot read from stream: stream may have been closed or is attempting to be read from"
							+ "multiple times?");
		}

		byte[] data = null;
		try (final ZipInputStream zis = new ZipInputStream(inputStream)) {
			while (true) {
				final ZipEntry zipEntry = zis.getNextEntry();
				if (zipEntry != null && zipEntry.getName().equals(name)) {
					if (zipEntry.isDirectory() || zipEntry.getSize() > Integer.MAX_VALUE)
						throw new IllegalArgumentException();
					final int size = (int) (zipEntry.getSize());
					if (size >= 0) { // known size
						data = IOUtils.readFully(zis, size);
					} else { // unknown size
						final ByteArrayOutputStream bout = new ByteArrayOutputStream();
						IOUtils.copy(zis, bout);
						data = bout.toByteArray();
					}
					break;
				}
			}

			return data;
		}
	}

	protected void saveToZip(File file, String name, byte[] data) throws IOException {
		
		File tempFile = File.createTempFile(file.getName(), null);
		tempFile.delete();
		boolean renameOk = file.renameTo(tempFile);
		if (!renameOk) {
			throw new RuntimeException(
					"could not rename the file " + file.getAbsolutePath() + " to " + tempFile.getAbsolutePath());
		}
		
		ZipInputStream inputStream = new ZipInputStream(new FileInputStream(tempFile));
		ZipOutputStream outputStream = new ZipOutputStream(new CloseShieldOutputStream(new FileOutputStream(file)));

		ZipEntry entry = inputStream.getNextEntry();
		while (entry != null) {
			String entryName = entry.getName();
			boolean toBeDeleted = false;
			if (name.indexOf(entryName) != -1) {
				toBeDeleted = true;
			}
			if (!toBeDeleted) {
				outputStream.putNextEntry(new ZipEntry(entryName));
				IOUtils.copy(inputStream, outputStream);
			}
			entry = inputStream.getNextEntry();
		}
		inputStream.close();
		
		ZipEntry newEntry = new ZipEntry(name);
		outputStream.putNextEntry(newEntry);
		outputStream.write(data);

		inputStream.close();
		outputStream.close();
		tempFile.delete();
	}

	protected void saveToZip(File file, File filetoSave) throws IOException {
		saveToZip(file, filetoSave.getName(), Files.readAllBytes(filetoSave.toPath()));
	}


	@SuppressWarnings("serial")
	@Data
	public class Configuration implements Serializable {
	}

}
