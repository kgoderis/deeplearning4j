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

package org.deeplearning4j.rl4j.agent.learning.history;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import lombok.Value;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.bytedeco.javacv.*;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.rl4j.environment.observation.Observation;
import org.deeplearning4j.rl4j.util.VideoRecorder;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

@Slf4j
public class VideoHistoryProcessor implements HistoryProcessor<Observation> {

	@Getter
	final private Configuration conf;
	private CircularFifoQueue<INDArray> history;
	private VideoRecorder videoRecorder;

	public VideoHistoryProcessor(Configuration conf) {
		this.conf = conf;
		history = new CircularFifoQueue<>(conf.getHistoryLength());
	}

	public void add(Observation obs) {
		INDArray processed = transform(obs.getData());
		history.add(processed);
	}

	public void startMonitor(File file) {
		if (videoRecorder == null) {
			videoRecorder = VideoRecorder.builder(conf.height, conf.width).build();
		}

		try {
			videoRecorder.startRecording(file);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void stopMonitor() {
		if (videoRecorder != null) {
			try {
				videoRecorder.stopRecording();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public boolean isMonitoring() {
		return videoRecorder != null && videoRecorder.isRecording();
	}

	public void record(Observation pixelArray) {
		if (isMonitoring()) {
			// before accessing the raw pointer, we need to make sure that array is actual
			// on the host side
			Nd4j.getAffinityManager().ensureLocation(pixelArray.getData(), AffinityManager.Location.HOST);

			try {
				videoRecorder.record(pixelArray.getData());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public List<Observation> getHistory() {
		ArrayList<Observation> list = new ArrayList<Observation>();
		for (int i = 0; i < conf.getHistoryLength(); i++) {
			list.add((Observation) history.get(i));
		}
		return list;
	}

	private INDArray transform(INDArray raw) {
		long[] shape = raw.shape();

		// before accessing the raw pointer, we need to make sure that array is actual
		// on the host side
		Nd4j.getAffinityManager().ensureLocation(raw, AffinityManager.Location.HOST);

		Mat ocvmat = new Mat((int) shape[0], (int) shape[1], CV_32FC(3), raw.data().pointer());
		Mat cvmat = new Mat(shape[0], shape[1], CV_8UC(3));
		ocvmat.convertTo(cvmat, CV_8UC(3), 255.0, 0.0);
		cvtColor(cvmat, cvmat, COLOR_RGB2GRAY);
		Mat resized = new Mat(conf.getRescaledHeight(), conf.getRescaledWidth(), CV_8UC(1));
		resize(cvmat, resized, new Size(conf.getRescaledWidth(), conf.getRescaledHeight()));
		// show(resized);
		// waitKP();
		// Crop by croppingHeight, croppingHeight
		Mat cropped = resized.apply(
				new Rect(conf.getOffsetX(), conf.getOffsetY(), conf.getCroppingWidth(), conf.getCroppingHeight()));
		// System.out.println(conf.getCroppingWidth() + " " +
		// cropped.data().asBuffer().array().length);

		INDArray out = null;
		try {
			out = new NativeImageLoader(conf.getCroppingHeight(), conf.getCroppingWidth()).asMatrix(cropped);
		} catch (IOException e) {
			e.printStackTrace();
		}
		// System.out.println(out.shapeInfoToString());
		out = out.reshape(1, conf.getCroppingHeight(), conf.getCroppingWidth());
		INDArray compressed = out.castTo(DataType.UBYTE);
		return compressed;
	}

	public double getScale() {
		return 255;
	}

	public void waitKP() {
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void show(Mat m) {
		OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
		CanvasFrame canvas = new CanvasFrame("LOL", 1);
		canvas.showImage(converter.convert(m));
	}

	@Value
	public static class Configuration {
		int height;
		int width;
		int historyLength;
		int rescaledWidth;
		int rescaledHeight;
		int croppingWidth;
		int croppingHeight;
		int offsetX;
		int offsetY;
		int skipFrame;

		public int[] getShape() {
			return new int[] { getHistoryLength(), getRescaledHeight(), getRescaledWidth() };
		}
	}

	/**
	 * @return An instance of a builder
	 */
	public static Builder builder() {
		return new Builder();
	}

	public static class Builder {

        int height = 84;
        int width = 84;
		int historyLength = 4;
		int rescaledWidth = 84;
		int rescaledHeight = 84;
		int croppingWidth = 84;
		int croppingHeight = 84;
		int offsetX = 0;
		int offsetY = 0;
		int skipFrame = 4;
		
		public Builder height(int historyLength) {
			Preconditions.checkNotNull(height, "The height must not be null");

			this.height = height;
			return this;
		}
		
		public Builder width(int width) {
			Preconditions.checkNotNull(width, "The width must not be null");

			this.width = width;
			return this;
		}
		
		public Builder historyLength(int historyLength) {
			Preconditions.checkNotNull(historyLength, "The historyLength must not be null");

			this.historyLength = historyLength;
			return this;
		}

		public Builder rescaledWidth(int rescaledWidth) {
			Preconditions.checkNotNull(rescaledWidth, "The rescaledWidth must not be null");

			this.rescaledWidth = rescaledWidth;
			return this;
		}

		public Builder rescaledHeight(int rescaledHeight) {
			Preconditions.checkNotNull(rescaledHeight, "The rescaledHeight must not be null");

			this.rescaledHeight = rescaledHeight;
			return this;
		}

		public Builder croppingWidth(int croppingWidth) {
			Preconditions.checkNotNull(croppingWidth, "The croppingWidth must not be null");

			this.croppingWidth = croppingWidth;
			return this;
		}

		public Builder croppingHeight(int croppingHeight) {
			Preconditions.checkNotNull(croppingHeight, "The croppingHeight must not be null");

			this.croppingHeight = croppingHeight;
			return this;
		}

		public Builder offsetX(int offsetX) {
			Preconditions.checkNotNull(offsetX, "The offsetX must not be null");

			this.offsetX = offsetX;
			return this;
		}

		public Builder offsetY(int offsetY) {
			Preconditions.checkNotNull(offsetY, "The offsetY must not be null");

			this.offsetY = offsetY;
			return this;
		}

		public Builder skipFrame(int skipFrame) {
			Preconditions.checkNotNull(skipFrame, "The skipFrame must not be null");

			this.skipFrame = skipFrame;
			return this;
		}

		public VideoHistoryProcessor build() {
			return new VideoHistoryProcessor(new Configuration(height, width, historyLength, rescaledWidth, rescaledHeight,
					croppingWidth, croppingHeight, offsetX, offsetY, skipFrame));
		}

	}

}
