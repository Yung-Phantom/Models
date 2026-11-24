package models.ml.Preprocessing.Audio;

import models.ml.Preprocessing.Features.AbstractDenseFeatureExtractor;
import models.ml.Preprocessing.Features.FeatureVectors;
import models.ml.Utils.Json;
import models.ml.Utils.JsonFields;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Feature extractor that computes spectral statistics from audio frames.
 * <p>
 * For each frame: computes RMS, zero-crossing rate, spectral centroid, and spectral rolloff.
 * Aggregates across frames: mean and standard deviation (population) of each statistic.
 * Output feature vector (size 8): [mean_rms, std_rms, mean_zcr, std_zcr, mean_centroid, std_centroid, mean_rolloff, std_rolloff].
 * <p>
 * Independent of sample rate; operates on raw frames.
 * <p>
 * Not thread-safe.
 */
public final class SpectralStatsExtractor extends AbstractDenseFeatureExtractor<AudioData> {

    private final int frameSize;
    private final int hop;
    private final double[] hannWindow;

    /**
     * Creates a spectral statistics feature extractor.
     *
     * @param frameSize frame size in samples; must be a power of two and > 0
     * @param hop       hop size in samples; must be > 0
     * @throws IllegalArgumentException if frameSize is not a power of two, or {@code <=} 0, or hop {@code <=} 0
     */
    public SpectralStatsExtractor(int frameSize, int hop) {
        if (frameSize <= 0) {
            throw new IllegalArgumentException("frameSize must be > 0, got " + frameSize);
        }
        if ((frameSize & (frameSize - 1)) != 0) {
            throw new IllegalArgumentException("frameSize must be a power of two, got " + frameSize);
        }
        if (hop <= 0) {
            throw new IllegalArgumentException("hop must be > 0, got " + hop);
        }
        this.frameSize = frameSize;
        this.hop = hop;
        this.hannWindow = Windows.hann(frameSize);
    }

    /**
     * Returns the frame size.
     */
    public int getFrameSize() {
        return frameSize;
    }

    /**
     * Returns the hop size.
     */
    public int getHop() {
        return hop;
    }

    @Override
    public int getFeatureSize() {
        return 8;
    }

    @Override
    protected double[] extractDense(AudioData audio) {
        double[] samples = audio.getSamples();
        var frames = Framing.frames(samples, frameSize, hop);
        int numFrames = frames.size();

        double[] rmsVals = new double[numFrames];
        double[] zcrVals = new double[numFrames];
        double[] centroidVals = new double[numFrames];
        double[] rolloffVals = new double[numFrames];

        for (int i = 0; i < numFrames; i++) {
            double[] frame = frames.get(i);
            // Apply Hann window using Windows.apply (modifies frame in-place)
            double[] windowed = Arrays.copyOf(frame, frame.length);
            Windows.apply(windowed, hannWindow);

            // RMS
            double sumSq = 0.0;
            for (double v : frame) {
                sumSq += v * v;
            }
            rmsVals[i] = Math.sqrt(sumSq / frameSize);

            // Zero-crossing rate
            int zcrCount = 0;
            for (int j = 1; j < frameSize; j++) {
                if (frame[j - 1] * frame[j] < 0) {
                    zcrCount++;
                }
            }
            zcrVals[i] = (double) zcrCount / (frameSize - 1);

            // FFT and spectral features
            double[] mag = Fft.magnitudeSpectrum(windowed, frameSize);
            int N = mag.length; // number of frequency bins including Nyquist (frameSize/2 + 1)

            double sumMag = 0.0;
            double weightedSum = 0.0;
            for (int k = 0; k < N; k++) {
                sumMag += mag[k];
                weightedSum += k * mag[k];
            }

            if (sumMag > 0) {
                centroidVals[i] = weightedSum / sumMag / (N - 1);
            } else {
                centroidVals[i] = 0.0;
            }

            // Rolloff: find smallest k where cumsum >= 0.85 * sumMag
            double cumsum = 0.0;
            double target = 0.85 * sumMag;
            int rolloffBin = 0;
            if (sumMag > 0) {
                for (int k = 0; k < N; k++) {
                    cumsum += mag[k];
                    if (cumsum >= target) {
                        rolloffBin = k;
                        break;
                    }
                }
                rolloffVals[i] = (double) rolloffBin / (N - 1);
            } else {
                rolloffVals[i] = 0.0;
            }
        }

        // Compute mean and std for each statistic using FrameStats
        double[] result = new double[8];
        result[0] = FrameStats.mean(rmsVals);   // mean_rms
        result[1] = FrameStats.std(rmsVals);    // std_rms
        result[2] = FrameStats.mean(zcrVals);   // mean_zcr
        result[3] = FrameStats.std(zcrVals);    // std_zcr
        result[4] = FrameStats.mean(centroidVals); // mean_centroid
        result[5] = FrameStats.std(centroidVals);  // std_centroid
        result[6] = FrameStats.mean(rolloffVals);  // mean_rolloff
        result[7] = FrameStats.std(rolloffVals);   // std_rolloff

        return result;
    }

    @Override
    protected String doToJson() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("type", "spectralStats");
        map.put("frameSize", frameSize);
        map.put("hop", hop);
        map.put("fitted", true);
        return Json.write(map);
    }

    /**
     * Reconstructs a SpectralStatsExtractor from its JSON representation.
     *
     * @param json JSON string produced by {@link #toJson()}; must not be null or empty
     * @return new SpectralStatsExtractor instance with the same configuration
     * @throws IllegalArgumentException if JSON is malformed, missing fields, or has invalid values
     */
    public static SpectralStatsExtractor fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }
        Object parsed = Json.parse(json);
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("JSON root must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> map = (Map<String, Object>) parsed;

        String type = JsonFields.requireString(map, "type");
        if (!"spectralStats".equals(type)) {
            throw new IllegalArgumentException("Expected type 'spectralStats', got '" + type + "'");
        }

        int frameSize = JsonFields.requireInt(map, "frameSize");
        int hop = JsonFields.requireInt(map, "hop");
        boolean fitted = JsonFields.requireBoolean(map, "fitted");

        // Rely on constructor for validation (like MfccExtractor)
        SpectralStatsExtractor extractor = new SpectralStatsExtractor(frameSize, hop);
        if (fitted) {
            extractor.markFitted();
        }
        return extractor;
    }
}