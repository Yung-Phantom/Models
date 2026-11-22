package models.ml.Preprocessing.Audio;

import models.ml.Utils.Json;
import models.ml.Utils.JsonFields;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Feature extractor that computes Mel-Frequency Cepstral Coefficients (MFCCs).
 * <p>
 * Pipeline: resample to targetSampleRate -> frame -> Hann window -> FFT -> power spectrum ->
 * Mel filterbank -> log(energy + 1e-10) -> DCT-II -> keep first numCoeffs coefficients.
 * Output: mean and population std per coefficient, coefficient-major order.
 * Feature vector size = 2 * numCoeffs.
 * <p>
 * Not thread-safe.
 */
public final class MfccExtractor extends AbstractAudioFeatureExtractor {

    private final int targetSampleRate;
    private final int frameSize;
    private final int hop;
    private final int numFilters;
    private final int numCoeffs;

    // Precomputed filterbank and Hann window
    private final double[][] filterbank;
    private final double[] hannWindow;

    /**
     * Creates an MFCC feature extractor.
     *
     * @param targetSampleRate target sample rate for resampling; must be > 0
     * @param frameSize        frame size in samples (power of two); must be > 0
     * @param hop              hop size in samples; must be > 0
     * @param numFilters       number of Mel filters; must be >= 1
     * @param numCoeffs        number of MFCC coefficients to keep; must be in [1, numFilters]
     * @throws IllegalArgumentException if any parameter is invalid
     */
    public MfccExtractor(int targetSampleRate, int frameSize, int hop, int numFilters, int numCoeffs) {
        if (targetSampleRate <= 0) {
            throw new IllegalArgumentException("targetSampleRate must be > 0, got " + targetSampleRate);
        }
        if (frameSize <= 0) {
            throw new IllegalArgumentException("frameSize must be > 0, got " + frameSize);
        }
        if ((frameSize & (frameSize - 1)) != 0) {
            throw new IllegalArgumentException("frameSize must be a power of two, got " + frameSize);
        }
        if (hop <= 0) {
            throw new IllegalArgumentException("hop must be > 0, got " + hop);
        }
        if (numFilters <= 0) {
            throw new IllegalArgumentException("numFilters must be >= 1, got " + numFilters);
        }
        if (numCoeffs <= 0 || numCoeffs > numFilters) {
            throw new IllegalArgumentException("numCoeffs must be in [1, " + numFilters + "], got " + numCoeffs);
        }
        this.targetSampleRate = targetSampleRate;
        this.frameSize = frameSize;
        this.hop = hop;
        this.numFilters = numFilters;
        this.numCoeffs = numCoeffs;

        // Precompute filterbank and Hann window once
        this.filterbank = MelMath.createFilterbank(targetSampleRate, frameSize, numFilters);
        this.hannWindow = Windows.hann(frameSize);
    }

    /**
     * Returns the target sample rate.
     */
    public int getTargetSampleRate() {
        return targetSampleRate;
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

    /**
     * Returns the number of Mel filters.
     */
    public int getNumFilters() {
        return numFilters;
    }

    /**
     * Returns the number of MFCC coefficients.
     */
    public int getNumCoeffs() {
        return numCoeffs;
    }

    @Override
    public int getFeatureSize() {
        return 2 * numCoeffs;
    }

    @Override
    protected double[] extractDense(AudioData audio) {
        // Resample if needed
        AudioData resampled = audio.getSampleRate() == targetSampleRate
                ? audio : AudioOps.resample(audio, targetSampleRate);

        double[] samples = resampled.getSamples();
        var frames = Framing.frames(samples, frameSize, hop);
        int numFrames = frames.size();

        double[][] coeffsFrames = new double[numFrames][numCoeffs];

        for (int i = 0; i < numFrames; i++) {
            double[] frame = frames.get(i);
            // Apply Hann window using Windows.apply (modifies copy in-place)
            double[] windowed = Arrays.copyOf(frame, frame.length);
            Windows.apply(windowed, hannWindow);

            // Power spectrum
            double[] mag = Fft.magnitudeSpectrum(windowed, frameSize);
            double[] power = new double[mag.length];
            for (int k = 0; k < mag.length; k++) {
                power[k] = mag[k] * mag[k];
            }

            // Mel filterbank
            double[] energies = new double[numFilters];
            for (int m = 0; m < numFilters; m++) {
                double sum = 0.0;
                for (int k = 0; k < power.length; k++) {
                    sum += filterbank[m][k] * power[k];
                }
                energies[m] = Math.log(sum + 1e-10);
            }

            // DCT
            double[] coeffs = MelMath.dct(energies, numCoeffs);
            coeffsFrames[i] = coeffs;
        }

        // Compute mean and std per coefficient (coefficient-major)
        double[] result = new double[2 * numCoeffs];
        for (int c = 0; c < numCoeffs; c++) {
            double sum = 0.0;
            for (int i = 0; i < numFrames; i++) {
                sum += coeffsFrames[i][c];
            }
            double mean = sum / numFrames;

            double sumSq = 0.0;
            for (int i = 0; i < numFrames; i++) {
                double diff = coeffsFrames[i][c] - mean;
                sumSq += diff * diff;
            }
            double std = Math.sqrt(sumSq / numFrames);

            result[2 * c] = mean;
            result[2 * c + 1] = std;
        }

        return result;
    }

    @Override
    protected String doToJson() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("type", "mfcc");
        map.put("targetSampleRate", targetSampleRate);
        map.put("frameSize", frameSize);
        map.put("hop", hop);
        map.put("numFilters", numFilters);
        map.put("numCoeffs", numCoeffs);
        map.put("fitted", true);
        return Json.write(map);
    }

    /**
     * Reconstructs an MfccExtractor from its JSON representation.
     *
     * @param json JSON string produced by {@link #toJson()}; must not be null or empty
     * @return new MfccExtractor instance with the same configuration
     * @throws IllegalArgumentException if JSON is malformed, missing fields, or has invalid values
     */
    public static MfccExtractor fromJson(String json) {
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
        if (!"mfcc".equals(type)) {
            throw new IllegalArgumentException("Expected type 'mfcc', got '" + type + "'");
        }

        int targetSampleRate = JsonFields.requireInt(map, "targetSampleRate");
        int frameSize = JsonFields.requireInt(map, "frameSize");
        int hop = JsonFields.requireInt(map, "hop");
        int numFilters = JsonFields.requireInt(map, "numFilters");
        int numCoeffs = JsonFields.requireInt(map, "numCoeffs");
        boolean fitted = JsonFields.requireBoolean(map, "fitted");

        // Rely on constructor for validation
        MfccExtractor extractor = new MfccExtractor(targetSampleRate, frameSize, hop, numFilters, numCoeffs);
        if (fitted) {
            extractor.markFitted();
        }
        return extractor;
    }
}