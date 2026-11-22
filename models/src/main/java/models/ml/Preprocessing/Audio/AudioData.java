package models.ml.Preprocessing.Audio;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable container for audio samples.
 * <p>
 * Samples are stored in [-1.0, 1.0] range.
 * Not thread-safe; immutable after construction.
 */
public final class AudioData {

    private final int sampleRate;
    private final double[] samples;

    private AudioData(int sampleRate, double[] samples) {
        this.sampleRate = sampleRate;
        this.samples = samples;
    }

    /**
     * Creates an AudioData instance with validation.
     *
     * @param sampleRate sample rate in Hz; must be > 0
     * @param samples    audio samples; must not be null or empty, all values must be finite and in [-1.0, 1.0]
     * @return new AudioData instance
     * @throws IllegalArgumentException if any validation fails
     */
    public static AudioData of(int sampleRate, double[] samples) {
        if (sampleRate <= 0) {
            throw new IllegalArgumentException("sampleRate must be > 0, got " + sampleRate);
        }
        if (samples == null) {
            throw new IllegalArgumentException("samples array cannot be null");
        }
        if (samples.length == 0) {
            throw new IllegalArgumentException("samples array cannot be empty");
        }
        for (int i = 0; i < samples.length; i++) {
            double v = samples[i];
            if (Double.isNaN(v) || Double.isInfinite(v) || v < -1.0 || v > 1.0) {
                throw new IllegalArgumentException(
                        "sample at index " + i + " must be a finite number in [-1.0, 1.0], got " + v);
            }
        }
        // Defensive copy
        double[] copy = Arrays.copyOf(samples, samples.length);
        return new AudioData(sampleRate, copy);
    }

    /**
     * Returns the sample rate in Hz.
     */
    public int getSampleRate() {
        return sampleRate;
    }

    /**
     * Returns a copy of the samples array.
     *
     * @return copy of samples
     */
    public double[] getSamples() {
        return Arrays.copyOf(samples, samples.length);
    }

    /**
     * Returns the number of samples.
     */
    public int length() {
        return samples.length;
    }

    /**
     * Returns the duration in seconds.
     */
    public double getDurationSeconds() {
        return (double) samples.length / sampleRate;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof AudioData)) return false;
        AudioData that = (AudioData) o;
        return sampleRate == that.sampleRate
                && Arrays.equals(samples, that.samples);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(sampleRate);
        result = 31 * result + Arrays.hashCode(samples);
        return result;
    }

    @Override
    public String toString() {
        return "AudioData[sampleRate=" + sampleRate + ", length=" + samples.length + "]";
    }
}