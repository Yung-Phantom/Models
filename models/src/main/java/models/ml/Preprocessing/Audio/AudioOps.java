package models.ml.Preprocessing.Audio;

/**
 * Stateless, pure audio processing operations on {@link AudioData} and raw arrays.
 * <p>
 * Not instantiable; all methods are static.
 * Not thread-safe; callers must ensure thread confinement.
 */
public final class AudioOps {

    private AudioOps() {}

    /**
     * Resamples audio data to a target sample rate using linear interpolation.
     *
     * @param audio       input audio; must not be null
     * @param targetRate  target sample rate in Hz; must be > 0
     * @return resampled AudioData with same duration (approximately)
     * @throws IllegalArgumentException if audio is null or targetRate <= 0
     */
    public static AudioData resample(AudioData audio, int targetRate) {
        if (audio == null) {
            throw new IllegalArgumentException("audio cannot be null");
        }
        if (targetRate <= 0) {
            throw new IllegalArgumentException("targetRate must be > 0, got " + targetRate);
        }
        int sourceRate = audio.getSampleRate();
        if (sourceRate == targetRate) {
            return audio;
        }
        double[] src = audio.getSamples();
        int n = src.length;
        int newLen = Math.max(1, (int) Math.round((double) n * targetRate / sourceRate));
        double[] dst = new double[newLen];
        for (int i = 0; i < newLen; i++) {
            double srcPos = i * (double) sourceRate / targetRate;
            int srcIdx = (int) Math.floor(srcPos);
            double frac = srcPos - srcIdx;
            if (srcIdx >= n - 1) {
                dst[i] = src[n - 1];
            } else {
                dst[i] = src[srcIdx] + frac * (src[srcIdx + 1] - src[srcIdx]);
            }
        }
        return AudioData.of(targetRate, dst);
    }

    /**
     * Truncates or zero-pads an array to the specified length.
     *
     * @param x   input array; must not be null
     * @param len target length; must be > 0
     * @return array of length {@code len}
     * @throws IllegalArgumentException if x is null or len <= 0
     */
    public static double[] fixLength(double[] x, int len) {
        if (x == null) {
            throw new IllegalArgumentException("input array cannot be null");
        }
        if (len <= 0) {
            throw new IllegalArgumentException("len must be > 0, got " + len);
        }
        double[] result = new double[len];
        int copyLen = Math.min(x.length, len);
        System.arraycopy(x, 0, result, 0, copyLen);
        return result;
    }
}