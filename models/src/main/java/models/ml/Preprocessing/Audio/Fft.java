package models.ml.Preprocessing.Audio;

/**
 * Fast Fourier Transform utilities.
 * <p>
 * Not instantiable; all methods are static.
 * Not thread-safe; callers must ensure thread confinement.
 */
public final class Fft {

    private Fft() {}

    /**
     * Returns the magnitude spectrum of a real-valued frame.
     * <p>
     * The input is zero-padded or truncated to {@code fftSize} (must be a power of two
     * and {@code >= frame.length}). An iterative radix-2 Cooley-Tukey FFT is used.
     *
     * @param frame    real-valued time-domain samples; must not be null
     * @param fftSize  FFT size (power of two, >= frame.length); must be > 0
     * @return magnitudes for bins 0..fftSize/2 (length fftSize/2 + 1)
     * @throws IllegalArgumentException if frame is null, fftSize is not a power of two,
     *         fftSize < frame.length, or fftSize <= 0
     */
    public static double[] magnitudeSpectrum(double[] frame, int fftSize) {
        if (frame == null) {
            throw new IllegalArgumentException("frame cannot be null");
        }
        if (fftSize <= 0) {
            throw new IllegalArgumentException("fftSize must be > 0, got " + fftSize);
        }
        if ((fftSize & (fftSize - 1)) != 0) {
            throw new IllegalArgumentException("fftSize must be a power of two, got " + fftSize);
        }
        if (fftSize < frame.length) {
            throw new IllegalArgumentException("fftSize (" + fftSize + ") must be >= frame.length (" + frame.length + ")");
        }

        // Initialize complex array: real = frame (padded), imag = 0
        double[] real = new double[fftSize];
        double[] imag = new double[fftSize];
        System.arraycopy(frame, 0, real, 0, frame.length);

        // Iterative radix-2 FFT (in-place)
        // Bit-reversal permutation
        for (int i = 1, j = 0; i < fftSize; i++) {
            int bit = fftSize >> 1;
            for (; j >= bit; bit >>= 1) {
                j -= bit;
            }
            j += bit;
            if (i < j) {
                double tmpR = real[i];
                double tmpI = imag[i];
                real[i] = real[j];
                imag[i] = imag[j];
                real[j] = tmpR;
                imag[j] = tmpI;
            }
        }

        // Cooley-Tukey iterative
        for (int len = 2; len <= fftSize; len <<= 1) {
            double angle = -2.0 * Math.PI / len;
            double wlenR = Math.cos(angle);
            double wlenI = Math.sin(angle);
            for (int i = 0; i < fftSize; i += len) {
                double wR = 1.0;
                double wI = 0.0;
                for (int j = 0; j < len / 2; j++) {
                    int u = i + j;
                    int v = i + j + len / 2;
                    double tR = wR * real[v] - wI * imag[v];
                    double tI = wR * imag[v] + wI * real[v];
                    real[v] = real[u] - tR;
                    imag[v] = imag[u] - tI;
                    real[u] = real[u] + tR;
                    imag[u] = imag[u] + tI;
                    // Multiply w by wlen
                    double nwR = wR * wlenR - wI * wlenI;
                    double nwI = wR * wlenI + wI * wlenR;
                    wR = nwR;
                    wI = nwI;
                }
            }
        }

        // Compute magnitudes for bins 0..fftSize/2
        int numBins = fftSize / 2 + 1;
        double[] magnitudes = new double[numBins];
        for (int k = 0; k < numBins; k++) {
            magnitudes[k] = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
        }
        return magnitudes;
    }
}