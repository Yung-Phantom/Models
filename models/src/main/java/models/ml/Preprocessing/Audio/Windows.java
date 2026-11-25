package models.ml.Preprocessing.Audio;

/**
 * Window functions for audio processing.
 * <p>
 * Not instantiable; all methods are static.
 * Not thread-safe; callers must ensure thread confinement.
 */
public final class Windows {

    private Windows() {}

    /**
     * Returns a periodic Hann window of the specified length.
     * <p>
     * Formula: w[i] = 0.5 - 0.5 * cos(2 * pi * i / n) for i = 0..n-1.
     * The window is periodic (i.e., w[0] = w[n] = 0 conceptually, with w[n-1] != 0).
     *
     * @param n window length; must be > 0
     * @return Hann window of length n
     * @throws IllegalArgumentException if n {@code <=} 0
     */
    public static double[] hann(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be > 0, got " + n);
        }
        double[] w = new double[n];
        if (n == 1) {
            w[0] = 1.0;
            return w;
        }
        for (int i = 0; i < n; i++) {
            w[i] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / n);
        }
        return w;
    }

    /**
     * Applies a window to a frame in-place.
     *
     * @param frame the frame to window; must not be null
     * @param window the window coefficients; must not be null and length must equal frame.length
     * @return the windowed frame (same array as input, modified in-place)
     * @throws IllegalArgumentException if frame or window is null, or lengths don't match
     */
    public static double[] apply(double[] frame, double[] window) {
        if (frame == null) {
            throw new IllegalArgumentException("frame cannot be null");
        }
        if (window == null) {
            throw new IllegalArgumentException("window cannot be null");
        }
        if (frame.length != window.length) {
            throw new IllegalArgumentException("frame length (" + frame.length + ") must equal window length (" + window.length + ")");
        }
        for (int i = 0; i < frame.length; i++) {
            frame[i] *= window[i];
        }
        return frame;
    }
}