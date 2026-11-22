package models.ml.Preprocessing.Audio;

import java.util.ArrayList;
import java.util.List;

/**
 * Framing utilities for audio processing.
 * <p>
 * Not instantiable; all methods are static.
 * Not thread-safe; callers must ensure thread confinement.
 */
public final class Framing {

    private Framing() {}

    /**
     * Splits a signal into overlapping frames.
     * <p>
     * If {@code x.length < frameSize}, returns a single zero-padded frame.
     * Otherwise returns {@code 1 + (x.length - frameSize) / hop} frames,
     * starting at indices 0, hop, 2*hop, ...; trailing partial frames are dropped.
     *
     * @param x         input signal; must not be null
     * @param frameSize frame size in samples; must be > 0
     * @param hop       hop size in samples; must be > 0
     * @return list of frames (each a double[] of length frameSize)
     * @throws IllegalArgumentException if x is null, frameSize <= 0, or hop <= 0
     */
    public static List<double[]> frames(double[] x, int frameSize, int hop) {
        if (x == null) {
            throw new IllegalArgumentException("x cannot be null");
        }
        if (frameSize <= 0) {
            throw new IllegalArgumentException("frameSize must be > 0, got " + frameSize);
        }
        if (hop <= 0) {
            throw new IllegalArgumentException("hop must be > 0, got " + hop);
        }

        List<double[]> result = new ArrayList<>();
        if (x.length < frameSize) {
            // Single zero-padded frame
            double[] frame = new double[frameSize];
            System.arraycopy(x, 0, frame, 0, x.length);
            result.add(frame);
            return result;
        }

        int numFrames = 1 + (x.length - frameSize) / hop;
        for (int i = 0; i < numFrames; i++) {
            int start = i * hop;
            double[] frame = new double[frameSize];
            System.arraycopy(x, start, frame, 0, frameSize);
            result.add(frame);
        }
        return result;
    }
}