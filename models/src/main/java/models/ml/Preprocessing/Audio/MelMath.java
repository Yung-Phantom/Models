package models.ml.Preprocessing.Audio;

import java.util.Arrays;

/**
 * Mel-scale and filterbank utilities for MFCC computation.
 * <p>
 * Not instantiable; all methods are static.
 * Not thread-safe; callers must ensure thread confinement.
 */
public final class MelMath {

    private MelMath() {}

    /**
     * Converts frequency in Hz to Mel scale.
     * Formula: mel = 2595 * log10(1 + f / 700)
     *
     * @param f frequency in Hz; must be >= 0
     * @return Mel value
     */
    public static double hzToMel(double f) {
        if (f < 0) {
            throw new IllegalArgumentException("frequency must be >= 0, got " + f);
        }
        return 2595.0 * Math.log10(1.0 + f / 700.0);
    }

    /**
     * Converts Mel scale back to frequency in Hz.
     * Formula: f = 700 * (10^(mel/2595) - 1)
     *
     * @param mel Mel value
     * @return frequency in Hz
     */
    public static double melToHz(double mel) {
        return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
    }

    /**
     * Creates a Mel filterbank.
     *
     * @param sampleRate  sample rate in Hz; must be > 0
     * @param fftSize     FFT size (power of two); must be > 0
     * @param numFilters  number of triangular filters; must be > 0
     * @return 2D array [numFilters][numBins] where numBins = fftSize/2 + 1;
     *         each row is a triangular filter
     * @throws IllegalArgumentException if any parameter <= 0
     */
    public static double[][] createFilterbank(int sampleRate, int fftSize, int numFilters) {
        if (sampleRate <= 0) {
            throw new IllegalArgumentException("sampleRate must be > 0");
        }
        if (fftSize <= 0) {
            throw new IllegalArgumentException("fftSize must be > 0");
        }
        if (numFilters <= 0) {
            throw new IllegalArgumentException("numFilters must be > 0");
        }

        int numBins = fftSize / 2 + 1;
        double[][] filterbank = new double[numFilters][numBins];

        double melMin = hzToMel(0);
        double melMax = hzToMel(sampleRate / 2.0);
        double[] melEdges = new double[numFilters + 2];
        for (int i = 0; i < numFilters + 2; i++) {
            melEdges[i] = melMin + (double) i * (melMax - melMin) / (numFilters + 1);
        }
        double[] hzEdges = new double[numFilters + 2];
        for (int i = 0; i < numFilters + 2; i++) {
            hzEdges[i] = melToHz(melEdges[i]);
        }
        double[] binFreqs = new double[numBins];
        for (int k = 0; k < numBins; k++) {
            binFreqs[k] = (double) k * sampleRate / fftSize;
        }

        for (int m = 0; m < numFilters; m++) {
            double fLeft = hzEdges[m];
            double fCenter = hzEdges[m + 1];
            double fRight = hzEdges[m + 2];

            for (int k = 0; k < numBins; k++) {
                double f = binFreqs[k];
                if (f >= fLeft && f <= fCenter) {
                    filterbank[m][k] = (f - fLeft) / (fCenter - fLeft);
                } else if (f > fCenter && f <= fRight) {
                    filterbank[m][k] = (fRight - f) / (fRight - fCenter);
                }
            }
        }
        return filterbank;
    }

    /**
     * Applies DCT-II (unnormalized) to the log filterbank energies.
     *
     * @param logEnergies array of length M (numFilters); must not be null
     * @param numCoeffs   number of coefficients to keep (1 <= numCoeffs <= M); must be > 0 and <= logEnergies.length
     * @return array of length numCoeffs containing c_n for n=0..numCoeffs-1
     *         where c_n = sum_{m=0}^{M-1} logE_m * cos(pi * n * (m + 0.5) / M)
     * @throws IllegalArgumentException if logEnergies is null, numCoeffs < 1, or numCoeffs > logEnergies.length
     */
    public static double[] dct(double[] logEnergies, int numCoeffs) {
        if (logEnergies == null) {
            throw new IllegalArgumentException("logEnergies cannot be null");
        }
        int M = logEnergies.length;
        if (numCoeffs <= 0 || numCoeffs > M) {
            throw new IllegalArgumentException("numCoeffs must be in [1, " + M + "], got " + numCoeffs);
        }
        double[] coeffs = new double[numCoeffs];
        for (int n = 0; n < numCoeffs; n++) {
            double sum = 0.0;
            for (int m = 0; m < M; m++) {
                sum += logEnergies[m] * Math.cos(Math.PI * n * (m + 0.5) / M);
            }
            coeffs[n] = sum;
        }
        return coeffs;
    }
}