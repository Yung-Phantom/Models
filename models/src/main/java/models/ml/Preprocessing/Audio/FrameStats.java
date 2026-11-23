package models.ml.Preprocessing.Audio;

/**
 * Utility class for computing frame statistics (mean and population standard deviation).
 * <p>
 * Not instantiable; all methods are static.
 * Not thread-safe.
 */
public final class FrameStats {

    private FrameStats() {}

    /**
     * Computes the arithmetic mean of an array.
     *
     * @param values array of values; must not be null or empty
     * @return mean of the values
     * @throws IllegalArgumentException if values is null or empty
     */
    public static double mean(double[] values) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("values array cannot be null or empty");
        }
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }

    /**
     * Computes the population standard deviation of an array.
     *
     * @param values array of values; must not be null or empty
     * @return population standard deviation
     * @throws IllegalArgumentException if values is null or empty
     */
    public static double std(double[] values) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("values array cannot be null or empty");
        }
        double m = mean(values);
        double sumSq = 0.0;
        for (double v : values) {
            double diff = v - m;
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq / values.length);
    }
}