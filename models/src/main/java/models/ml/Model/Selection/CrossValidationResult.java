package models.ml.Model.Selection;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Immutable result of a k-fold cross-validation run.
 * <p>
 * Contains per-fold metric values, plus mean and sample standard deviation per metric.
 * All collections are unmodifiable.
 *
 * @since 1.0
 */
public final class CrossValidationResult {

    private final Map<String, List<Double>> foldMetrics;
    private final Map<String, Double> meanMetrics;
    private final Map<String, Double> stdMetrics;

    /**
     * Creates a cross-validation result from per-fold metric values.
     *
     * @param foldMetrics map of metric name to list of per-fold values (length = k)
     * @throws IllegalArgumentException if foldMetrics is null, empty, or has inconsistent fold counts
     */
    public CrossValidationResult(Map<String, List<Double>> foldMetrics) {
        if (foldMetrics == null || foldMetrics.isEmpty()) {
            throw new IllegalArgumentException("foldMetrics cannot be null or empty");
        }

        int k = -1;
        for (List<Double> values : foldMetrics.values()) {
            if (values == null || values.isEmpty()) {
                throw new IllegalArgumentException("Metric values cannot be null or empty");
            }
            if (k == -1) {
                k = values.size();
            } else if (values.size() != k) {
                throw new IllegalArgumentException("All metrics must have the same number of folds");
            }
        }
        if (k < 2) {
            throw new IllegalArgumentException("Must have at least 2 folds");
        }

        this.foldMetrics = Collections.unmodifiableMap(new LinkedHashMap<>(foldMetrics.entrySet().stream()
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        e -> Collections.unmodifiableList(new java.util.ArrayList<>(e.getValue())),
                        (a, b) -> a,
                        LinkedHashMap::new
                ))));
        this.meanMetrics = Collections.unmodifiableMap(computeMeans(this.foldMetrics));
        this.stdMetrics = Collections.unmodifiableMap(computeStdDevs(this.foldMetrics, this.meanMetrics));
    }

    private static Map<String, Double> computeMeans(Map<String, List<Double>> foldMetrics) {
        return foldMetrics.entrySet().stream()
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        e -> e.getValue().stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN),
                        (a, b) -> a,
                        LinkedHashMap::new
                ));
    }

    private static Map<String, Double> computeStdDevs(Map<String, List<Double>> foldMetrics, Map<String, Double> means) {
        return foldMetrics.entrySet().stream()
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        e -> {
                            double mean = means.get(e.getKey());
                            List<Double> values = e.getValue();
                            if (values.size() <= 1) return 0.0;
                            double sumSq = values.stream()
                                    .mapToDouble(v -> (v - mean) * (v - mean))
                                    .sum();
                            return Math.sqrt(sumSq / (values.size() - 1));
                        },
                        (a, b) -> a,
                        LinkedHashMap::new
                ));
    }

    /**
     * Returns the per-fold values for a given metric.
     *
     * @param metricName metric name
     * @return unmodifiable list of per-fold values, or null if metric not present
     */
    public List<Double> getFoldValues(String metricName) {
        return foldMetrics.get(metricName);
    }

    /**
     * Returns the mean value for a given metric across all folds.
     *
     * @param metricName metric name
     * @return mean value, or NaN if metric not present
     */
    public double getMean(String metricName) {
        return meanMetrics.getOrDefault(metricName, Double.NaN);
    }

    /**
     * Returns the sample standard deviation for a given metric across all folds.
     *
     * @param metricName metric name
     * @return sample standard deviation, or NaN if metric not present
     */
    public double getStdDev(String metricName) {
        return stdMetrics.getOrDefault(metricName, Double.NaN);
    }

    /**
     * Returns all metric names present in this result.
     *
     * @return unmodifiable set of metric names
     */
    public java.util.Set<String> getMetricNames() {
        return Collections.unmodifiableSet(foldMetrics.keySet());
    }

    /**
     * Returns the number of folds (k).
     *
     * @return number of folds
     */
    public int getK() {
        if (foldMetrics.isEmpty()) return 0;
        return foldMetrics.values().iterator().next().size();
    }

    /**
     * Returns a string representation showing mean ± std for each metric.
     *
     * @return formatted string
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("CrossValidationResult (k=").append(getK()).append("):\n");
        for (String metric : foldMetrics.keySet()) {
            double mean = meanMetrics.get(metric);
            double std = stdMetrics.get(metric);
            sb.append("  ").append(metric).append(": ")
                    .append(String.format("%.6f", mean)).append(" ± ")
                    .append(String.format("%.6f", std)).append("\n");
        }
        return sb.toString().trim();
    }

    /**
     * Returns a detailed string with per-fold values.
     *
     * @return formatted string with fold breakdown
     */
    public String toDetailedString() {
        StringBuilder sb = new StringBuilder();
        sb.append("CrossValidationResult (k=").append(getK()).append("):\n");
        for (String metric : foldMetrics.keySet()) {
            sb.append("  ").append(metric).append(":\n");
            List<Double> values = foldMetrics.get(metric);
            for (int i = 0; i < values.size(); i++) {
                sb.append("    Fold ").append(i + 1).append(": ")
                        .append(String.format("%.6f", values.get(i))).append("\n");
            }
            sb.append("    Mean: ").append(String.format("%.6f", meanMetrics.get(metric)))
                    .append(", StdDev: ").append(String.format("%.6f", stdMetrics.get(metric))).append("\n");
        }
        return sb.toString().trim();
    }
}