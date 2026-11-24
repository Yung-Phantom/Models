package models.ml.Preprocessing.Features;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utility class for converting between sparse and dense feature vector
 * representations.
 * <p>
 * Not instantiable; all methods are static.
 * Not thread-safe; callers must ensure thread confinement.
 */
public final class FeatureVectors {

    private FeatureVectors() {
    }

    /**
     * Converts a sparse feature vector to a dense array of the given size.
     *
     * @param sparse sparse map from feature index to value
     * @param size   the feature dimensionality; must be non-negative
     * @return dense array of length {@code size}
     * @throws IllegalArgumentException if sparse is null, size is negative,
     *                                  a sparse key is {@code <} 0 or {@code >=}
     *                                  size, or a sparse key/value is null
     */
    public static double[] toDense(Map<Integer, Double> sparse, int size) {
        if (sparse == null) {
            throw new IllegalArgumentException("sparse map cannot be null");
        }
        if (size < 0) {
            throw new IllegalArgumentException("size cannot be negative: " + size);
        }
        double[] dense = new double[size];
        for (Map.Entry<Integer, Double> e : sparse.entrySet()) {
            Integer idx = e.getKey();
            Double val = e.getValue();
            if (idx == null) {
                throw new IllegalArgumentException("sparse map key cannot be null");
            }
            if (val == null) {
                throw new IllegalArgumentException("sparse map value cannot be null");
            }
            if (idx < 0 || idx >= size) {
                throw new IllegalArgumentException(
                        "sparse key out of range [0, " + size + "): " + idx);
            }
            dense[idx] = val;
        }
        return dense;
    }

    /**
     * Converts a list of sparse feature vectors to a dense matrix.
     *
     * @param rows list of sparse maps; must not be null
     * @param size the feature dimensionality; must be non-negative
     * @return dense matrix (one row per input)
     * @throws IllegalArgumentException if rows is null, size is negative,
     *                                  a row is null, a sparse key is {@code <} 0
     *                                  or {@code >=} size, or a sparse key/value is
     *                                  null
     */
    public static double[][] toDense(List<Map<Integer, Double>> rows, int size) {
        if (rows == null) {
            throw new IllegalArgumentException("rows list cannot be null");
        }
        if (size < 0) {
            throw new IllegalArgumentException("size cannot be negative: " + size);
        }
        double[][] dense = new double[rows.size()][size];
        for (int i = 0; i < rows.size(); i++) {
            Map<Integer, Double> row = rows.get(i);
            if (row == null) {
                throw new IllegalArgumentException("row at index " + i + " cannot be null");
            }
            for (Map.Entry<Integer, Double> e : row.entrySet()) {
                Integer idx = e.getKey();
                Double val = e.getValue();
                if (idx == null) {
                    throw new IllegalArgumentException("sparse map key at row " + i + " cannot be null");
                }
                if (val == null) {
                    throw new IllegalArgumentException("sparse map value at row " + i + " cannot be null");
                }
                if (idx < 0 || idx >= size) {
                    throw new IllegalArgumentException(
                            "sparse key out of range [0, " + size + ") at row " + i + ": " + idx);
                }
                dense[i][idx] = val;
            }
        }
        return dense;
    }

    /**
     * Converts a dense vector to a sparse map, dropping exact zeros.
     * Always returns a mutable HashMap (never an unmodifiable or empty singleton).
     *
     * @param row dense array; must not be null
     * @return sparse map with only non-zero entries
     * @throws IllegalArgumentException if row is null
     */
    public static Map<Integer, Double> toSparse(double[] row) {
        if (row == null) {
            throw new IllegalArgumentException("row cannot be null");
        }
        Map<Integer, Double> sparse = new HashMap<>();
        for (int i = 0; i < row.length; i++) {
            double v = row[i];
            if (v != 0.0) {
                sparse.put(i, v);
            }
        }
        return sparse;
    }

    /**
     * Converts a dense matrix to a list of sparse maps, dropping exact zeros.
     *
     * @param rows dense matrix; must not be null
     * @return list of sparse maps (one per row)
     * @throws IllegalArgumentException if rows is null
     */
    public static List<Map<Integer, Double>> toSparse(double[][] rows) {
        if (rows == null) {
            throw new IllegalArgumentException("rows cannot be null");
        }
        List<Map<Integer, Double>> result = new ArrayList<>(rows.length);
        for (double[] row : rows) {
            result.add(toSparse(row));
        }
        return result;
    }
}