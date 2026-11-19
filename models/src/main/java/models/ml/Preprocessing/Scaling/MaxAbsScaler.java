package models.ml.Preprocessing.Scaling;

import java.util.*;
import java.util.Arrays;

public class MaxAbsScaler implements Transformer {
    private double[] maxAbs;
    private int numFeatures;
    private boolean fitted = false;

    public MaxAbsScaler() {}

    @Override
    public void fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input matrix cannot be null or empty");
        }
        numFeatures = X[0].length;
        maxAbs = new double[numFeatures];
        Arrays.fill(maxAbs, 0.0);

        for (double[] row : X) {
            if (row.length != numFeatures) {
                throw new IllegalArgumentException("All rows must have the same number of features");
            }
            for (int j = 0; j < numFeatures; j++) {
                double val = row[j];
                if (!Double.isNaN(val)) {
                    double absVal = Math.abs(val);
                    if (absVal > maxAbs[j]) {
                        maxAbs[j] = absVal;
                    }
                }
            }
        }

        // Handle features with maxAbs = 0 (all zeros or all NaN)
        for (int j = 0; j < numFeatures; j++) {
            if (maxAbs[j] == 0.0) {
                maxAbs[j] = 1.0; // No scaling
            }
        }

        fitted = true;
    }

    public void fitSparse(List<Map<Integer, Double>> X) {
        if (X == null || X.isEmpty()) {
            throw new IllegalArgumentException("Input cannot be null or empty");
        }

        // Determine numFeatures as max key + 1
        int maxKey = -1;
        for (Map<Integer, Double> row : X) {
            if (row == null) continue;
            for (Integer key : row.keySet()) {
                if (key < 0) {
                    throw new IllegalArgumentException("Negative feature index: " + key);
                }
                if (key > maxKey) {
                    maxKey = key;
                }
            }
        }
        numFeatures = maxKey + 1;
        if (numFeatures == 0) {
            throw new IllegalArgumentException("No features found in sparse data");
        }

        maxAbs = new double[numFeatures];
        Arrays.fill(maxAbs, 0.0);

        for (Map<Integer, Double> row : X) {
            if (row == null) continue;
            for (Map.Entry<Integer, Double> e : row.entrySet()) {
                int idx = e.getKey();
                double val = e.getValue();
                if (!Double.isNaN(val)) {
                    double absVal = Math.abs(val);
                    if (absVal > maxAbs[idx]) {
                        maxAbs[idx] = absVal;
                    }
                }
            }
        }

        // Handle features with maxAbs = 0 (never seen or all zeros/NaN)
        for (int j = 0; j < numFeatures; j++) {
            if (maxAbs[j] == 0.0) {
                maxAbs[j] = 1.0; // No scaling
            }
        }

        fitted = true;
    }

    public List<Map<Integer, Double>> fitTransformSparse(List<Map<Integer, Double>> X) {
        fitSparse(X);
        return transformSparse(X);
    }

    @Override
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }
        if (X == null) {
            return new double[0][];
        }
        double[][] result = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            result[i] = transform(X[i]);
        }
        return result;
    }

    @Override
    public double[] transform(double[] row) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }
        if (row == null || row.length != numFeatures) {
            throw new IllegalArgumentException("Row must have exactly " + numFeatures + " features");
        }
        double[] result = new double[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            double val = row[j];
            if (Double.isNaN(val)) {
                result[j] = Double.NaN;
            } else {
                result[j] = val / maxAbs[j];
            }
        }
        return result;
    }

    @Override
    public double[][] fitTransform(double[][] X) {
        fit(X);
        return transform(X);
    }

    public Map<Integer, Double> transformSparse(Map<Integer, Double> row) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }
        if (row == null) {
            return new HashMap<>();
        }
        Map<Integer, Double> result = new HashMap<>();
        for (Map.Entry<Integer, Double> e : row.entrySet()) {
            int idx = e.getKey();
            double val = e.getValue();
            if (idx < 0) {
                throw new IllegalArgumentException("Negative feature index: " + idx);
            }
            if (idx >= numFeatures) {
                // Drop unseen features (no learned scale)
                continue;
            }
            if (Double.isNaN(val)) {
                // NaN passes through unchanged (same as dense path)
                result.put(idx, Double.NaN);
            } else {
                double scaled = val / maxAbs[idx];
                // Only keep non-zero entries to maintain sparsity
                if (scaled != 0.0) {
                    result.put(idx, scaled);
                }
            }
        }
        return result;
    }

    public List<Map<Integer, Double>> transformSparse(List<Map<Integer, Double>> X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }
        if (X == null) {
            return new ArrayList<>();
        }
        List<Map<Integer, Double>> result = new ArrayList<>(X.size());
        for (Map<Integer, Double> row : X) {
            result.add(transformSparse(row));
        }
        return result;
    }

    @Override
    public String toJson() {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before toJson");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\"maxAbs\":[");
        for (int j = 0; j < maxAbs.length; j++) {
            sb.append(maxAbs[j]);
            if (j < maxAbs.length - 1) sb.append(",");
        }
        sb.append("],\"numFeatures\":").append(numFeatures).append("}");
        return sb.toString();
    }

    public static MaxAbsScaler fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }

        MaxAbsScaler scaler = new MaxAbsScaler();

        // Parse maxAbs - required field
        int maxAbsStart = json.indexOf("\"maxAbs\":[");
        int maxAbsEnd = json.indexOf("]", maxAbsStart);
        if (maxAbsStart < 0 || maxAbsEnd <= maxAbsStart) {
            throw new IllegalArgumentException("Missing or malformed 'maxAbs' field");
        }
        String maxAbsStr = json.substring(maxAbsStart + 9, maxAbsEnd).trim();
        maxAbsStr = maxAbsStr.replace("[", "").replace("]", "");
        if (maxAbsStr.isEmpty()) {
            throw new IllegalArgumentException("'maxAbs' array cannot be empty");
        }
        String[] parts = maxAbsStr.split(",");
        scaler.maxAbs = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                scaler.maxAbs[i] = Double.parseDouble(parts[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid number in 'maxAbs': " + parts[i].trim());
            }
        }

        // Parse numFeatures - required field
        int numFeaturesStart = json.indexOf("\"numFeatures\":");
        if (numFeaturesStart < 0) {
            throw new IllegalArgumentException("Missing 'numFeatures' field");
        }
        int start = numFeaturesStart + 14;
        int end = json.indexOf("}", start);
        if (end < 0) end = json.length();
        String numStr = json.substring(start, end).trim();
        try {
            scaler.numFeatures = Integer.parseInt(numStr);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid 'numFeatures': " + numStr);
        }

        if (scaler.maxAbs.length != scaler.numFeatures) {
            throw new IllegalArgumentException("'maxAbs' length (" + scaler.maxAbs.length + ") does not match 'numFeatures' (" + scaler.numFeatures + ")");
        }

        scaler.fitted = true;
        return scaler;
    }
}