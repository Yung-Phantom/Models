package models.ml.Preprocessing.Scaling;

import models.ml.Utils.Json;

import java.util.List;
import java.util.Map;

public class ZScoreScaler implements Transformer {
    private double[] means;
    private double[] stds;
    private int numFeatures;
    private boolean fitted = false;

    public ZScoreScaler() {}

    @Override
    public void fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input matrix cannot be null or empty");
        }
        numFeatures = X[0].length;
        means = new double[numFeatures];
        stds = new double[numFeatures];
        int[] counts = new int[numFeatures];

        // First pass: compute means
        for (double[] row : X) {
            if (row.length != numFeatures) {
                throw new IllegalArgumentException("All rows must have the same number of features");
            }
            for (int j = 0; j < numFeatures; j++) {
                double val = row[j];
                if (!Double.isNaN(val)) {
                    means[j] += val;
                    counts[j]++;
                }
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            if (counts[j] > 0) {
                means[j] /= counts[j];
            } else {
                means[j] = 0.0;
            }
        }

        // Second pass: compute stds
        for (double[] row : X) {
            for (int j = 0; j < numFeatures; j++) {
                double val = row[j];
                if (!Double.isNaN(val)) {
                    double diff = val - means[j];
                    stds[j] += diff * diff;
                }
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            if (counts[j] > 1) {
                stds[j] = Math.sqrt(stds[j] / (counts[j] - 1));
            } else if (counts[j] == 1) {
                stds[j] = 0.0;
            } else {
                stds[j] = 0.0;
            }
            // If std is 0 (constant feature), use scale 1
            if (stds[j] == 0.0) {
                stds[j] = 1.0;
            }
        }

        fitted = true;
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
                result[j] = (val - means[j]) / stds[j];
            }
        }
        return result;
    }

    @Override
    public double[][] fitTransform(double[][] X) {
        fit(X);
        return transform(X);
    }

    @Override
    public String toJson() {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before toJson");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\"means\":[");
        for (int j = 0; j < means.length; j++) {
            sb.append(means[j]);
            if (j < means.length - 1) sb.append(",");
        }
        sb.append("],\"stds\":[");
        for (int j = 0; j < stds.length; j++) {
            sb.append(stds[j]);
            if (j < stds.length - 1) sb.append(",");
        }
        sb.append("],\"numFeatures\":").append(numFeatures).append("}");
        return sb.toString();
    }

    public static ZScoreScaler fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }

        Object parsed;
        try {
            parsed = Json.parse(json);
        } catch (IllegalArgumentException e) {
            String msg = e.getMessage();
            // Provide more specific error messages for common cases
            if (msg != null) {
                if (msg.contains("Expected digit") || msg.contains("Invalid number")) {
                    if (json.contains("\"means\"")) {
                        throw new IllegalArgumentException("Invalid number in 'means': " + msg);
                    }
                    if (json.contains("\"stds\"")) {
                        throw new IllegalArgumentException("Invalid number in 'stds': " + msg);
                    }
                }
            }
            throw new IllegalArgumentException("Invalid JSON: " + msg);
        }
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("JSON root must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> map = (Map<String, Object>) parsed;

        // Parse means - required field
        if (!map.containsKey("means")) {
            throw new IllegalArgumentException("Missing required field: means");
        }
        Object meansObj = map.get("means");
        if (!(meansObj instanceof List)) {
            throw new IllegalArgumentException("'means' must be an array");
        }
        @SuppressWarnings("unchecked")
        List<Object> meansList = (List<Object>) meansObj;
        if (meansList.isEmpty()) {
            throw new IllegalArgumentException("'means' array cannot be empty");
        }
        ZScoreScaler scaler = new ZScoreScaler();
        scaler.means = new double[meansList.size()];
        for (int i = 0; i < meansList.size(); i++) {
            Object v = meansList.get(i);
            if (!(v instanceof Number)) {
                throw new IllegalArgumentException("Invalid number in 'means' at index " + i);
            }
            scaler.means[i] = ((Number) v).doubleValue();
        }

        // Parse stds - required field
        if (!map.containsKey("stds")) {
            throw new IllegalArgumentException("Missing required field: stds");
        }
        Object stdsObj = map.get("stds");
        if (!(stdsObj instanceof List)) {
            throw new IllegalArgumentException("'stds' must be an array");
        }
        @SuppressWarnings("unchecked")
        List<Object> stdsList = (List<Object>) stdsObj;
        if (stdsList.isEmpty()) {
            throw new IllegalArgumentException("'stds' array cannot be empty");
        }
        scaler.stds = new double[stdsList.size()];
        for (int i = 0; i < stdsList.size(); i++) {
            Object v = stdsList.get(i);
            if (!(v instanceof Number)) {
                throw new IllegalArgumentException("Invalid number in 'stds' at index " + i);
            }
            scaler.stds[i] = ((Number) v).doubleValue();
        }

        // Parse numFeatures - required field
        if (!map.containsKey("numFeatures")) {
            throw new IllegalArgumentException("Missing required field: numFeatures");
        }
        Object numFeaturesObj = map.get("numFeatures");
        if (!(numFeaturesObj instanceof Number)) {
            throw new IllegalArgumentException("'numFeatures' must be a number");
        }
        double numFeaturesDouble = ((Number) numFeaturesObj).doubleValue();
        if (numFeaturesDouble != Math.floor(numFeaturesDouble)) {
            throw new IllegalArgumentException("'numFeatures' must be a whole number, got " + numFeaturesDouble);
        }
        scaler.numFeatures = ((Number) numFeaturesObj).intValue();

        if (scaler.means.length != scaler.numFeatures || scaler.stds.length != scaler.numFeatures) {
            throw new IllegalArgumentException("'means'/'stds' length does not match 'numFeatures'");
        }

        scaler.fitted = true;
        return scaler;
    }
}