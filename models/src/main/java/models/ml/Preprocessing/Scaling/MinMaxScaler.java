package models.ml.Preprocessing.Scaling;

import models.ml.Utils.Json;
import models.ml.Utils.JsonFields;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class MinMaxScaler implements Transformer {
    private double[] mins;
    private double[] maxs;
    private int numFeatures;
    private boolean fitted = false;

    public MinMaxScaler() {}

    @Override
    public void fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input matrix cannot be null or empty");
        }
        numFeatures = X[0].length;
        mins = new double[numFeatures];
        maxs = new double[numFeatures];
        boolean[] hasValue = new boolean[numFeatures];

        Arrays.fill(mins, Double.POSITIVE_INFINITY);
        Arrays.fill(maxs, Double.NEGATIVE_INFINITY);

        for (double[] row : X) {
            if (row.length != numFeatures) {
                throw new IllegalArgumentException("All rows must have the same number of features");
            }
            for (int j = 0; j < numFeatures; j++) {
                double val = row[j];
                if (!Double.isNaN(val)) {
                    hasValue[j] = true;
                    if (val < mins[j]) mins[j] = val;
                    if (val > maxs[j]) maxs[j] = val;
                }
            }
        }

        // Handle features with no valid values (all NaN)
        for (int j = 0; j < numFeatures; j++) {
            if (!hasValue[j]) {
                mins[j] = 0.0;
                maxs[j] = 1.0; // Will map to 0
            } else if (maxs[j] == mins[j]) {
                // Constant feature - maps to 0
                maxs[j] = mins[j] + 1.0;
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
                result[j] = (val - mins[j]) / (maxs[j] - mins[j]);
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
        sb.append("{\"mins\":[");
        for (int j = 0; j < mins.length; j++) {
            sb.append(mins[j]);
            if (j < mins.length - 1) sb.append(",");
        }
        sb.append("],\"maxs\":[");
        for (int j = 0; j < maxs.length; j++) {
            sb.append(maxs[j]);
            if (j < maxs.length - 1) sb.append(",");
        }
        sb.append("],\"numFeatures\":").append(numFeatures).append("}");
        return sb.toString();
    }

    public static MinMaxScaler fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }

        Object parsed;
        try {
            parsed = Json.parse(json);
        } catch (IllegalArgumentException e) {
            String msg = e.getMessage();
            if (msg != null) {
                if (msg.contains("Expected digit") || msg.contains("Invalid number")) {
                    if (json.contains("\"mins\"")) {
                        throw new IllegalArgumentException("Invalid number in 'mins': " + msg);
                    }
                    if (json.contains("\"maxs\"")) {
                        throw new IllegalArgumentException("Invalid number in 'maxs': " + msg);
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

        // Parse mins - required field
        if (!map.containsKey("mins")) {
            throw new IllegalArgumentException("Missing required field: mins");
        }
        Object minsObj = map.get("mins");
        if (!(minsObj instanceof List)) {
            throw new IllegalArgumentException("'mins' must be an array");
        }
        @SuppressWarnings("unchecked")
        List<Object> minsList = (List<Object>) minsObj;
        if (minsList.isEmpty()) {
            throw new IllegalArgumentException("'mins' array cannot be empty");
        }
        MinMaxScaler scaler = new MinMaxScaler();
        scaler.mins = new double[minsList.size()];
        for (int i = 0; i < minsList.size(); i++) {
            Object v = minsList.get(i);
            if (!(v instanceof Number)) {
                throw new IllegalArgumentException("Invalid number in 'mins' at index " + i);
            }
            scaler.mins[i] = ((Number) v).doubleValue();
        }

        // Parse maxs - required field
        if (!map.containsKey("maxs")) {
            throw new IllegalArgumentException("Missing required field: maxs");
        }
        Object maxsObj = map.get("maxs");
        if (!(maxsObj instanceof List)) {
            throw new IllegalArgumentException("'maxs' must be an array");
        }
        @SuppressWarnings("unchecked")
        List<Object> maxsList = (List<Object>) maxsObj;
        if (maxsList.isEmpty()) {
            throw new IllegalArgumentException("'maxs' array cannot be empty");
        }
        scaler.maxs = new double[maxsList.size()];
        for (int i = 0; i < maxsList.size(); i++) {
            Object v = maxsList.get(i);
            if (!(v instanceof Number)) {
                throw new IllegalArgumentException("Invalid number in 'maxs' at index " + i);
            }
            scaler.maxs[i] = ((Number) v).doubleValue();
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

        if (scaler.mins.length != scaler.numFeatures || scaler.maxs.length != scaler.numFeatures) {
            throw new IllegalArgumentException("'mins'/'maxs' length does not match 'numFeatures'");
        }

        scaler.fitted = true;
        return scaler;
    }
}