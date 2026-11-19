package models.ml.Preprocessing.Scaling;

import java.util.Arrays;

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

        ZScoreScaler scaler = new ZScoreScaler();

        // Parse means - required field
        int meansStart = json.indexOf("\"means\":[");
        int meansEnd = json.indexOf("]", meansStart);
        if (meansStart < 0 || meansEnd <= meansStart) {
            throw new IllegalArgumentException("Missing or malformed 'means' field");
        }
        String meansStr = json.substring(meansStart + 9, meansEnd).trim();
        meansStr = meansStr.replace("[", "").replace("]", "");
        if (meansStr.isEmpty()) {
            throw new IllegalArgumentException("'means' array cannot be empty");
        }
        String[] parts = meansStr.split(",");
        scaler.means = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                scaler.means[i] = Double.parseDouble(parts[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid number in 'means': " + parts[i].trim());
            }
        }

        // Parse stds - required field
        int stdsStart = json.indexOf("\"stds\":[");
        int stdsEnd = json.indexOf("]", stdsStart);
        if (stdsStart < 0 || stdsEnd <= stdsStart) {
            throw new IllegalArgumentException("Missing or malformed 'stds' field");
        }
        String stdsStr = json.substring(stdsStart + 8, stdsEnd).trim();
        stdsStr = stdsStr.replace("[", "").replace("]", "");
        if (stdsStr.isEmpty()) {
            throw new IllegalArgumentException("'stds' array cannot be empty");
        }
        parts = stdsStr.split(",");
        scaler.stds = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                scaler.stds[i] = Double.parseDouble(parts[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid number in 'stds': " + parts[i].trim());
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

        if (scaler.means.length != scaler.numFeatures || scaler.stds.length != scaler.numFeatures) {
            throw new IllegalArgumentException("'means'/'stds' length does not match 'numFeatures'");
        }

        scaler.fitted = true;
        return scaler;
    }
}