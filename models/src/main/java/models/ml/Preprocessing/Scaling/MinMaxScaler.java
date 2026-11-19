package models.ml.Preprocessing.Scaling;

import java.util.Arrays;

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

        MinMaxScaler scaler = new MinMaxScaler();

        // Parse mins - required field
        int minsStart = json.indexOf("\"mins\":[");
        int minsEnd = json.indexOf("]", minsStart);
        if (minsStart < 0 || minsEnd <= minsStart) {
            throw new IllegalArgumentException("Missing or malformed 'mins' field");
        }
        String minsStr = json.substring(minsStart + 8, minsEnd).trim();
        minsStr = minsStr.replace("[", "").replace("]", "");
        if (minsStr.isEmpty()) {
            throw new IllegalArgumentException("'mins' array cannot be empty");
        }
        String[] parts = minsStr.split(",");
        scaler.mins = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                scaler.mins[i] = Double.parseDouble(parts[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid number in 'mins': " + parts[i].trim());
            }
        }

        // Parse maxs - required field
        int maxsStart = json.indexOf("\"maxs\":[");
        int maxsEnd = json.indexOf("]", maxsStart);
        if (maxsStart < 0 || maxsEnd <= maxsStart) {
            throw new IllegalArgumentException("Missing or malformed 'maxs' field");
        }
        String maxsStr = json.substring(maxsStart + 8, maxsEnd).trim();
        maxsStr = maxsStr.replace("[", "").replace("]", "");
        if (maxsStr.isEmpty()) {
            throw new IllegalArgumentException("'maxs' array cannot be empty");
        }
        parts = maxsStr.split(",");
        scaler.maxs = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                scaler.maxs[i] = Double.parseDouble(parts[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid number in 'maxs': " + parts[i].trim());
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

        if (scaler.mins.length != scaler.numFeatures || scaler.maxs.length != scaler.numFeatures) {
            throw new IllegalArgumentException("'mins'/'maxs' length does not match 'numFeatures'");
        }

        scaler.fitted = true;
        return scaler;
    }
}