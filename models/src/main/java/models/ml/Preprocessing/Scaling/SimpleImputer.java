package models.ml.Preprocessing.Scaling;

import java.util.*;

public class SimpleImputer implements Transformer {
    public enum Strategy {
        MEAN, MEDIAN, MOST_FREQUENT, CONSTANT
    }

    private Strategy strategy;
    private double constantValue;
    private double[] fillValues;
    private int numFeatures;
    private boolean fitted = false;

    public SimpleImputer(Strategy strategy) {
        this(strategy, 0.0);
    }

    public SimpleImputer(Strategy strategy, double constantValue) {
        if (strategy == null) {
            throw new IllegalArgumentException("Strategy cannot be null");
        }
        this.strategy = strategy;
        this.constantValue = constantValue;
    }

    @Override
    public void fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input matrix cannot be null or empty");
        }
        numFeatures = X[0].length;
        fillValues = new double[numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            List<Double> values = new ArrayList<>();
            for (double[] row : X) {
                if (row.length != numFeatures) {
                    throw new IllegalArgumentException("All rows must have the same number of features");
                }
                double val = row[j];
                if (!Double.isNaN(val)) {
                    values.add(val);
                }
            }

            if (values.isEmpty()) {
                // All NaN column - fallback to 0 or constant
                fillValues[j] = (strategy == Strategy.CONSTANT) ? constantValue : 0.0;
            } else {
                switch (strategy) {
                    case MEAN:
                        double sum = 0;
                        for (double v : values) sum += v;
                        fillValues[j] = sum / values.size();
                        break;
                    case MEDIAN:
                        values.sort(null);
                        int mid = values.size() / 2;
                        if (values.size() % 2 == 0) {
                            fillValues[j] = (values.get(mid - 1) + values.get(mid)) / 2.0;
                        } else {
                            fillValues[j] = values.get(mid);
                        }
                        break;
                    case MOST_FREQUENT:
                        Map<Double, Integer> counts = new HashMap<>();
                        for (double v : values) {
                            counts.put(v, counts.getOrDefault(v, 0) + 1);
                        }
                        double mostFreq = values.get(0);
                        int maxCount = 0;
                        for (Map.Entry<Double, Integer> e : counts.entrySet()) {
                            if (e.getValue() > maxCount || (e.getValue() == maxCount && e.getKey() < mostFreq)) {
                                maxCount = e.getValue();
                                mostFreq = e.getKey();
                            }
                        }
                        fillValues[j] = mostFreq;
                        break;
                    case CONSTANT:
                        fillValues[j] = constantValue;
                        break;
                }
            }
        }

        fitted = true;
    }

    @Override
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Imputer must be fitted before transform");
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
            throw new IllegalStateException("Imputer must be fitted before transform");
        }
        if (row == null || row.length != numFeatures) {
            throw new IllegalArgumentException("Row must have exactly " + numFeatures + " features");
        }
        double[] result = new double[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            double val = row[j];
            if (Double.isNaN(val)) {
                result[j] = fillValues[j];
            } else {
                result[j] = val;
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
            throw new IllegalStateException("Imputer must be fitted before toJson");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\"strategy\":\"").append(strategy.name()).append("\"");
        sb.append(",\"constantValue\":").append(constantValue);
        sb.append(",\"fillValues\":[");
        for (int j = 0; j < fillValues.length; j++) {
            sb.append(fillValues[j]);
            if (j < fillValues.length - 1) sb.append(",");
        }
        sb.append("],\"numFeatures\":").append(numFeatures).append("}");
        return sb.toString();
    }

    public static SimpleImputer fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }

        // Parse strategy - required field
        Strategy strategy = null;
        int strategyStart = json.indexOf("\"strategy\":\"");
        if (strategyStart < 0) {
            throw new IllegalArgumentException("Missing 'strategy' field");
        }
        int start = strategyStart + 12;
        int end = json.indexOf("\"", start);
        if (end <= start) {
            throw new IllegalArgumentException("Malformed 'strategy' field");
        }
        try {
            strategy = Strategy.valueOf(json.substring(start, end));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid strategy: " + json.substring(start, end));
        }

        // Parse constantValue - required field
        double constantValue = 0.0;
        int constStart = json.indexOf("\"constantValue\":");
        if (constStart < 0) {
            throw new IllegalArgumentException("Missing 'constantValue' field");
        }
        start = constStart + 16;
        end = json.indexOf(",", start);
        if (end < 0) end = json.indexOf("}", start);
        if (end <= start) {
            throw new IllegalArgumentException("Malformed 'constantValue' field");
        }
        String valStr = json.substring(start, end).trim();
        try {
            constantValue = Double.parseDouble(valStr);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid 'constantValue': " + valStr);
        }

        SimpleImputer imputer = new SimpleImputer(strategy, constantValue);

        // Parse fillValues - required field
        int fillStart = json.indexOf("\"fillValues\":[");
        int fillEnd = json.indexOf("]", fillStart);
        if (fillStart < 0 || fillEnd <= fillStart) {
            throw new IllegalArgumentException("Missing or malformed 'fillValues' field");
        }
        String fillStr = json.substring(fillStart + 13, fillEnd).trim();
        fillStr = fillStr.replace("[", "").replace("]", "");
        if (fillStr.isEmpty()) {
            throw new IllegalArgumentException("'fillValues' array cannot be empty");
        }
        String[] parts = fillStr.split(",");
        imputer.fillValues = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                imputer.fillValues[i] = Double.parseDouble(parts[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid number in 'fillValues': " + parts[i].trim());
            }
        }
        imputer.numFeatures = parts.length;
        imputer.fitted = true;

        // Parse numFeatures - required field
        int numFeaturesStart = json.indexOf("\"numFeatures\":");
        if (numFeaturesStart < 0) {
            throw new IllegalArgumentException("Missing 'numFeatures' field");
        }
        start = numFeaturesStart + 14;
        end = json.indexOf("}", start);
        if (end < 0) end = json.length();
        String numStr = json.substring(start, end).trim();
        try {
            imputer.numFeatures = Integer.parseInt(numStr);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid 'numFeatures': " + numStr);
        }

        if (imputer.fillValues.length != imputer.numFeatures) {
            throw new IllegalArgumentException("'fillValues' length does not match 'numFeatures'");
        }

        return imputer;
    }
}