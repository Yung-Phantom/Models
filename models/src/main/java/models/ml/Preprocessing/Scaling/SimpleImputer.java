package models.ml.Preprocessing.Scaling;

import models.ml.Utils.Json;
import models.ml.Utils.JsonFields;

import java.util.*;
import java.util.List;
import java.util.Map;

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

        Object parsed;
        try {
            parsed = Json.parse(json);
        } catch (IllegalArgumentException e) {
            String msg = e.getMessage();
            if (msg != null) {
                if (msg.contains("Expected digit") || msg.contains("Invalid number")) {
                    if (json.contains("\"fillValues\"")) {
                        throw new IllegalArgumentException("Invalid number in 'fillValues': " + msg);
                    }
                    if (json.contains("\"constantValue\"")) {
                        throw new IllegalArgumentException("Invalid 'constantValue': " + msg);
                    }
                    if (json.contains("\"numFeatures\"")) {
                        throw new IllegalArgumentException("Invalid 'numFeatures': " + msg);
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

        // Parse strategy - required field
        if (!map.containsKey("strategy")) {
            throw new IllegalArgumentException("Missing required field: strategy");
        }
        String strategyStr = JsonFields.requireString(map, "strategy");
        Strategy strategy;
        try {
            strategy = Strategy.valueOf(strategyStr);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid strategy: " + strategyStr);
        }

        // Parse constantValue - required field
        if (!map.containsKey("constantValue")) {
            throw new IllegalArgumentException("Missing required field: constantValue");
        }
        Object constantValueObj = map.get("constantValue");
        if (!(constantValueObj instanceof Number)) {
            throw new IllegalArgumentException("'constantValue' must be a number");
        }
        double constantValue = ((Number) constantValueObj).doubleValue();

        SimpleImputer imputer = new SimpleImputer(strategy, constantValue);

        // Parse fillValues - required field
        if (!map.containsKey("fillValues")) {
            throw new IllegalArgumentException("Missing required field: fillValues");
        }
        Object fillValuesObj = map.get("fillValues");
        if (!(fillValuesObj instanceof List)) {
            throw new IllegalArgumentException("'fillValues' must be an array");
        }
        @SuppressWarnings("unchecked")
        List<Object> fillValuesList = (List<Object>) fillValuesObj;
        if (fillValuesList.isEmpty()) {
            throw new IllegalArgumentException("'fillValues' array cannot be empty");
        }
        imputer.fillValues = new double[fillValuesList.size()];
        for (int i = 0; i < fillValuesList.size(); i++) {
            Object v = fillValuesList.get(i);
            if (!(v instanceof Number)) {
                throw new IllegalArgumentException("Invalid number in 'fillValues' at index " + i);
            }
            imputer.fillValues[i] = ((Number) v).doubleValue();
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
        imputer.numFeatures = ((Number) numFeaturesObj).intValue();

        if (imputer.fillValues.length != imputer.numFeatures) {
            throw new IllegalArgumentException("'fillValues' length does not match 'numFeatures'");
        }

        imputer.fitted = true;
        return imputer;
    }
}