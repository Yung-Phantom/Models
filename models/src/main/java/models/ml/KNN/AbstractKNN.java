package models.ml.KNN;

import java.util.*;

public class AbstractKNN {
    public double[][] dataset;
    public List<Map<Integer, Double>> sparseDataset;
    private boolean sparse = false;

    public String method;
    public int p;

    public int numSamples;
    public int numFeatures;

    // public double[] weights;
    public double[][] distances;

    public AbstractKNN(List<Map<Integer, Double>> sparseDataset, String method, int p) {
        
        this.sparse = true;
        if (method == null)
            throw new IllegalArgumentException("Method cannot be null");

        if (sparseDataset == null || sparseDataset.isEmpty())
            throw new IllegalArgumentException("Empty sparse dataset");

        for (Map<Integer, Double> row : sparseDataset) {
            if (row == null)
                throw new IllegalArgumentException("Sparse dataset contains null row");
            if (row.size() != numFeatures)
                throw new IllegalArgumentException("Inconsistent row length");
            for (double val : row.values()) {
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException("Dataset contains NaN or Infinity");
                }
            }
        }

        this.sparseDataset = sparseDataset;
        this.numSamples = sparseDataset.size();

        this.method = method.trim().toLowerCase();
        if (!Set.of("euclidean", "manhattan", "minkowski", "cosine").contains(this.method))
            throw new IllegalArgumentException("Unsupported method: " + method);

        this.p = p;
    }

    public AbstractKNN(double[][] dataset, String method, int p) {
        
        this.sparse = false;
        if (dataset == null || dataset.length == 0)
            throw new IllegalArgumentException("Empty dataset");

        if (method == null)
            throw new IllegalArgumentException("Method cannot be null");

        this.dataset = dataset;

        this.numFeatures = dataset[0].length;
        for (double[] row : dataset) {
            if (row == null)
                throw new IllegalArgumentException("Dense dataset contains null row");
            if (row.length != numFeatures)
                throw new IllegalArgumentException("Inconsistent row length");
            for (double val : row) {
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException("Dataset contains NaN or Infinity");
                }
            }
        }

        this.numSamples = dataset.length;

        this.method = method.trim().toLowerCase();
        if (!Set.of("euclidean", "manhattan", "minkowski", "cosine").contains(this.method))
            throw new IllegalArgumentException("Unsupported method: " + method);

        this.p = p;
    }

    public double distance(double[] x, double[] y) {
        if (sparse) {
            throw new IllegalStateException("Dense distance called in sparse mode");
        }
        if (x == null || y == null)
            throw new IllegalArgumentException("Vectors cannot be null");
        if (x.length != numFeatures || y.length != numFeatures)
            throw new IllegalArgumentException("Vector dimension mismatch");

        switch (method) {
            case "euclidean":
                return euclidean(x, y);
            case "manhattan":
                return manhattan(x, y);
            case "minkowski":
                return minkowski(x, y, p);
            case "cosine":
                return cosine(x, y);
            default:
                throw new IllegalArgumentException("Unsupported method: " + method);
        }
    }

    public double distance(Map<Integer, ? extends Number> x, Map<Integer, ? extends Number> y) {
        if (!sparse) {
            throw new IllegalStateException("Sparse distance called in dense mode");
        }
        if (x == null || y == null)
            throw new IllegalArgumentException("Vectors cannot be null");
        if (x.size() != numFeatures || y.size() != numFeatures)
            throw new IllegalArgumentException("Vector dimension mismatch");

        switch (method) {
            case "euclidean":
                return euclideanSparse(x, y);
            case "manhattan":
                return manhattanSparse(x, y);
            case "minkowski":
                return minkowskiSparse(x, y, p);
            case "cosine":
                return cosineSparse(x, y);
            default:
                throw new IllegalArgumentException("Unsupported method: " + method);
        }
    }

    private double euclidean(double[] x, double[] y) {

        double sum = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            double diff = x[i] - y[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    private double manhattan(double[] x, double[] y) {

        double sum = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            sum += Math.abs(x[i] - y[i]);
        }
        return sum;
    }

    private double minkowski(double[] x, double[] y, int p) {
        if (p <= 0)
            throw new IllegalArgumentException("p must be > 0");

        double sum = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            sum += Math.pow(Math.abs(x[i] - y[i]), p);
        }
        return Math.pow(sum, 1.0 / p);
    }

    private double cosine(double[] x, double[] y) {

        double dot = 0.0, normX = 0.0, normY = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            dot += x[i] * y[i];
            normX += x[i] * x[i];
            normY += y[i] * y[i];
        }

        if (normX == 0.0 && normY == 0.0)
            return 0.0;

        if (normX == 0.0 || normY == 0.0)
            return 1.0;

        double denom = Math.sqrt(normX) * Math.sqrt(normY);
        return 1.0 - (dot / denom);
    }

    private double euclideanSparse(Map<Integer, ? extends Number> x, Map<Integer, ? extends Number> y) {
        double sum = 0.0;

        // iterate smaller map
        Map<Integer, ? extends Number> a = x.size() < y.size() ? x : y;
        Map<Integer, ? extends Number> b = (a == x) ? y : x;

        for (var e : a.entrySet()) {
            double diff = e.getValue().doubleValue();
            Number val = b.get(e.getKey());
            if (val != null) {
                diff -= val.doubleValue();
            }
            sum += diff * diff;
        }

        for (var e : b.entrySet()) {
            if (!a.containsKey(e.getKey())) {
                double v = e.getValue().doubleValue();
                sum += v * v;
            }
        }

        return Math.sqrt(sum);
    }

    private double manhattanSparse(Map<Integer, ? extends Number> x,
            Map<Integer, ? extends Number> y) {
        double sum = 0.0;

        Map<Integer, ? extends Number> a = x.size() < y.size() ? x : y;
        Map<Integer, ? extends Number> b = (a == x) ? y : x;

        for (var e : a.entrySet()) {
            double diff = e.getValue().doubleValue();
            Number val = b.get(e.getKey());
            if (val != null) {
                diff -= val.doubleValue();
            }
            sum += Math.abs(diff);
        }

        for (var e : b.entrySet()) {
            if (!a.containsKey(e.getKey())) {
                sum += Math.abs(e.getValue().doubleValue());
            }
        }

        return sum;
    }

    private double minkowskiSparse(Map<Integer, ? extends Number> x, Map<Integer, ? extends Number> y, int p) {
        if (p <= 0)
            throw new IllegalArgumentException("p must be > 0");

        double sum = 0.0;

        Map<Integer, ? extends Number> a = x.size() < y.size() ? x : y;
        Map<Integer, ? extends Number> b = (a == x) ? y : x;

        for (var e : a.entrySet()) {
            double diff = Math.abs(e.getValue().doubleValue());
            Number val = b.get(e.getKey());
            if (val != null) {
                diff = Math.abs(e.getValue().doubleValue() - val.doubleValue());
            }
            sum += Math.pow(diff, p);
        }

        for (var e : b.entrySet()) {
            if (!a.containsKey(e.getKey())) {
                sum += Math.pow(Math.abs(e.getValue().doubleValue()), p);
            }
        }

        return Math.pow(sum, 1.0 / p);
    }

    private double cosineSparse(Map<Integer, ? extends Number> x,
            Map<Integer, ? extends Number> y) {
        double dot = 0.0, normX = 0.0, normY = 0.0;

        for (var e : x.entrySet()) {
            double xv = e.getValue().doubleValue();
            Number yVal = y.get(e.getKey());
            double yv = yVal != null ? yVal.doubleValue() : 0.0;
            dot += xv * yv;
            normX += xv * xv;
        }
        for (var e : y.entrySet()) {
            double v = e.getValue().doubleValue();
            normY += v * v;
        }

        if (normX == 0.0 && normY == 0.0)
            return 0.0;

        if (normX == 0.0 || normY == 0.0)
            return 1.0;

        double denom = Math.sqrt(normX) * Math.sqrt(normY);

        return 1.0 - (dot / denom);
    }

    public double[][] getDistances(double[][] query) {
        if (query == null) {
            throw new NullPointerException("Query cannot be null");
        }
        if (query.length == 0) {
            throw new IllegalArgumentException("Query cannot be empty");
        }
        for (double[] q : query) {
            for (double val : q) {
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException("Query contains NaN or Infinity");
                }
            }
        }

        distances = new double[query.length][numSamples];
        for (int np = 0; np < query.length; np++) {
            for (int i = 0; i < numSamples; i++) {
                distances[np][i] = distance(query[np], dataset[i]);
            }
        }
        return distances;
    }

    public double[][] getDistances(List<Map<Integer, Double>> query) {
        if (query == null) {
            throw new NullPointerException("Query cannot be null");
        }
        if (query.isEmpty()) {
            throw new IllegalArgumentException("Query cannot be empty");
        }

        for (Map<Integer, Double> q : query) {
            for (double val : q.values()) {
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException("Query contains NaN or Infinity");
                }
            }
        }

        double[][] out = new double[query.size()][numSamples];

        for (int i = 0; i < query.size(); i++) {
            Map<Integer, Double> q = query.get(i);
            for (int j = 0; j < numSamples; j++) {
                out[i][j] = distance(q, sparseDataset.get(j));
            }
        }
        return out;
    }

}