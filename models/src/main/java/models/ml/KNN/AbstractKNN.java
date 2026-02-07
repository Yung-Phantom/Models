package models.ml.KNN;

import java.util.List;
import java.util.Map;

/**
 * Abstract class for K-Nearest Neighbours algorithm.
 * 
 * @author Kotei Justice
 * @version 1.2
 */
public class AbstractKNN {
    public double[][] dataset;
    public List<Map<Integer, Double>> sparseDataset;
    public int numSamples;
    public int numFeatures;
    public double[] weights;
    public double[][] distances;
    public String method;
    public int p;
    private boolean sparse = false;

    /**
     * Constructor.
     * 
     * @param dataset The dataset to use.
     * @param method  The method to use, either "euclidean", "manhattan",
     *                "minkowski", or "cosine".
     */
    public AbstractKNN(double[][] dataset, String method) {
        this(dataset, method, 2);
    }

    /**
     * Constructor.
     * 
     * @param dataset The dataset to use.
     * @param method  The method to use, either "euclidean", "manhattan",
     *                "minkowski", or "cosine".
     * @param p       The power to use for Minkowski distance.
     */
    public AbstractKNN(List<Map<Integer, Double>> sparseDataset, String method, int p) {
        this.sparseDataset = sparseDataset;
        this.numSamples = sparseDataset.size();
        this.method = method.toLowerCase();
        this.p = p;
        this.sparse = true;
    }

    public AbstractKNN(List<Map<Integer, Double>> sparseDataset, String method) {
        this(sparseDataset, method, 2);
    }

    /**
     * Constructor.
     * 
     * @param dataset The dataset to use.
     * @param method  The method to use, either "euclidean", "manhattan",
     *                "minkowski", or "cosine".
     * @param p       The power to use for Minkowski distance.
     */
    public AbstractKNN(double[][] dataset, String method, int p) {
        this.dataset = dataset;
        this.numFeatures = dataset[0].length;
        this.numSamples = dataset.length;
        this.method = method.toLowerCase();
        this.p = p;
        this.sparse = false;
    }

    /**
     * Compute the distance between two points.
     * 
     * @param x The first point.
     * @param y The second point.
     * @return The distance between the two points.
     */
    public double distance(double[] x, double[] y) {
        if (sparse) {
            throw new IllegalStateException("Dense distance called in sparse mode");
        }

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

    public double distance(Map<Integer, ? extends Number> x,
            Map<Integer, ? extends Number> y) {
        if (!sparse) {
            throw new IllegalStateException("Sparse distance called in dense mode");
        }

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

    /**
     * Compute the Euclidean distance between two points.
     * 
     * @param x The first point.
     * @param y The second point.
     * @return The Euclidean distance between the two points.
     */
    private double euclidean(double[] x, double[] y) {
        double sum = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            double diff = x[i] - y[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Compute the Manhattan distance between two points.
     * 
     * @param x The first point.
     * @param y The second point.
     * @return The Manhattan distance between the two points.
     */
    private double manhattan(double[] x, double[] y) {
        double sum = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            sum += Math.abs(x[i] - y[i]);
        }
        return sum;
    }

    /**
     * Compute the Minkowski distance between two points.
     * 
     * @param x The first point.
     * @param y The second point.
     * @param p The power to use.
     * @return The Minkowski distance between the two points.
     */
    private double minkowski(double[] x, double[] y, int p) {
        double sum = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            sum += Math.pow(Math.abs(x[i] - y[i]), p);
        }
        return Math.pow(sum, 1.0 / p);
    }

    /**
     * Compute the cosine distance between two points.
     * 
     * @param x The first point.
     * @param y The second point.
     * @return The cosine distance between the two points.
     */
    private double cosine(double[] x, double[] y) {
        double dot = 0.0, normX = 0.0, normY = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            dot += x[i] * y[i];
            normX += x[i] * x[i];
            normY += y[i] * y[i];
        }
        return 1.0 - (dot / (Math.sqrt(normX) * Math.sqrt(normY)));
    }

    private double euclideanSparse(Map<Integer, ? extends Number> x,
            Map<Integer, ? extends Number> y) {
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

    private double minkowskiSparse(Map<Integer, ? extends Number> x,
            Map<Integer, ? extends Number> y,
            int p) {
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

        return 1.0 - (dot / (Math.sqrt(normX) * Math.sqrt(normY)));
    }

    /**
     * Compute the distances between all points in the query set and all points in
     * the dataset.
     * 
     * @param query The query set.
     * @return The matrix of distances, where distances[i][j] is the distance
     *         between the i-th query point and the j-th dataset point.
     */
    public double[][] getDistances(double[][] query) {
        if (sparse)
            throw new IllegalStateException("Dense getDistances called in sparse mode");
        distances = new double[query.length][numSamples];
        for (int np = 0; np < query.length; np++) {
            for (int i = 0; i < numSamples; i++) {
                distances[np][i] = distance(query[np], dataset[i]);
            }
        }
        return distances;
    }

    /**
     * Compute distances between sparse query vectors and sparse dataset vectors.
     *
     * @param query Sparse query set
     * @return Distance matrix [query][dataset]
     */
    public double[][] getDistances(List<Map<Integer, Double>> query) {
        if (!sparse)
            throw new IllegalStateException("Sparse getDistances called in dense mode");

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