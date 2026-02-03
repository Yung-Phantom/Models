package models.ml.KNN;

/**
 * Abstract class for K-Nearest Neighbours algorithm.
 * 
 * @author Kotei Justice
 * @version 1.2
 */
public class AbstractKNN {
    public double[][] dataset;
    public int numSamples;
    public int numFeatures;
    public double[] weights;
    public double[][] distances;
    public String method;
    public int p;

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
    public AbstractKNN(double[][] dataset, String method, int p) {
        this.dataset = dataset;
        this.numFeatures = dataset[0].length - 1;
        this.numSamples = dataset.length;
        this.method = method.toLowerCase();
        this.p = p;
    }

    /**
     * Compute the distance between two points.
     * 
     * @param x The first point.
     * @param y The second point.
     * @return The distance between the two points.
     */
    public double distance(double[] x, double[] y) {
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

    /**
     * Compute the distances between all points in the query set and all points in
     * the dataset.
     * 
     * @param query The query set.
     * @return The matrix of distances, where distances[i][j] is the distance
     *         between the i-th query point and the j-th dataset point.
     */
    public double[][] getDistances(double[][] query) {
        distances = new double[query.length][numSamples];
        for (int np = 0; np < query.length; np++) {
            for (int i = 0; i < numSamples; i++) {
                distances[np][i] = distance(query[np], dataset[i]);
            }
        }
        return distances;
    }

}