package models.ml.KNN;

/**
 * Abstract class for K-Nearest Neighbors algorithm.
 * 
 * @author Kotei Justice
 * @version 1.1
 */
public class AbstractKNN {
    private double[][] dataset;
    private double[][] points;
    public int numSamples;
    private int numFeatures;
    private double[] weights;
    private double[][] distances;

    /**
     * Constructor for the AbstractKNN class.
     * 
     * @param dataset the dataset containing all the points.
     * @param points  the new point(s) to be compared with the dataset.
     * @param method  the method to be used for the KNN algorithm.
     */
    public AbstractKNN(double[][] dataset, double[][] points, String method) {
        this(dataset, points, method, 2);
    }

    /**
     * Constructor for the AbstractKNN class.
     * 
     * @param dataset the dataset containing all the points.
     * @param points  the new point(s) to be compared with the dataset.
     * @param method  the method to be used for the KNN algorithm.
     * @param p       the value of p for the Minkowski distance.
     */
    public AbstractKNN(double[][] dataset, double[][] points, String method, int p) {
        this.dataset = dataset;
        this.points = points;
        this.numFeatures = dataset[0].length - 1;
        this.numSamples = dataset.length;

        String normalized = method.toLowerCase();
        switch (normalized) {
            case "euclidean":
                computeEuclideanDistance();
                break;
            case "manhattan":
                computeManhattanDistance();
                break;
            case "minkowski":
                computeMinkowskiDistance(p);
                break;
            case "cosine":
                computeCosineDistance();
                break;
            default:
                System.out.println("Method not supported: " + method);
                break;
        }
    }

    /** Euclidean distance for all queries */
    public double[][] computeEuclideanDistance() {
        distances = new double[points.length][numSamples];
        for (int np = 0; np < points.length; np++) {
            for (int i = 0; i < numSamples; i++) {
                double sum = 0.0;
                for (int j = 0; j < numFeatures; j++) {
                    double diff = dataset[i][j] - points[np][j];
                    sum += diff * diff;
                }
                distances[np][i] = Math.sqrt(sum);
            }
        }
        return distances;
    }

    /** Manhattan distance for all queries */
    public double[][] computeManhattanDistance() {
        distances = new double[points.length][numSamples];
        for (int np = 0; np < points.length; np++) {
            for (int i = 0; i < numSamples; i++) {
                double sum = 0.0;
                for (int j = 0; j < numFeatures; j++) {
                    sum += Math.abs(dataset[i][j] - points[np][j]);
                }
                distances[np][i] = sum;
            }
        }
        return distances;
    }

    /** Minkowski distance for all queries */
    public double[][] computeMinkowskiDistance(int p) {
        distances = new double[points.length][numSamples];
        for (int np = 0; np < points.length; np++) {
            for (int i = 0; i < numSamples; i++) {
                double sum = 0.0;
                for (int j = 0; j < numFeatures; j++) {
                    sum += Math.pow(Math.abs(dataset[i][j] - points[np][j]), p);
                }
                distances[np][i] = Math.pow(sum, 1.0 / p);
            }
        }
        return distances;
    }

    /** Cosine distance for all queries */
    public double[][] computeCosineDistance() {
        distances = new double[points.length][numSamples];
        for (int np = 0; np < points.length; np++) {
            for (int i = 0; i < numSamples; i++) {
                double dot = 0.0, normX = 0.0, normY = 0.0;
                for (int j = 0; j < numFeatures; j++) {
                    dot += dataset[i][j] * points[np][j];
                    normX += dataset[i][j] * dataset[i][j];
                    normY += points[np][j] * points[np][j];
                }
                double similarity = dot / (Math.sqrt(normX) * Math.sqrt(normY));
                distances[np][i] = 1.0 - similarity;
            }
        }
        return distances;
    }

    /** Getter for distances */
    public double[][] getDistances() {
        return distances;
    }
}
