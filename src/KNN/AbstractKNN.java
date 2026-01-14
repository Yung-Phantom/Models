package KNN;

/**
 * Abstract class for K-Nearest Neighbors algorithm.
 * 
 * @author Kotei Justice
 * @version 1.1
 */
public class AbstractKNN {
    private double[][] dataset;
    private double[][] newPoint;
    private double[][] distances; // now 2D: queries × dataset
    private int dataLength;

    /**
     * Constructor for the AbstractKNN class.
     * 
     * @param dataset  the dataset containing all the points.
     * @param newPoint the new point(s) to be compared with the dataset.
     * @param method   the method to be used for the KNN algorithm.
     */
    public AbstractKNN(double[][] dataset, double[][] newPoint, String method) {
        this(dataset, newPoint, method, 2);
    }

    /**
     * Constructor for the AbstractKNN class.
     * 
     * @param dataset  the dataset containing all the points.
     * @param newPoint the new point(s) to be compared with the dataset.
     * @param method   the method to be used for the KNN algorithm.
     * @param p        the value of p for the Minkowski distance.
     */
    public AbstractKNN(double[][] dataset, double[][] newPoint, String method, int p) {
        this.dataset = dataset;
        this.newPoint = newPoint;
        this.dataLength = dataset[0].length - 1;

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
        distances = new double[newPoint.length][dataset.length];
        for (int np = 0; np < newPoint.length; np++) {
            for (int i = 0; i < dataset.length; i++) {
                double sum = 0.0;
                for (int j = 0; j < dataLength; j++) {
                    double diff = dataset[i][j] - newPoint[np][j];
                    sum += diff * diff;
                }
                distances[np][i] = Math.sqrt(sum);
            }
        }
        return distances;
    }

    /** Manhattan distance for all queries */
    public double[][] computeManhattanDistance() {
        distances = new double[newPoint.length][dataset.length];
        for (int np = 0; np < newPoint.length; np++) {
            for (int i = 0; i < dataset.length; i++) {
                double sum = 0.0;
                for (int j = 0; j < dataLength; j++) {
                    sum += Math.abs(dataset[i][j] - newPoint[np][j]);
                }
                distances[np][i] = sum;
            }
        }
        return distances;
    }

    /** Minkowski distance for all queries */
    public double[][] computeMinkowskiDistance(int p) {
        distances = new double[newPoint.length][dataset.length];
        for (int np = 0; np < newPoint.length; np++) {
            for (int i = 0; i < dataset.length; i++) {
                double sum = 0.0;
                for (int j = 0; j < dataLength; j++) {
                    sum += Math.pow(Math.abs(dataset[i][j] - newPoint[np][j]), p);
                }
                distances[np][i] = Math.pow(sum, 1.0 / p);
            }
        }
        return distances;
    }

    /** Cosine distance for all queries */
    public double[][] computeCosineDistance() {
        distances = new double[newPoint.length][dataset.length];
        for (int np = 0; np < newPoint.length; np++) {
            for (int i = 0; i < dataset.length; i++) {
                double dot = 0.0, normX = 0.0, normY = 0.0;
                for (int j = 0; j < dataLength; j++) {
                    dot += dataset[i][j] * newPoint[np][j];
                    normX += dataset[i][j] * dataset[i][j];
                    normY += newPoint[np][j] * newPoint[np][j];
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
