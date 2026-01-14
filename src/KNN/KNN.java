package KNN;

import java.util.*;

/**
 * K-Nearest Neighbors algorithm.
 * 
 * @author Kotei Justice
 * @version 1.1
 */
public class KNN {
    /**
     * Dataset used for the algorithm.
     */
    public double[][] dataset;

    /**
     * Points to be queried.
     */
    public double[][] points;

    /**
     * Object containing the implementation of the KNN algorithm.
     */
    public AbstractKNN knn;

    /**
     * Distances between all query points and dataset points.
     */
    public double[][] distances;

    /**
     * Number of nearest neighbors to consider.
     */
    public int k;

    /**
     * Constructor for the KNN algorithm.
     * 
     * @param dataset dataset used for the algorithm
     * @param points  points to be queried
     * @param method  distance metric to be used
     */
    public KNN(double[][] dataset, double[][] points, String method) {
        this.dataset = dataset;
        this.points = points;
        this.knn = new AbstractKNN(dataset, points, method);

        // compute optimal k after knn is ready
        this.k = optimal();
    }

    /**
     * Constructor for the KNN algorithm with a given k.
     * 
     * @param dataset dataset used for the algorithm
     * @param points  points to be queried
     * @param method  distance metric to be used
     * @param k       number of nearest neighbors to consider
     */
    public KNN(double[][] dataset, double[][] points, String method, int k) {
        this.dataset = dataset;
        this.points = points;
        this.k = k;
        this.knn = new AbstractKNN(dataset, points, method);
    }

    /**
     * Get distances between all query points and dataset points.
     * 
     * @return distances between all query points and dataset points
     */
    public double[][] getDistances() {
        distances = knn.getDistances();
        return distances;
    }

    /**
     * Get the k nearest neighbors for a specific query point.
     * 
     * @param queryIndex index of the query point
     * @return k nearest neighbors of the query point
     */
    public LinkedHashMap<Integer, double[]> getNeighbours(int queryIndex) {
        if (distances == null)
            distances = knn.getDistances();

        LinkedHashMap<Integer, double[]> neighbors = new LinkedHashMap<>(k);
        HashMap<Integer, Double> neighborDistances = new HashMap<>();

        for (int i = 0; i < dataset.length; i++) {
            double d = distances[queryIndex][i];

            if (neighbors.size() < k) {
                neighbors.put(i, dataset[i]);
                neighborDistances.put(i, d);
            } else {
                // find farthest neighbor
                int worstIndex = -1;
                double worstDistance = Double.NEGATIVE_INFINITY;
                for (Map.Entry<Integer, Double> entry : neighborDistances.entrySet()) {
                    if (entry.getValue() > worstDistance) {
                        worstDistance = entry.getValue();
                        worstIndex = entry.getKey();
                    }
                }
                // replace if closer
                if (d < worstDistance) {
                    neighbors.remove(worstIndex);
                    neighborDistances.remove(worstIndex);
                    neighbors.put(i, dataset[i]);
                    neighborDistances.put(i, d);
                }
            }
        }

        // order neighbors by distance ascending
        List<Map.Entry<Integer, Double>> ordered = new ArrayList<>(neighborDistances.entrySet());
        ordered.sort(Comparator.comparingDouble(Map.Entry::getValue));

        LinkedHashMap<Integer, double[]> orderedNeighbors = new LinkedHashMap<>();
        for (Map.Entry<Integer, Double> e : ordered) {
            orderedNeighbors.put(e.getKey(), dataset[e.getKey()]);
        }
        return orderedNeighbors;
    }

    /**
     * Predict the label for a specific query point.
     * 
     * @param queryIndex index of the query point
     * @return predicted label for the query point
     */
    public int majority(int queryIndex) {
        LinkedHashMap<Integer, double[]> neighbours = getNeighbours(queryIndex);

        HashMap<Integer, Integer> labelCounts = new HashMap<>();
        for (double[] point : neighbours.values()) {
            int label = (int) point[point.length - 1];
            labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
        }

        int predicted = -1, maxCount = -1;
        for (Map.Entry<Integer, Integer> e : labelCounts.entrySet()) {
            if (e.getValue() > maxCount) {
                maxCount = e.getValue();
                predicted = e.getKey();
            }
        }
        return predicted;
    }

    /**
     * Predict class probabilities for a specific query point.
     * 
     * @param queryIndex index of the query point
     * @return map of label -> probability
     */
    public Map<Integer, Double> probability(int queryIndex) {
        LinkedHashMap<Integer, double[]> neighbours = getNeighbours(queryIndex);

        // Count occurrences of each label
        HashMap<Integer, Integer> labelCounts = new HashMap<>();
        for (double[] point : neighbours.values()) {
            int label = (int) point[point.length - 1];
            labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
        }

        // Convert counts to probabilities
        Map<Integer, Double> probabilities = new HashMap<>();
        for (Map.Entry<Integer, Integer> e : labelCounts.entrySet()) {
            probabilities.put(e.getKey(), (double) e.getValue() / k);
        }

        return probabilities;
    }

    /**
     * Compute the accuracy of the algorithm.
     * 
     * @return accuracy of the algorithm
     */
    public double accuracy() {
        int correct = 0;
        if (distances == null)
            distances = knn.getDistances();

        for (int q = 0; q < points.length; q++) {
            int predicted = majority(q);
            int actual = (int) points[q][points[q].length - 1];
            if (predicted == actual)
                correct++;
        }
        return (double) correct / points.length;
    }

    /**
     * Find the optimal k for the algorithm.
     * 
     * @return optimal k for the algorithm
     */
    public int optimal() {
        double maxAccuracy = 0.0;
        int optimalK = 1;

        for (int i = 1; i <= dataset.length; i++) {
            this.k = i;
            double acc = accuracy();
            if (acc > maxAccuracy) {
                maxAccuracy = acc;
                optimalK = i;
            }
        }
        return optimalK;
    }

    public int predictMajority(int queryIndex) {
        return majority(queryIndex);
    }

    public Map<Integer, Double> predictProbability(int queryIndex) {
        return probability(queryIndex);
    }

    public int[] predictAllMajority() {
        int[] labels = new int[points.length];
        for (int i = 0; i < points.length; i++) {
            labels[i] = majority(i);
        }
        return labels;
    }

    public List<Map<Integer, Double>> predictAllProbability() {
        List<Map<Integer, Double>> probList = new ArrayList<>();
        for (int i = 0; i < points.length; i++) {
            probList.add(probability(i));
        }
        return probList;
    }
}
//knnregressor
//getneighbours should return distances too
//research weighted knn
//accuracy should work for both classification and regression
//research GridSearchCV / cross_val_score
//f1 score
//confusion matrix