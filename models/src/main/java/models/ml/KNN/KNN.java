package models.ml.KNN;

import java.util.*;

/**
 * Optimized K-Nearest Neighbours algorithm.
 * 
 * @author Kotei Justice
 * @version 1.3
 * 
 *          This class provides an optimized K-Nearest Neighbours algorithm. It
 *          uses a priority queue to keep track of the top K nearest neighbours.
 *          The algorithm is optimized for speed and memory efficiency.
 */
public class KNN {

    private final double[][] dataset;
    private final double[][] points;
    private final AbstractKNN knn;
    private int k;

    /**
     * Constructor for the KNN algorithm with a specified method.
     * Uses the provided value of K, or computes the optimal value if k == 0.
     *
     * @param dataset The dataset to use.
     * @param points  The points to query.
     * @param method  The distance method to use (e.g., "euclidean", "manhattan",
     *                "minkowski").
     * @param k       The initial value of K .
     */
    public KNN(double[][] dataset, double[][] points, String method, int k) {
        this.dataset = dataset;
        this.points = points;
        this.k = k;
        if (k == 0)
            this.k=optimal();
        this.knn = new AbstractKNN(dataset, method);
    }

    /**
     * Constructor for the KNN algorithm with Euclidean distance.
     *
     * @param dataset The dataset to use.
     * @param points  The points to query.
     * @param k       The value of K to use.
     */
    public KNN(double[][] dataset, double[][] points, int k) {
        this(dataset, points, "euclidean", k);
    }

    /**
     * Constructor for the KNN algorithm with a specified method and Minkowski
     * parameter p.
     * Uses the provided value of K, or computes the optimal value if k == 0.
     * 
     * @param dataset The dataset to use.
     * @param points  The points to query.
     * @param method  The distance method to use (e.g., "euclidean", "manhattan",
     *                "minkowski").
     * @param k       The value of K to use.
     * @param p       The Minkowski distance parameter.
     */
    public KNN(double[][] dataset, double[][] points, String method, int k, int p) {
        this.dataset = dataset;
        this.points = points;
        this.k = k;
        if (k == 0)
            this.k=optimal();
        this.knn = new AbstractKNN(dataset, method, p);
    }

    /**
     * Constructor for the KNN algorithm with Minkowski distance and parameter p.
     *
     * @param dataset The dataset to use.
     * @param points  The points to query.
     * @param k       The value of K to use.
     * @param p       The Minkowski distance parameter.
     */
    public KNN(double[][] dataset, double[][] points, int k, int p) {
        this(dataset, points, "minkowski", k, p);
    }

    /**
     * Constructor for the KNN algorithm with a specified method.
     * Automatically determines the optimal value of K.
     * 
     * @param dataset The dataset to use.
     * @param points  The points to query.
     * @param method  The distance method to use (e.g., "euclidean", "manhattan",
     *                "minkowski").
     */
    public KNN(double[][] dataset, double[][] points, String method) {
        this.dataset = dataset;
        this.points = points;
        this.knn = new AbstractKNN(dataset, method);
        this.k = optimal();
    }

    /**
     * Constructor for the KNN algorithm with Euclidean distance.
     * Automatically determines the optimal value of K.
     * 
     * @param dataset The dataset to use.
     * @param points  The points to query.
     */
    public KNN(double[][] dataset, double[][] points) {
        this(dataset, points, "euclidean");
    }

    /**
     * Get the K nearest neighbours of a given query point.
     * 
     * @param queryIndex The index of the query point.
     * @return A map containing the K nearest neighbours and their distances.
     */
    public LinkedHashMap<Integer, Double> getNeighboursWithDistance(int queryIndex) {
        double[] query = points[queryIndex];

        PriorityQueue<Map.Entry<Integer, Double>> pq = new PriorityQueue<>(
                Comparator.comparingDouble(Map.Entry<Integer, Double>::getValue).reversed());

        for (int i = 0; i < dataset.length; i++) {
            double d = knn.distance(query, dataset[i]);

            if (pq.size() < k) {
                pq.offer(new AbstractMap.SimpleEntry<>(i, d));
            } else if (d < pq.peek().getValue()) {
                pq.poll();
                pq.offer(new AbstractMap.SimpleEntry<>(i, d));
            }
        }

        List<Map.Entry<Integer, Double>> ordered = new ArrayList<>(pq);
        ordered.sort(Comparator.comparingDouble(Map.Entry::getValue));

        LinkedHashMap<Integer, Double> neighbours = new LinkedHashMap<>();
        for (Map.Entry<Integer, Double> e : ordered) {
            neighbours.put(e.getKey(), e.getValue());
        }
        return neighbours;
    }

    /**
     * Predict the label of a given query point using the majority vote of its K
     * nearest neighbours.
     * 
     * @param queryIndex The index of the query point.
     * @return The predicted label.
     */
    public int majority(int queryIndex) {
        Map<Integer, Double> neighbours = getNeighboursWithDistance(queryIndex);

        int predicted = -1, maxCount = -1;
        HashMap<Integer, Integer> labelCounts = new HashMap<>();

        for (int idx : neighbours.keySet()) {
            int label = (int) dataset[idx][dataset[idx].length - 1];
            int count = labelCounts.getOrDefault(label, 0) + 1;
            labelCounts.put(label, count);

            if (count > maxCount) {
                maxCount = count;
                predicted = label;
            }
        }
        return predicted;
    }

    /**
     * Predict the probability distribution of a given query point using the
     * majority vote of its K nearest neighbours.
     * 
     * @param queryIndex The index of the query point.
     * @return A map containing the predicted probability distribution.
     */
    public Map<Integer, Double> probability(int queryIndex) {
        Map<Integer, Double> neighbours = getNeighboursWithDistance(queryIndex);

        HashMap<Integer, Integer> counts = new HashMap<>();
        for (int idx : neighbours.keySet()) {
            int label = (int) dataset[idx][dataset[idx].length - 1];
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }

        Map<Integer, Double> probs = new HashMap<>();
        for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
            probs.put(e.getKey(), (double) e.getValue() / k);
        }
        return probs;
    }

    /**
     * Compute the accuracy of the algorithm using the leave-one-out
     * cross-validation method.
     * 
     * @return The accuracy of the algorithm.
     */
    public double accuracy() {
        int correct = 0;
        for (int i = 0; i < points.length; i++) {
            if (majority(i) == (int) points[i][points[i].length - 1]) {
                correct++;
            }
        }
        return (double) correct / points.length;
    }

    /**
     * Compute the optimal value of K using leave-one-out cross-validation.
     * Searches values of K from 1 up to sqrt(dataset length).
     *
     * @return The optimal value of K.
     */
    public int optimal() {
        double best = 0.0;
        int bestK = 1;

        int maxK = (int) Math.sqrt(dataset.length);
        for (int i = 1; i <= maxK; i++) {
            this.k = i;
            double acc = accuracy();
            if (acc > best) {
                best = acc;
                bestK = i;
            }
        }
        return bestK;
    }

    /**
     * Predict the labels of all query points using the majority vote of their K
     * nearest neighbours.
     * 
     * @return The predicted labels.
     */
    public int[] predictAllMajority() {
        int[] out = new int[points.length];
        for (int i = 0; i < points.length; i++) {
            out[i] = majority(i);
        }
        return out;
    }

    /**
     * Predict the probability distributions of all query points using the majority
     * vote of their K nearest neighbours.
     * 
     * @return A list containing the predicted probability distributions.
     */
    public List<Map<Integer, Double>> predictAllProbability() {
        List<Map<Integer, Double>> list = new ArrayList<>();
        for (int i = 0; i < points.length; i++) {
            list.add(probability(i));
        }
        return list;
    }
}