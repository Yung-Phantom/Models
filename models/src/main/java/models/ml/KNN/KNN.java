package models.ml.KNN;

import java.util.*;

public class KNN {

    private double[][] dataset;
    private double[][] points;

    private List<Map<Integer, Double>> sparseDataset;
    private List<Map<Integer, Double>> sparsePoints;

    private boolean sparse;

    private double[] trainingLabels;
    private double[] testLabels;

    private int k;
    private int p;

    private String method;

    private AbstractKNN knn;

    public KNN() {
    }

    private KNN(double[][] dataset, double[] trainingLabels, double[][] points, double[] testLabels, String method,
            int k, int p) {
        this.sparse = false;
        if (dataset == null) {
            throw new IllegalArgumentException("Dataset cannot be null.");
        }
        if (points == null) {
            throw new IllegalArgumentException("Points cannot be null.");
        }
        if (dataset.length != trainingLabels.length) {
            throw new IllegalArgumentException("Dataset length must match training labels length.");
        }
        if (points.length != testLabels.length) {
            throw new IllegalArgumentException("Points length must match test labels length.");
        }
        this.dataset = dataset;
        this.points = points;
        this.trainingLabels = trainingLabels;
        this.testLabels = testLabels;

        this.method = method != null ? method : "euclidean";
        this.p = p > 0 ? p : 2;

        this.knn = new AbstractKNN(this.dataset, this.method, this.p);

        this.k = k > 0 ? k : optimal();
    }

    public KNN(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels,
            List<Map<Integer, Double>> sparsePoints, double[] testLabels, String method, int k, Integer p) {
        this.sparse = true;
        if (sparseDataset == null) {
            throw new IllegalArgumentException("Sparse dataset cannot be null.");
        }
        if (sparsePoints == null) {
            throw new IllegalArgumentException("Sparse points cannot be null.");
        }
        if (trainingLabels == null) {
            throw new IllegalArgumentException("Training labels cannot be null.");
        }
        if (testLabels == null) {
            throw new IllegalArgumentException("Test labels cannot be null.");
        }
        this.sparseDataset = sparseDataset;
        this.sparsePoints = sparsePoints;
        this.trainingLabels = trainingLabels;
        this.testLabels = testLabels;

        this.method = method != null ? method : "cosine";
        this.p = (p != null && p > 0) ? p : 2;

        this.knn = new AbstractKNN(this.sparseDataset, this.method, this.p);

        this.k = (k > 0) ? k : optimal();
    }

    public KNN(double[][] dataset, double[] trainingLabels, double[][] points, double[] testLabels, int k) {
        this(dataset, trainingLabels, points, testLabels, "euclidean", k, 2);
    }

    public KNN(List<Map<Integer, Double>> sparseDataset, List<Map<Integer, Double>> sparsePoints,
            double[] trainingLabels, double[] testLabels, int k) {
        this(sparseDataset, trainingLabels, sparsePoints, testLabels, "cosine", k, 2);
    }

    public void fit(double[][] dataset, double[] trainingLabels, String method) {
        this.sparse = false;
        this.dataset = dataset;
        this.trainingLabels = trainingLabels;
        this.method = method;
        this.knn = new AbstractKNN(this.dataset, this.method, this.p);
    }

    public void fit(double[][] dataset, double[] trainingLabels) {
        fit(dataset, trainingLabels, this.method);
    }

    public void fit(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels, String method) {
        this.sparse = true;
        this.sparseDataset = sparseDataset;
        this.trainingLabels = trainingLabels;
        this.method = method;
        this.knn = new AbstractKNN(this.sparseDataset, this.method, this.p);
    }

    public void fit(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels) {
        fit(sparseDataset, trainingLabels, this.method);
    }

    public void setK(int k) {
        if (k <= 0)
            throw new IllegalArgumentException("k must be > 0");
        this.k = k;
    }

    public void setDistanceMethod(String method) {
        this.method = method;
        if (!sparse) {
            this.knn = new AbstractKNN(dataset, method,this.p);
        } else {
            this.knn = new AbstractKNN(sparseDataset, method, this.p);
        }
    }

    public void setPoints(double[][] points, double[] testLabels) {
        if (points == null) {
            throw new NullPointerException("Points cannot be null");
        }
        if (testLabels == null) {
            throw new NullPointerException("Test labels cannot be null");
        }
        if (points.length != testLabels.length) {
            throw new IllegalArgumentException("Points length must match test labels length.");
        }

        this.points = points;
        this.testLabels = testLabels;
    }

    public void setPoints(List<Map<Integer, Double>> sparsePoints, double[] testLabels) {
        if (sparsePoints == null) {
            throw new NullPointerException("Sparse points cannot be null");
        }
        if (testLabels == null) {
            throw new NullPointerException("Test labels cannot be null");
        }
        if (sparsePoints.size() != testLabels.length) {
            throw new IllegalArgumentException("Sparse points size must match test labels length.");
        }

        this.sparsePoints = sparsePoints;
        this.testLabels = testLabels;
    }

    public void refreshKNN() {
        if (!sparse && dataset != null) {
            this.knn = new AbstractKNN(this.dataset, this.method, this.p);
        } else if (sparse && sparseDataset != null) {
            this.knn = new AbstractKNN(this.sparseDataset, this.method, this.p);
        }
    }

    public void setTrainingLabels(double[] trainingLabels) {
        if (trainingLabels.length != dataset.length || trainingLabels.length != sparseDataset.size()) {
            throw new IllegalArgumentException("Training labels length must match dataset length. The size should be "
                    + dataset.length + " but got " + trainingLabels.length);
        }
        this.trainingLabels = trainingLabels;
    }

    public void setTestLabels(double[] testLabels) {
        if (testLabels.length != points.length || testLabels.length != sparsePoints.size()) {
            throw new IllegalArgumentException("Test labels length must match points length. The size should be "
                    + points.length + " but got " + testLabels.length);
        }
        this.testLabels = testLabels;
    }

    public void recalcOptimalK() {
        if (sparse) {
            this.k = optimal();
        } else {
            this.k = optimal();
        }
    }

    public LinkedHashMap<Integer, Double> getNeighboursWithDistance(int queryIndex) {
        int n = sparse ? sparseDataset.size() : dataset.length;
        if (queryIndex < 0 || queryIndex >= n) {
            throw new IndexOutOfBoundsException("Query index out of bounds.");
        }
        Map<Integer, Double> distances = new HashMap<>();

        for (int i = 0; i < n; i++) {
            double d;
            if (sparse) {
                d = knn.distance(sparsePoints.get(queryIndex), sparseDataset.get(i));
            } else {
                d = knn.distance(points[queryIndex], dataset[i]);
            }
            distances.put(i, d);
        }

        return distances.entrySet().stream().sorted(Map.Entry.comparingByValue()).limit(k).collect(LinkedHashMap::new,
                (m, e) -> m.put(e.getKey(), e.getValue()), LinkedHashMap::putAll);
    }

    public int majority(int queryIndex) {
        Map<Integer, Double> neighbours = getNeighboursWithDistance(queryIndex);
        Map<Integer, Integer> counts = new HashMap<>();

        for (int idx : neighbours.keySet()) {
            int label = (int) trainingLabels[idx];
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }

        return counts.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    }

    public Map<Integer, Double> probability(int queryIndex) {
        Map<Integer, Double> neighbours = getNeighboursWithDistance(queryIndex);
        Map<Integer, Integer> counts = new HashMap<>();

        for (int idx : neighbours.keySet()) {
            int label = (int) trainingLabels[idx];
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }

        Map<Integer, Double> probs = new HashMap<>();
        for (var e : counts.entrySet()) {
            probs.put(e.getKey(), e.getValue() / (double) k);
        }
        return probs;
    }

    public double accuracy() {
        if (testLabels == null) {
            throw new IllegalStateException("No labels for query/test points available.");
        }

        int correct = 0;
        int n = testLabels.length;

        for (int i = 0; i < n; i++) {
            if (majority(i) == (int) testLabels[i]) {
                correct++;
            }
        }

        return (double) correct / n;
    }

    public int optimal() {
        double best = 0.0;
        int bestK = 3;

        int length = sparse ? sparseDataset.size() : dataset.length;
        int maxK = (int) Math.sqrt(length);
        for (int i = 3; i <= maxK; i++) {
            if (i % 2 == 0)
                continue;
            this.k = i;
            double acc = accuracy();
            if (acc > best) {
                best = acc;
                bestK = i;
            }
        }
        return bestK;
    }

    public int[] predictAllMajority() {
        int n = sparse ? sparsePoints.size() : points.length;
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = majority(i);
        }
        return out;
    }

    public List<Map<Integer, Double>> predictAllProbability() {
        int n = sparse ? sparsePoints.size() : points.length;
        List<Map<Integer, Double>> list = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            list.add(probability(i));
        }
        return list;
    }
}