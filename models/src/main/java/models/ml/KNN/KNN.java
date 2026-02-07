package models.ml.KNN;

import java.util.*;

public class KNN {

    private double[][] dataset;
    private double[][] points;
    private List<Map<Integer, Double>> sparseDataset;
    private List<Map<Integer, Double>> sparsePoints;
    private double[] trainingLabels;
    private double[] testLabels;
    private AbstractKNN knn;
    private int k;
    private boolean sparse;
    private String method;

    // Dense master constructor
    private KNN(double[][] dataset, double[][] points, double[] trainingLabels, double[] testLabels,
            String method, int k, Integer p) {
        if (dataset == null)
            throw new IllegalArgumentException("Dense dataset cannot be null.");

        this.sparse = false;
        this.dataset = extractFeatures(dataset);
        this.points = extractFeatures(points);
        this.trainingLabels = trainingLabels != null ? trainingLabels : extractLabels(dataset);
        this.testLabels = testLabels != null ? testLabels : extractLabels(points);
        this.method = method != null ? method : "euclidean";

        this.knn = (p != null) ? new AbstractKNN(this.dataset, this.method, p)
                : new AbstractKNN(this.dataset, this.method);

        this.k = (k > 0) ? k : (this.testLabels != null ? optimalDense() : 1);
    }

    // Sparse master constructor
    public KNN(List<Map<Integer, Double>> sparseDataset, List<Map<Integer, Double>> sparsePoints,
            double[] trainingLabels, double[] testLabels, String method, int k, Integer p) {
        if (sparseDataset == null)
            throw new IllegalArgumentException("Sparse dataset cannot be null.");

        this.sparse = true;
        this.sparseDataset = sparseDataset;
        this.sparsePoints = sparsePoints;
        this.trainingLabels = trainingLabels;
        this.testLabels = testLabels;
        this.method = method != null ? method : "euclidean";

        this.knn = (p != null) ? new AbstractKNN(this.sparseDataset, this.method, p)
                : new AbstractKNN(this.sparseDataset, this.method);

        this.k = (k > 0) ? k : (this.testLabels != null ? optimalSparse() : 1);
    }

    public KNN(double[][] dataset, double[][] points, int k) {
        this(dataset, points, null, null, "euclidean", k, null);
    }

    public KNN(double[][] dataset, double[][] points, String method, int k, int p) {
        this(dataset, points, null, null, method, k, p);
    }

    public KNN(List<Map<Integer, Double>> sparseDataset, List<Map<Integer, Double>> sparsePoints,
            double[] trainingLabels, double[] testLabels, int k) {
        this(sparseDataset, sparsePoints, trainingLabels, testLabels, "euclidean", k, null);
    }

    public void fit(double[][] dataset, double[] labels, String method) {
        this.dataset = extractFeatures(dataset);
        this.method = method;
        this.trainingLabels = labels != null ? labels : extractLabels(dataset);
        this.sparse = false;
        this.knn = new AbstractKNN(this.dataset, method);
    }

    public void fit(double[][] dataset, double[] labels) {
        fit(dataset, labels, method);
    }

    public void fit(List<Map<Integer, Double>> sparseDataset, double[] labels, String method) {
        this.sparseDataset = sparseDataset;
        this.trainingLabels = labels;
        this.method = method;
        this.sparse = true;
        this.knn = new AbstractKNN(this.sparseDataset, method);
    }

    public void fit(List<Map<Integer, Double>> sparseDataset, double[] labels) {
        fit(sparseDataset, labels, method);
    }

    public void setK(int k) {
        if (k <= 0)
            throw new IllegalArgumentException("k must be > 0");
        this.k = k;
    }

    public void setDistanceMethod(String method) {
        this.method = method;
        if (!sparse) {
            this.knn = new AbstractKNN(dataset, method);
        } else {
            this.knn = new AbstractKNN(sparseDataset, method);
        }
    }

    public void setPoints(double[][] points) {
        if (points != null && testLabels != null && points.length != testLabels.length)
            throw new IllegalArgumentException("Points length must match test labels length.");
        this.points = extractFeatures(points);
        this.testLabels = extractLabels(points);
    }

    public void setPoints(List<Map<Integer, Double>> sparsePoints, double[] testLabels) {
        if (sparsePoints != null && testLabels != null && sparsePoints.size() != testLabels.length)
            throw new IllegalArgumentException("Sparse points size must match test labels length.");
    
        this.sparsePoints = sparsePoints;
        this.testLabels = testLabels;
    }

    public void refreshKNN() {
        if (!sparse && dataset != null) {
            this.knn = new AbstractKNN(dataset, method);
        } else if (sparse && sparseDataset != null) {
            this.knn = new AbstractKNN(sparseDataset, method);
        }
    }

    public void setTrainingLabels(double[] labels) {
        this.trainingLabels = labels;
    }

    public void setTestLabels(double[] labels) {
        this.testLabels = labels;
    }

    public void recalcOptimalK() {
        if (sparse) {
            this.k = optimalSparse();
        } else {
            this.k = optimalDense();
        }
    }

    public LinkedHashMap<Integer, Double> getNeighboursWithDistance(int queryIndex) {
        int n = sparse ? sparseDataset.size() : dataset.length;
        Map<Integer, Double> distances = new HashMap<>();

        for (int i = 0; i < n; i++) {
            double d = sparse ? knn.distance(sparsePoints.get(queryIndex), sparseDataset.get(i))
                    : knn.distance(points[queryIndex], dataset[i]);
            distances.put(i, d);
        }

        return distances.entrySet().stream()
                .sorted(Map.Entry.comparingByValue())
                .limit(k)
                .collect(LinkedHashMap::new,
                        (m, e) -> m.put(e.getKey(), e.getValue()),
                        LinkedHashMap::putAll);
    }

    public int majority(int queryIndex) {
        Map<Integer, Double> neighbours = getNeighboursWithDistance(queryIndex);
        Map<Integer, Integer> counts = new HashMap<>();

        for (int idx : neighbours.keySet()) {
            int label = (int) trainingLabels[idx];

            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }

        return counts.entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .get()
                .getKey();
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

    public int optimalDense() {
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

    private int optimalSparse() {
        double best = 0;
        int bestK = 1;

        int maxK = (int) Math.sqrt(sparseDataset.size());
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

    private double[][] extractFeatures(double[][] dataset) {
        if (dataset[0].length < 2)
            throw new IllegalArgumentException("Dataset must have at least one feature and one label column.");
        
        double[][] datasetIn = new double[dataset.length][dataset[0].length - 1];
        for (int i = 0; i < dataset.length; i++) {
            System.arraycopy(dataset[i], 0, datasetIn[i], 0, dataset[0].length - 1);
        }
        return datasetIn;
    }

    private double[] extractLabels(double[][] dataset) {
        if (dataset[0].length < 2)
            throw new IllegalArgumentException("Dataset must have at least one feature and one label column.");
        
        double[] labels = new double[dataset.length];
        for (int i = 0; i < dataset.length; i++) {
            labels[i] = dataset[i][dataset[0].length - 1];
            if (labels[i] != (int) labels[i])
                throw new IllegalArgumentException("Labels must be integers");
        }
        return labels;
    }

}