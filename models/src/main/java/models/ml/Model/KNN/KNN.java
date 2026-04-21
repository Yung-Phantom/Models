package models.ml.Model.KNN;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.DatasetHandler.helpers.DatasetSplit;
import models.ml.Model.Metrics;
import models.ml.Model.Model;
import models.ml.Model.KNN.AbstractKNN;

public class KNN extends Model {
    public enum KNNMETHODS {
        EUCLIDEAN, MANHATTAN, MINKOWSKI, COSINE,
    }

    private static final int DEFAULT_K = 3;
    private static final int DEFAULT_P = 2;
    private static final String DEFAULT_DENSE_METHOD = "euclidean";
    private static final String DEFAULT_SPARSE_METHOD = "cosine";

    public String method;
    private int k;
    public int p;
    private AbstractKNN knn;

    public KNN() {
    }

    public KNN(double[][] trainingDataset, int k, int p, String method) {
        super(trainingDataset);
        initDenseParams(k, p, method);
        initDenseKNN();
    }

    public KNN(double[][] trainingDataset, double[] trainingLabels, int k, int p, String method) {
        super(trainingDataset, trainingLabels);
        initDenseParams(k, p, method);
        initDenseKNN();
    }

    public KNN(double[][] trainingDataset, double[][] testDataset, int k, int p, String method) {
        super(trainingDataset, testDataset);
        initDenseParams(k, p, method);
        initDenseKNN();
    }

    public KNN(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels, int k,
            int p, String method) {
        super(trainingDataset, trainingLabels, testDataset, testLabels);
        initDenseParams(k, p, method);
        initDenseKNN();
    }

    public KNN(List<Map<Integer, Double>> trainingDataset, int k, int p, String method) {
        super(trainingDataset);
        initSparseParams(k, p, method);
        initSparseKNN();
    }

    public KNN(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, int k, int p, String method) {
        super(trainingDataset, trainingLabels);
        initSparseParams(k, p, method);
        initSparseKNN();
    }

    public KNN(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset, int k, int p,
            String method) {
        super(trainingDataset, testDataset);
        initSparseParams(k, p, method);
        initSparseKNN();
    }

    public KNN(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, int k, int p, String method) {
        super(trainingDataset, trainingLabels, testDataset, testLabels);
        initSparseParams(k, p, method);
        initSparseKNN();
    }

    @Override
    public void fit(double[][] trainingDataset) {
        if (trainingDataset == null || trainingDataset.length == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        this.denseTrainingDataset = this.scaler.extractFeatures(trainingDataset);
        this.trainingLabels = this.scaler.extractLabels(trainingDataset);
        initDenseParams();
        initDenseKNN();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, int k, int p) {
        this.k = k;
        this.p = p;
        this.fit(trainingDataset);
    }

    public void fit(double[][] trainingDataset, String method) {
        this.method = method;
        this.fit(trainingDataset);
    }

    public void fit(double[][] trainingDataset, int k, int p, String method) {
        this.k = k;
        this.p = p;
        this.method = method;
        this.fit(trainingDataset);
    }

    @Override
    public void fit(double[][] trainingDataset, double[] trainingLabels) {
        if (trainingDataset == null || trainingDataset.length == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        if (trainingLabels == null || trainingLabels.length == 0) {
            throw new IllegalArgumentException("trainingLabels cannot be empty");
        }
        this.denseTrainingDataset = trainingDataset;
        this.trainingLabels = trainingLabels;
        initDenseParams();
        initDenseKNN();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, int k, int p) {
        this.k = k;
        this.p = p;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, String method) {
        this.method = method;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, int k, int p, String method) {
        this.k = k;
        this.p = p;
        this.method = method;
        this.fit(trainingDataset, trainingLabels);
    }

    @Override
    public void fit(double[][] trainingDataset, double[][] testDataset) {
        if (trainingDataset == null || trainingDataset.length == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        if (testDataset == null || testDataset.length == 0) {
            throw new IllegalArgumentException("testDataset cannot be empty");
        }
        this.denseTrainingDataset = this.scaler.extractFeatures(trainingDataset);
        this.denseTestDataset = this.scaler.extractFeatures(testDataset);
        this.trainingLabels = this.scaler.extractLabels(trainingDataset);
        this.testLabels = this.scaler.extractLabels(testDataset);
        initDenseParams();
        initDenseKNN();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, double[][] testDataset, int k, int p) {
        this.k = k;
        this.p = p;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(double[][] trainingDataset, double[][] testDataset, String method) {
        this.method = method;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(double[][] trainingDataset, double[][] testDataset, int k, int p, String method) {
        this.k = k;
        this.p = p;
        this.method = method;
        this.fit(trainingDataset, testDataset);
    }

    @Override
    public void fit(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels) {
        if (trainingDataset == null || trainingDataset.length == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        if (trainingLabels == null || trainingLabels.length == 0) {
            throw new IllegalArgumentException("trainingLabels cannot be empty");
        }
        if (testDataset == null || testDataset.length == 0) {
            throw new IllegalArgumentException("testDataset cannot be empty");
        }
        if (testLabels == null || testLabels.length == 0) {
            throw new IllegalArgumentException("testLabels cannot be empty");
        }
        this.denseTrainingDataset = trainingDataset;
        this.denseTestDataset = testDataset;
        this.trainingLabels = trainingLabels;
        this.testLabels = testLabels;
        initDenseParams();
        initDenseKNN();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            int k, int p) {
        this.k = k;
        this.p = p;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            String method) {
        this.method = method;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            int k, int p, String method) {
        this.k = k;
        this.p = p;
        this.method = method;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    @Override
    public void fit(List<Map<Integer, Double>> trainingDataset) {
        if (trainingDataset == null || trainingDataset.size() == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        this.sparseTrainingDataset = trainingDataset;
        initSparseParams();
        initSparseKNN();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, int k, int p) {
        this.k = k;
        this.p = p;
        this.fit(trainingDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, String method) {
        this.method = method;
        this.fit(trainingDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, int k, int p, String method) {
        this.k = k;
        this.p = p;
        this.method = method;
        this.fit(trainingDataset);
    }

    @Override
    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels) {
        if (trainingDataset == null || trainingDataset.size() == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        this.sparseTrainingDataset = trainingDataset;
        this.trainingLabels = trainingLabels;
        initSparseParams();
        initSparseKNN();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, int k, int p) {
        this.k = k;
        this.p = p;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, String method) {
        this.method = method;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, int k, int p, String method) {
        this.k = k;
        this.p = p;
        this.method = method;
        this.fit(trainingDataset, trainingLabels);
    }

    @Override
    public void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset) {
        if (trainingDataset == null || trainingDataset.size() == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        this.sparseTrainingDataset = trainingDataset;
        this.sparseTestDataset = testDataset;
        initSparseParams();
        initSparseKNN();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset, int k, int p) {
        this.k = k;
        this.p = p;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset, String method) {
        this.method = method;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset,
            List<Map<Integer, Double>> testDataset, int k, int p, String method) {
        this.k = k;
        this.p = p;
        this.method = method;
        this.fit(trainingDataset, testDataset);
    }

    @Override
    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels) {
        if (trainingDataset == null || trainingDataset.size() == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        this.sparseTrainingDataset = trainingDataset;
        this.sparseTestDataset = testDataset;
        this.trainingLabels = trainingLabels;
        this.testLabels = testLabels;
        initSparseParams();
        initSparseKNN();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, int k, int p) {
        this.k = k;
        this.p = p;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, String method) {
        this.method = method;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, int k, int p, String method) {
        this.k = k;
        this.p = p;
        this.method = method;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    @Override
    public void fit(DatasetLoader loader, int splitPercent, String splitStrategy) {
        double splitDecimal = splitPercent / 100.0;
        this.fit(loader, splitDecimal, splitStrategy);
    }

    @Override
    public void fit(DatasetLoader loader, double splitDecimal, String splitStrategy) {
        if (loader == null || loader.getDataset() == null) {
            throw new IllegalArgumentException("DatasetLoader or underlying dataset cannot be null");
        }

        DatasetSplit splitData = loader.split(splitDecimal, splitStrategy);

        this.denseTrainingDataset = this.scaler.extractFeatures(splitData.train);
        this.trainingLabels = this.scaler.extractLabels(splitData.train);

        this.denseTestDataset = this.scaler.extractFeatures(splitData.test);
        this.testLabels = this.scaler.extractLabels(splitData.test);

        initDenseParams();
        initDenseKNN();
        this.sparse = false;
    }

    @Override
    public int[] predictLabels() {
        int n = this.sparse ? this.sparseTestDataset.size() : this.denseTestDataset.length;
        int[] predictions = new int[n];
        for (int i = 0; i < n; i++) {
            predictions[i] = majority(i);
        }
        return predictions;
    }

    @Override
    public List<Map<Integer, Double>> predictProbabilities() {
        int n = this.sparse ? this.sparseTestDataset.size() : this.denseTestDataset.length;

        List<Map<Integer, Double>> probabilities = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            probabilities.add(probability(i));
        }
        return probabilities;
    }

    @Override
    public int[][] predictMultiLabel(double threshold) {
        int n = sparse ? sparseTestDataset.size() : denseTestDataset.length;
        int[][] predictions = new int[n][];

        for (int i = 0; i < n; i++) {
            Map<Integer, Double> probMap = probability(i);

            List<Integer> labels = new ArrayList<>();
            for (var e : probMap.entrySet()) {
                if (e.getValue() >= threshold) {
                    labels.add(e.getKey());
                }
            }
            predictions[i] = labels.stream().mapToInt(Integer::intValue).toArray();
        }
        return predictions;
    }

    public Map<String, Double> evaluate(double[][] dataset, double[] labels, String[] metrics) {
        if (dataset == null || labels == null || dataset.length != labels.length) {
            throw new IllegalArgumentException("Dataset and labels must match in length.");
        }

        double[][] originalDenseTest = this.denseTestDataset;
        boolean originalSparseFlag = this.sparse;

        this.denseTestDataset = dataset;
        this.sparse = false;

        int[] predictions = predictLabels();
        Map<String, Double> results = calculateMetrics(predictions, labels, metrics);

        this.denseTestDataset = originalDenseTest;
        this.sparse = originalSparseFlag;

        return results;
    }

    @Override
    public Map<String, Double> evaluate(String[] metrics) {
        if (sparse) {
            return evaluate(this.sparseTestDataset, this.testLabels, metrics);
        }
        return evaluate(this.denseTestDataset, this.testLabels, metrics);
    }

    @Override
    public Map<String, Double> evaluate(List<Map<Integer, Double>> dataset, double[] labels, String[] metrics) {
        if (dataset == null || labels == null || dataset.size() != labels.length) {
            throw new IllegalArgumentException("Dataset and labels must match in size.");
        }

        // Temporarily swap state
        List<Map<Integer, Double>> originalSparseTest = this.sparseTestDataset;
        boolean originalSparseFlag = this.sparse;

        this.sparseTestDataset = dataset;
        this.sparse = true;

        int[] predictions = predictLabels();
        Map<String, Double> results = calculateMetrics(predictions, labels, metrics);

        // Restore state
        this.sparseTestDataset = originalSparseTest;
        this.sparse = originalSparseFlag;

        return results;
    }

    @Override
    public void save(String filename) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            // Save core parameters
            oos.writeObject(this.k);
            oos.writeObject(this.p);
            oos.writeObject(this.method);
            oos.writeObject(this.sparse);

            // Save datasets and labels
            if (sparse) {
                oos.writeObject(this.sparseTrainingDataset);
                oos.writeObject(this.sparseTestDataset);
            } else {
                oos.writeObject(this.denseTrainingDataset);
                oos.writeObject(this.denseTestDataset);
            }
            oos.writeObject(this.trainingLabels);
            oos.writeObject(this.testLabels);
        } catch (IOException e) {
            throw new RuntimeException("Error saving KNN model: " + e.getMessage(), e);
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public void load(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            // Load core parameters
            this.k = (int) ois.readObject();
            this.p = (int) ois.readObject();
            this.method = (String) ois.readObject();
            this.sparse = (boolean) ois.readObject();

            // Load datasets and labels
            if (sparse) {
                this.sparseTrainingDataset = (List<Map<Integer, Double>>) ois.readObject();
                this.sparseTestDataset = (List<Map<Integer, Double>>) ois.readObject();
            } else {
                this.denseTrainingDataset = (double[][]) ois.readObject();
                this.denseTestDataset = (double[][]) ois.readObject();
            }
            this.trainingLabels = (double[]) ois.readObject();
            this.testLabels = (double[]) ois.readObject();

            refresh();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException("Error loading KNN model: " + e.getMessage(), e);
        }
    }

    @Override
    public String getModelType() {
        return "KNN";
    }

    @Override
    public String getVersion() {
        return "1.0.0";
    }

    @Override
    public void refresh() {
        if (!sparse && denseTrainingDataset != null) {
            this.knn = new AbstractKNN(this.denseTrainingDataset, this.method, this.p);
        } else if (sparse && sparseTrainingDataset != null) {
            this.knn = new AbstractKNN(this.sparseTrainingDataset, this.method, this.p);
        } else {
            throw new IllegalArgumentException("Dataset cannot be null.");
        }
    }

    // initialization helpers
    private void initDenseParams(int k, int p, String method) {
        this.k = (k > 0) ? k : DEFAULT_K;
        this.p = (p > 0) ? p : DEFAULT_P;
        this.method = (method != null && !method.isBlank()) ? method.trim().toLowerCase() : DEFAULT_DENSE_METHOD;
    }

    private void initDenseParams() {
        this.k = (k > 0) ? k : DEFAULT_K;
        this.p = (p > 0) ? p : DEFAULT_P;
        this.method = (method != null && !method.isBlank()) ? method.trim().toLowerCase() : DEFAULT_DENSE_METHOD;
    }

    private void initSparseParams(int k, int p, String method) {
        this.k = (k > 0) ? k : DEFAULT_K;
        this.p = (p > 0) ? p : DEFAULT_P;
        this.method = (method != null && !method.isBlank()) ? method.trim().toLowerCase() : DEFAULT_SPARSE_METHOD;
    }

    private void initSparseParams() {
        this.k = (k > 0) ? k : DEFAULT_K;
        this.p = (p > 0) ? p : DEFAULT_P;
        this.method = (method != null && !method.isBlank()) ? method.trim().toLowerCase() : DEFAULT_SPARSE_METHOD;
    }

    private void initDenseKNN() {
        this.knn = new AbstractKNN(this.denseTrainingDataset, this.method, this.p);
    }

    private void initSparseKNN() {
        this.knn = new AbstractKNN(this.sparseTrainingDataset, this.method, this.p);
    }

    public LinkedHashMap<Integer, Double> getNeighboursWithDistance(int queryIndex) {
        int n = sparse ? sparseTrainingDataset.size() : denseTrainingDataset.length;
        if (queryIndex < 0 || queryIndex >= n) {
            throw new IndexOutOfBoundsException("Query index out of bounds.");
        }
        Map<Integer, Double> distances = new HashMap<>();

        for (int i = 0; i < n; i++) {
            double d;
            if (sparse) {
                d = knn.distance(sparseTestDataset.get(queryIndex), sparseTrainingDataset.get(i));
            } else {
                d = knn.distance(denseTestDataset[queryIndex], denseTrainingDataset[i]);
            }
            distances.put(i, d);
        }

        return distances.entrySet()
                .stream()
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

        return counts.entrySet().stream()
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

    private Map<String, Double> calculateMetrics(int[] predictions, double[] labels, String[] metrics) {
        Map<String, Double> results = new HashMap<>();
        int[] actualInt = Arrays.stream(labels).mapToInt(d -> (int) d).toArray();
        double[] predDouble = Arrays.stream(predictions).asDoubleStream().toArray();

        for (String metric : metrics) {
            switch (metric.toLowerCase()) {
                case "accuracy":
                    results.put("accuracy", Metrics.accuracy(predictions, actualInt));
                    break;
                case "precision":
                    results.put("precision", Metrics.precision(predictions, actualInt, 1));
                    break;
                case "recall":
                    results.put("recall", Metrics.recall(predictions, actualInt, 1));
                    break;
                case "mse":
                    results.put("mse", Metrics.mse(predDouble, labels));
                    break;
                case "rmse":
                    results.put("rmse", Metrics.rmse(predDouble, labels));
                    break;
                case "mae":
                    results.put("mae", Metrics.mae(predDouble, labels));
                    break;
            }
        }
        return results;
    }
}