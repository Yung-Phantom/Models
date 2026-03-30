package models.ml.Model.NaiveBayes;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.DatasetHandler.helpers.DatasetSplit;
import models.ml.Model.Metrics;
import models.ml.Model.Model;
import models.ml.Model.NaiveBayes.AbstractNaiveBayes;

public class NaiveBayes extends Model {
    public enum NBMETHODS {
        GAUSSIAN, MULTINOMIAL, BERNOULLI
    }

    private static final String DEFAULT_DENSE_METHOD = "gaussian";
    private static final String DEFAULT_SPARSE_METHOD = "multinomial";
    private static final double DEFAULT_ALPHA = 1.0;
    public String method;
    public double alpha;
    private AbstractNaiveBayes nb;

    public NaiveBayes() {
    }

    public NaiveBayes(double[][] trainingDataset, String method, double alpha) {
        super(trainingDataset);
        initDenseParams(method, alpha);
        initNB();
    }

    public NaiveBayes(double[][] trainingDataset, double[] trainingLabels, String method, double alpha) {
        super(trainingDataset, trainingLabels);
        initDenseParams(method, alpha);
        initNB();
    }

    public NaiveBayes(double[][] trainingDataset, double[][] testDataset, String method, double alpha) {
        super(trainingDataset, testDataset);
        initDenseParams(method, alpha);
        initNB();
    }

    public NaiveBayes(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            String method, double alpha) {
        super(trainingDataset, trainingLabels, testDataset, testLabels);
        initDenseParams(method, alpha);
        initNB();
    }

    public NaiveBayes(List<Map<Integer, Double>> trainingDataset, String method, double alpha) {
        super(trainingDataset);
        initSparseParams(method, alpha);
        initNB();
    }

    public NaiveBayes(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, String method,
            double alpha) {
        super(trainingDataset, trainingLabels);
        initSparseParams(method, alpha);
        initNB();
    }

    public NaiveBayes(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset,
            String method, double alpha) {
        super(trainingDataset, testDataset);
        initSparseParams(method, alpha);
        initNB();
    }

    public NaiveBayes(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, String method, double alpha) {
        super(trainingDataset, trainingLabels, testDataset, testLabels);
        initSparseParams(method, alpha);
        initNB();
    }

    @Override
    public void fit(double[][] trainingDataset) {
        if (trainingDataset == null || trainingDataset.length == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        this.denseTrainingDataset = this.scaler.extractFeatures(trainingDataset);
        this.trainingLabels = this.scaler.extractLabels(trainingDataset);
        initDenseParams();
        initNB();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, double alpha) {
        this.alpha = alpha;
        this.fit(trainingDataset);
    }

    public void fit(double[][] trainingDataset, String method) {
        this.method = method;
        this.fit(trainingDataset);
    }

    public void fit(double[][] trainingDataset, double alpha, String method) {
        this.alpha = alpha;
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
        initNB();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double alpha) {
        this.alpha = alpha;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, String method) {
        this.method = method;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double alpha, String method) {
        this.alpha = alpha;
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
        initNB();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, double[][] testDataset, double alpha) {
        this.alpha = alpha;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(double[][] trainingDataset, double[][] testDataset, String method) {
        this.method = method;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(double[][] trainingDataset, double[][] testDataset, double alpha, String method) {
        this.alpha = alpha;
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
        initNB();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            double alpha) {
        this.alpha = alpha;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            String method) {
        this.method = method;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            double alpha, String method) {
        this.alpha = alpha;
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
            initNB();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double alpha) {
        this.alpha = alpha;
        this.fit(trainingDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, String method) {
        this.method = method;
        this.fit(trainingDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double alpha, String method) {
        this.alpha = alpha;
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
        initNB();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, double alpha) {
        this.alpha = alpha;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, String method) {
        this.method = method;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, double alpha, String method) {
        this.alpha = alpha;
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
        initNB();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset, double alpha) {
        this.alpha = alpha;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset, String method) {
        this.method = method;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset,
            List<Map<Integer, Double>> testDataset, double alpha, String method) {
        this.alpha = alpha;
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
        initNB();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, double alpha) {
        this.alpha = alpha;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, double alpha, String method) {
        this.alpha = alpha;
        this.method = method;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, String method) {
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
        initNB();
        this.sparse = false;
    }

    @Override
    public int[] predictLabels() {
        if (nb == null)
            throw new IllegalStateException("Model must be fitted before prediction.");

        // Determine which dataset to predict on (Test set)
        int n = sparse ? sparseTestDataset.size() : denseTestDataset.length;
        int[] predictions = new int[n];

        for (int i = 0; i < n; i++) {
            Map<Integer, Double> probs = sparse ? nb.computeProbabilities(sparseTestDataset.get(i))
                    : nb.computeProbabilities(denseTestDataset[i]);

            predictions[i] = getMaxProbabilityClass(probs);
        }
        return predictions;
    }

    @Override
    public List<Map<Integer, Double>> predictProbabilities() {
        if (nb == null)
            throw new IllegalStateException("Model must be fitted before prediction.");

        List<Map<Integer, Double>> allProbabilities = new ArrayList<>();
        int n = sparse ? sparseTestDataset.size() : denseTestDataset.length;

        for (int i = 0; i < n; i++) {
            allProbabilities.add(sparse ? nb.computeProbabilities(sparseTestDataset.get(i))
                    : nb.computeProbabilities(denseTestDataset[i]));
        }
        return allProbabilities;
    }

    @Override
    public int[][] predictMultiLabel(double threshold) {
        List<Map<Integer, Double>> probs = predictProbabilities();
        int[][] multiLabels = new int[probs.size()][];

        for (int i = 0; i < probs.size(); i++) {
            multiLabels[i] = probs.get(i).entrySet().stream()
                    .filter(entry -> entry.getValue() >= threshold)
                    .mapToInt(Map.Entry::getKey)
                    .toArray();
        }
        return multiLabels;
    }

    @Override
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
            oos.writeObject(this.alpha);
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
            throw new RuntimeException("Error saving Naive Bayes model: " + e.getMessage(), e);
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public void load(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            // Load core parameters
            this.alpha = (double) ois.readObject();
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
            throw new RuntimeException("Error loading Naive Bayes model: " + e.getMessage(), e);
        }
    }

    @Override
    public void partialFit(double[][] batchDataset, double[] batchLabels) {
        if (batchDataset == null || batchDataset.length == 0) {
            throw new IllegalArgumentException("batchDataset cannot be empty");
        }
        if (batchLabels == null || batchLabels.length != batchDataset.length) {
            throw new IllegalArgumentException("batchLabels must match batchDataset length");
        }

        // Merge new batch into existing dense dataset
        int oldSize = (denseTrainingDataset != null) ? denseTrainingDataset.length : 0;
        int newSize = oldSize + batchDataset.length;

        double[][] mergedData = new double[newSize][];
        double[] mergedLabels = new double[newSize];

        if (oldSize > 0) {
            System.arraycopy(denseTrainingDataset, 0, mergedData, 0, oldSize);
            System.arraycopy(trainingLabels, 0, mergedLabels, 0, oldSize);
        }
        System.arraycopy(batchDataset, 0, mergedData, oldSize, batchDataset.length);
        System.arraycopy(batchLabels, 0, mergedLabels, oldSize, batchLabels.length);

        this.denseTrainingDataset = mergedData;
        this.trainingLabels = mergedLabels;
        this.sparse = false;

        refresh();
    }

    @Override
    public void partialFit(List<Map<Integer, Double>> batchDataset, double[] batchLabels) {
        if (batchDataset == null || batchDataset.isEmpty()) {
            throw new IllegalArgumentException("batchDataset cannot be empty");
        }
        if (batchLabels == null || batchLabels.length != batchDataset.size()) {
            throw new IllegalArgumentException("batchLabels must match batchDataset size");
        }

        // Merge new batch into existing sparse dataset
        if (sparseTrainingDataset == null) {
            sparseTrainingDataset = new ArrayList<>();
        }
        sparseTrainingDataset.addAll(batchDataset);

        int oldSize = (trainingLabels != null) ? trainingLabels.length : 0;
        int newSize = oldSize + batchLabels.length;

        double[] mergedLabels = new double[newSize];
        if (oldSize > 0) {
            System.arraycopy(trainingLabels, 0, mergedLabels, 0, oldSize);
        }
        System.arraycopy(batchLabels, 0, mergedLabels, oldSize, batchLabels.length);

        this.trainingLabels = mergedLabels;
        this.sparse = true;

        refresh();
    }

    @Override
    public String getModelType() {
        return "Naive Bayes";
    }

    @Override
    public String getVersion() {
        return "1.0.0";
    }

    @Override
    public void refresh() {
        if (!sparse && denseTrainingDataset != null) {
            this.nb = new AbstractNaiveBayes(this.denseTrainingDataset, this.trainingLabels, this.method, this.alpha);
        } else if (sparse && sparseTrainingDataset != null) {
            this.nb = new AbstractNaiveBayes(this.sparseTrainingDataset, this.trainingLabels, this.method, this.alpha);
        } else {
            throw new IllegalArgumentException("Dataset cannot be null.");
        }
    }

    private void initDenseParams(String method, double alpha) {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_DENSE_METHOD;
        this.alpha = (alpha > 0) ? alpha : DEFAULT_ALPHA;
        this.sparse = false;
    }

    private void initSparseParams(String method, double alpha) {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_SPARSE_METHOD;
        this.alpha = (alpha > 0) ? alpha : DEFAULT_ALPHA;
        this.sparse = true;
    }
    private void initDenseParams() {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_DENSE_METHOD;
        this.alpha = (alpha > 0) ? alpha : DEFAULT_ALPHA;
        this.sparse = false;
    }

    private void initSparseParams() {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_SPARSE_METHOD;
        this.alpha = (alpha > 0) ? alpha : DEFAULT_ALPHA;
        this.sparse = true;
    }

    private void initNB() {
        if (!sparse && denseTrainingDataset != null) {
            this.nb = new AbstractNaiveBayes(this.denseTrainingDataset, this.trainingLabels, this.method, this.alpha);
        } else if (sparse && sparseTrainingDataset != null) {
            this.nb = new AbstractNaiveBayes(this.sparseTrainingDataset, this.trainingLabels, this.method, this.alpha);
        }
    }

    private int getMaxProbabilityClass(Map<Integer, Double> probabilities) {
        return probabilities.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElseThrow(() -> new RuntimeException("Probability map is empty"));
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
