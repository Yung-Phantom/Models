package models.ml.Model.LogisticRegression;

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
import models.ml.Preprocessing.Scaling.StandardScaler;

public class LogisticRegression extends Model {
    public enum LRMETHODS {
        BINARY, MULTINOMIAL, ORDINAL
    }

    private static final String DEFAULT_DENSE_METHOD = "binary";
    private static final String DEFAULT_SPARSE_METHOD = "binary";
    private static final double DEFAULT_LEARNING_RATE = 0.01;
    private static final int DEFAULT_EPOCHS = 1000;

    public String method;
    public double learningRate;
    public int epochs;
    private AbstractLogisticRegression lr;

    public LogisticRegression() {
    }

    public LogisticRegression(double[][] trainingDataset, String method, double learningRate, int epochs) {
        super(trainingDataset);
        initDenseParams(method, learningRate, epochs);
        initLR();
    }

    public LogisticRegression(double[][] trainingDataset, double[] trainingLabels, String method, double learningRate, int epochs) {
        super(trainingDataset, trainingLabels);
        initDenseParams(method, learningRate, epochs);
        initLR();
    }

    public LogisticRegression(double[][] trainingDataset, double[][] testDataset, String method, double learningRate, int epochs) {
        super(trainingDataset, testDataset);
        initDenseParams(method, learningRate, epochs);
        initLR();
    }

    public LogisticRegression(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            String method, double learningRate, int epochs) {
        super(trainingDataset, trainingLabels, testDataset, testLabels);
        initDenseParams(method, learningRate, epochs);
        initLR();
    }

    public LogisticRegression(List<Map<Integer, Double>> trainingDataset, String method, double learningRate, int epochs) {
        super(trainingDataset);
        initSparseParams(method, learningRate, epochs);
        initLR();
    }

    public LogisticRegression(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, String method,
            double learningRate, int epochs) {
        super(trainingDataset, trainingLabels);
        initSparseParams(method, learningRate, epochs);
        initLR();
    }

    public LogisticRegression(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset,
            String method, double learningRate, int epochs) {
        super(trainingDataset, testDataset);
        initSparseParams(method, learningRate, epochs);
        initLR();
    }

    public LogisticRegression(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, String method, double learningRate, int epochs) {
        super(trainingDataset, trainingLabels, testDataset, testLabels);
        initSparseParams(method, learningRate, epochs);
        initLR();
    }

    public LogisticRegression(DatasetLoader loader, int splitPercent, String splitStrategy,
            String method, double learningRate, int epochs) {
        super(loader, splitPercent, splitStrategy);
        initDenseParams(method, learningRate, epochs);
        initLR();
    }

    public LogisticRegression(DatasetLoader loader, double splitDecimal, String splitStrategy,
            String method, double learningRate, int epochs) {
        super(loader, splitDecimal, splitStrategy);
        initDenseParams(method, learningRate, epochs);
        initLR();
    }

    @Override
    public void fit(double[][] trainingDataset) {
        if (trainingDataset == null || trainingDataset.length == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        this.denseTrainingDataset = this.scaler.extractFeatures(trainingDataset);
        this.trainingLabels = this.scaler.extractLabels(trainingDataset);
        initDenseParams();
        initLR();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, String method) {
        this.method = method;
        this.fit(trainingDataset);
    }

    public void fit(double[][] trainingDataset, double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.fit(trainingDataset);
    }

    public void fit(double[][] trainingDataset, String method, double learningRate, int epochs) {
        this.method = method;
        this.learningRate = learningRate;
        this.epochs = epochs;
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
        initLR();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, String method) {
        this.method = method;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, String method, double learningRate, int epochs) {
        this.method = method;
        this.learningRate = learningRate;
        this.epochs = epochs;
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
        initLR();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, double[][] testDataset, String method) {
        this.method = method;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(double[][] trainingDataset, double[][] testDataset, double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(double[][] trainingDataset, double[][] testDataset, String method, double learningRate, int epochs) {
        this.method = method;
        this.learningRate = learningRate;
        this.epochs = epochs;
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
        initLR();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            String method) {
        this.method = method;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            String method, double learningRate, int epochs) {
        this.method = method;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    @Override
    public void fit(List<Map<Integer, Double>> trainingDataset) {
        if (trainingDataset == null || trainingDataset.size() == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        if (this.trainingLabels == null || this.trainingLabels.length == 0) {
            throw new IllegalArgumentException("trainingLabels must be set before calling fit() without labels for sparse data. Use fit(trainingDataset, trainingLabels) or setTrainingLabels().");
        }
        this.sparseTrainingDataset = trainingDataset;
        initSparseParams();
        initLR();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, String method) {
        this.method = method;
        this.fit(trainingDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.fit(trainingDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, String method, double learningRate, int epochs) {
        this.method = method;
        this.learningRate = learningRate;
        this.epochs = epochs;
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
        initLR();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, String method) {
        this.method = method;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.fit(trainingDataset, trainingLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, String method, double learningRate, int epochs) {
        this.method = method;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.fit(trainingDataset, trainingLabels);
    }

    @Override
    public void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset) {
        if (trainingDataset == null || trainingDataset.size() == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        if (this.trainingLabels == null || this.trainingLabels.length == 0) {
            throw new IllegalArgumentException("trainingLabels must be set before calling fit() without labels for sparse data. Use fit(trainingDataset, trainingLabels) or setTrainingLabels().");
        }
        this.sparseTrainingDataset = trainingDataset;
        this.sparseTestDataset = testDataset;
        initSparseParams();
        initLR();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset, String method) {
        this.method = method;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset,
            double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.fit(trainingDataset, testDataset);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset,
            String method, double learningRate, int epochs) {
        this.method = method;
        this.learningRate = learningRate;
        this.epochs = epochs;
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
        initLR();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, String method) {
        this.method = method;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, String method, double learningRate, int epochs) {
        this.method = method;
        this.learningRate = learningRate;
        this.epochs = epochs;
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
        initLR();
        this.sparse = false;
    }

    @Override
    public double[] predictLabels() {
        if (lr == null)
            throw new IllegalStateException("Model must be fitted before prediction.");

        int n = sparse ? sparseTestDataset.size() : denseTestDataset.length;
        double[] predictions = new double[n];

        for (int i = 0; i < n; i++) {
            predictions[i] = sparse ? predict(sparseTestDataset.get(i)) : predict(denseTestDataset[i]);
        }
        return predictions;
    }

    @Override
    public double predict(double[] row) {
        if (lr == null)
            throw new IllegalStateException("Model must be fitted before prediction.");
        requireDense();
        if (row == null)
            throw new IllegalArgumentException("Row cannot be null");
        if (row.length < lr.numFeatures)
            throw new IllegalArgumentException("Row must have at least " + lr.numFeatures + " features");
        return lr.predictClass(row);
    }

    @Override
    public double predict(Map<Integer, Double> row) {
        if (lr == null)
            throw new IllegalStateException("Model must be fitted before prediction.");
        requireSparse();
        if (row == null)
            throw new IllegalArgumentException("Row cannot be null");
        if (row.isEmpty())
            throw new IllegalArgumentException("Row cannot be empty");
        return lr.predictClass(row);
    }

    @Override
    public List<Map<Integer, Double>> predictProbabilities() {
        if (lr == null)
            throw new IllegalStateException("Model must be fitted before prediction.");

        List<Map<Integer, Double>> allProbabilities = new ArrayList<>();
        int n = sparse ? sparseTestDataset.size() : denseTestDataset.length;

        for (int i = 0; i < n; i++) {
            allProbabilities.add(sparse ? lr.predictProbabilities(sparseTestDataset.get(i))
                    : lr.predictProbabilities(denseTestDataset[i]));
        }
        return allProbabilities;
    }

    @Override
    public double[][] predictMultiLabel(double threshold) {
        List<Map<Integer, Double>> probs = predictProbabilities();
        double[][] multiLabels = new double[probs.size()][];

        for (int i = 0; i < probs.size(); i++) {
            List<Double> labels = new ArrayList<>();
            for (var e : probs.get(i).entrySet()) {
                if (e.getValue() >= threshold) {
                    labels.add((double) e.getKey());
                }
            }
            multiLabels[i] = labels.stream().mapToDouble(Double::doubleValue).toArray();
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

        double[] predictions = predictLabels();
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

        double[] predictions = predictLabels();
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
            oos.writeObject(this.method);
            oos.writeObject(this.learningRate);
            oos.writeObject(this.epochs);
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
            throw new RuntimeException("Error saving Logistic Regression model: " + e.getMessage(), e);
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public void load(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            // Load core parameters
            this.method = (String) ois.readObject();
            this.learningRate = (double) ois.readObject();
            this.epochs = (int) ois.readObject();
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
            throw new RuntimeException("Error loading Logistic Regression model: " + e.getMessage(), e);
        }
    }

    @Override
    public String getModelType() {
        return "Logistic Regression";
    }

    @Override
    public String getVersion() {
        return "1.0.0";
    }

    @Override
    public void refresh() {
        if (!sparse && denseTrainingDataset != null) {
            this.lr = new AbstractLogisticRegression(this.denseTrainingDataset, this.trainingLabels, this.method,
                    this.learningRate, this.epochs);
        } else if (sparse && sparseTrainingDataset != null) {
            // For sparse, we need numFeatures - infer from training data
            int numFeatures = sparseTrainingDataset.stream()
                    .flatMap(m -> m.keySet().stream())
                    .max(Integer::compare)
                    .orElseThrow(() -> new IllegalArgumentException("Sparse dataset must contain features"))
                    + 1;
            this.lr = new AbstractLogisticRegression(this.sparseTrainingDataset, this.trainingLabels, numFeatures,
                    this.method, this.learningRate, this.epochs);
        } else {
            throw new IllegalArgumentException("Dataset cannot be null.");
        }
    }

    // --- Partial fit support ---
    // The base Model.partialFit() merges data and calls refresh(), which works for us.

    // --- Deprecated methods for backward compatibility ---

    /**
     * @deprecated Use {@link #fit(List, double[])} instead.
     */
    @Deprecated
    public void fit(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels, int numFeatures, String method) {
        fit(sparseDataset, trainingLabels);
        this.method = method;
    }

    /**
     * @deprecated Use {@link #refresh()} instead.
     */
    @Deprecated
    public void refreshLogisticRegression() {
        refresh();
    }

    /**
     * @deprecated Use {@link #predictLabels()} instead.
     */
    @Deprecated
    public int predict(int i) {
        if (!sparse && (denseTestDataset == null || denseTestDataset.length == 0)) {
            throw new IllegalStateException("No dense query points available for prediction.");
        }
        if (sparse && (sparseTestDataset == null || sparseTestDataset.isEmpty())) {
            throw new IllegalStateException("No sparse query points available for prediction.");
        }
        return sparse ? lr.predictClass(sparseTestDataset.get(i)) : lr.predictClass(denseTestDataset[i]);
    }

    /**
     * @deprecated Use {@link #predictLabels()} instead.
     */
    @Deprecated
    public double[] predictAll() {
        return predictLabels();
    }

    /**
     * @deprecated Use {@link #evaluate(String[])} with "accuracy" metric instead.
     */
    @Deprecated
    public double accuracy() {
        if (testLabels == null)
            throw new IllegalStateException("No test labels available");

        int correct = 0;
        double[] preds = predictLabels();
        for (int i = 0; i < Math.min(testLabels.length, preds.length); i++) {
            if (preds[i] == testLabels[i]) {
                correct++;
            }
        }
        return (double) correct / Math.max(1, testLabels.length);
    }

    /**
     * @deprecated Use {@link #evaluate(String[])} with "mse" metric instead.
     */
    @Deprecated
    public double mse() {
        if (testLabels == null)
            throw new IllegalStateException("No test labels available");
        double sum = 0.0;
        double[] preds = predictLabels();
        for (int i = 0; i < Math.min(testLabels.length, preds.length); i++) {
            double error = preds[i] - testLabels[i];
            sum += error * error;
        }
        return sum / Math.max(1, testLabels.length);
    }

    /**
     * @deprecated Use {@link #evaluate(String[])} with "r2" metric instead (not yet in Metrics).
     */
    @Deprecated
    public double r2() {
        if (testLabels == null)
            throw new IllegalStateException("No test labels available");

        double mean = 0.0;
        for (double y : testLabels)
            mean += y;
        mean /= testLabels.length;

        double ssTot = 0.0, ssRes = 0.0;
        double[] preds = predictLabels();
        for (int i = 0; i < Math.min(testLabels.length, preds.length); i++) {
            double y = testLabels[i];
            double yHat = preds[i];
            ssTot += (y - mean) * (y - mean);
            ssRes += (y - yHat) * (y - yHat);
        }
        return ssTot == 0.0 ? 0.0 : 1.0 - (ssRes / ssTot);
    }

    /**
     * @deprecated Use {@link #fit(double[][], double[])} then set test data separately.
     */
    @Deprecated
    public void setPoints(double[][] points, double[] testLabels) {
        if (points == null) {
            throw new IllegalArgumentException("Points cannot be null.");
        }
        if (points.length == 0) {
            throw new IllegalArgumentException("Points cannot be empty.");
        }
        if (sparse) {
            throw new IllegalStateException("Dense points not allowed in sparse mode.");
        }
        if (testLabels == null) {
            throw new IllegalArgumentException("Test labels cannot be null.");
        }
        if (testLabels.length == 0) {
            throw new IllegalArgumentException("Test labels cannot be empty.");
        }
        if (testLabels.length != points.length) {
            throw new IllegalArgumentException("Test labels length must match points length.");
        }
        this.denseTestDataset = points;
        this.testLabels = testLabels;
    }

    /**
     * @deprecated Use {@link #fit(List, double[])} then set test data separately.
     */
    @Deprecated
    public void setSparsePoints(List<Map<Integer, Double>> sparsePoints, double[] testLabels) {
        if (sparsePoints == null) {
            throw new IllegalArgumentException("Sparse points cannot be null.");
        }
        if (sparsePoints.isEmpty()) {
            throw new IllegalArgumentException("Sparse points cannot be empty.");
        }
        if (!sparse) {
            throw new IllegalStateException("Sparse points not allowed in dense mode.");
        }
        if (testLabels == null) {
            throw new IllegalArgumentException("Test labels cannot be null.");
        }
        if (testLabels.length == 0) {
            throw new IllegalArgumentException("Test labels cannot be empty.");
        }
        if (testLabels.length != sparsePoints.size()) {
            throw new IllegalArgumentException("Test labels length must match points length.");
        }
        this.sparseTestDataset = sparsePoints;
        this.testLabels = testLabels;
    }

    /**
     * @deprecated Use {@link #fit(double[][], double[])} instead.
     */
    @Deprecated
    public void setTrainingLabels(double[] trainingLabels) {
        if (trainingLabels == null) {
            throw new IllegalArgumentException("Training labels cannot be null.");
        }
        if (trainingLabels.length == 0) {
            throw new IllegalArgumentException("Training labels cannot be empty.");
        }
        this.trainingLabels = trainingLabels;
        refresh();
    }

    /**
     * @deprecated Use test data from fit() or set test data directly.
     */
    @Deprecated
    public void setTestLabels(double[] testLabels) {
        if (testLabels == null) {
            throw new IllegalArgumentException("Test labels cannot be null.");
        }
        if (testLabels.length == 0) {
            throw new IllegalArgumentException("Test labels cannot be empty.");
        }
        this.testLabels = testLabels;
    }

    /**
     * @deprecated Set method via constructor or fit() overload.
     */
    @Deprecated
    public void setMethod(String method) {
        if (method == null) {
            throw new IllegalArgumentException("Method cannot be null.");
        }
        this.method = method;
        refresh();
    }

    /**
     * @deprecated Set learning rate via constructor or fit() overload.
     */
    @Deprecated
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be positive and <= 1.");
        }
        this.learningRate = learningRate;
        refresh();
    }

    /**
     * @deprecated Set epochs via constructor or fit() overload.
     */
    @Deprecated
    public void setEpochs(int epochs) {
        this.epochs = epochs;
        refresh();
    }

    // --- Initialization helpers ---

    private void initDenseParams(String method, double learningRate, int epochs) {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_DENSE_METHOD;
        this.learningRate = (learningRate > 0 && learningRate <= 1) ? learningRate : DEFAULT_LEARNING_RATE;
        this.epochs = (epochs > 0) ? epochs : DEFAULT_EPOCHS;
        this.sparse = false;
    }

    private void initSparseParams(String method, double learningRate, int epochs) {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_SPARSE_METHOD;
        this.learningRate = (learningRate > 0 && learningRate <= 1) ? learningRate : DEFAULT_LEARNING_RATE;
        this.epochs = (epochs > 0) ? epochs : DEFAULT_EPOCHS;
        this.sparse = true;
    }

    private void initDenseParams() {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_DENSE_METHOD;
        this.learningRate = (learningRate > 0 && learningRate <= 1) ? learningRate : DEFAULT_LEARNING_RATE;
        this.epochs = (epochs > 0) ? epochs : DEFAULT_EPOCHS;
        this.sparse = false;
    }

    private void initSparseParams() {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_SPARSE_METHOD;
        this.learningRate = (learningRate > 0 && learningRate <= 1) ? learningRate : DEFAULT_LEARNING_RATE;
        this.epochs = (epochs > 0) ? epochs : DEFAULT_EPOCHS;
        this.sparse = true;
    }

    private void initLR() {
        if (!sparse && denseTrainingDataset != null) {
            this.lr = new AbstractLogisticRegression(this.denseTrainingDataset, this.trainingLabels, this.method,
                    this.learningRate, this.epochs);
        } else if (sparse && sparseTrainingDataset != null) {
            int numFeatures = sparseTrainingDataset.stream()
                    .flatMap(m -> m.keySet().stream())
                    .max(Integer::compare)
                    .orElseThrow(() -> new IllegalArgumentException("Sparse dataset must contain features"))
                    + 1;
            this.lr = new AbstractLogisticRegression(this.sparseTrainingDataset, this.trainingLabels, numFeatures,
                    this.method, this.learningRate, this.epochs);
        }
    }

    private Map<String, Double> calculateMetrics(double[] predictions, double[] labels, String[] metrics) {
        Map<String, Double> results = new HashMap<>();
        int[] actualInt = Arrays.stream(labels).mapToInt(d -> (int) d).toArray();
        int[] predInt = Arrays.stream(predictions).mapToInt(d -> (int) d).toArray();
        double[] predDouble = predictions;

        for (String metric : metrics) {
            switch (metric.toLowerCase()) {
                case "accuracy":
                    results.put("accuracy", Metrics.accuracy(predInt, actualInt));
                    break;
                case "precision":
                    results.put("precision", Metrics.precision(predInt, actualInt, 1));
                    break;
                case "recall":
                    results.put("recall", Metrics.recall(predInt, actualInt, 1));
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