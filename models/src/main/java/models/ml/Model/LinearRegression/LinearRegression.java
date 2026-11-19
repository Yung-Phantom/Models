package models.ml.Model.LinearRegression;

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

public class LinearRegression extends Model {
    public enum LRMETHODS {
        NORMAL, GRADIENT_DESCENT
    }

    private static final String DEFAULT_DENSE_METHOD = "normal";
    private static final String DEFAULT_SPARSE_METHOD = "gd";
    private static final double DEFAULT_LEARNING_RATE = 0.01;
    private static final int DEFAULT_EPOCHS = 1000;

    public String method;
    public int epochs;
    public double learningRate;
    private AbstractLinearRegression lr;

    public LinearRegression() {
    }

    public LinearRegression(double[][] trainingDataset, String method, double learningRate, int epochs) {
        super(trainingDataset);
        initDenseParams(method, learningRate, epochs);
        initLR();
    }

    public LinearRegression(double[][] trainingDataset, double[] trainingLabels, String method, double learningRate, int epochs) {
        super(trainingDataset, trainingLabels);
        initDenseParams(method, learningRate, epochs);
        initLR();
    }

    public LinearRegression(double[][] trainingDataset, double[][] testDataset, String method, double learningRate, int epochs) {
        super(trainingDataset, testDataset);
        initDenseParams(method, learningRate, epochs);
        initLR();
    }

    public LinearRegression(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            String method, double learningRate, int epochs) {
        super(trainingDataset, trainingLabels, testDataset, testLabels);
        initDenseParams(method, learningRate, epochs);
        initLR();
    }

    public LinearRegression(List<Map<Integer, Double>> trainingDataset, String method, double learningRate, int epochs) {
        super(trainingDataset);
        initSparseParams(method, learningRate, epochs);
        initLR();
    }

    public LinearRegression(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, String method,
            double learningRate, int epochs) {
        super(trainingDataset, trainingLabels);
        initSparseParams(method, learningRate, epochs);
        initLR();
    }

    public LinearRegression(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset,
            String method, double learningRate, int epochs) {
        super(trainingDataset, testDataset);
        initSparseParams(method, learningRate, epochs);
        initLR();
    }

    public LinearRegression(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, String method, double learningRate, int epochs) {
        super(trainingDataset, trainingLabels, testDataset, testLabels);
        initSparseParams(method, learningRate, epochs);
        initLR();
    }

    public LinearRegression(DatasetLoader loader, int splitPercent, String splitStrategy,
            String method, double learningRate, int epochs) {
        super(loader, splitPercent, splitStrategy);
        initDenseParams(method, learningRate, epochs);
        initLR();
    }

    public LinearRegression(DatasetLoader loader, double splitDecimal, String splitStrategy,
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
        this.trainingLabels = this.scaler.extractLabelsRegression(trainingDataset);
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
        this.trainingLabels = this.scaler.extractLabelsRegression(trainingDataset);
        this.testLabels = this.scaler.extractLabelsRegression(testDataset);
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

    public void fit(List<Map<Integer, Double>> trainingDataset,
            List<Map<Integer, Double>> testDataset, String method, double learningRate, int epochs) {
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
        this.trainingLabels = this.scaler.extractLabelsRegression(splitData.train);

        this.denseTestDataset = this.scaler.extractFeatures(splitData.test);
        this.testLabels = this.scaler.extractLabelsRegression(splitData.test);
        initDenseParams();
        initLR();
        this.sparse = false;
    }

    @Override
    public double[] predictLabels() {
        if (lr == null)
            throw new IllegalStateException("Model must be fitted before prediction.");

        if (!sparse && (denseTestDataset == null || denseTestDataset.length == 0)) {
            throw new IllegalStateException("No test dataset available for prediction. Call fit() with test data or use evaluate() with explicit dataset.");
        }
        if (sparse && (sparseTestDataset == null || sparseTestDataset.isEmpty())) {
            throw new IllegalStateException("No test dataset available for prediction. Call fit() with test data or use evaluate() with explicit dataset.");
        }

        int n = sparse ? sparseTestDataset.size() : denseTestDataset.length;
        double[] predictions = new double[n];

        for (int i = 0; i < n; i++) {
            predictions[i] = sparse ? lr.predictSparseRow(sparseTestDataset.get(i)) : lr.predictRow(denseTestDataset[i]);
        }
        return predictions;
    }

    @Override
    public List<Map<Integer, Double>> predictProbabilities() {
        throw new UnsupportedOperationException("predictProbabilities is not supported for regression");
    }

    @Override
    public double[][] predictMultiLabel(double threshold) {
        throw new UnsupportedOperationException("predictMultiLabel is not supported for regression");
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
            oos.writeObject(this.epochs);
            oos.writeObject(this.method);
            oos.writeObject(this.learningRate);
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
            throw new RuntimeException("Error saving LinearRegression model: " + e.getMessage(), e);
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public void load(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            // Load core parameters
            this.epochs = (int) ois.readObject();
            this.method = (String) ois.readObject();
            this.learningRate = (double) ois.readObject();
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
            throw new RuntimeException("Error loading LinearRegression model: " + e.getMessage(), e);
        }
    }

    @Override
    public String getModelType() {
        return "Linear Regression";
    }

    @Override
    public String getVersion() {
        return "1.0.0";
    }

    @Override
    public void refresh() {
        if (!sparse && denseTrainingDataset != null) {
            this.lr = new AbstractLinearRegression(this.denseTrainingDataset, this.trainingLabels, this.method,
                    this.learningRate,
                    this.epochs);
        } else if (sparse && sparseTrainingDataset != null) {
            // numFeatures is inferred in AbstractLinearRegression sparse constructor
            this.lr = new AbstractLinearRegression(this.sparseTrainingDataset, this.trainingLabels, this.method,
                    this.learningRate,
                    this.epochs);
        } else {
            throw new IllegalArgumentException("Dataset cannot be null.");
        }
    }

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
            this.lr = new AbstractLinearRegression(this.denseTrainingDataset, this.trainingLabels, this.method,
                    this.learningRate, this.epochs);
        } else if (sparse && sparseTrainingDataset != null) {
            this.lr = new AbstractLinearRegression(this.sparseTrainingDataset, this.trainingLabels, this.method,
                    this.learningRate, this.epochs);
        }
    }

    private Map<String, Double> calculateMetrics(double[] predictions, double[] labels, String[] metrics) {
        Map<String, Double> results = new HashMap<>();

        for (String metric : metrics) {
            switch (metric.toLowerCase()) {
                case "mse":
                    results.put("mse", Metrics.mse(predictions, labels));
                    break;
                case "rmse":
                    results.put("rmse", Metrics.rmse(predictions, labels));
                    break;
                case "mae":
                    results.put("mae", Metrics.mae(predictions, labels));
                    break;
                case "r2":
                    results.put("r2", Metrics.r2(predictions, labels));
                    break;
                case "accuracy":
                case "precision":
                case "recall":
                case "specificity":
                case "f1score":
                case "balancedaccuracy":
                case "mcc":
                    throw new IllegalArgumentException("Classification metric '" + metric + "' not supported for regression. Use: mse, rmse, mae, r2");
                default:
                    throw new IllegalArgumentException("Unknown metric: " + metric + ". Supported regression metrics: mse, rmse, mae, r2");
            }
        }
        return results;
    }
}