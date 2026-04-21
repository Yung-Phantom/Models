package models.ml.Model.SVM;

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

public class SVM extends Model {
    public enum SVMMETHODS {
        LINEARSVC, SVC
    }

    private static final String DEFAULT_DENSE_METHOD = "linearsvc";
    private static final String DEFAULT_SPARSE_METHOD = "svc";
    private static final String DEFAULT_KERNEL = "linearkernel";
    private static final double DEFAULT_C = 1.0;
    private static final double DEFAULT_LEARNING_RATE = 0.01;
    private static final int DEFAULT_EPOCHS = 1000;
    private static final double DEFAULT_GAMMA = 0.1;
    private static final int DEFAULT_DEGREE = 3;
    private static final double DEFAULT_COEF0 = 1.0;

    public String method;
    public String kernel;
    public double C;
    public double learningRate;
    public int epochs;
    public double gamma;
    public int degree;
    public double coef0;
    private AbstractSVM svm;

    public SVM() {
    }

    public SVM(double[][] trainingDataset, String method, String kernel,
            double C, double learningRate, int epochs, double gamma, int degree, double coef0) {
        super(trainingDataset);
        initDenseParams(method, kernel, C, learningRate, epochs, gamma, degree, coef0);
        initSVM();
    }

    public SVM(double[][] trainingDataset) {
        this(trainingDataset, DEFAULT_DENSE_METHOD, DEFAULT_KERNEL,
                DEFAULT_C, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_GAMMA, DEFAULT_DEGREE, DEFAULT_COEF0);
    }

    public SVM(double[][] trainingDataset, double[] trainingLabels,
            String method, String kernel,
            double C, double learningRate, int epochs, double gamma, int degree, double coef0) {
        super(trainingDataset, trainingLabels);
        initDenseParams(method, kernel, C, learningRate, epochs, gamma, degree, coef0);
        initSVM();
    }

    public SVM(double[][] trainingDataset, double[] trainingLabels) {
        this(trainingDataset, trainingLabels, DEFAULT_DENSE_METHOD, DEFAULT_KERNEL,
                DEFAULT_C, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_GAMMA, DEFAULT_DEGREE, DEFAULT_COEF0);
    }

    public SVM(double[][] trainingDataset, double[][] testDataset, String method, String kernel,
            double C, double learningRate, int epochs, double gamma, int degree, double coef0) {
        super(trainingDataset, testDataset);
        initDenseParams(method, kernel, C, learningRate, epochs, gamma, degree, coef0);
        initSVM();
    }

    public SVM(double[][] trainingDataset, double[][] testDataset) {
        this(trainingDataset, testDataset, DEFAULT_DENSE_METHOD, DEFAULT_KERNEL,
                DEFAULT_C, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_GAMMA, DEFAULT_DEGREE, DEFAULT_COEF0);
    }

    public SVM(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels,
            String method, String kernel,
            double C, double learningRate, int epochs, double gamma, int degree, double coef0) {
        super(trainingDataset, trainingLabels, testDataset, testLabels);
        initDenseParams(method, kernel, C, learningRate, epochs, gamma, degree, coef0);
        initSVM();
    }

    public SVM(double[][] trainingDataset, double[] trainingLabels,
            double[][] testDataset, double[] testLabels) {
        this(trainingDataset, trainingLabels, testDataset, testLabels,
                DEFAULT_DENSE_METHOD, DEFAULT_KERNEL,
                DEFAULT_C, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_GAMMA, DEFAULT_DEGREE, DEFAULT_COEF0);
    }

    public SVM(List<Map<Integer, Double>> trainingDataset,
            String method, String kernel,
            double C, double learningRate, int epochs, double gamma, int degree, double coef0) {
        super(trainingDataset);
        initSparseParams(method, kernel, C, learningRate, epochs, gamma, degree, coef0);
        initSVM();
    }

    public SVM(List<Map<Integer, Double>> trainingDataset) {
        this(trainingDataset, DEFAULT_SPARSE_METHOD, DEFAULT_KERNEL,
                DEFAULT_C, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_GAMMA, DEFAULT_DEGREE, DEFAULT_COEF0);
    }

    public SVM(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            String method, String kernel,
            double C, double learningRate, int epochs, double gamma, int degree, double coef0) {
        super(trainingDataset, trainingLabels);
        initSparseParams(method, kernel, C, learningRate, epochs, gamma, degree, coef0);
        initSVM();
    }

    public SVM(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels) {
        this(trainingDataset, trainingLabels, DEFAULT_SPARSE_METHOD, DEFAULT_KERNEL,
                DEFAULT_C, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_GAMMA, DEFAULT_DEGREE, DEFAULT_COEF0);
    }

    public SVM(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset,
            String method, String kernel,
            double C, double learningRate, int epochs, double gamma, int degree, double coef0) {
        super(trainingDataset, testDataset);
        initSparseParams(method, kernel, C, learningRate, epochs, gamma, degree, coef0);
        initSVM();
    }

    public SVM(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset) {
        this(trainingDataset, testDataset, DEFAULT_SPARSE_METHOD, DEFAULT_KERNEL,
                DEFAULT_C, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_GAMMA, DEFAULT_DEGREE, DEFAULT_COEF0);
    }

    public SVM(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels,
            String method, String kernel,
            double C, double learningRate, int epochs, double gamma, int degree, double coef0) {
        super(trainingDataset, trainingLabels, testDataset, testLabels);
        initSparseParams(method, kernel, C, learningRate, epochs, gamma, degree, coef0);
        initSVM();
    }

    public SVM(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels) {
        this(trainingDataset, trainingLabels, testDataset, testLabels,
                DEFAULT_SPARSE_METHOD, DEFAULT_KERNEL,
                DEFAULT_C, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS, DEFAULT_GAMMA, DEFAULT_DEGREE, DEFAULT_COEF0);
    }

    @Override
    public void fit(double[][] trainingDataset) {
        if (trainingDataset == null || trainingDataset.length == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        this.denseTrainingDataset = this.scaler.extractFeatures(trainingDataset);
        this.trainingLabels = this.scaler.extractLabels(trainingDataset);
        initDenseParams();
        initSVM();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset, String method, String kernel, double C, double learningRate,
            int epochs) {
        this.method = method;
        this.kernel = kernel;
        this.C = C;
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
        initSVM();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset,
            double[] trainingLabels, String method, String kernel, double C, double learningRate,
            int epochs) {
        this.method = method;
        this.kernel = kernel;
        this.C = C;
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
        initSVM();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset,
            double[][] testDataset, String method, String kernel, double C, double learningRate,
            int epochs) {
        this.method = method;
        this.kernel = kernel;
        this.C = C;
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
        initSVM();
        this.sparse = false;
    }

    public void fit(double[][] trainingDataset,
            double[] trainingLabels, double[][] testDataset, double[] testLabels, String method, String kernel,
            double C, double learningRate, int epochs) {
        this.method = method;
        this.kernel = kernel;
        this.C = C;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.fit(trainingDataset, trainingLabels, testDataset, testLabels);
    }

    @Override
    public void fit(List<Map<Integer, Double>> trainingDataset) {
        if (trainingDataset == null || trainingDataset.size() == 0) {
            throw new IllegalArgumentException("trainingDataset cannot be empty");
        }
        this.sparseTrainingDataset = trainingDataset;
        initSparseParams();
        initSVM();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, String method, String kernel,
            double C, double learningRate, int epochs) {
        this.method = method;
        this.kernel = kernel;
        this.C = C;
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
        initSVM();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, String method, String kernel,
            double C, double learningRate, int epochs) {
        this.method = method;
        this.kernel = kernel;
        this.C = C;
        this.learningRate = learningRate;
        this.epochs = epochs;
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
        initSVM();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset, String method,
            String kernel,
            double C, double learningRate, int epochs) {
        this.method = method;
        this.kernel = kernel;
        this.C = C;
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
        initSVM();
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels, String method, String kernel, double C,
            double learningRate, int epochs) {
        this.method = method;
        this.kernel = kernel;
        this.C = C;
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
        initSVM();
        this.sparse = false;
    }

    @Override
    public int[] predictLabels() {
        if (svm == null) {
            throw new IllegalStateException("Model must be fitted before prediction.");
        }

        int n = sparse ? sparseTestDataset.size() : denseTestDataset.length;
        int[] predictions = new int[n];

        for (int i = 0; i < n; i++) {
            predictions[i] = sparse
                    ? svm.predictSparse(sparseTestDataset.get(i))
                    : svm.predict(denseTestDataset[i]);
        }
        return predictions;
    }

    @Override
    public List<Map<Integer, Double>> predictProbabilities() {
        if (svm == null) {
            throw new IllegalStateException("Model must be fitted before prediction.");
        }

        List<Map<Integer, Double>> probabilities = new ArrayList<>();
        int n = sparse ? sparseTestDataset.size() : denseTestDataset.length;

        for (int i = 0; i < n; i++) {
            double score = sparse
                    ? svm.linearDecisionSparse(svm.getWeights(), sparseTestDataset.get(i), svm.getBias())
                    : svm.predictScore(denseTestDataset[i]);

            // Sigmoid transform
            double probPositive = 1.0 / (1.0 + Math.exp(-score));
            double probNegative = 1.0 - probPositive;

            Map<Integer, Double> probMap = new HashMap<>();
            probMap.put(1, probPositive);
            probMap.put(-1, probNegative);

            probabilities.add(probMap);
        }
        return probabilities;
    }

    @Override
    public int[][] predictMultiLabel(double threshold) {
        List<Map<Integer, Double>> probs = predictProbabilities();
        int[][] multiLabels = new int[probs.size()][];

        for (int i = 0; i < probs.size(); i++) {
            List<Integer> labels = new ArrayList<>();
            for (var e : probs.get(i).entrySet()) {
                if (e.getValue() >= threshold) {
                    labels.add(e.getKey());
                }
            }
            multiLabels[i] = labels.stream().mapToInt(Integer::intValue).toArray();
        }
        return multiLabels;
    }

    @Override
    public Map<String, Double> evaluate(double[][] dataset, double[] labels, String[] metrics) {
        if (dataset == null || labels == null || dataset.length != labels.length) {
            throw new IllegalArgumentException("Dataset and labels must match in length.");
        }

        // Temporarily swap state
        double[][] originalDenseTest = this.denseTestDataset;
        boolean originalSparseFlag = this.sparse;

        this.denseTestDataset = dataset;
        this.sparse = false;

        int[] predictions = predictLabels();
        Map<String, Double> results = calculateMetrics(predictions, labels, metrics);

        // Restore state
        this.denseTestDataset = originalDenseTest;
        this.sparse = originalSparseFlag;

        return results;
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
    public Map<String, Double> evaluate(String[] metrics) {
        if (sparse) {
            return evaluate(this.sparseTestDataset, this.testLabels, metrics);
        }
        return evaluate(this.denseTestDataset, this.testLabels, metrics);
    }

    @Override
    public void save(String filename) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            // Save core parameters
            oos.writeObject(this.method);
            oos.writeObject(this.kernel);
            oos.writeObject(this.C);
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
            throw new RuntimeException("Error saving SVM model: " + e.getMessage(), e);
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    public void load(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            // Load core parameters
            this.method = (String) ois.readObject();
            this.kernel = (String) ois.readObject();
            this.C = (double) ois.readObject();
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

            // Rebuild the SVM
            refresh();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException("Error loading SVM model: " + e.getMessage(), e);
        }
    }

    @Override
    public String getModelType() {
        return "Support Vector Machine(SVM)";
    }

    @Override
    public String getVersion() {
        return "1.0.0";
    }

    @Override
    public void refresh() {
        if (!sparse && denseTrainingDataset != null) {
            this.svm = new AbstractSVM(denseTrainingDataset, trainingLabels, C, learningRate, epochs, kernel, method,
                    gamma, degree, coef0);
        } else if (sparse && sparseTrainingDataset != null) {
            this.svm = new AbstractSVM(sparseTrainingDataset, trainingLabels, C, learningRate, epochs, kernel,
                    method, gamma, degree, coef0);
        } else {
            throw new IllegalArgumentException("Dataset cannot be null.");
        }
    }

    private void initDenseParams(String method, String kernel, double C, double learningRate, int epochs,
            double gamma, int degree, double coef0) {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_DENSE_METHOD;
        this.kernel = (kernel != null) ? kernel.toLowerCase() : DEFAULT_KERNEL;
        this.C = (C > 0) ? C : DEFAULT_C;
        this.learningRate = (learningRate > 0) ? learningRate : DEFAULT_LEARNING_RATE;
        this.epochs = (epochs > 0) ? epochs : DEFAULT_EPOCHS;
        this.gamma = gamma;
        this.degree = degree;
        this.coef0 = coef0;
        this.sparse = false;
    }

    private void initSparseParams(String method, String kernel, double C, double learningRate, int epochs,
            double gamma, int degree, double coef0) {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_SPARSE_METHOD;
        this.kernel = (kernel != null) ? kernel.toLowerCase() : DEFAULT_KERNEL;
        this.C = (C > 0) ? C : DEFAULT_C;
        this.learningRate = (learningRate > 0) ? learningRate : DEFAULT_LEARNING_RATE;
        this.epochs = (epochs > 0) ? epochs : DEFAULT_EPOCHS;
        this.gamma = gamma;
        this.degree = degree;
        this.coef0 = coef0;
        this.sparse = true;
    }

    private void initDenseParams() {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_DENSE_METHOD;
        this.kernel = (kernel != null) ? kernel.toLowerCase() : DEFAULT_KERNEL;
        this.C = (C > 0) ? C : DEFAULT_C;
        this.learningRate = (learningRate > 0) ? learningRate : DEFAULT_LEARNING_RATE;
        this.epochs = (epochs > 0) ? epochs : DEFAULT_EPOCHS;
        this.sparse = false;
    }

    private void initSparseParams() {
        this.method = (method != null) ? method.toLowerCase() : DEFAULT_SPARSE_METHOD;
        this.kernel = (kernel != null) ? kernel.toLowerCase() : DEFAULT_KERNEL;
        this.C = (C > 0) ? C : DEFAULT_C;
        this.learningRate = (learningRate > 0) ? learningRate : DEFAULT_LEARNING_RATE;
        this.epochs = (epochs > 0) ? epochs : DEFAULT_EPOCHS;
        this.sparse = true;
    }

    private void initSVM() {
        if (!sparse && denseTrainingDataset != null) {
            this.svm = new AbstractSVM(denseTrainingDataset, trainingLabels, C, learningRate, epochs, kernel, method,
                    gamma, degree, coef0);
        } else if (sparse && sparseTrainingDataset != null) {
            this.svm = new AbstractSVM(sparseTrainingDataset, trainingLabels, C, learningRate, epochs, kernel,
                    method, gamma, degree, coef0);
        } else {
            throw new IllegalArgumentException("Dataset cannot be null.");
        }
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
