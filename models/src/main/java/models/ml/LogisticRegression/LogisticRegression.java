package models.ml.LogisticRegression;

import java.util.List;
import java.util.Map;

public class LogisticRegression {
    public double[][] dataset;
    public double[][] points;

    public List<Map<Integer, Double>> sparseDataset;
    public List<Map<Integer, Double>> sparsePoints;

    public double[] trainingLabels;
    public double[] testLabels;

    public AbstractLogisticRegression lr;
    public boolean sparse;
    public String method;
    public double learningRate;
    public int epochs;
    public int numFeatures;

    public LogisticRegression() {
    }

    public LogisticRegression(double[][] dataset, double[] trainingLabels, double[][] points, double[] testLabels,
            String method, double learningRate, int epochs) {

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
        this.method = method;
        this.learningRate = learningRate;
        this.epochs = epochs;

        refresh();
    }

    public LogisticRegression(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels,
            List<Map<Integer, Double>> sparsePoints,
            double[] testLabels, int numFeatures, String method, double learningRate,
            int epochs) {

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

        this.method = method;
        this.learningRate = learningRate;
        this.epochs = epochs;

        this.numFeatures = numFeatures;

        refresh();
    }

    public LogisticRegression(double[][] dataset, double[] trainingLabels, double[][] points, double[] testLabels) {
        this(dataset, trainingLabels, points, testLabels, "binary", 0.01, 1000);
    }

    public LogisticRegression(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels,
            List<Map<Integer, Double>> sparsePoints, double[] testLabels, int numFeatures) {
        this(sparseDataset, trainingLabels, sparsePoints, testLabels, numFeatures, "binary", 0.01, 1000);
    }

    public void fit(double[][] dataset, double[] trainingLabels, String method) {
        if (sparse)
            throw new IllegalStateException("Use the right fit for sparse models.");
        if (dataset == null || dataset.length == 0) {
            throw new IllegalArgumentException("Dataset cannot be null or empty.");
        }
        if (trainingLabels == null || trainingLabels.length == 0) {
            throw new IllegalArgumentException("Training labels cannot be null or empty.");
        }

        if (method == null) {
            throw new IllegalArgumentException("Method cannot be null.");
        }

        if (!sparse && trainingLabels.length != dataset.length) {
            throw new IllegalArgumentException("Training labels length must match dataset length. Expected "
                    + dataset.length + " but got " + trainingLabels.length);
        }

        this.dataset = dataset;
        this.trainingLabels = trainingLabels;
        this.method = method;
        this.sparse = false;
        refresh();
    }

    public void fit(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels, int numFeatures, String method) {
        if (!sparse)
            throw new IllegalStateException("Use fit for dense models.");
        if (sparseDataset == null || sparseDataset.size() == 0) {
            throw new IllegalArgumentException("sparseDataset cannot be null or empty.");
        }
        if (trainingLabels == null || trainingLabels.length == 0) {
            throw new IllegalArgumentException("Training labels cannot be null or empty.");
        }
        if (numFeatures <= 0) {
            throw new IllegalArgumentException("numFeatures must be positive.");
        }
        if (method == null) {
            throw new IllegalArgumentException("Method cannot be null.");
        }

        if (sparse && trainingLabels.length != sparseDataset.size()) {
            throw new IllegalArgumentException("Training labels length must match sparse dataset size. Expected "
                    + sparseDataset.size() + " but got " + trainingLabels.length);
        }
        this.sparseDataset = sparseDataset;
        this.trainingLabels = trainingLabels;
        this.numFeatures = numFeatures;
        this.method = method;
        this.sparse = true;
        for (Map<Integer, Double> row : sparseDataset) {
            for (Integer featureIndex : row.keySet()) {
                if (featureIndex + 1 > numFeatures) {
                    numFeatures = featureIndex + 1;
                }
            }
        }
        refresh();
    }

    public void refresh() {
        if (!sparse) {
            lr = new AbstractLogisticRegression(dataset, trainingLabels, method, learningRate, epochs);
        } else {
            lr = new AbstractLogisticRegression(sparseDataset, trainingLabels, numFeatures, method, learningRate,
                    epochs);
        }
    }

    public int predict(int i) {
        if (!sparse && (points == null || points.length == 0)) {
            throw new IllegalStateException("No dense query points available for prediction.");
        }
        if (sparse && (sparsePoints == null || sparsePoints.isEmpty())) {
            throw new IllegalStateException("No sparse query points available for prediction.");
        }
        return sparse ? lr.predictClass(sparsePoints.get(i)) : lr.predictClass(points[i]);
    }

    public int[] predictAll() {
        int n = sparse ? sparsePoints.size() : points.length;
        int[] preds = new int[n];
        for (int i = 0; i < n; i++) {
            preds[i] = predict(i);
        }
        return preds;
    }

    public double accuracy() {
        if (testLabels == null)
            throw new IllegalStateException("No test labels available");

        int correct = 0;
        for (int i = 0; i < testLabels.length; i++) {
            if (predict(i) == (int) testLabels[i]) {
                correct++;
            }
        }
        return (double) correct / testLabels.length;
    }

    public double mse() {
        if (testLabels == null)
            throw new IllegalStateException("No test labels available");
        double sum = 0.0;
        for (int i = 0; i < testLabels.length; i++) {
            double error = predict(i) - testLabels[i];
            sum += error * error;
        }
        return sum / testLabels.length;
    }

    public double r2() {
        if (testLabels == null)
            throw new IllegalStateException("No test labels available");

        double mean = 0.0;
        for (double y : testLabels)
            mean += y;
        mean /= testLabels.length;

        double ssTot = 0.0, ssRes = 0.0;
        for (int i = 0; i < testLabels.length; i++) {
            double y = testLabels[i];
            double yHat = predict(i);
            ssTot += (y - mean) * (y - mean);
            ssRes += (y - yHat) * (y - yHat);
        }
        return ssTot == 0.0 ? 0.0 : 1.0 - (ssRes / ssTot);
    }

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
        this.points = points;
        this.testLabels = testLabels;
    }

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
        if (testLabels.length != points.length) {
            throw new IllegalArgumentException("Test labels length must match points length.");
        }
        this.sparsePoints = sparsePoints;
        this.testLabels = testLabels;
    }

    public void setTrainingLabels(double[] trainingLabels) {
        if (trainingLabels == null) {
            throw new IllegalArgumentException("Training labels cannot be null.");
        }
        if (trainingLabels.length == 0) {
            throw new IllegalArgumentException("Training labels cannot be empty.");
        }
        if (!sparse && trainingLabels.length != dataset.length) {
            throw new IllegalArgumentException("Training labels length must match dataset length.");
        }
        if (sparse && trainingLabels.length != sparseDataset.size()) {
            throw new IllegalArgumentException("Training labels length must match sparse dataset size.");
        }
        this.trainingLabels = trainingLabels;
        this.lr = new AbstractLogisticRegression(this.dataset, this.trainingLabels, this.method, this.learningRate,
                this.epochs);
    }

    public void setTestLabels(double[] testLabels) {
        if (testLabels == null) {
            throw new IllegalArgumentException("Test labels cannot be null.");
        }
        if (testLabels.length == 0) {
            throw new IllegalArgumentException("Test labels cannot be empty.");
        }
        if (!sparse && testLabels.length != points.length) {
            throw new IllegalArgumentException("Test labels length must match points length.");
        }
        if (sparse && testLabels.length != sparsePoints.size()) {
            throw new IllegalArgumentException("Test labels length must match sparse points size.");
        }
        this.testLabels = testLabels;
    }

    public void setMethod(String method) {
        if (method == null) {
            throw new IllegalArgumentException("Method cannot be null.");
        }
        this.method = method;
        this.lr = new AbstractLogisticRegression(this.dataset, this.trainingLabels, this.method, this.learningRate,
                this.epochs);
    }

    public void setLearningRate(double learningRate) {
        if (learningRate <= 0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be positive.");
        }
        this.learningRate = learningRate;
        this.lr = new AbstractLogisticRegression(this.dataset, this.trainingLabels, this.method, this.learningRate,
                this.epochs);
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
        this.lr = new AbstractLogisticRegression(this.dataset, this.trainingLabels, this.method, this.learningRate,
                this.epochs);
    }

    public void refreshLogisticRegression() {
        if (!sparse && dataset != null) {
            this.lr = new AbstractLogisticRegression(this.dataset, this.trainingLabels, this.method, this.learningRate,
                    this.epochs);
        } else if (sparse && sparseDataset != null) {
            this.lr = new AbstractLogisticRegression(this.sparseDataset, this.trainingLabels, this.numFeatures,
                    this.method,
                    this.learningRate, this.epochs);
        }
    }

}
