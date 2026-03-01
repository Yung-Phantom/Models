package models.ml.SVM;

import java.util.Map;

/**
 * Wrapper class for Support Vector Machine (SVM).
 * Handles predictions and evaluation metrics.
 * 
 * @author Justice
 * @version 1.1
 */
public class SVM {

    public double[][] dataset;
    public double[][] points;
    public Map<Integer, Double>[] sparseDataset;
    public Map<Integer, Double>[] sparsePoints;

    public double[] trainingLabels;
    public double[] testLabels;

    public AbstractSVM svm;
    public boolean sparse;

    public double C;
    public double learningRate;
    public int epochs;
    public String kernel;
    public String method;

    public SVM(double[][] dataset, double[] trainingLabels, double[][] points, double[] testLabels,
            double C, double learningRate, int epochs,
            String kernel, String method) {
        if (dataset == null)
            throw new IllegalArgumentException("Dataset cannot be null.");
        if (trainingLabels == null || trainingLabels.length == 0)
            throw new IllegalArgumentException("Training labels cannot be null or empty.");

        if (testLabels == null || testLabels.length == 0)
            throw new IllegalArgumentException("Test labels cannot be null or empty.");

        if (dataset.length == 0)
            throw new IllegalArgumentException("Dataset cannot be empty.");

        if (points == null || points.length == 0)
            throw new IllegalArgumentException("Points cannot be null or empty.");

        if (trainingLabels.length != dataset.length)
            throw new IllegalArgumentException("Training labels length must match dataset length.");

        if (testLabels.length != points.length)
            throw new IllegalArgumentException("Test labels length must match points length.");

        if (dataset[0].length == 0)
            throw new IllegalArgumentException("Dataset must contain at least one feature.");
        this.sparse = false;
        this.dataset = dataset;
        this.points = points;

        this.trainingLabels = trainingLabels;
        this.testLabels = testLabels;

        normalizeLabels(trainingLabels);
        normalizeLabels(testLabels);

        this.C = C;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.kernel = kernel != null ? kernel : "linearKernel";
        this.method = method != null ? method : "linearsvc";

        this.svm = new AbstractSVM(this.dataset, this.trainingLabels, C, learningRate, epochs, kernel, method);
    }

    public SVM(double[][] dataset, double[] trainingLabels, double[][] points, double[] testLabels) {
        this(dataset, trainingLabels, points, testLabels, 1.0, 0.01, 1000, "linearKernel", "linearsvc");
    }

    public SVM(Map<Integer, Double>[] sparseDataset, double[] trainingLabels, Map<Integer, Double>[] sparsePoints,
            double[] testLabels, int numFeatures, double C,
            double learningRate, int epochs, String method) {

        if (sparseDataset == null)
            throw new IllegalArgumentException("Sparse dataset cannot be null.");
        if (trainingLabels == null || trainingLabels.length == 0)
            throw new IllegalArgumentException("Training labels cannot be null or empty.");

        if (testLabels == null || testLabels.length == 0)
            throw new IllegalArgumentException("Test labels cannot be null or empty.");

        if (sparsePoints == null || sparsePoints.length == 0)
            throw new IllegalArgumentException("Sparse points cannot be null or empty.");

        if (trainingLabels.length != sparseDataset.length)
            throw new IllegalArgumentException("Training labels length must match sparse dataset length.");

        if (testLabels.length != sparsePoints.length)
            throw new IllegalArgumentException("Test labels length must match sparse points length.");

        if (numFeatures <= 0)
            throw new IllegalArgumentException("Number of features must be positive.");
        this.sparse = true;

        this.sparseDataset = sparseDataset;
        this.sparsePoints = sparsePoints;
        this.trainingLabels = trainingLabels;
        this.testLabels = testLabels;

        normalizeLabels(this.trainingLabels);
        normalizeLabels(this.testLabels);

        this.C = C;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.method = method != null ? method : "linearsvc";

        this.svm = new AbstractSVM(sparseDataset, this.trainingLabels, numFeatures, C,
                learningRate, epochs, this.method);
    }

    /** Predict label for a single query point */
    public int predict(int queryIndex) {
        if (!sparse) {
            if (points == null || points.length == 0)
                throw new IllegalStateException("Points not initialized.");
            if (queryIndex < 0 || queryIndex >= points.length)
                throw new IllegalArgumentException("Query index out of bounds.");
        } else {
            if (sparsePoints == null || sparsePoints.length == 0)
                throw new IllegalStateException("Sparse points not initialized.");
            if (queryIndex < 0 || queryIndex >= sparsePoints.length)
                throw new IllegalArgumentException("Query index out of bounds.");
        }
        if (!sparse) {
            return svm.predict(points[queryIndex]);
        } else {
            return svm.predictSparse(sparsePoints[queryIndex]);
        }
    }

    /** Predict labels for all query points */
    public int[] predictAll() {
        int n = sparse ? sparsePoints.length : points.length;
        int[] preds = new int[n];
        for (int i = 0; i < n; i++) {
            preds[i] = predict(i);
        }
        return preds;
    }

    /** Compute accuracy on points points */
    public double accuracy() {
        if (testLabels == null || testLabels.length == 0)
            throw new IllegalStateException("Test labels cannot be null or empty.");

        if (!sparse && testLabels.length != points.length)
            throw new IllegalStateException("Test labels length must match points length.");

        if (sparse && testLabels.length != sparsePoints.length)
            throw new IllegalStateException("Test labels length must match sparse points length.");
        int correct = 0;
        int n = testLabels.length;
        for (int i = 0; i < n; i++) {
            if (predict(i) == (int) testLabels[i])
                correct++;
        }
        return (double) correct / n;
    }

    private void normalizeLabels(double[] labels) {
        if (labels == null)
            return;

        for (int i = 0; i < labels.length; i++) {
            labels[i] = (labels[i] == 0) ? -1 : 1;
        }
    }
}
