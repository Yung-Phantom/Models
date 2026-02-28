package models.ml.NaiveBayes;

import java.util.*;

public class NaiveBayes {
    private double[][] dataset;
    private double[][] points;

    private List<Map<Integer, Double>> sparseDataset;
    private List<Map<Integer, Double>> sparsePoints;

    private double[] trainingLabels;
    private double[] testLabels;

    private AbstractNaiveBayes nb;
    private boolean sparse;
    private String method;
    private double alpha;

    // ---------------- Constructors ----------------

    public NaiveBayes() {}

    public NaiveBayes(double[][] dataset, double[] trainingLabels, double[][] points, double[] testLabels,
                      String method, double alpha) {
        this.sparse = false;
        validateDenseInputs(dataset, trainingLabels, points, testLabels);

        this.dataset = dataset;
        this.points = points;
        this.trainingLabels = trainingLabels;
        this.testLabels = testLabels;
        this.method = method;
        this.alpha = alpha;

        refreshNaiveBayes();
    }

    public NaiveBayes(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels,
                      List<Map<Integer, Double>> sparsePoints, double[] testLabels,
                      String method, double alpha) {
        this.sparse = true;
        validateSparseInputs(sparseDataset, sparsePoints, trainingLabels, testLabels);

        this.sparseDataset = sparseDataset;
        this.sparsePoints = sparsePoints;
        this.trainingLabels = trainingLabels;
        this.testLabels = testLabels;
        this.method = method;
        this.alpha = alpha;

        refreshNaiveBayes();
    }

    public NaiveBayes(double[][] dataset, double[] trainingLabels, double[][] points, double[] testLabels) {
        this(dataset, trainingLabels, points, testLabels, "gaussian", 1.0);
    }

    public NaiveBayes(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels,
                      List<Map<Integer, Double>> sparsePoints, double[] testLabels) {
        this(sparseDataset, trainingLabels, sparsePoints, testLabels, "gaussian", 1.0);
    }

    // ---------------- Fit Methods ----------------

    public void fit(double[][] dataset, double[] trainingLabels, String method, double alpha) {
        if (sparse) throw new IllegalStateException("Use the sparse fit method for sparse models.");
        validateDenseInputs(dataset, trainingLabels, this.points, this.testLabels);

        this.dataset = dataset;
        this.trainingLabels = trainingLabels;
        this.method = method;
        this.alpha = alpha;
        this.sparse = false;

        refreshNaiveBayes();
    }

    public void fit(double[][] dataset, double[] trainingLabels) {
        fit(dataset, trainingLabels, this.method, this.alpha);
    }

    public void fit(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels, String method, double alpha) {
        if (!sparse) throw new IllegalStateException("Use the dense fit method for dense models.");
        validateSparseInputs(sparseDataset, this.sparsePoints, trainingLabels, this.testLabels);

        this.sparseDataset = sparseDataset;
        this.trainingLabels = trainingLabels;
        this.method = method;
        this.alpha = alpha;
        this.sparse = true;

        refreshNaiveBayes();
    }

    public void fit(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels) {
        fit(sparseDataset, trainingLabels, this.method, this.alpha);
    }

    // ---------------- Predict Methods ----------------

    public int predict(int queryIndex) {
        return getMaxProbabilityClass(predictProbability(queryIndex));
    }

    public Map<Integer, Double> predictProbability(int queryIndex) {
        return sparse ? nb.computeProbabilities(sparsePoints.get(queryIndex))
                      : nb.computeProbabilities(points[queryIndex]);
    }

    public int[] predictAll() {
        int n = sparse ? sparsePoints.size() : points.length;
        int[] preds = new int[n];
        for (int i = 0; i < n; i++) preds[i] = predict(i);
        return preds;
    }

    public double accuracy() {
        if (testLabels == null) throw new IllegalStateException("No test labels available");

        int correct = 0;
        for (int i = 0; i < testLabels.length; i++) {
            if (predict(i) == (int) testLabels[i]) correct++;
        }
        return (double) correct / testLabels.length;
    }

    // ---------------- Setter Methods ----------------

    public void setMethod(String method) {
        if (method == null) throw new IllegalArgumentException("Method cannot be null.");
        this.method = method;
        refreshNaiveBayes();
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
        refreshNaiveBayes();
    }

    public void setTrainingLabels(double[] trainingLabels) {
        if (trainingLabels == null || trainingLabels.length == 0)
            throw new IllegalArgumentException("Training labels cannot be null or empty.");
        if (!sparse && trainingLabels.length != dataset.length)
            throw new IllegalArgumentException("Training labels length must match dataset length.");
        if (sparse && trainingLabels.length != sparseDataset.size())
            throw new IllegalArgumentException("Training labels length must match sparse dataset size.");

        this.trainingLabels = trainingLabels;
        refreshNaiveBayes();
    }

    public void setTestLabels(double[] testLabels) {
        if (testLabels == null || testLabels.length == 0)
            throw new IllegalArgumentException("Test labels cannot be null or empty.");
        if (!sparse && testLabels.length != points.length)
            throw new IllegalArgumentException("Test labels length must match points length.");
        if (sparse && testLabels.length != sparsePoints.size())
            throw new IllegalArgumentException("Test labels length must match sparse points size.");

        this.testLabels = testLabels;
    }

    public void setPoints(double[][] points, double[] testLabels) {
        this.points = points;
        this.testLabels = testLabels;
    }

    public void setSparsePoints(List<Map<Integer, Double>> sparsePoints) {
        this.sparsePoints = sparsePoints;
    }

    // ---------------- Helper Methods ----------------

    private int getMaxProbabilityClass(Map<Integer, Double> probabilities) {
        return probabilities.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElseThrow(() -> new IllegalArgumentException("No class found"));
    }

    private void validateDenseInputs(double[][] dataset, double[] trainingLabels, double[][] points, double[] testLabels) {
        if (dataset == null) throw new IllegalArgumentException("Dataset cannot be null.");
        if (points == null) throw new IllegalArgumentException("Points cannot be null.");
        if (trainingLabels == null) throw new IllegalArgumentException("Training labels cannot be null.");
        if (testLabels == null) throw new IllegalArgumentException("Test labels cannot be null.");
        if (dataset.length != trainingLabels.length)
            throw new IllegalArgumentException("Dataset length must match training labels length.");
        if (points.length != testLabels.length)
            throw new IllegalArgumentException("Points length must match test labels length.");
    }

    private void validateSparseInputs(List<Map<Integer, Double>> sparseDataset, List<Map<Integer, Double>> sparsePoints,
                                      double[] trainingLabels, double[] testLabels) {
        if (sparseDataset == null) throw new IllegalArgumentException("Sparse dataset cannot be null.");
        if (sparsePoints == null) throw new IllegalArgumentException("Sparse points cannot be null.");
        if (trainingLabels == null) throw new IllegalArgumentException("Training labels cannot be null.");
        if (testLabels == null) throw new IllegalArgumentException("Test labels cannot be null.");
    }

    public void refreshNaiveBayes() {
        if (sparse && sparseDataset != null) {
            nb = new AbstractNaiveBayes(sparseDataset, trainingLabels, method, alpha);
        } else if (!sparse && dataset != null) {
            nb = new AbstractNaiveBayes(dataset, trainingLabels, method, alpha);
        }
    }
}