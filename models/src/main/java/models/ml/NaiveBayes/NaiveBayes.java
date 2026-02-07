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

    // Dense constructor
    public NaiveBayes(double[][] dataset, double[][] points, String method, double alpha) {
        this.dataset = dataset;
        this.points = points;
        this.sparse = false;
        this.method = method != null ? method : "gaussian";
        this.alpha = alpha;
        this.trainingLabels = extractLabels(dataset);
        this.testLabels = extractLabels(points);
        this.nb = new AbstractNaiveBayes(dataset, trainingLabels, method, alpha);
    }

    // Sparse constructor
    public NaiveBayes(List<Map<Integer, Double>> sparseDataset, List<Map<Integer, Double>> sparsePoints,
            double[] trainingLabels, double[] testLabels, String method, double alpha) {
        this.sparseDataset = sparseDataset;
        this.sparsePoints = sparsePoints;
        this.trainingLabels = trainingLabels;
        this.testLabels = testLabels;
        this.sparse = true;
        this.method = method != null ? method : "gaussian";
        this.alpha = alpha;
        this.nb = new AbstractNaiveBayes(sparseDataset, trainingLabels, method, alpha);
    }

    // Convenience constructor for dense dataset with default Gaussian method and
    // alpha=1.0
    public NaiveBayes(double[][] dataset, double[][] points) {
        this(dataset, points, "gaussian", 1.0);
    }

    // Convenience constructor for sparse dataset with default Gaussian method and
    // alpha=1.0
    public NaiveBayes(List<Map<Integer, Double>> sparseDataset, List<Map<Integer, Double>> sparsePoints) {
        this(sparseDataset, sparsePoints, null, null, "gaussian", 1.0);
    }

    // Method to fit the model on the training data (Dense)
    public void fit(double[][] dataset, double[] labels, String method, double alpha) {
        this.dataset = dataset;
        this.trainingLabels = labels != null ? labels : extractLabels(dataset);
        this.nb = new AbstractNaiveBayes(dataset, trainingLabels, method, alpha);
        this.sparse = false;
    }

    public void fit(double[][] dataset, double[] labels) {
        fit(dataset, labels, method, alpha);
    }

    // Method to fit the model on the training data (Sparse)
    public void fit(List<Map<Integer, Double>> sparseDataset, double[] labels, String method, double alpha) {
        this.sparseDataset = sparseDataset;
        this.trainingLabels = labels;
        this.nb = new AbstractNaiveBayes(sparseDataset, trainingLabels, method, alpha);
        this.sparse = true;
    }

    public void fit(List<Map<Integer, Double>> sparseDataset, double[] labels) {
        fit(sparseDataset, labels, method, alpha);
    }

    private int getMaxProbabilityClass(Map<Integer, Double> probabilities) {
        return probabilities.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElseThrow(() -> new IllegalArgumentException("No class found"));
    }

    public int predict(int queryIndex) {
        Map<Integer, Double> probabilities = predictProbability(queryIndex);
        return getMaxProbabilityClass(probabilities);
    }

    public Map<Integer, Double> predictProbability(int queryIndex) {
        if (sparse) {
            return nb.computeProbabilities(sparsePoints.get(queryIndex));
        } else {
            return nb.computeProbabilities(points[queryIndex]);
        }
    }

    public int[] predictAll() {
        int[] preds = new int[points.length];
        for (int i = 0; i < points.length; i++)
            preds[i] = predict(i);
        return preds;
    }

    public double accuracy() {
        int correct = 0;
        for (int i = 0; i < points.length; i++) {
            if (predict(i) == (int) points[i][points[i].length - 1])
                correct++;
        }
        return (double) correct / points.length;
    }

    private double[] extractLabels(double[][] dataset) {
        double[] labels = new double[dataset.length];
        for (int i = 0; i < dataset.length; i++) {
            labels[i] = dataset[i][dataset[i].length - 1];
        }
        return labels;
    }

    private double[] extractLabels(List<Map<Integer, Double>> sparseDataset) {
        double[] labels = new double[sparseDataset.size()];
        for (int i = 0; i < sparseDataset.size(); i++) {
            labels[i] = (double) sparseDataset.get(i).get(sparseDataset.get(i).size() - 1);
        }
        return labels;
    }
    
    public void setMethod(String method) {
        this.method = method;
        if (sparse) {
            this.nb = new AbstractNaiveBayes(sparseDataset, trainingLabels, method, alpha);
        } else {
            this.nb = new AbstractNaiveBayes(dataset, trainingLabels, method, alpha);
        }
    }
    
    public void setAlpha(double alpha) {
        this.alpha = alpha;
        if (sparse) {
            this.nb = new AbstractNaiveBayes(sparseDataset, trainingLabels, method, alpha);
        } else {
            this.nb = new AbstractNaiveBayes(dataset, trainingLabels, method, alpha);
        }
    }
    
    public void setTrainingLabels(double[] trainingLabels) {
        this.trainingLabels = trainingLabels;
        if (sparse) {
            this.nb = new AbstractNaiveBayes(sparseDataset, trainingLabels, method, alpha);
        } else {
            this.nb = new AbstractNaiveBayes(dataset, trainingLabels, method, alpha);
        }
    }
    
    public void setTestLabels(double[] testLabels) {
        this.testLabels = testLabels;
    }
    
    public void setDataset(double[][] dataset) {
        this.dataset = dataset;
        this.trainingLabels = extractLabels(dataset);
        if (sparse) {
            this.nb = new AbstractNaiveBayes(sparseDataset, trainingLabels, method, alpha);
        } else {
            this.nb = new AbstractNaiveBayes(dataset, trainingLabels, method, alpha);
        }
    }
    
    public void setSparseDataset(List<Map<Integer, Double>> sparseDataset) {
        this.sparseDataset = sparseDataset;
        this.trainingLabels = extractLabels(sparseDataset);
        this.nb = new AbstractNaiveBayes(sparseDataset, trainingLabels, method, alpha);
    }
    public void setPoints(double[][] points) {
        this.points = points;
        this.testLabels = extractLabels(points);
    }
    
    public void setSparsePoints(List<Map<Integer, Double>> sparsePoints) {
        this.sparsePoints = sparsePoints;
    }
}
