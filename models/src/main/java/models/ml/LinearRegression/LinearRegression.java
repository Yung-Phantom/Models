package models.ml.LinearRegression;

import java.util.*;

public class LinearRegression {

    public double[][] dataset;
    public double[][] points;

    public List<Map<Integer, Double>> sparseDataset;
    public List<Map<Integer, Double>> sparsePoints;

    public boolean sparse;

    public double[] trainingLabels;
    public double[] testLabels;

    public String method;

    public double learningRate;
    public int epochs;

    public AbstractLinearRegression lr;

    public LinearRegression() {
    }

    public LinearRegression(double[][] dataset, double[] trainingLabels, double[][] points, double[] testLabels,
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

        this.lr = new AbstractLinearRegression(this.dataset, this.trainingLabels, this.method, learningRate, epochs);
    }

    public LinearRegression(double[][] dataset, double[] trainingLabels, double[][] points, double[] testLabels) {
        this(dataset, trainingLabels, points, testLabels, "normal", 0.01, 1000);
    }
    
    public LinearRegression(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels,
            List<Map<Integer, Double>> sparsePoints, double[] testLabels, int numFeatures, String method,
            double learningRate, int epochs) {
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

        this.lr = new AbstractLinearRegression(sparseDataset, trainingLabels, numFeatures, method, learningRate,
                epochs);
    }
    public LinearRegression(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels,
            List<Map<Integer, Double>> sparsePoints, double[] testLabels, int numFeatures) {
        this(sparseDataset, trainingLabels, sparsePoints, testLabels, numFeatures, "gradientdescent", 0.01, 1000);
    }
    

    public void fit(double[][] dataset, double[] trainingLabels, String method) {
        if (sparse)
            throw new IllegalStateException("Use fitSparse for sparse models.");
        this.dataset = dataset;
        this.trainingLabels = trainingLabels;
        this.method = method;
        this.lr = new AbstractLinearRegression(this.dataset, this.trainingLabels, this.method, learningRate, epochs);
    }

    public void fit(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels, int numFeatures, String method) {
        if (!sparse)
            throw new IllegalStateException("Use fit for dense models.");
        this.sparseDataset = sparseDataset;
        this.trainingLabels = trainingLabels;
        this.method = method;
        for (Map<Integer, Double> row : sparseDataset) {
            for (Integer featureIndex : row.keySet()) {
                if (featureIndex + 1 > numFeatures) {
                    numFeatures = featureIndex + 1;
                }
            }
        }
        this.lr = new AbstractLinearRegression(this.sparseDataset, this.trainingLabels, numFeatures, this.method,
                learningRate, epochs);
    }

    public double predict(double[] row) {
        if (sparse)
            throw new IllegalStateException("Use predictSparse for sparse models.");
        return lr.predictRow(row);
    }

    public double[] predictAll() {
        if (points == null)
            throw new IllegalStateException("No query points provided.");
        return lr.predict(points);
    }

    public double predictSparseRow(Map<Integer, Double> x) {
        if (!sparse)
            throw new IllegalStateException("Use predict for dense models.");
        return lr.predictSparseRow(x);
    }

    public double[] predictAllSparse() {
        if (!sparse || sparsePoints == null)
            throw new IllegalStateException("No sparse query points provided.");
        return lr.predictSparse(sparsePoints);
    }

    /**
     * Mean Squared Error (MSE)
     * 
     * @return MSE
     */
    public double mse() {
        if (testLabels == null)
            throw new IllegalStateException("No test labels provided.");

        double sum = 0.0;

        if (sparse) {
            double[] preds = predictAllSparse();
            for (int i = 0; i < preds.length; i++) {
                double err = preds[i] - testLabels[i];
                sum += err * err;
            }
        } else {
            double[] preds = predictAll();
            for (int i = 0; i < preds.length; i++) {
                double err = preds[i] - testLabels[i];
                sum += err * err;
            }
        }
        return sum / testLabels.length;
    }

    /**
     * R² score
     * 
     * @return R² score
     */
    public double r2() {
        if (testLabels == null)
            throw new IllegalStateException("No test labels provided.");

        double mean = 0.0;
        for (double y : testLabels)
            mean += y;
        mean /= testLabels.length;

        double ssTot = 0.0, ssRes = 0.0;

        double[] preds = sparse ? predictAllSparse() : predictAll();

        for (int i = 0; i < preds.length; i++) {
            double y = testLabels[i];
            ssTot += (y - mean) * (y - mean);
            ssRes += (y - preds[i]) * (y - preds[i]);
        }

        return 1.0 - ssRes / ssTot;
    }

    public double[] getWeights() {
        return lr.getWeights();
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
        this.lr = new AbstractLinearRegression(this.dataset, this.trainingLabels, this.method, this.learningRate,
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
        this.lr = new AbstractLinearRegression(this.dataset, this.trainingLabels, this.method, this.learningRate,
                this.epochs);
    }

    public void setLearningRate(double learningRate) {
        if (learningRate <= 0 || learningRate > 1.0) {
            throw new IllegalArgumentException("Learning rate must be positive.");
        }
        this.learningRate = learningRate;
        this.lr = new AbstractLinearRegression(this.dataset, this.trainingLabels, this.method, this.learningRate,
                this.epochs);
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
        this.lr = new AbstractLinearRegression(this.dataset, this.trainingLabels, this.method, this.learningRate,
                this.epochs);
    }

    public void refreshLinearRegression(int numFeatures) {
        if (!sparse && dataset != null) {
            this.lr = new AbstractLinearRegression(this.dataset, this.trainingLabels, this.method, this.learningRate,
                    this.epochs);
        } else if (sparse && sparseDataset != null) {
            this.lr = new AbstractLinearRegression(this.sparseDataset, this.trainingLabels, numFeatures, this.method,
                    this.learningRate, this.epochs);
        }
    }
    
}
