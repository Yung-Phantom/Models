package models.ml.Model.LinearRegression;

import java.util.*;

public class AbstractLinearRegression {
    public double[][] dataset;
    public List<Map<Integer, Double>> sparseDataset;
    public double[] trainingLabels;

    public boolean sparse;
    public String method;

    public int numSamples;
    public int numFeatures;

    public double learningRate;
    public int epochs;
    /**
     * L2 regularization parameter (ridge penalty). Must be non-negative and finite.
     * Default is 0.0 (no regularization). The bias/intercept term is never penalized.
     * For normal equation: adds lambda to diagonal of X^T X (excluding bias).
     * For gradient descent: adds lambda * w_j / n to gradient of each non-bias weight.
     */
    public double lambda;

    public double[] weights;

    public AbstractLinearRegression(double[][] dataset, double[] trainingLabels, String method, double learningRate,
            int epochs) {
        this(dataset, trainingLabels, method, learningRate, epochs, 0.0);
    }

    public AbstractLinearRegression(double[][] dataset, double[] trainingLabels, String method, double learningRate,
            int epochs, double lambda) {

        this.sparse = false;
        if (dataset == null || dataset.length == 0)
            throw new IllegalArgumentException("Empty dataset");

        if (method == null)
            throw new IllegalArgumentException("Method cannot be null");

        this.dataset = dataset;

        this.numFeatures = dataset[0].length;
        for (double[] row : dataset) {
            if (row == null)
                throw new IllegalArgumentException("Dense dataset contains null row");
            if (row.length != numFeatures)
                throw new IllegalArgumentException("Inconsistent row length");
            for (double val : row) {
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException("Dataset contains NaN or Infinity");
                }
            }
        }

        this.numSamples = dataset.length;

        this.method = method.trim().toLowerCase();
        if (!Set.of("n", "normal", "gd", "gradientdescent").contains(this.method))
            throw new IllegalArgumentException("Unsupported method: " + method);

        this.trainingLabels = trainingLabels;
        if (trainingLabels.length != numSamples)
            throw new IllegalArgumentException("Labels size must match number of samples.");

        if (learningRate <= 0 || learningRate > 1) {
            throw new IllegalArgumentException("Learning rate must be between 0 and 1");
        }
        if (epochs <= 0) {
            throw new IllegalArgumentException("Epochs must be a positive integer");
        }
        if (lambda < 0 || Double.isNaN(lambda) || Double.isInfinite(lambda)) {
            throw new IllegalArgumentException("Lambda must be a non-negative finite value");
        }
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.lambda = lambda;

        fit();
    }

    public AbstractLinearRegression(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels,
            String method, double learningRate, int epochs) {
        this(sparseDataset, trainingLabels, method, learningRate, epochs, 0.0);
    }

    public AbstractLinearRegression(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels,
            String method, double learningRate, int epochs, double lambda) {
        this.sparse = true;

        if (method == null)
            throw new IllegalArgumentException("Method cannot be null");

        if (sparseDataset == null || sparseDataset.isEmpty())
            throw new IllegalArgumentException("Empty sparse dataset");

        for (Map<Integer, Double> row : sparseDataset) {
            if (row == null)
                throw new IllegalArgumentException("Sparse dataset contains null row");
            for (Map.Entry<Integer, Double> e : row.entrySet()) {
                int key = e.getKey();
                double val = e.getValue();
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException("Sparse dataset contains NaN or Infinity");
                }
            }
        }

        this.sparseDataset = sparseDataset;
        this.trainingLabels = trainingLabels;
        this.numSamples = sparseDataset.size();

        // Infer numFeatures from max feature index + 1
        this.numFeatures = sparseDataset.stream()
                .flatMap(m -> m.keySet().stream())
                .max(Integer::compare)
                .orElseThrow(() -> new IllegalArgumentException("Sparse dataset must contain features"))
                + 1;

        this.method = method.trim().toLowerCase();
        if (!Set.of("gd", "gradientdescent").contains(this.method))
            throw new IllegalArgumentException("Unsupported method for sparse data: " + method + ". Only 'gd' or 'gradientdescent' supported for sparse.");

        if (learningRate <= 0 || learningRate > 1) {
            throw new IllegalArgumentException("Learning rate must be between 0 and 1");
        }
        if (epochs <= 0) {
            throw new IllegalArgumentException("Epochs must be a positive integer");
        }
        if (lambda < 0 || Double.isNaN(lambda) || Double.isInfinite(lambda)) {
            throw new IllegalArgumentException("Lambda must be a non-negative finite value");
        }
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.lambda = lambda;

        if (trainingLabels.length != numSamples)
            throw new IllegalArgumentException("Labels size must match number of samples.");

        fit();
    }

    public void fit() {
        if (this.sparse) {
            fitGradientDescentSparse();
            validateWeights();
        } else {
            switch (method) {
                case "normal":
                case "n":
                    fitNormalEquation();
                    validateWeights();
                    break;
                case "gradientdescent":
                case "gd":
                    fitGradientDescent(learningRate, epochs);
                    validateWeights();
                    break;
                default:
                    System.out.println("Method not supported: " + method);
                    break;
            }
        }

    }

    public double[] fitNormalEquation() {
        double[][] designMatrix = buildDesignMatrix();

        double[][] Xt = transpose(designMatrix);
        double[][] XtX = multiply(Xt, designMatrix);

        // Add L2 regularization: X^T X + lambda * I (excluding bias term)
        if (lambda > 0) {
            for (int i = 1; i < XtX.length; i++) {
                XtX[i][i] += lambda;
            }
        }

        double[][] XtXInv;
        try {
            XtXInv = invert(XtX);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("Normal equation failed: X^T X + lambda*I is singular (collinear features). " +
                    "Increase lambda, remove collinear features, or use gradient descent.", e);
        }
        double[] XtY = multiply(Xt, trainingLabels);

        weights = multiply(XtXInv, XtY);
        return weights;
    }

    public double[] fitGradientDescent(double learningRate, int epochs) {
        weights = new double[numFeatures + 1];

        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] gradients = new double[weights.length];

            for (int i = 0; i < numSamples; i++) {
                double prediction = predictRow(dataset[i]);
                double error = prediction - trainingLabels[i];

                gradients[0] += error;
                for (int j = 0; j < numFeatures; j++) {
                    gradients[j + 1] += error * dataset[i][j];
                }
            }

            // Apply L2 regularization: add lambda * w_j to gradient for non-bias weights
            // Gradient is normalized by numSamples, so penalty is lambda * w_j / numSamples
            for (int j = 0; j < weights.length; j++) {
                double penalty = (j == 0) ? 0.0 : lambda * weights[j] / numSamples;
                weights[j] -= learningRate * (gradients[j] / numSamples + penalty);
            }
        }
        return weights;
    }

    private void fitGradientDescentSparse() {
        weights = new double[numFeatures + 1];

        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] gradients = new double[weights.length];

            for (int i = 0; i < numSamples; i++) {
                Map<Integer, Double> xi = sparseDataset.get(i);
                double prediction = predictSparseRow(xi);
                double error = prediction - trainingLabels[i];

                gradients[0] += error;

                for (var feat : xi.entrySet()) {
                    gradients[feat.getKey() + 1] += error * feat.getValue();
                }
            }

            // Apply L2 regularization: add lambda * w_j to gradient for non-bias weights
            // Gradient is normalized by numSamples, so penalty is lambda * w_j / numSamples
            for (int j = 0; j < weights.length; j++) {
                double penalty = (j == 0) ? 0.0 : lambda * weights[j] / numSamples;
                weights[j] -= learningRate * (gradients[j] / numSamples + penalty);
            }
        }
    }

    public double predictRow(double[] row) {
        if (sparse)
            throw new IllegalStateException("Dense prediction called in sparse mode");
        if (row == null) {
            throw new IllegalArgumentException("Row cannot be null");
        }
        if (row.length == 0) {
            throw new IllegalArgumentException("Row cannot be empty");
        }
        if (row.length < numFeatures) {
            throw new IllegalArgumentException("Row must have at least " + numFeatures + " features");
        }
        for (double val : row) {
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException("Row contains NaN or Infinity");
            }
        }

        double y = weights[0];
        for (int j = 0; j < numFeatures; j++) {
            if (j < row.length) {
                y += weights[j + 1] * row[j];
            } else {
                y += weights[j + 1] * 0;
            }
        }
        return y;
    }

    public double predictSparseRow(Map<Integer, Double> x) {
        if (!sparse)
            throw new IllegalStateException("Sparse prediction called in dense mode");
        if (x == null)
            throw new IllegalArgumentException("Row cannot be null");
        // Empty row is valid - predicts from bias alone
        for (Map.Entry<Integer, Double> e : x.entrySet()) {
            int key = e.getKey();
            double val = e.getValue();
            if (key < 0) {
                throw new IllegalArgumentException("Sparse vector contains negative feature index: " + key);
            }
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException("Sparse vector contains NaN or Infinity");
            }
        }

        double y = weights[0];
        for (var e : x.entrySet()) {
            int idx = e.getKey();
            if (idx >= 0 && idx < numFeatures) {
                y += weights[idx + 1] * e.getValue();
            }
            // Ignore keys >= numFeatures (unseen features contribute nothing)
        }
        return y;
    }

    public double[] predict(double[][] query) {
        double[] predictions = new double[query.length];
        for (int i = 0; i < query.length; i++) {
            predictions[i] = predictRow(query[i]);
        }
        return predictions;
    }

    public double[] predictSparse(List<Map<Integer, Double>> query) {
        double[] preds = new double[query.size()];
        for (int i = 0; i < query.size(); i++) {
            preds[i] = predictSparseRow(query.get(i));
        }
        return preds;
    }

    public double[][] buildDesignMatrix() {
        double[][] designMatrix = new double[numSamples][numFeatures + 1];
        for (int i = 0; i < numSamples; i++) {
            designMatrix[i][0] = 1.0;
            for (int j = 0; j < numFeatures; j++) {
                designMatrix[i][j + 1] = dataset[i][j];
            }
        }
        return designMatrix;
    }

    public static double[][] transpose(double[][] A) {
        double[][] T = new double[A[0].length][A.length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++) {
                T[j][i] = A[i][j];
            }
        return T;
    }

    public static double[][] multiply(double[][] A, double[][] B) {
        double[][] C = new double[A.length][B[0].length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < B[0].length; j++)
                for (int k = 0; k < B.length; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
        return C;
    }

    public static double[] multiply(double[][] A, double[] x) {
        double[] y = new double[A.length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < x.length; j++) {
                y[i] += A[i][j] * x[j];
            }
        return y;
    }

    public static double[][] invert(double[][] A) {
        int n = A.length;
        double[][] I = new double[n][n];
        double[][] B = new double[n][n];

        for (int i = 0; i < n; i++) {
            I[i][i] = 1;
            System.arraycopy(A[i], 0, B[i], 0, n);
        }

        for (int i = 0; i < n; i++) {
            double pivot = B[i][i];
            if (Math.abs(pivot) < 1e-12) {
                throw new ArithmeticException("Matrix is singular");
            }

            for (int j = 0; j < n; j++) {
                B[i][j] /= pivot;
                I[i][j] /= pivot;
            }
            for (int k = 0; k < n; k++) {
                if (k == i)
                    continue;
                double factor = B[k][i];
                for (int j = 0; j < n; j++) {
                    B[k][j] -= factor * B[i][j];
                    I[k][j] -= factor * I[i][j];
                }
            }
        }
        return I;
    }

    private void validateWeights() {
        for (double w : weights) {
            if (Double.isNaN(w) || Double.isInfinite(w)) {
                throw new IllegalStateException("Model weights contain NaN or Infinity after training");
            }
        }
    }

    public double[] getWeights() {
        return weights;
    }

    public double getLambda() {
        return lambda;
    }
}