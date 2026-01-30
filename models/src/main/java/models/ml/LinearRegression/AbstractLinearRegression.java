package models.ml.LinearRegression;

/**
 * Abstract class for Linear Regression computations.
 * Supports Normal Equation and Gradient Descent.
 * 
 * Convention:
 * - Last column of dataset is the target (y)
 * - All other columns are features (X)
 * 
 * @author Kotei Justice
 * @version 1.0
 */
public class AbstractLinearRegression {

    protected double[][] dataset;
    private String method;
    protected int numSamples;
    protected int numFeatures;

    protected double[] weights;

    public AbstractLinearRegression(double[][] dataset) {
        this(dataset, "normal", 1, 10);
    }

    public AbstractLinearRegression(double[][] dataset, String method) {
        this(dataset, method, 1, 10);
    }

    public AbstractLinearRegression(double[][] dataset, double learningRate, int epochs) {
        this(dataset, "normal", learningRate, epochs);
    }

    public AbstractLinearRegression(double[][] dataset, String method, double learningRate, int epochs) {
        this.dataset = dataset;
        this.method = method;
        this.numSamples = dataset.length;
        this.numFeatures = dataset[0].length - 1;

        String normalized = method.toLowerCase();
        switch (normalized) {
            case "normal":
            case "n":
                fitNormalEquation();
                break;
            case "gradientDescent":
            case "gd":
                fitGradientDescent(learningRate, epochs);
                break;
            default:
                System.out.println("Method not supported: " + method);
                break;
        }
    }

    /**
     * Train using the Normal Equation:
     * w = (X^T X)^-1 X^T y
     */
    public double[] fitNormalEquation() {
        double[][] X = buildDesignMatrix();
        double[] y = extractTarget();

        double[][] Xt = transpose(X);
        double[][] XtX = multiply(Xt, X);
        double[][] XtXInv = invert(XtX);
        double[] XtY = multiply(Xt, y);

        weights = multiply(XtXInv, XtY);
        return weights;
    }

    /**
     * Train using Gradient Descent.
     *
     * @param learningRate step size
     * @param epochs       number of iterations
     */
    public double[] fitGradientDescent(double learningRate, int epochs) {
        weights = new double[numFeatures + 1]; // bias included

        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] gradients = new double[weights.length];

            for (int i = 0; i < numSamples; i++) {
                double prediction = predictRow(dataset[i]);
                double error = prediction - dataset[i][numFeatures];

                gradients[0] += error; // bias gradient
                for (int j = 0; j < numFeatures; j++) {
                    gradients[j + 1] += error * dataset[i][j];
                }
            }

            for (int j = 0; j < weights.length; j++) {
                weights[j] -= learningRate * gradients[j] / numSamples;
            }
        }
        return weights;
    }

    /**
     * Predict a single row (training or test).
     */
    protected double predictRow(double[] row) {
        double y = weights[0]; // bias
        for (int j = 0; j < numFeatures; j++) {
            y += weights[j + 1] * row[j];
        }
        return y;
    }

    /* ===================== Helpers ===================== */

    protected double[][] buildDesignMatrix() {
        double[][] X = new double[numSamples][numFeatures + 1];
        for (int i = 0; i < numSamples; i++) {
            X[i][0] = 1.0; // bias
            for (int j = 0; j < numFeatures; j++) {
                X[i][j + 1] = dataset[i][j];
            }
        }
        return X;
    }

    protected double[] extractTarget() {
        double[] y = new double[numSamples];
        for (int i = 0; i < numSamples; i++) {
            y[i] = dataset[i][numFeatures];
        }
        return y;
    }

    /* ---------- Linear Algebra ---------- */

    protected static double[][] transpose(double[][] A) {
        double[][] T = new double[A[0].length][A.length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                T[j][i] = A[i][j];
        return T;
    }

    protected static double[][] multiply(double[][] A, double[][] B) {
        double[][] C = new double[A.length][B[0].length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < B[0].length; j++)
                for (int k = 0; k < B.length; k++)
                    C[i][j] += A[i][k] * B[k][j];
        return C;
    }

    protected static double[] multiply(double[][] A, double[] x) {
        double[] y = new double[A.length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < x.length; j++)
                y[i] += A[i][j] * x[j];
        return y;
    }

    protected static double[][] invert(double[][] A) {
        int n = A.length;
        double[][] I = new double[n][n];
        double[][] B = new double[n][n];

        for (int i = 0; i < n; i++) {
            I[i][i] = 1;
            System.arraycopy(A[i], 0, B[i], 0, n);
        }

        for (int i = 0; i < n; i++) {
            double pivot = B[i][i];
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

    public double[] getWeights() {
        return weights;
    }
}
