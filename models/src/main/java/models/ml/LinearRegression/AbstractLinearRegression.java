package models.ml.LinearRegression;

/**
 * Abstract class for Linear Regression.
 * Implements linear regression using normal equation and gradient descent.
 *
 * @author Justice
 * @version 1.1
 */
public class AbstractLinearRegression {

    /**
     * Dataset for linear regression.
     */
    public double[][] dataset;

    /**
     * Method for linear regression.
     * Supports "normal" and "gradientDescent".
     */
    public String method;

    /**
     * Number of samples in the dataset.
     */
    public int numSamples;

    /**
     * Number of features in the dataset.
     */
    public int numFeatures;

    /**
     * Weights for linear regression.
     */
    public double[] weights;

    /**
     * Constructor.
     * @param dataset dataset for linear regression
     */
    public AbstractLinearRegression(double[][] dataset) {
        this(dataset, "normal", 1, 10);
    }

    /**
     * Constructor.
     * @param dataset dataset for linear regression
     * @param method method for linear regression
     */
    public AbstractLinearRegression(double[][] dataset, String method) {
        this(dataset, method, 1, 10);
    }

    /**
     * Constructor.
     * @param dataset dataset for linear regression
     * @param learningRate learning rate for gradient descent
     * @param epochs number of epochs for gradient descent
     */
    public AbstractLinearRegression(double[][] dataset, double learningRate, int epochs) {
        this(dataset, "normal", learningRate, epochs);
    }

    /**
     * Constructor.
     * @param dataset dataset for linear regression
     * @param method method for linear regression
     * @param learningRate learning rate for gradient descent
     * @param epochs number of epochs for gradient descent
     */
    public AbstractLinearRegression(double[][] dataset, String method, double learningRate, int epochs) {
        this.dataset = dataset;
        this.method = method;
        this.numSamples = dataset.length;
        this.numFeatures = dataset[0].length - 1;

        switch (method.toLowerCase()) {
            case "normal":
            case "n":
                fitNormalEquation();
                break;
            case "gradientdescent":
            case "gd":
                fitGradientDescent(learningRate, epochs);
                break;
            default:
                System.out.println("Method not supported: " + method);
                break;
        }
    }

    /**
     * Fits linear regression model using normal equation.
     * @return weights for linear regression
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
     * Fits linear regression model using gradient descent.
     * @param learningRate learning rate for gradient descent
     * @param epochs number of epochs for gradient descent
     * @return weights for linear regression
     */
    public double[] fitGradientDescent(double learningRate, int epochs) {
        weights = new double[numFeatures + 1];

        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] gradients = new double[weights.length];

            for (int i = 0; i < numSamples; i++) {
                double prediction = predictRow(dataset[i]);
                double error = prediction - dataset[i][numFeatures];

                gradients[0] += error;
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
     * Predicts label for a single query point.
     * @param row query point
     * @return label for query point
     */
    public double predictRow(double[] row) {
        double y = weights[0];
        for (int j = 0; j < numFeatures; j++) {
            y += weights[j + 1] * row[j];
        }
        return y;
    }


    /**
     * Predicts labels for a set of query points.
     * @param X set of query points
     * @return labels for query points
     */
    public double[] predict(double[][] X) {
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictRow(X[i]);
        }
        return predictions;
    }
    
    /**
     * Builds design matrix for linear regression.
     * @return design matrix
     */
    public double[][] buildDesignMatrix() {
        double[][] X = new double[numSamples][numFeatures + 1];
        for (int i = 0; i < numSamples; i++) {
            X[i][0] = 1.0;
            for (int j = 0; j < numFeatures; j++) {
                X[i][j + 1] = dataset[i][j];
            }
        }
        return X;
    }

    /**
     * Extracts target from dataset.
     * @return target
     */
    public double[] extractTarget() {
        double[] y = new double[numSamples];
        for (int i = 0; i < numSamples; i++) {
            y[i] = dataset[i][numFeatures];
        }
        return y;
    }

    /**
     * Transposes matrix.
     * @param A matrix
     * @return transposed matrix
     */
    public static double[][] transpose(double[][] A) {
        double[][] T = new double[A[0].length][A.length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++) {
                T[j][i] = A[i][j];
            }
        return T;
    }

    /**
     * Multiplies matrix.
     * @param A first matrix
     * @param B second matrix
     * @return product matrix
     */
    public static double[][] multiply(double[][] A, double[][] B) {
        double[][] C = new double[A.length][B[0].length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < B[0].length; j++)
                for (int k = 0; k < B.length; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
        return C;
    }

    /**
     * Multiplies matrix and vector.
     * @param A matrix
     * @param x vector
     * @return product vector
     */
    public static double[] multiply(double[][] A, double[] x) {
        double[] y = new double[A.length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < x.length; j++) {
                y[i] += A[i][j] * x[j];
            }
        return y;
    }

    /**
     * Inverts matrix.
     * @param A matrix
     * @return inverted matrix
     */
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

    /**
     * Gets weights for linear regression.
     * @return weights
     */
    public double[] getWeights() {
        return weights;
    }
}