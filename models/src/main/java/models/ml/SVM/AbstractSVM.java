package models.ml.SVM;

import java.util.Map;

/**
 * Abstract class for Support Vector Machine (SVM).
 * Implements linear SVM using hinge loss and gradient descent.
 * Kernel functions included for extension.
 * 
 * @author Justice
 * @version 1.1
 */
public class AbstractSVM {

    protected double[][] dataset;
    protected int numSamples;
    protected int numFeatures;

    protected Map<Integer, Double>[] sparseDataset;
    protected double[] sparseLabels;

    protected double[] labels;
    protected boolean sparse;

    protected double[] weights;
    protected double bias;
    protected double C;
    protected double learningRate;
    protected int epochs;
    protected String kernel;
    protected String method;

    protected double[] alpha;

    public AbstractSVM(double[][] dataset, double[] labels) {
        this(dataset,labels , 0.1, 0.01, 10, "linearKernel", "linearsvc");
    }

    public AbstractSVM(double[][] dataset,
            double[] labels, double C, double learningRate, int epochs,
            String kernel, String method) {
        if (dataset == null || dataset.length == 0)
            throw new IllegalArgumentException("Dataset cannot be null or empty.");

        if (labels == null || labels.length == 0)
            throw new IllegalArgumentException("Labels cannot be null or empty.");

        if (dataset.length != labels.length)
            throw new IllegalArgumentException("Dataset length must match labels length.");

        if (dataset[0] == null || dataset[0].length == 0)
            throw new IllegalArgumentException("Dataset must contain at least one feature.");

        if (C <= 0)
            throw new IllegalArgumentException("C must be positive.");

        if (learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive.");

        if (epochs <= 0)
            throw new IllegalArgumentException("Epochs must be positive.");

        if (kernel == null)
            throw new IllegalArgumentException("Kernel cannot be null.");

        if (method == null)
            throw new IllegalArgumentException("Method cannot be null.");
        this.dataset = dataset;
        this.labels = labels;
        this.numSamples = dataset.length;
        this.numFeatures = dataset[0].length;

        this.C = C;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.kernel = kernel.toLowerCase();
        this.method = method.toLowerCase();

        this.weights = new double[numFeatures];
        this.bias = 0.0;
        this.alpha = new double[numSamples];
        switch (this.method) {
            case "svc":
                computeSVC();
                break;
            case "linearsvc":
                computeLinearSVC();
                break;
            default:
                System.out.println("Method not supported: " + method);
        }
    }

    public AbstractSVM(Map<Integer, Double>[] sparseDataset, double[] labels, int numFeatures, double C,
            double learningRate, int epochs, String method) {
                if (sparseDataset == null || sparseDataset.length == 0)
            throw new IllegalArgumentException("Sparse dataset cannot be null or empty.");

        if (labels == null || labels.length == 0)
            throw new IllegalArgumentException("Labels cannot be null or empty.");

        if (sparseDataset.length != labels.length)
            throw new IllegalArgumentException("Sparse dataset length must match labels length.");

        if (numFeatures <= 0)
            throw new IllegalArgumentException("Number of features must be positive.");

        if (C <= 0)
            throw new IllegalArgumentException("C must be positive.");

        if (learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive.");

        if (epochs <= 0)
            throw new IllegalArgumentException("Epochs must be positive.");

        if (method == null)
            throw new IllegalArgumentException("Method cannot be null.");

        this.sparseDataset = sparseDataset;
        this.sparseLabels = labels;
        this.numSamples = labels.length;
        this.numFeatures = numFeatures;

        this.C = C;
        this.learningRate = learningRate;
        this.epochs = epochs;
        // this.kernel = kernel.toLowerCase();
        this.method = method.toLowerCase();

        this.weights = new double[numFeatures];
        this.bias = 0.0;
        this.alpha = new double[numSamples];
        computeLinearSVCSparse();
    }

    public void computeLinearSVC() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < numSamples; i++) {

                double[] x = getX(i);
                double y = labels[i];

                double fx = linearDecision(weights, x, bias);

                if (y * fx < 1) {
                    for (int j = 0; j < weights.length; j++) {
                        weights[j] -= learningRate * (weights[j] - C * y * x[j]);
                    }
                    bias += learningRate * C * y;
                } else {
                    for (int j = 0; j < weights.length; j++) {
                        weights[j] -= learningRate * weights[j];
                    }
                }
            }
        }
    }

    public void computeLinearSVCSparse() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < numSamples; i++) {
                Map<Integer, Double> x = sparseDataset[i];
                double y = sparseLabels[i];
                double fx = linearDecisionSparse(weights, x, bias);

                if (y * fx < 1) {
                    for (Map.Entry<Integer, Double> entry : x.entrySet()) {
                        int idx = entry.getKey();
                        double val = entry.getValue();
                        weights[idx] -= learningRate * (weights[idx] - C * y * val);
                    }
                    bias += learningRate * C * y;
                } else {
                    for (int j = 0; j < weights.length; j++)
                        weights[j] -= learningRate * weights[j];
                }
            }
        }
    }

    public void computeSVC() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < numSamples; i++) {

                double yi = dataset[i][numFeatures];
                double Ei = decisionDual(i) - yi;

                if ((yi * Ei < -1e-3 && alpha[i] < C) ||
                        (yi * Ei > 1e-3 && alpha[i] > 0)) {

                    int j = (i + 1) % numSamples;

                    double yj = dataset[j][numFeatures];
                    double Ej = decisionDual(j) - yj;

                    // double aiOld = alpha[i];
                    double ajOld = alpha[j];

                    double kii = kernel(getX(i), getX(i));
                    double kjj = kernel(getX(j), getX(j));
                    double kij = kernel(getX(i), getX(j));

                    double eta = kii + kjj - 2 * kij;
                    if (eta <= 0)
                        continue;

                    alpha[j] += yj * (Ei - Ej) / eta;
                    alpha[j] = Math.max(0, Math.min(C, alpha[j]));
                    alpha[i] += yi * yj * (ajOld - alpha[j]);
                }
            }
        }

        if (kernel.equals("linearkernel")) {
            for (int i = 0; i < numSamples; i++) {
                double y = dataset[i][numFeatures];
                double[] x = getX(i);
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += alpha[i] * y * x[j];
                }
            }
        }
    }

    protected double[] getX(int i) {
        double[] x = new double[numFeatures];
        System.arraycopy(dataset[i], 0, x, 0, numFeatures);
        return x;
    }

    protected double decisionDual(int i) {
        double sum = 0.0;
        double[] xi = getX(i);
        for (int j = 0; j < numSamples; j++) {
            if (alpha[j] == 0)
                continue;
            double yj = dataset[j][numFeatures];
            sum += alpha[j] * yj * kernel(getX(j), xi);
        }
        return sum + bias;
    }

    protected double linearDecision(double[] w, double[] x, double b) {
        double sum = 0.0;
        for (int i = 0; i < w.length; i++) {
            sum += w[i] * x[i];
        }
        return sum + b;
    }

    protected double linearDecisionSparse(double[] w, Map<Integer, Double> x, double b) {
        double sum = 0.0;
        for (Map.Entry<Integer, Double> entry : x.entrySet())
            sum += w[entry.getKey()] * entry.getValue();
        return sum + b;
    }
    protected double kernel(double[] x, double z[]) {
        switch (kernel) {
            case "polynomial":
                return polynomialKernel(x, z, 1.0, 3);
            case "rbf":
                return rbfKernel(x, z, 0.5);
            default:
                return linearKernel(x, z);
        }
    }

    public double linearKernel(double[] x, double[] z) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * z[i];
        return sum;
    }

    public double polynomialKernel(double[] x, double[] z, double c, int d) {
        return Math.pow(linearKernel(x, z) + c, d);
    }

    public double rbfKernel(double[] x, double[] z, double gamma) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            double diff = x[i] - z[i];
            sum += diff * diff;
        }
        return Math.exp(-gamma * sum);
    }

    public double predictScore(double[] row) {
        return linearDecision(weights, row, bias);
    }
    
    public int predictSparse(Map<Integer, Double> row) {
        if (row == null)
            throw new IllegalArgumentException("Sparse row cannot be null.");

        if (row.isEmpty())
            throw new IllegalArgumentException("Sparse row cannot be empty.");
        return linearDecisionSparse(weights, row, bias) >= 0 ? 1 : -1;
    }

    public int predict(double[] row) {
        if (row == null)
            throw new IllegalArgumentException("Row cannot be null.");

        if (row.length != numFeatures)
            throw new IllegalArgumentException("Row must have exactly " + numFeatures + " features.");
        return predictScore(row) >= 0 ? 1 : -1;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }
}
