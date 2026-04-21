package models.ml.Model.SVM;

import java.util.ArrayList;
import java.util.List;
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

    protected List<Map<Integer, Double>> sparseDataset;

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

    private boolean normalizedLabels = false;
    private boolean originalZeroOne = false;

    protected double gamma;
    protected int degree;
    protected double coef0;

    public AbstractSVM(double[][] dataset,
            double[] labels, double C, double learningRate, int epochs,
            String kernel, String method, double gamma, int degree, double coef0) {
        this.sparse = false;
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
        if (gamma <= 0)
            throw new IllegalArgumentException("Gamma must be positive.");
        if (degree <= 0)
            throw new IllegalArgumentException("Degree must be positive.");
        if (coef0 < 0)
            throw new IllegalArgumentException("Coef0 must be non-negative.");
        this.dataset = dataset;
        this.labels = normalizeLabels(labels);
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

        this.gamma = gamma;
        this.degree = degree;
        this.coef0 = coef0;
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

    public AbstractSVM(List<Map<Integer, Double>> sparseDataset, double[] labels, double C,
            double learningRate, int epochs, String kernel, String method, double gamma, int degree, double coef0) {
        this.sparse = true;
        if (sparseDataset == null || sparseDataset.size() == 0)
            throw new IllegalArgumentException("Sparse dataset cannot be null or empty.");

        if (labels == null || labels.length == 0)
            throw new IllegalArgumentException("Labels cannot be null or empty.");

        if (sparseDataset.size() != labels.length)
            throw new IllegalArgumentException("Sparse dataset length must match labels length.");

        if (C <= 0)
            throw new IllegalArgumentException("C must be positive.");

        if (learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive.");

        if (epochs <= 0)
            throw new IllegalArgumentException("Epochs must be positive.");

        if (method == null)
            throw new IllegalArgumentException("Method cannot be null.");

        this.sparseDataset = sparseDataset;
        this.labels = normalizeLabels(labels);
        this.numSamples = labels.length;
        this.numFeatures = sparseDataset.stream()
                .flatMap(m -> m.keySet().stream())
                .max(Integer::compare)
                .orElseThrow(() -> new IllegalArgumentException("Sparse dataset must contain features"))
                + 1;

        this.C = C;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.kernel = kernel.toLowerCase();
        this.method = method.toLowerCase();

        this.weights = new double[numFeatures];
        this.bias = 0.0;
        this.alpha = new double[numSamples];

        this.gamma = gamma;
        this.degree = degree;
        this.coef0 = coef0;
        switch (this.method) {
            case "svc":
                computeSVC();
                break;
            case "linearsvc":
                computeLinearSVC();
            default:
                System.out.println("Method not supported: " + method);
        }
    }

    public void computeLinearSVC() {
        double eta = learningRate;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < numSamples; i++) {
                double y = labels[i];

                double fx = bias;

                if (sparse) {
                    Map<Integer, Double> x = sparseDataset.get(i);
                    for (var entry : x.entrySet()) {
                        fx += weights[entry.getKey()] * entry.getValue();
                    }
                    double multiplier = (y * fx < 1) ? C * y : 0;
                    for (var entry : x.entrySet()) {
                        int idx = entry.getKey();
                        double val = entry.getValue();
                        weights[idx] = weights[idx] - eta * (weights[idx] - multiplier * val);
                    }
                } else {
                    double[] x = dataset[i];
                    for (int j = 0; j < numFeatures; j++) {
                        fx += weights[j] * x[j];
                    }
                    double multiplier = (y * fx < 1) ? C * y : 0;
                    for (int j = 0; j < numFeatures; j++) {
                        weights[j] = weights[j] - eta * (weights[j] - multiplier * x[j]);
                    }
                }

                bias += (y * fx < 1) ? eta * C * y : 0;
            }
        }
    }

    public void computeSVC() {
        double[][] K = new double[numSamples][numSamples];
        for (int i = 0; i < numSamples; i++) {
            double[] xi = getX(i);
            for (int j = i; j < numSamples; j++) {
                double[] xj = getX(j);
                double val = kernel(xi, xj);
                K[i][j] = val;
                K[j][i] = val;
            }
        }

        // Dual coordinate ascent (SMO-like simplified loop)
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < numSamples; i++) {
                double Ei = predictScoreDual(i, K) - labels[i];
                double yi = labels[i];

                if ((yi * Ei < -1e-3 && alpha[i] < C) || (yi * Ei > 1e-3 && alpha[i] > 0)) {
                    int j = (i + 1) % numSamples;
                    double Ej = predictScoreDual(j, K) - labels[j];

                    double kii = K[i][i];
                    double kjj = K[j][j];
                    double kij = K[i][j];

                    double eta = kii + kjj - 2 * kij;
                    if (eta <= 0)
                        continue;

                    double alphaIOld = alpha[i];
                    double alphaJOld = alpha[j];
                    alpha[j] += labels[j] * (Ei - Ej) / eta;
                    alpha[j] = Math.max(0, Math.min(C, alpha[j]));
                    alpha[i] += labels[i] * labels[j] * (alphaJOld - alpha[j]);

                    updateBias(i, j, Ei, Ej, labels[i], labels[j], kii, kjj, kij, alphaIOld, alphaJOld);
                }
            }
        }

        // If linear kernel, build primal weights from alphas
        if ("linearkernel".equalsIgnoreCase(kernel)) {
            for (int i = 0; i < numSamples; i++) {
                double y = labels[i];
                double[] x = getX(i);
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += alpha[i] * y * x[j];
                }
            }
        }
    }

    protected double predictScoreDual(int i, double[][] K) {
        double sum = 0.0;
        for (int j = 0; j < numSamples; j++) {
            if (alpha[j] == 0)
                continue;
            sum += alpha[j] * labels[j] * K[j][i];
        }
        return sum + bias;
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
            double yj = labels[j];
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
                return polynomialKernel(x, z);
            case "rbf":
                return rbfKernel(x, z);
            case "linearkernel":
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

    public double polynomialKernel(double[] x, double[] z) {
        return Math.pow(linearKernel(x, z) + coef0, degree);
    }

    public double rbfKernel(double[] x, double[] z) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            double diff = x[i] - z[i];
            sum += diff * diff;
        }
        return Math.exp(-gamma * sum);
    }

    public double predictScore(double[] row) {
        if ("svc".equalsIgnoreCase(this.method)) {
            double sum = 0.0;

            for (int i = 0; i < numSamples; i++) {
                if (alpha[i] == 0)
                    continue;
                double[] xi = getX(i);
                sum += alpha[i] * labels[i] * kernel(xi, row);
            }

            return sum + bias;
        }
        return linearDecision(weights, row, bias);
    }

    public int predictSparse(Map<Integer, Double> row) {
        if (row == null)
            throw new IllegalArgumentException("Sparse row cannot be null.");

        if (row.isEmpty())
            throw new IllegalArgumentException("Sparse row cannot be empty.");
        int raw = linearDecisionSparse(weights, row, bias) >= 0 ? 1 : -1;
        return restoreLabel(raw);
    }

    public int predict(double[] row) {
        if (row == null)
            throw new IllegalArgumentException("Row cannot be null.");

        if (row.length != numFeatures)
            throw new IllegalArgumentException("Row must have exactly " + numFeatures + " features.");
        int raw = predictScore(row) >= 0 ? 1 : -1;
        return restoreLabel(raw);
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    private double[] normalizeLabels(double[] y) {
        double[] out = new double[y.length];

        boolean hasZero = false, hasOne = false, hasNeg = false;

        for (double v : y) {
            if (v == 0)
                hasZero = true;
            if (v == 1)
                hasOne = true;
            if (v == -1)
                hasNeg = true;
        }

        // Detect {0,1}
        if (hasZero && hasOne && !hasNeg) {
            originalZeroOne = true;
            normalizedLabels = true;

            for (int i = 0; i < y.length; i++)
                out[i] = (y[i] == 0) ? -1 : 1;

            return out;
        }

        return y; // already fine
    }

    private int restoreLabel(int pred) {
        if (originalZeroOne)
            return pred == -1 ? 0 : 1;
        return pred;
    }

    private void updateBias(int i, int j, double Ei, double Ej, double yi, double yj,
            double kii, double kjj, double kij,
            double aiOld, double ajOld) {

        double bi = bias - Ei
                - yi * (alpha[i] - aiOld) * kii
                - yj * (alpha[j] - ajOld) * kij;

        double bj = bias - Ej
                - yi * (alpha[i] - aiOld) * kij
                - yj * (alpha[j] - ajOld) * kjj;

        if (alpha[i] > 0 && alpha[i] < C)
            bias = bi;
        else if (alpha[j] > 0 && alpha[j] < C)
            bias = bj;
        else
            bias = (bi + bj) / 2.0;
    }
}
