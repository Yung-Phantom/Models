package models.ml.SVM;

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

    protected double[] weights;
    protected double bias;
    protected double C;
    protected double learningRate;
    protected int epochs;
    protected String kernel;
    protected String method;

    protected double[] alpha;

    public AbstractSVM(double[][] dataset) {
        this(dataset, 0.1, 0.01, 10, "linearKernel", "linearsvc");
    }

    public AbstractSVM(double[][] dataset, int epochs, String kernel, String method) {
        this(dataset, 0.1, 0.01, epochs, kernel, method);
    }

    public AbstractSVM(double[][] dataset, int epochs) {
        this(dataset, 0.1, 0.01, epochs, "linearKernel", "linearsvc");
    }

    public AbstractSVM(double[][] dataset, String kernel, String method) {
        this(dataset, 0.1, 0.01, 10, kernel, method);
    }

    public AbstractSVM(double[][] dataset, double C, double learningRate, String kernel, String method) {
        this(dataset, C, learningRate, 10, kernel, method);
    }

    public AbstractSVM(double[][] dataset, double C, double learningRate) {
        this(dataset, C, learningRate, 10, "linearKernel", "linearsvc");
    }

    public AbstractSVM(double[][] dataset, double C, double learningRate, int epochs) {
        this(dataset, C, learningRate, epochs, "linearKernel", "linearsvc");
    }

    public AbstractSVM(double[][] dataset, double C, double learningRate, int epochs,
            String kernel, String method) {

        this.dataset = dataset;
        this.numSamples = dataset.length;
        this.numFeatures = dataset[0].length - 1;

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

    public void computeLinearSVC() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < numSamples; i++) {

                double[] x = getX(i);
                double y = dataset[i][numFeatures];

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

                    //double aiOld = alpha[i];
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

    public int predict(double[] row) {
        return predictScore(row) >= 0 ? 1 : -1;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }
}
