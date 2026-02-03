package models.ml.LogisticRegression;

/**
 * Abstract class for Logistic Regression.
 * Provides implementations for binary, multinomial, and ordinal logistic
 * regression
 * using gradient descent optimization.
 * 
 * @author Justice
 * @version 1.0
 */
public class AbstractLogisticRegression {

    /** The dataset to fit the model to. */
    public double[][] dataset;

    /** The number of samples in the dataset. */
    public int numSamples;

    /** The number of features in the dataset. */
    public int numFeatures;

    /** The weights for the logistic regression model. */
    public double[] weights;

    /** The learning rate for gradient descent. */
    public double learningRate;

    /** The number of epochs for gradient descent. */
    public int epochs;

    /**
     * The weights for the multinomial logistic regression model. Each row
     * represents the weights for a class, and each column represents the weights
     * for a feature.
     */
    public double[][] multiWeights;

    /**
     * The thresholds for the logistic regression model. Each element
     * represents the threshold for a class. The threshold is used to
     * determine the class label for a given input vector.
     */
    public double[] thresholds;

    public String method;

    /**
     * Constructor for logistic regression with default settings.
     * Defaults to binary logistic regression with learningRate = 0.01 and epochs =
     * 1000.
     *
     * @param dataset The dataset to fit the model to.
     */
    public AbstractLogisticRegression(double[][] dataset) {
        this(dataset, "binary", 0.01, 1000);
    }

    /**
     * Constructor for logistic regression with specified method.
     * Defaults to learningRate = 0.01 and epochs = 1000.
     *
     * @param dataset The dataset to fit the model to.
     * @param method  The method to use ("binary", "multinomial", "ordinal").
     */
    public AbstractLogisticRegression(double[][] dataset, String method) {
        this(dataset, method, 0.01, 1000);
    }

    /**
     * Constructor for logistic regression with specified learning parameters.
     * Defaults to binary logistic regression.
     *
     * @param dataset      The dataset to fit the model to.
     * @param learningRate The learning rate for gradient descent.
     * @param epochs       The number of epochs for gradient descent.
     */
    public AbstractLogisticRegression(double[][] dataset, double learningRate, int epochs) {
        this(dataset, "binary", learningRate, epochs);
    }

    /**
     * Constructor for logistic regression with full specification.
     *
     * @param dataset      The dataset to fit the model to.
     * @param method       The method to use ("binary", "multinomial", "ordinal").
     * @param learningRate The learning rate for gradient descent.
     * @param epochs       The number of epochs for gradient descent.
     */
    public AbstractLogisticRegression(double[][] dataset, String method, double learningRate, int epochs) {
        this.dataset = dataset;
        this.method = method;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.numSamples = dataset.length;
        this.numFeatures = dataset[0].length - 1;

        switch (method.toLowerCase()) {
            case "binary":
            case "b":
                fitBinaryLogistic();
                break;
            case "multinomial":
            case "m":
                fitMultinomialLogistic();
                break;
            case "ordinal":
            case "o":
                fitOrdinalLogistic();
                break;
            default:
                throw new IllegalArgumentException("Unsupported logistic regression method: " + method);
        }
    }

    /**
     * Computes the sigmoid of a given value
     *
     * @param z the value of which to compute the sigmoid
     * @return the sigmoid of z
     *
     */
    public double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /**
     * Fits a binary logistic regression model using gradient descent.
     * Updates the weights based on the dataset.
     */
    public void fitBinaryLogistic() {
        weights = new double[numFeatures + 1];
        double[] gradients = new double[weights.length];

        for (int epoch = 0; epoch < epochs; epoch++) {

            for (int j = 0; j < gradients.length; j++)
                gradients[j] = 0.0;

            for (int i = 0; i < numSamples; i++) {
                double[] row = dataset[i];
                double y = row[numFeatures]; // 0 or 1

                double z = predictRow(row);
                double p = sigmoid(z);

                double error = p - y;

                gradients[0] += error;
                for (int j = 0; j < numFeatures; j++)
                    gradients[j + 1] += error * row[j];
            }

            for (int j = 0; j < weights.length; j++)
                weights[j] -= learningRate * gradients[j] / numSamples;
        }
    }

    /**
     * Fits a multinomial logistic regression model using gradient descent.
     * Updates the weights based on the dataset.
     */

    public void fitMultinomialLogistic() {
        int numClasses = 0;
        for (double[] row : dataset) {
            numClasses = Math.max(numClasses, (int) row[numFeatures] + 1);
        }

        multiWeights = new double[numClasses][numFeatures + 1];
        double[][] gradients = new double[numClasses][numFeatures + 1];

        for (int epoch = 0; epoch < epochs; epoch++) {

            for (int c = 0; c < numClasses; c++) {
                for (int j = 0; j <= numFeatures; j++) {
                    gradients[c][j] = 0.0;
                }
            }

            for (int i = 0; i < numSamples; i++) {
                double[] row = dataset[i];
                int y = (int) row[numFeatures];

                double[] scores = new double[numClasses];
                for (int c = 0; c < numClasses; c++) {
                    scores[c] = multiWeights[c][0];
                    for (int j = 0; j < numFeatures; j++) {
                        scores[c] += multiWeights[c][j + 1] * row[j];
                    }
                }

                double[] probs = softmax(scores);

                for (int c = 0; c < numClasses; c++) {
                    double error = probs[c] - (c == y ? 1.0 : 0.0);
                    gradients[c][0] += error;
                    for (int j = 0; j < numFeatures; j++) {
                        gradients[c][j + 1] += error * row[j];
                    }
                }
            }

            for (int c = 0; c < numClasses; c++) {
                for (int j = 0; j <= numFeatures; j++) {
                    multiWeights[c][j] -= learningRate * gradients[c][j] / numSamples;
                }
            }
        }
    }

    /**
     * Fits an ordinal logistic regression model using gradient descent.
     * Updates the weights based on the dataset.
     */

    public void fitOrdinalLogistic() {
        int numClasses = 0;
        for (double[] row : dataset)
            numClasses = Math.max(numClasses, (int) row[numFeatures] + 1);

        weights = new double[numFeatures + 1];
        thresholds = new double[numClasses - 1];

        double[] gradW = new double[weights.length];
        double[] gradT = new double[thresholds.length];

        for (int epoch = 0; epoch < epochs; epoch++) {

            for (int j = 0; j < gradW.length; j++)
                gradW[j] = 0.0;
            for (int j = 0; j < gradT.length; j++)
                gradT[j] = 0.0;

            for (int i = 0; i < numSamples; i++) {
                double[] row = dataset[i];
                int y = (int) row[numFeatures];

                double z = predictRow(row);

                for (int k = 0; k < thresholds.length; k++) {
                    double p = sigmoid(thresholds[k] - z);
                    double t = (y <= k) ? 1.0 : 0.0;
                    double error = p - t;

                    gradT[k] += error;
                    gradW[0] -= error;
                    for (int j = 0; j < numFeatures; j++)
                        gradW[j + 1] -= error * row[j];
                }
            }

            for (int j = 0; j < weights.length; j++)
                weights[j] -= learningRate * gradW[j] / numSamples;

            for (int j = 0; j < thresholds.length; j++)
                thresholds[j] -= learningRate * gradT[j] / numSamples;
        }
    }

    /**
     * Computes the softmax function for a given vector.
     *
     * @param z Input vector.
     * @return Probability distribution after applying softmax.
     */
    public double[] softmax(double[] z) {
        double sum = 0.0;
        double[] probabilities = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            probabilities[i] = Math.exp(z[i]);
            sum += probabilities[i];
        }
        for (int i = 0; i < z.length; i++) {
            probabilities[i] /= sum;
        }
        return probabilities;
    }

    /**
     * Predicts the raw linear combination (z) for a single row.
     * Subclasses may apply sigmoid or softmax to convert into probabilities.
     *
     * @param row Input feature row.
     * @return Raw prediction value (z).
     */
    public double predictRow(double[] row) {
        double z = weights[0];
        for (int j = 0; j < numFeatures; j++) {
            z += weights[j + 1] * row[j];
        }
        return z;
    }

    public double[] predictScoresMultinomial(double[] row) {
        int numClasses = multiWeights.length;
        double[] scores = new double[numClasses];

        for (int c = 0; c < numClasses; c++) {
            scores[c] = multiWeights[c][0];
            for (int j = 0; j < numFeatures; j++) {
                scores[c] += multiWeights[c][j + 1] * row[j];
            }
        }
        return scores;
    }

    public double[] predictProbabilitiesOrdinal(double[] row) {
        int K = thresholds.length + 1;
        double[] cum = new double[K - 1];
        double[] probs = new double[K];

        double z = predictRow(row);

        // cumulative probabilities P(Y <= k)
        for (int k = 0; k < K - 1; k++) {
            cum[k] = sigmoid(thresholds[k] - z);
        }

        // convert cumulative -> class probabilities
        probs[0] = cum[0];
        for (int k = 1; k < K - 1; k++) {
            probs[k] = cum[k] - cum[k - 1];
        }
        probs[K - 1] = 1.0 - cum[K - 2];

        return probs;
    }

    public int predictClass(double[] row) {
        switch (method) {
            case "binary":
            case "b":
                return sigmoid(predictRow(row)) >= 0.5 ? 1 : 0;

            case "multinomial":
            case "m": {
                double[] probs = softmax(predictScoresMultinomial(row));
                int best = 0;
                for (int i = 1; i < probs.length; i++)
                    if (probs[i] > probs[best])
                        best = i;
                return best;
            }

            case "ordinal":
            case "o": {
                double[] probs = predictProbabilitiesOrdinal(row);
                int best = 0;
                for (int i = 1; i < probs.length; i++)
                    if (probs[i] > probs[best])
                        best = i;
                return best;
            }

            default:
                throw new UnsupportedOperationException();
        }
    }

    // Getter for weights
    public double[] getWeights() {
        return weights;
    }
}
