package models.ml.Model.LogisticRegression;

import java.util.*;

public class AbstractLogisticRegression {
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
     * For ordinal mode, thresholds are not penalized.
     * Gradient descent: adds lambda * w_j / n to gradient of each non-bias weight.
     */
    public double lambda;

    public double[] weights;
    public double[][] multiWeights;
    public double[] thresholds;

    public AbstractLogisticRegression(double[][] dataset, double[] trainingLabels, String method, double learningRate,
            int epochs) {
        this(dataset, trainingLabels, method, learningRate, epochs, 0.0);
    }

    public AbstractLogisticRegression(double[][] dataset, double[] trainingLabels, String method, double learningRate,
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
        if (!Set.of("b", "binary", "m", "multinomial", "o", "ordinal").contains(this.method))
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

    public AbstractLogisticRegression(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels,
            int numFeatures, String method, double learningRate, int epochs) {
        this(sparseDataset, trainingLabels, numFeatures, method, learningRate, epochs, 0.0);
    }

    public AbstractLogisticRegression(List<Map<Integer, Double>> sparseDataset, double[] trainingLabels,
            int numFeatures, String method, double learningRate, int epochs, double lambda) {
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
                if (key < 0 || key >= numFeatures) {
                    throw new IllegalArgumentException("Sparse dataset contains invalid feature index: " + key);
                }
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException("Sparse dataset contains NaN or Infinity");
                }
            }
        }

        this.sparseDataset = sparseDataset;
        this.trainingLabels = trainingLabels;
        this.numSamples = sparseDataset.size();
        this.numFeatures = numFeatures;

        this.method = method.trim().toLowerCase();
        if (!Set.of("b", "binary", "m", "multinomial", "o", "ordinal").contains(this.method))
            throw new IllegalArgumentException("Unsupported method: " + method);

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

    private void fit() {
        switch (method) {
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
                throw new UnsupportedOperationException("Unsupported method: " + method);
        }
    }

    public double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public void fitBinaryLogistic() {
        weights = new double[numFeatures + 1];
        double[] gradients = new double[weights.length];

        for (int epoch = 0; epoch < epochs; epoch++) {

            for (int j = 0; j < gradients.length; j++)
                gradients[j] = 0.0;

            for (int i = 0; i < numSamples; i++) {

                double z;
                if (sparse) {
                    z = predictRowSparse(sparseDataset.get(i));
                } else {
                    z = predictRow(dataset[i]);
                }
                double p = sigmoid(z);

                double error = p - trainingLabels[i];

                gradients[0] += error;
                if (!sparse) {
                    for (int j = 0; j < numFeatures; j++)
                        gradients[j + 1] += error * dataset[i][j];
                } else {
                    for (var e : sparseDataset.get(i).entrySet()) {
                        int idx = e.getKey();
                        if (idx >= 0 && idx < numFeatures) {
                            gradients[idx + 1] += error * e.getValue();
                        }
                    }
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

    public void fitMultinomialLogistic() {
        int numClasses = 0;
        for (double label : trainingLabels) {
            numClasses = Math.max(numClasses, (int) label + 1);
        }

        multiWeights = new double[numClasses][numFeatures + 1];
        double[][] gradients = new double[numClasses][numFeatures + 1];

        for (int epoch = 0; epoch < epochs; epoch++) {

            for (int c = 0; c < numClasses; c++)
                for (int j = 0; j <= numFeatures; j++)
                    gradients[c][j] = 0.0;

            for (int i = 0; i < numSamples; i++) {
                int y = (int) trainingLabels[i];

                double[] scores = new double[numClasses];

                // ----- scores -----
                for (int c = 0; c < numClasses; c++) {
                    scores[c] = multiWeights[c][0];

                    if (!sparse) {
                        for (int j = 0; j < numFeatures; j++)
                            scores[c] += multiWeights[c][j + 1] * dataset[i][j];
                    } else {
                        for (var e : sparseDataset.get(i).entrySet()) {
                            int idx = e.getKey();
                            if (idx >= 0 && idx < numFeatures)
                                scores[c] += multiWeights[c][idx + 1] * e.getValue();
                        }
                    }
                }

                double[] probs = softmax(scores);

                // ----- gradients -----
                for (int c = 0; c < numClasses; c++) {
                    double error = probs[c] - (c == y ? 1.0 : 0.0);
                    gradients[c][0] += error;

                    if (!sparse) {
                        for (int j = 0; j < numFeatures; j++)
                            gradients[c][j + 1] += error * dataset[i][j];
                    } else {
                        for (var e : sparseDataset.get(i).entrySet()) {
                            int idx = e.getKey();
                            if (idx >= 0 && idx < numFeatures)
                                gradients[c][idx + 1] += error * e.getValue();
                        }
                    }
                }
            }

            // Apply L2 regularization: add lambda * w_cj to gradient for non-bias weights
            // Gradient is normalized by numSamples, so penalty is lambda * w_cj / numSamples
            for (int c = 0; c < numClasses; c++) {
                for (int j = 0; j <= numFeatures; j++) {
                    double penalty = (j == 0) ? 0.0 : lambda * multiWeights[c][j] / numSamples;
                    multiWeights[c][j] -= learningRate * (gradients[c][j] / numSamples + penalty);
                }
            }
        }
    }

    public void fitOrdinalLogistic() {
        int numClasses = 0;
        for (double label : trainingLabels)
            numClasses = Math.max(numClasses, (int) label + 1);

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
                double z = sparse
                        ? predictRowSparse(sparseDataset.get(i))
                        : predictRow(dataset[i]);

                for (int k = 0; k < thresholds.length; k++) {
                    double p = sigmoid(thresholds[k] - z);
                    double t = (trainingLabels[i] <= k) ? 1.0 : 0.0;
                    double error = p - t;

                    gradT[k] += error;
                    gradW[0] -= error;
                    if (!sparse) {
                        for (int j = 0; j < numFeatures; j++) {
                            gradW[j + 1] -= error * dataset[i][j];
                        }
                    } else {
                        for (var e : sparseDataset.get(i).entrySet()) {
                            int idx = e.getKey();
                            if (idx >= 0 && idx < numFeatures) {
                                gradW[idx + 1] -= error * e.getValue();
                            }
                        }
                    }
                }
            }

            // Apply L2 regularization to weights (not thresholds): add lambda * w_j to gradient for non-bias weights
            // Gradient is normalized by numSamples, so penalty is lambda * w_j / numSamples
            for (int j = 0; j < weights.length; j++) {
                double penalty = (j == 0) ? 0.0 : lambda * weights[j] / numSamples;
                weights[j] -= learningRate * (gradW[j] / numSamples + penalty);
            }

            // Thresholds are not penalized
            for (int j = 0; j < thresholds.length; j++) {
                thresholds[j] -= learningRate * gradT[j] / numSamples;
            }
        }
    }

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
        double z = weights[0];
        for (int j = 0; j < numFeatures; j++) {
            z += weights[j + 1] * row[j];
        }
        return z;
    }

    public double predictRowSparse(Map<Integer, Double> row) {
        if (!sparse)
            throw new IllegalStateException("Sparse prediction called in dense mode");
        if (row == null)
            throw new IllegalArgumentException("Row cannot be null");
        // Empty row is valid - predicts from bias alone
        for (Map.Entry<Integer, Double> e : row.entrySet()) {
            int key = e.getKey();
            double val = e.getValue();
            if (key < 0) {
                throw new IllegalArgumentException("Sparse vector contains negative feature index: " + key);
            }
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException("Sparse vector contains NaN or Infinity");
            }
        }
        double z = weights[0];
        for (var e : row.entrySet()) {
            int idx = e.getKey();
            if (idx >= 0 && idx < numFeatures) {
                z += weights[idx + 1] * e.getValue();
            }
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

    public double[] predictScoresMultinomialSparse(Map<Integer, Double> row) {
        int numClasses = multiWeights.length;
        double[] scores = new double[numClasses];

        if (row == null)
            throw new IllegalArgumentException("Row cannot be null");
        // Empty row is valid - predicts from bias alone
        for (Map.Entry<Integer, Double> e : row.entrySet()) {
            int key = e.getKey();
            double val = e.getValue();
            if (key < 0) {
                throw new IllegalArgumentException("Sparse vector contains negative feature index: " + key);
            }
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException("Sparse vector contains NaN or Infinity");
            }
        }
        for (int c = 0; c < numClasses; c++) {
            scores[c] = multiWeights[c][0];
            for (var e : row.entrySet()) {
                int idx = e.getKey();
                if (idx >= 0 && idx < numFeatures) {
                    scores[c] += multiWeights[c][idx + 1] * e.getValue();
                }
            }
        }
        return scores;
    }

    public double[] predictProbabilitiesOrdinal(double[] row) {
        int K = thresholds.length + 1;
        double[] cumulativeProbability = new double[K - 1];
        double[] probs = new double[K];

        double z = predictRow(row);

        for (int k = 0; k < K - 1; k++) {
            cumulativeProbability[k] = sigmoid(thresholds[k] - z);
        }

        probs[0] = cumulativeProbability[0];
        for (int k = 1; k < K - 1; k++) {
            probs[k] = cumulativeProbability[k] - cumulativeProbability[k - 1];
        }
        probs[K - 1] = 1.0 - cumulativeProbability[K - 2];

        return probs;
    }

    public double[] predictProbabilitiesOrdinalSparse(Map<Integer, Double> row) {
        int K = thresholds.length + 1;
        double[] cumulativeProbability = new double[K - 1];
        double[] probs = new double[K];

        double z = predictRowSparse(row);

        for (int k = 0; k < K - 1; k++) {
            cumulativeProbability[k] = sigmoid(thresholds[k] - z);
        }

        probs[0] = cumulativeProbability[0];
        for (int k = 1; k < K - 1; k++) {
            probs[k] = cumulativeProbability[k] - cumulativeProbability[k - 1];
        }
        probs[K - 1] = 1.0 - cumulativeProbability[K - 2];

        return probs;
    }

    public Map<Integer, Double> predictProbabilities(double[] row) {
        switch (method) {
            case "binary":
            case "b": {
                double p = sigmoid(predictRow(row));
                Map<Integer, Double> probs = new HashMap<>();
                probs.put(1, p);
                probs.put(0, 1.0 - p);
                return probs;
            }
            case "multinomial":
            case "m": {
                double[] probsArray = softmax(predictScoresMultinomial(row));
                Map<Integer, Double> probs = new HashMap<>();
                for (int i = 0; i < probsArray.length; i++) {
                    probs.put(i, probsArray[i]);
                }
                return probs;
            }
            case "ordinal":
            case "o": {
                double[] probsArray = predictProbabilitiesOrdinal(row);
                Map<Integer, Double> probs = new HashMap<>();
                for (int i = 0; i < probsArray.length; i++) {
                    probs.put(i, probsArray[i]);
                }
                return probs;
            }
            default:
                throw new UnsupportedOperationException();
        }
    }

    public Map<Integer, Double> predictProbabilities(Map<Integer, Double> row) {
        switch (method) {
            case "binary":
            case "b": {
                double p = sigmoid(predictRowSparse(row));
                Map<Integer, Double> probs = new HashMap<>();
                probs.put(1, p);
                probs.put(0, 1.0 - p);
                return probs;
            }
            case "multinomial":
            case "m": {
                double[] probsArray = softmax(predictScoresMultinomialSparse(row));
                Map<Integer, Double> probs = new HashMap<>();
                for (int i = 0; i < probsArray.length; i++) {
                    probs.put(i, probsArray[i]);
                }
                return probs;
            }
            case "ordinal":
            case "o": {
                double[] probsArray = predictProbabilitiesOrdinalSparse(row);
                Map<Integer, Double> probs = new HashMap<>();
                for (int i = 0; i < probsArray.length; i++) {
                    probs.put(i, probsArray[i]);
                }
                return probs;
            }
            default:
                throw new UnsupportedOperationException();
        }
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

    public int predictClass(Map<Integer, Double> row) {
        switch (method) {
            case "binary":
            case "b":
                return sigmoid(predictRowSparse(row)) >= 0.5 ? 1 : 0;

            case "multinomial":
            case "m": {
                return predictMultinomialClass(row);
            }
            case "ordinal":
            case "o": {
                return predictOrdinalClass(row);
            }

            default:
                throw new UnsupportedOperationException();
        }
    }

    public int predictMultinomialClass(Map<Integer, Double> row) {
        if (row == null)
            throw new IllegalArgumentException("Row cannot be null");
        // Empty row is valid - predicts from bias alone
        for (Map.Entry<Integer, Double> e : row.entrySet()) {
            int key = e.getKey();
            double val = e.getValue();
            if (key < 0) {
                throw new IllegalArgumentException("Sparse vector contains negative feature index: " + key);
            }
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException("Sparse vector contains NaN or Infinity");
            }
        }
        int numClasses = multiWeights.length;
        double[] scores = new double[numClasses];

        for (int c = 0; c < numClasses; c++) {
            scores[c] = multiWeights[c][0];
            for (var e : row.entrySet()) {
                int idx = e.getKey();
                if (idx >= 0 && idx < numFeatures) {
                    scores[c] += multiWeights[c][idx + 1] * e.getValue();
                }
            }
        }

        double[] probs = softmax(scores);
        int best = 0;
        for (int i = 1; i < probs.length; i++)
            if (probs[i] > probs[best])
                best = i;
        return best;
    }

    public int predictOrdinalClass(Map<Integer, Double> row) {
        int K = thresholds.length + 1;
        double[] cumulativeProbability = new double[K - 1];
        double[] probs = new double[K];

        double z = predictRowSparse(row);

        for (int k = 0; k < K - 1; k++)
            cumulativeProbability[k] = sigmoid(thresholds[k] - z);

        probs[0] = cumulativeProbability[0];
        for (int k = 1; k < K - 1; k++)
            probs[k] = cumulativeProbability[k] - cumulativeProbability[k - 1];
        probs[K - 1] = 1.0 - cumulativeProbability[K - 2];

        int best = 0;
        for (int i = 1; i < probs.length; i++)
            if (probs[i] > probs[best])
                best = i;
        return best;
    }

    public double[] getWeights() {
        return weights;
    }

    public double[][] getMultiWeights() {
        return multiWeights;
    }

    public double[] getThresholds() {
        return thresholds;
    }

    public String getMethod() {
        return method;
    }

    public boolean isSparse() {
        return sparse;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public int getEpochs() {
        return epochs;
    }

    public double getLambda() {
        return lambda;
    }
}