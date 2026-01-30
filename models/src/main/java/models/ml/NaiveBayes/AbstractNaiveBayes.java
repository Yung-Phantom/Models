package models.ml.NaiveBayes;

import java.util.*;

public class AbstractNaiveBayes {
    private double[][] dataset;
    private double[][] points;
    private int numFeatures;
    private Map<Integer, Integer> classCounts = new HashMap<>();
    private int totalSamples = 0;
    private String method;
    private double alpha;

    private Map<Integer, double[]> means = new HashMap<>();
    private Map<Integer, double[]> variances = new HashMap<>();

    private Map<Integer, double[]> multinomialFeatureCounts = new HashMap<>();
    private Map<Integer, Double> multinomialTotalFeatureCount = new HashMap<>();

    private Map<Integer, double[]> bernoulliFeatureCounts = new HashMap<>();

    public AbstractNaiveBayes(double[][] dataset, double[][] points, String method) {
        this(dataset, points, method, 1.0);
    }

    public AbstractNaiveBayes(double[][] dataset, double[][] points, String method, double alpha) {
        this.dataset = dataset;
        this.points = points;
        this.numFeatures = dataset[0].length - 1;
        this.method = method.toLowerCase();
        this.alpha = alpha;

        // compute class counts and prepare parameters
        computeClassCounts();
        switch (this.method) {
            case "gaussian":
                computeGaussianParameters();
                break;
            case "multinomial":
                computeMultinomialParameters();
                break;
            case "bernoulli":
                computeBernoulliParameters();
                break;
            default:
                System.out.println("Method not supported: " + method);
        }
    }

    private void computeClassCounts() {
        for (double[] row : dataset) {
            int label = (int) row[numFeatures];
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
            totalSamples++;
        }
    }

    private void computeGaussianParameters() {
        Map<Integer, double[]> sums = new HashMap<>();
        Map<Integer, double[]> sqSums = new HashMap<>();

        for (int label : classCounts.keySet()) {
            sums.put(label, new double[numFeatures]);
            sqSums.put(label, new double[numFeatures]);
        }

        for (double[] row : dataset) {
            int label = (int) row[numFeatures];
            double[] s = sums.get(label);
            double[] ss = sqSums.get(label);
            for (int j = 0; j < numFeatures; j++) {
                s[j] += row[j];
                ss[j] += row[j] * row[j];
            }
        }

        for (int label : classCounts.keySet()) {
            int n = classCounts.get(label);
            double[] mean = new double[numFeatures];
            double[] var = new double[numFeatures];
            double[] s = sums.get(label);
            double[] ss = sqSums.get(label);
            for (int j = 0; j < numFeatures; j++) {
                mean[j] = s[j] / n;
                var[j] = (ss[j] / n) - (mean[j] * mean[j]);
                if (var[j] <= 1e-9)
                    var[j] = 1e-9;
            }
            means.put(label, mean);
            variances.put(label, var);
        }
        
    }

    private void computeMultinomialParameters() {
        for (int label : classCounts.keySet()) {
            multinomialFeatureCounts.put(label, new double[numFeatures]);
            multinomialTotalFeatureCount.put(label, 0.0);
        }

        for (double[] row : dataset) {
            int label = (int) row[numFeatures];
            double[] counts = multinomialFeatureCounts.get(label);
            double total = multinomialTotalFeatureCount.get(label);
            for (int j = 0; j < numFeatures; j++) {
                counts[j] += row[j];
                total += row[j];
            }
            multinomialTotalFeatureCount.put(label, total);
        }
    }

    private void computeBernoulliParameters() {
        for (int label : classCounts.keySet()) {
            bernoulliFeatureCounts.put(label, new double[numFeatures]);
        }

        for (double[] row : dataset) {
            int label = (int) row[numFeatures];
            double[] counts = bernoulliFeatureCounts.get(label);
            for (int j = 0; j < numFeatures; j++) {
                if (row[j] != 0.0)
                    counts[j] += 1.0;
            }
        }
    }

    /**
     * Returns unnormalized log-probabilities (label -> log P(label) + log
     * P(x|label))
     */
    public Map<Integer, Double> computeLogProbabilities(double[] query) {
        Map<Integer, Double> logProbs = new HashMap<>();

        for (int label : classCounts.keySet()) {
            double prior = Math.log((double) classCounts.get(label) / totalSamples);
            double logLikelihood = 0.0;

            switch (method) {
                case "gaussian":
                    logLikelihood = gaussianLogLikelihood(label, query);
                    break;
                case "multinomial":
                    logLikelihood = multinomialLogLikelihood(label, query);
                    break;
                case "bernoulli":
                    logLikelihood = bernoulliLogLikelihood(label, query);
                    break;
                default:
                    throw new IllegalStateException("Unsupported method: " + method);
            }

            logProbs.put(label, prior + logLikelihood);
        }

        return logProbs;
    }

    /**
     * Returns normalized probabilities (label -> P(label | x)), sums to 1.
     * Uses log-sum-exp for numerical stability.
     */
    public Map<Integer, Double> computeProbabilities(double[] query) {
        Map<Integer, Double> logProbs = computeLogProbabilities(query);

        // compute log-sum-exp
        double lse = logSumExp(logProbs.values());

        Map<Integer, Double> probs = new HashMap<>();
        for (Map.Entry<Integer, Double> e : logProbs.entrySet()) {
            probs.put(e.getKey(), Math.exp(e.getValue() - lse));
        }
        return probs;
    }

    private double logSumExp(Collection<Double> values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : values)
            if (v > max)
                max = v;
        if (max == Double.NEGATIVE_INFINITY)
            return Double.NEGATIVE_INFINITY;
        double sum = 0.0;
        for (double v : values)
            sum += Math.exp(v - max);
        return max + Math.log(sum);
    }

    private double gaussianLogLikelihood(int label, double[] query) {
        double[] mean = means.get(label);
        double[] var = variances.get(label);
        double sum = 0.0;
        for (int j = 0; j < numFeatures; j++) {
            double x = query[j];
            double m = mean[j];
            double v = var[j];
            sum += -0.5 * Math.log(2 * Math.PI * v) - ((x - m) * (x - m)) / (2 * v);
        }
        return sum;
    }

    private double multinomialLogLikelihood(int label, double[] query) {
        double[] counts = multinomialFeatureCounts.get(label);
        double totalCount = multinomialTotalFeatureCount.get(label);
        double V = numFeatures; // number of features
        double sum = 0.0;
        for (int j = 0; j < numFeatures; j++) {
            double qCount = query[j];
            double prob = (counts[j] + alpha) / (totalCount + alpha * V);
            if (qCount > 0)
                sum += qCount * Math.log(prob);
        }
        return sum;
    }

    private double bernoulliLogLikelihood(int label, double[] query) {
        double[] counts = bernoulliFeatureCounts.get(label);
        int Nc = classCounts.get(label);
        double sum = 0.0;
        for (int j = 0; j < numFeatures; j++) {
            double x = query[j] != 0.0 ? 1.0 : 0.0;
            double p1 = (counts[j] + alpha) / (Nc + 2.0 * alpha);
            double p0 = 1.0 - p1;
            p1 = Math.max(p1, 1e-12);
            p0 = Math.max(p0, 1e-12);
            sum += x * Math.log(p1) + (1.0 - x) * Math.log(p0);
        }
        return sum;
    }

    public double[][] getDataset() {
        return dataset;
    }

    public double[][] getPoints() {
        return points;
    }

    // Optional getters for debugging/tests
    public Map<Integer, Integer> getClassCounts() {
        return Collections.unmodifiableMap(classCounts);
    }

    public Map<Integer, double[]> getMeans() {
        return Collections.unmodifiableMap(means);
    }

    public Map<Integer, double[]> getVariances() {
        return Collections.unmodifiableMap(variances);
    }
}
