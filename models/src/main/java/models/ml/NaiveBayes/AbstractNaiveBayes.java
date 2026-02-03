package models.ml.NaiveBayes;

import java.util.*;

public class AbstractNaiveBayes {
    public double[][] dataset;
    public int numFeatures;
    public String method;
    public double alpha;

    public Map<Integer, Integer> classCounts = new HashMap<>();
    public int totalSamples = 0;

    private int[] classes;
    private double[] logPriors;

    public Map<Integer, double[]> means = new HashMap<>();
    public Map<Integer, double[]> variances = new HashMap<>();

    public Map<Integer, double[]> multinomialFeatureCounts = new HashMap<>();
    public Map<Integer, Double> multinomialTotalFeatureCount = new HashMap<>();

    public Map<Integer, double[]> bernoulliFeatureCounts = new HashMap<>();

    public AbstractNaiveBayes(double[][] dataset, String method) {
        this(dataset, method, 1.0);
    }

    public AbstractNaiveBayes(double[][] dataset, String method, double alpha) {
        this.dataset = dataset;
        this.numFeatures = dataset[0].length - 1;
        this.method = method.toLowerCase();
        this.alpha = alpha;

        // compute class counts and prepare parameters
        computeClassCounts();
        cacheClassesAndPriors();
        switch (this.method) {
            case "gaussian":
            case "g":
                computeGaussianParameters();
                break;
            case "m":
            case "multinomial":
                computeMultinomialParameters();
                break;
            case "b":
            case "bernoulli":
                computeBernoulliParameters();
                break;
            default:
                System.out.println("Method not supported: " + method);
        }
    }

    public void computeClassCounts() {
        for (double[] row : dataset) {
            int label = (int) row[numFeatures];
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
            totalSamples++;
        }
    }

    private void cacheClassesAndPriors() {
        int k = classCounts.size();
        classes = new int[k];
        logPriors = new double[k];

        int i = 0;
        for (Map.Entry<Integer, Integer> e : classCounts.entrySet()) {
            classes[i] = e.getKey();
            logPriors[i] = Math.log((double) e.getValue() / totalSamples);
            i++;
        }
    }

    public void computeGaussianParameters() {
        Map<Integer, double[]> sums = new HashMap<>();
        Map<Integer, double[]> sqSums = new HashMap<>();

        for (int label : classes) {
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

        for (int label : classes) {
            int n = classCounts.get(label);
            double[] mean = new double[numFeatures];
            double[] variance = new double[numFeatures];
            double[] s = sums.get(label);
            double[] ss = sqSums.get(label);

            for (int j = 0; j < numFeatures; j++) {
                mean[j] = s[j] / n;
                variance[j] = (ss[j] / n) - mean[j] * mean[j];
                if (variance[j] < 1e-9)
                    variance[j] = 1e-9;
            }
            means.put(label, mean);
            variances.put(label, variance);
        }

    }

    public void computeMultinomialParameters() {
        for (int label : classes) {
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

    public void computeBernoulliParameters() {
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

    public Map<Integer, Double> computeLogProbabilities(double[] query) {
        Map<Integer, Double> logProbs = new HashMap<>();

        for (int i = 0; i < classes.length; i++) {
            int label = classes[i];
            double logLike;

            switch (method) {
                case "gaussian":
                case "g":
                    logLike = gaussianLogLikelihood(label, query);
                    break;
                case "multinomial":
                case "m":
                    logLike = multinomialLogLikelihood(label, query);
                    break;
                case "bernoulli":
                case "b":
                    logLike = bernoulliLogLikelihood(label, query);
                    break;
                default:
                    throw new IllegalStateException();
            }

            logProbs.put(label, logPriors[i] + logLike);
        }

        return logProbs;
    }

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

    public double logSumExp(Collection<Double> values) {
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

    public double gaussianLogLikelihood(int label, double[] query) {
        double[] mean = means.get(label);
        double[] variance = variances.get(label);
        double sum = 0.0;
        for (int j = 0; j < numFeatures; j++) {
            double d = query[j] - mean[j];
            sum += -0.5 * Math.log(2 * Math.PI * variance[j]) - (d * d) / (2 * variance[j]);
        }
        return sum;
    }

    public double multinomialLogLikelihood(int label, double[] query) {
        double[] counts = multinomialFeatureCounts.get(label);
        double totalCount = multinomialTotalFeatureCount.get(label);
        double V = numFeatures; // number of features
        double sum = 0.0;
        for (int j = 0; j < numFeatures; j++) {
            if (query[j] > 0)
                sum += query[j] * Math.log((counts[j] + alpha) / (totalCount + alpha * V));
        }
        return sum;
    }

    public double bernoulliLogLikelihood(int label, double[] query) {
        double[] counts = bernoulliFeatureCounts.get(label);
        int Nc = classCounts.get(label);
        double sum = 0.0;

        for (int j = 0; j < numFeatures; j++) {
            double x = query[j] != 0.0 ? 1.0 : 0.0;
            double p1 = (counts[j] + alpha) / (Nc + 2.0 * alpha);
            p1 = Math.max(p1, 1e-12);
            sum += x * Math.log(p1) + (1.0 - x) * Math.log(1.0 - p1);
        }
        return sum;
    }
}
