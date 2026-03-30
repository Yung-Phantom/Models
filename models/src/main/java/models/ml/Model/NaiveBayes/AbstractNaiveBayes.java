package models.ml.Model.NaiveBayes;

import java.util.*;

public class AbstractNaiveBayes {
    public double[][] dataset;
    public List<Map<Integer, Double>> sparseDataset;
    private double[] labels;
    private boolean sparse;
    public int numFeatures;
    public String method;
    public double alpha;

    public Map<Integer, Integer> classCounts = new HashMap<>();
    public int totalSamples = 0;

    private int[] classes;
    private double[] logPriors;

    public Map<Integer, double[]> means = new HashMap<>();
    public Map<Integer, double[]> variances = new HashMap<>();

    private final Map<Integer, double[]> featureCounts = new HashMap<>();
    private final Map<Integer, Double> totalFeatureCounts = new HashMap<>();

    /*
     * =========================
     * Dense constructor
     * =========================
     */
    public AbstractNaiveBayes(double[][] X, double[] labels, String method, double alpha) {
        this.dataset = X;
        this.labels = labels;
        this.method = method.toLowerCase();
        this.alpha = alpha;
        this.sparse = false;
        this.numFeatures = X[0].length;

        train();
    }

    /*
     * =========================
     * Sparse constructor
     * =========================
     */
    public AbstractNaiveBayes(List<Map<Integer, Double>> X, double[] labels,
            String method, double alpha) {
        this.sparseDataset = X;
        this.labels = labels;
        this.method = method.toLowerCase();
        this.alpha = alpha;
        this.sparse = true;

        this.numFeatures = inferFeatureCount(X);
        train();
    }

    private void train() {
        computeClassCounts();
        cacheClassesAndPriors();

        switch (method) {
            case "gaussian":
            case "g":
                if (sparse)
                    throw new IllegalArgumentException("Gaussian NB not supported for sparse data.");
                trainGaussian();
                break;

            case "multinomial":
            case "m":
                trainMultinomial();
                break;

            case "bernoulli":
            case "b":
                trainBernoulli();
                break;

            default:
                throw new IllegalArgumentException("Unsupported Naive Bayes method: " + method);
        }
    }

    public void computeClassCounts() {
        for (double label : labels) {
            int c = (int) label;
            classCounts.put(c, classCounts.getOrDefault(c, 0) + 1);
            totalSamples++;
        }
    }

    private void cacheClassesAndPriors() {
        int k = classCounts.size();
        classes = new int[k];
        logPriors = new double[k];

        int i = 0;
        for (var e : classCounts.entrySet()) {
            classes[i] = e.getKey();
            logPriors[i] = Math.log((double) e.getValue() / totalSamples);
            i++;
        }
    }

    private void trainGaussian() {
        Map<Integer, double[]> sums = new HashMap<>();
        Map<Integer, double[]> sqSums = new HashMap<>();

        for (int c : classes) {
            sums.put(c, new double[numFeatures]);
            sqSums.put(c, new double[numFeatures]);
        }

        for (int i = 0; i < dataset.length; i++) {
            int c = (int) labels[i];
            for (int j = 0; j < numFeatures; j++) {
                sums.get(c)[j] += dataset[i][j];
                sqSums.get(c)[j] += dataset[i][j] * dataset[i][j];
            }
        }

        for (int c : classes) {
            int n = classCounts.get(c);
            double[] mean = new double[numFeatures];
            double[] var = new double[numFeatures];

            for (int j = 0; j < numFeatures; j++) {
                mean[j] = sums.get(c)[j] / n;
                var[j] = sqSums.get(c)[j] / n - mean[j] * mean[j];
                if (var[j] < 1e-9)
                    var[j] = 1e-9;
            }
            means.put(c, mean);
            variances.put(c, var);
        }
    }

    private void trainMultinomial() {
        for (int c : classes) {
            featureCounts.put(c, new double[numFeatures]);
            totalFeatureCounts.put(c, 0.0);
        }

        if (!sparse) {
            for (int i = 0; i < dataset.length; i++) {
                int c = (int) labels[i];
                for (int j = 0; j < numFeatures; j++) {
                    featureCounts.get(c)[j] += dataset[i][j];
                    totalFeatureCounts.put(c,
                            totalFeatureCounts.get(c) + dataset[i][j]);
                }
            }
        } else {
            for (int i = 0; i < sparseDataset.size(); i++) {
                int c = (int) labels[i];
                for (var e : sparseDataset.get(i).entrySet()) {
                    featureCounts.get(c)[e.getKey()] += e.getValue();
                    totalFeatureCounts.put(c,
                            totalFeatureCounts.get(c) + e.getValue());
                }
            }
        }
    }

    private void trainBernoulli() {
        for (int c : classes) {
            featureCounts.put(c, new double[numFeatures]);
        }

        if (!sparse) {
            for (int i = 0; i < dataset.length; i++) {
                int c = (int) labels[i];
                for (int j = 0; j < numFeatures; j++) {
                    if (dataset[i][j] != 0.0)
                        featureCounts.get(c)[j]++;
                }
            }
        } else {
            for (int i = 0; i < sparseDataset.size(); i++) {
                int c = (int) labels[i];
                for (int j : sparseDataset.get(i).keySet()) {
                    featureCounts.get(c)[j]++;
                }
            }
        }
    }

    public Map<Integer, Double> computeProbabilities(double[] x) {
        if (sparse)
            throw new IllegalStateException("Dense query in sparse model.");
        return softmax(logPosteriorsDense(x));
    }

    public Map<Integer, Double> computeProbabilities(Map<Integer, Double> x) {
        if (!sparse)
            throw new IllegalStateException("Sparse query in dense model.");
        return softmax(logPosteriorsSparse(x));
    }

    private Map<Integer, Double> logPosteriorsDense(double[] x) {
        Map<Integer, Double> out = new HashMap<>();

        for (int i = 0; i < classes.length; i++) {
            int c = classes[i];
            double logLike;

            switch (method) {
                case "gaussian":
                case "g":
                    logLike = gaussianLogLike(c, x);
                    break;
                case "multinomial":
                case "m":
                    logLike = multinomialLogLike(c, x);
                    break;
                default:
                    logLike = bernoulliLogLikeDense(c, x);
            }

            out.put(c, logPriors[i] + logLike);
        }
        return out;
    }

    private Map<Integer, Double> logPosteriorsSparse(Map<Integer, Double> x) {
        Map<Integer, Double> out = new HashMap<>();

        for (int i = 0; i < classes.length; i++) {
            int c = classes[i];
            double logLike = multinomialLogLikeSparse(c, x);
            out.put(c, logPriors[i] + logLike);
        }
        return out;
    }

    private double gaussianLogLike(int c, double[] x) {
        double sum = 0;
        double[] mean = means.get(c);
        double[] var = variances.get(c);

        for (int j = 0; j < numFeatures; j++) {
            double d = x[j] - mean[j];
            sum += -0.5 * Math.log(2 * Math.PI * var[j])
                    - (d * d) / (2 * var[j]);
        }
        return sum;
    }

    private double multinomialLogLike(int c, double[] x) {
        double[] counts = featureCounts.get(c);
        double total = totalFeatureCounts.get(c);
        double V = numFeatures;

        double sum = 0;
        for (int j = 0; j < numFeatures; j++) {
            if (x[j] > 0)
                sum += x[j] * Math.log((counts[j] + alpha) / (total + alpha * V));
        }
        return sum;
    }

    private double multinomialLogLikeSparse(int c, Map<Integer, Double> x) {
        double[] counts = featureCounts.get(c);
        double total = totalFeatureCounts.get(c);
        double V = numFeatures;

        double sum = 0;
        for (var e : x.entrySet()) {
            sum += e.getValue()
                    * Math.log((counts[e.getKey()] + alpha) / (total + alpha * V));
        }
        return sum;
    }

    private double bernoulliLogLikeDense(int c, double[] x) {
        double[] counts = featureCounts.get(c);
        int Nc = classCounts.get(c);
        double sum = 0;

        for (int j = 0; j < numFeatures; j++) {
            double p = (counts[j] + alpha) / (Nc + 2 * alpha);
            p = Math.max(p, 1e-12);
            sum += (x[j] != 0 ? Math.log(p) : Math.log(1 - p));
        }
        return sum;
    }

    private double bernoulliLogLikeSparse(int c, Map<Integer, Double> x) {
        double[] counts = featureCounts.get(c);
        int Nc = classCounts.get(c);
        double sum = 0;

        for (var entry : x.entrySet()) {
            int feature = entry.getKey();
            double p = (counts[feature] + alpha) / (Nc + 2 * alpha);
            p = Math.max(p, 1e-12);
            sum += Math.log(p); // feature is present
        }

        // For features not present in the sparse data, use (1 - p) (feature is absent)
        for (int j = 0; j < numFeatures; j++) {
            if (!x.containsKey(j)) {
                double p = (counts[j] + alpha) / (Nc + 2 * alpha);
                p = Math.max(p, 1e-12);
                sum += Math.log(1 - p); // feature is absent
            }
        }
        return sum;
    }

    private Map<Integer, Double> softmax(Map<Integer, Double> logVals) {
        double max = Collections.max(logVals.values());
        double sum = 0;

        for (double v : logVals.values())
            sum += Math.exp(v - max);

        Map<Integer, Double> out = new HashMap<>();
        for (var e : logVals.entrySet())
            out.put(e.getKey(), Math.exp(e.getValue() - max) / sum);

        return out;
    }

    private int inferFeatureCount(List<Map<Integer, Double>> X) {
        int max = 0;
        for (var m : X)
            for (int k : m.keySet())
                max = Math.max(max, k + 1);
        return max;
    }
}
