package models.ml.NaiveBayes;

import java.util.*;

public class NaiveBayes {
    public double[][] dataset;
    public double[][] points;
    public AbstractNaiveBayes nb;

    public NaiveBayes(double[][] dataset, double[][] points, String method) {
        this.dataset = dataset;
        this.points = points;
        this.nb = new AbstractNaiveBayes(dataset, points, method);
    }

    public NaiveBayes(double[][] dataset, double[][] points, String method, double alpha) {
        this.dataset = dataset;
        this.points = points;
        this.nb = new AbstractNaiveBayes(dataset, points, method, alpha);
    }

    public int predict(int queryIndex) {
        double[] query = points[queryIndex];
        Map<Integer, Double> probs = nb.computeProbabilities(query);
        return probs.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .get()
                .getKey();
    }

    /** Returns normalized probabilities that sum to 1.0 */
    public Map<Integer, Double> predictProbability(int queryIndex) {
        double[] query = points[queryIndex];
        return nb.computeProbabilities(query);
    }

    /** Returns raw unnormalized log-probabilities (for debugging) */
    public Map<Integer, Double> predictLogProbability(int queryIndex) {
        double[] query = points[queryIndex];
        return nb.computeLogProbabilities(query);
    }

    public int[] predictAll() {
        int[] labels = new int[points.length];
        for (int i = 0; i < points.length; i++)
            labels[i] = predict(i);
        return labels;
    }

    public List<Map<Integer, Double>> predictAllProbability() {
        List<Map<Integer, Double>> list = new ArrayList<>();
        for (int i = 0; i < points.length; i++)
            list.add(predictProbability(i));
        return list;
    }

    public double accuracy() {
        int correct = 0;
        for (int i = 0; i < points.length; i++) {
            int predicted = predict(i);
            int actual = (int) points[i][points[i].length - 1];
            if (predicted == actual)
                correct++;
        }
        return (double) correct / points.length;
    }
}
