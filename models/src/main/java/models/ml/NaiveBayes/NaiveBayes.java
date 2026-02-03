package models.ml.NaiveBayes;

import java.util.*;

public class NaiveBayes {
    public double[][] dataset;
    public double[][] points;
    public AbstractNaiveBayes nb;

    public NaiveBayes(double[][] dataset, double[][] points) {
        this(dataset, points, "gaussian", 1.0);
    }
    public NaiveBayes(double[][] dataset, double[][] points, String method) {
        this(dataset, points, method, 1.0);
    }

    public NaiveBayes(double[][] dataset, double[][] points, String method, double alpha) {
        this.dataset = dataset;
        this.points = points;
        this.nb = new AbstractNaiveBayes(dataset, method, alpha);
    }

    public int predict(int queryIndex) {
        Map<Integer, Double> logProbs = predictProbability(queryIndex);

        int best = -1;
        double max = Double.NEGATIVE_INFINITY;

        for (Map.Entry<Integer, Double> e : logProbs.entrySet()) {
            if (e.getValue() > max) {
                max = e.getValue();
                best = e.getKey();
            }
        }
        return best;
    }

    /** Returns normalized probabilities that sum to 1.0 */
    public Map<Integer, Double> predictProbability(int i) {
        return nb.computeProbabilities(points[i]);
    }

    public int[] predictAll() {
        int[] preds = new int[points.length];
        for (int i = 0; i < points.length; i++)
            preds[i] = predict(i);
        return preds;
    }

    public double accuracy() {
        int correct = 0;
        for (int i = 0; i < points.length; i++) {
            if (predict(i) == (int) points[i][points[i].length - 1])
                correct++;
        }
        return (double) correct / points.length;
    }
}
