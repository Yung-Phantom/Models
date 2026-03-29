package models.ml.Model;

public class Metrics {
    
    // --- Classification Metrics ---
    public static double accuracy(int[] predictions, int[] labels) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == labels[i]) correct++;
        }
        return (double) correct / predictions.length;
    }

    public static double precision(int[] predictions, int[] labels, int positiveClass) {
        int tp = 0, fp = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == positiveClass) {
                if (labels[i] == positiveClass) tp++;
                else fp++;
            }
        }
        return tp + fp == 0 ? 0.0 : (double) tp / (tp + fp);
    }

    public static double recall(int[] predictions, int[] labels, int positiveClass) {
        int tp = 0, fn = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (labels[i] == positiveClass) {
                if (predictions[i] == positiveClass) tp++;
                else fn++;
            }
        }
        return tp + fn == 0 ? 0.0 : (double) tp / (tp + fn);
    }

    public static double specificity(int[] predictions, int[] labels, int positiveClass) {
        int tn = 0, fp = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (labels[i] != positiveClass) {
                if (predictions[i] != positiveClass) tn++;
                else fp++;
            }
        }
        return tn + fp == 0 ? 0.0 : (double) tn / (tn + fp);
    }

    public static double f1Score(int[] predictions, int[] labels, int positiveClass) {
        double prec = precision(predictions, labels, positiveClass);
        double rec = recall(predictions, labels, positiveClass);
        return (prec + rec == 0) ? 0.0 : 2 * (prec * rec) / (prec + rec);
    }

    public static double balancedAccuracy(int[] predictions, int[] labels, int positiveClass) {
        return (recall(predictions, labels, positiveClass) + specificity(predictions, labels, positiveClass)) / 2.0;
    }

    public static double mcc(int[] predictions, int[] labels, int positiveClass) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == positiveClass && labels[i] == positiveClass) tp++;
            else if (predictions[i] != positiveClass && labels[i] != positiveClass) tn++;
            else if (predictions[i] == positiveClass && labels[i] != positiveClass) fp++;
            else if (predictions[i] != positiveClass && labels[i] == positiveClass) fn++;
        }
        double numerator = (tp * tn - fp * fn);
        double denominator = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
        return denominator == 0 ? 0.0 : numerator / denominator;
    }

    // --- Regression Metrics ---
    public static double mse(double[] predictions, double[] labels) {
        double sum = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            double diff = predictions[i] - labels[i];
            sum += diff * diff;
        }
        return sum / predictions.length;
    }

    public static double rmse(double[] predictions, double[] labels) {
        return Math.sqrt(mse(predictions, labels));
    }

    public static double mae(double[] predictions, double[] labels) {
        double sum = 0.0;
        for (int i = 0; i < predictions.length; i++) {
            sum += Math.abs(predictions[i] - labels[i]);
        }
        return sum / predictions.length;
    }
}
