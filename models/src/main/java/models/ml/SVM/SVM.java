package models.ml.SVM;

/**
 * Wrapper class for Support Vector Machine (SVM).
 * Handles predictions and evaluation metrics.
 * 
 * @author Justice
 * @version 1.1
 */
public class SVM {

    public double[][] train;
    public double[][] test;
    public AbstractSVM svm;

    public SVM(double[][] train, double[][] test,
            double C, double learningRate, int epochs,
            String kernel, String method) {

        this.train = normalizeLabels(train);
        this.test = normalizeLabels(test);

        this.svm = new AbstractSVM(this.train, C, learningRate, epochs, kernel, method);
    }

    public SVM(double[][] train2, double[][] test2) {
        this(train2, test2, 0.1, 0.01, 10, "linearKernel", "svc");
    }

    /** Predict label for a single query point */
    public int predict(int queryIndex) {
        double[] x = new double[test[queryIndex].length - 1];
        System.arraycopy(test[queryIndex], 0, x, 0, x.length);

        if (x.length != svm.getWeights().length) {
            throw new IllegalStateException("Feature size mismatch");
        }

        return svm.predict(x);
    }

    /** Predict labels for all query points */
    public int[] predictAll() {
        int[] preds = new int[test.length];
        for (int i = 0; i < test.length; i++) {
            preds[i] = predict(i);
        }
        return preds;
    }

    /** Compute accuracy on test points */
    public double accuracy() {
        int correct = 0;
        for (int i = 0; i < test.length; i++) {
            int predicted = predict(i);
            int actual = (int) test[i][test[i].length - 1];
            if (predicted == actual)
                correct++;
        }
        return (double) correct / test.length;
    }

    private double[][] normalizeLabels(double[][] data) {
        double[][] normalized = new double[data.length][data[0].length];

        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, normalized[i], 0, data[i].length);

            double label = normalized[i][normalized[i].length - 1];
            normalized[i][normalized[i].length - 1] = (label == 0) ? -1 : 1;
        }
        return normalized;
    }
}
