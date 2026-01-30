package models.ml.LinearRegression;

/**
 * Linear Regression model.
 * 
 * Supports:
 * - Normal Equation
 * - Gradient Descent
 * 
 * @author Kotei Justice
 * @version 1.0
 */
public class LinearRegression {

    private AbstractLinearRegression lr;
    private double[][] dataset;
    private double[][] points;
    private String method;

    public LinearRegression(double[][] dataset, double[][] points) {
        this(dataset, points, "normal", 1, 10);
    }

    public LinearRegression(double[][] dataset, double[][] points, String method) {
        this(dataset, points, method, 1, 10);
    }

    public LinearRegression(double[][] dataset, double[][] points, double learningRate, int epochs) {
        this(dataset, points, "normal", learningRate, epochs);
    }

    public LinearRegression(double[][] dataset, double[][] points, String method, double learningRate, int epochs) {
        this.dataset = dataset;
        this.points = points;
        this.method = method;
        this.lr = new AbstractLinearRegression(dataset, method, learningRate, epochs);
    }

    public double[] predictAll() {
        double[] preds = new double[points.length];
        for (int i = 0; i < points.length; i++) {
            preds[i] = predict(points[i]);
        }
        return preds;
    }

    public double predict(double[] row) {
        double[] w = lr.getWeights();
        double y = w[0];
        for (int j = 0; j < row.length - 1; j++) {
            y += w[j + 1] * row[j];
        }
        return y;
    }

    public double mse() {
        double sum = 0.0;
        for (int i = 0; i < points.length; i++) {
            double err = predict(points[i]) - points[i][points[i].length - 1];
            sum += err * err;
        }
        return sum / points.length;
    }

    /**
     * R² score.
     */
    public double r2() {
        double mean = 0.0;
        for (double[] row : points)
            mean += row[row.length - 1];
        mean /= points.length;

        double ssTot = 0.0, ssRes = 0.0;
        for (double[] row : points) {
            double y = row[row.length - 1];
            double yHat = predict(row);
            ssTot += (y - mean) * (y - mean);
            ssRes += (y - yHat) * (y - yHat);
        }
        return 1.0 - (ssRes / ssTot);
    }
}
