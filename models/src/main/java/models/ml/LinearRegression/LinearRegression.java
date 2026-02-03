/**
 * Linear Regression Model
 *
 * <p>Supports multiple methods for linear regression: normal equation and gradient descent.
 * <p>Configurable learning rate and epochs.
 * <p>Evaluation: MSE, R² score
 *
 * @author Justice
 * @version 1.1
 */
package models.ml.LinearRegression;

public class LinearRegression {

    /**
     * Abstract Linear Regression Model
     */
    public AbstractLinearRegression lr;

    /**
     * Dataset
     */
    public double[][] dataset;

    /**
     * Points (features, labels)
     */
    public double[][] points;

    /**
     * Method for linear regression
     */
    public String method;

    /**
     * Constructor
     * 
     * @param dataset      dataset
     * @param points       points (features, labels)
     * @param method       method for linear regression
     * @param learningRate learning rate for gradient descent
     * @param epochs       number of epochs for gradient descent
     */
    public LinearRegression(double[][] dataset, double[][] points, String method, double learningRate, int epochs) {
        this.dataset = dataset;
        this.points = points;
        this.method = method.toLowerCase();
        this.lr = new AbstractLinearRegression(dataset, method, learningRate, epochs);
    }

    /**
     * Constructor
     * 
     * @param dataset dataset
     * @param points  points (features, labels)
     * @param method  method for linear regression
     */
    public LinearRegression(double[][] dataset, double[][] points, String method) {
        this(dataset, points, method, 1.0, 10);
    }

    /**
     * Constructor
     * 
     * @param dataset dataset
     * @param points  points (features, labels)
     */
    public LinearRegression(double[][] dataset, double[][] points) {
        this(dataset, points, "normal", 1.0, 10);
    }

    /**
     * Constructor
     * 
     * @param dataset      dataset
     * @param points       points (features, labels)
     * @param learningRate learning rate for gradient descent
     * @param epochs       number of epochs for gradient descent
     */
    public LinearRegression(double[][] dataset, double[][] points, double learningRate, int epochs) {
        this(dataset, points, "gradientDescent", learningRate, epochs);
    }

    /**
     * Predict for a single row
     * 
     * @param row row
     * @return prediction
     */
    public double predict(double[] row) {
        return lr.predictRow(row);
    }

    /**
     * Predict for all points
     * 
     * @return predictions
     */
    public double[] predictAll() {
        return lr.predict(points);
    }

    /**
     * Mean Squared Error (MSE)
     * 
     * @return MSE
     */
    public double mse() {
        double sum = 0.0;
        for (int i = 0; i < points.length; i++) {
            double err = predict(points[i]) - points[i][points[i].length - 1];
            sum += err * err;
        }
        return sum / points.length;
    }

    /**
     * R² score
     * 
     * @return R² score
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
