package models.ml.LogisticRegression;

public class LogisticRegression {
    /** The dataset used for training the logistic regression model. */
    public double[][] dataset;

    /** The points (query data) used for prediction and evaluation. */
    public double[][] points;

    /**
     * The underlying AbstractLogisticRegression instance that performs training.
     */
    public AbstractLogisticRegression lr;

    /**
     * Constructor for Logistic Regression with full specification.
     *
     * @param dataset      The dataset to fit the model to.
     * @param points       The query points to predict.
     * @param method       The regression method ("binary",
     *                     "multinomial","ordinal").
     * @param learningRate The learning rate for gradient descent.
     * @param epochs       The number of epochs for gradient descent.
     */
    public LogisticRegression(double[][] dataset, double[][] points, String method, double learningRate, int epochs) {
        this.dataset = dataset;
        this.points = points;
        this.lr = new AbstractLogisticRegression(dataset, method.toLowerCase(), learningRate, epochs);
    }

    /**
     * Constructor for Logistic Regression using binary classification.
     * Defaults to learningRate = 0.01 and epochs = 1000.
     * 
     * @param dataset The dataset to fit the model to.
     * @param points  The query points to predict.
     */
    public LogisticRegression(double[][] dataset, double[][] points) {
        this(dataset, points, "binary", 0.01, 1000);
    }

    /**
     * Constructor for Logistic Regression with specified method.
     * Defaults to learningRate = 0.01 and epochs = 1000.
     * 
     * @param dataset The dataset to fit the model to.
     * @param points  The query points to predict.
     * @param method  The regression method ("binary", "multinomial", "ordinal").
     */
    public LogisticRegression(double[][] dataset, double[][] points, String method) {
        this(dataset, points, method, 0.01, 1000);
    }

    /**
     * Constructor for Logistic Regression using binary.
     *
     * @param dataset      The dataset to fit the model to.
     * @param points       The query points to predict.
     * @param learningRate The learning rate for binary.
     * @param epochs       The number of epochs for binary.
     */
    public LogisticRegression(double[][] dataset, double[][] points, double learningRate, int epochs) {
        this(dataset, points, "binary", learningRate, epochs);
    }

    /**
     * Predicts the class label for a given query point. * Uses a threshold of 0.5
     * on the predicted probability.
     * 
     * @param i Index of the query point in the points array.
     * @return Predicted class label (0 or 1).
     */
    public int predict(int i) {
        return lr.predictClass(points[i]);
    }

    /**
     * Predicts class labels for all query points.
     *
     * @return Array of predicted class labels (0 or 1).
     */
    public int[] predictAll() {
        int[] preds = new int[points.length];
        for (int i = 0; i < points.length; i++) {
            preds[i] = predict(i);
        }
        return preds;
    }

    /**
     * Computes the accuracy of the logistic regression model.
     * Accuracy is the proportion of correctly predicted labels.
     *
     * @return Accuracy value between 0 and 1.
     */
    public double accuracy() {
        int correct = 0;
        for (int i = 0; i < points.length; i++) {
            double p = predict(i);
            int yHat = p >= 0.5 ? 1 : 0;
            if (yHat == (int) points[i][points[i].length - 1])
                correct++;
        }
        return (double) correct / points.length;
    }

    /**
     * Computes the Mean Squared Error (MSE) of the logistic regression model.
     * MSE is the average squared difference between predicted and actual labels.
     *
     * @return MSE value.
     */
    public double mse() {
        double sum = 0.0;
        for (int i = 0; i < points.length; i++) {
            double error = predict(i) - points[i][points[i].length - 1];
            sum += error * error;
        }
        return sum / points.length;
    }

    /**
     * Computes the R² (coefficient of determination) of the logistic regression
     * model.
     * R² measures how well the predictions approximate the actual labels.
     *
     * @return R² value between -∞ and 1, where 1 indicates a perfect fit.
     */
    public double r2() {
        double mean = 0.0;
        for (double[] row : points)
            mean += row[row.length - 1];
        mean /= points.length;

        double ssTot = 0.0, ssRes = 0.0;
        for (int i = 0; i < points.length; i++) {
            double y = points[i][points[i].length - 1];
            double yHat = predict(i);
            ssTot += (y - mean) * (y - mean);
            ssRes += (y - yHat) * (y - yHat);
        }
        return 1.0 - (ssRes / ssTot);
    }
}
