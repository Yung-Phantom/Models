package models.ml.Preprocessing.Scaling;

/**
 * Utility class for splitting datasets into features and labels.
 * <p>
 * Provides methods to extract feature matrices and label arrays from combined
 * datasets where the last column represents the label/target.
 * <p>
 * This class provides an alternative to the feature/label splitting functionality
 * in {@link StandardScaler}.
 */
public final class FeatureLabelSplitter {

    private FeatureLabelSplitter() {}

    /**
     * Extracts the feature matrix from a dataset where the last column is the label.
     *
     * @param dataset the input dataset (rows x (features + 1))
     * @return feature matrix (rows x features)
     * @throws IllegalArgumentException if dataset is null, empty, or has fewer than 2 columns
     */
    public static double[][] extractFeatures(double[][] dataset) {
        if (dataset == null || dataset.length == 0) {
            throw new IllegalArgumentException("Dataset cannot be null or empty");
        }
        if (dataset[0] == null || dataset[0].length < 2) {
            throw new IllegalArgumentException("Dataset must have at least one feature and one label column.");
        }

        int rows = dataset.length;
        int cols = dataset[0].length - 1;
        double[][] features = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            if (dataset[i] == null || dataset[i].length != dataset[0].length) {
                throw new IllegalArgumentException("All rows must have the same number of columns");
            }
            System.arraycopy(dataset[i], 0, features[i], 0, cols);
        }
        return features;
    }

    /**
     * Extracts the label array from a dataset where the last column is the label.
     * Labels are validated to be integers (for classification).
     *
     * @param dataset the input dataset (rows x (features + 1))
     * @return label array (rows)
     * @throws IllegalArgumentException if dataset is null, empty, has fewer than 2 columns,
     *         or labels are not integers
     */
    public static double[] extractLabels(double[][] dataset) {
        if (dataset == null || dataset.length == 0) {
            throw new IllegalArgumentException("Dataset cannot be null or empty");
        }
        if (dataset[0] == null || dataset[0].length < 2) {
            throw new IllegalArgumentException("Dataset must have at least one feature and one label column.");
        }

        int rows = dataset.length;
        double[] labels = new double[rows];
        for (int i = 0; i < rows; i++) {
            if (dataset[i] == null || dataset[i].length != dataset[0].length) {
                throw new IllegalArgumentException("All rows must have the same number of columns");
            }
            labels[i] = dataset[i][dataset[0].length - 1];
            if (labels[i] != (int) labels[i]) {
                throw new IllegalArgumentException("Labels must be integers");
            }
        }
        return labels;
    }

    /**
     * Extracts the label array from a dataset where the last column is the label.
     * Does not validate label types (for regression).
     *
     * @param dataset the input dataset (rows x (features + 1))
     * @return label array (rows)
     * @throws IllegalArgumentException if dataset is null, empty, or has fewer than 2 columns
     */
    public static double[] extractLabelsRegression(double[][] dataset) {
        if (dataset == null || dataset.length == 0) {
            throw new IllegalArgumentException("Dataset cannot be null or empty");
        }
        if (dataset[0] == null || dataset[0].length < 2) {
            throw new IllegalArgumentException("Dataset must have at least one feature and one label column.");
        }

        int rows = dataset.length;
        double[] labels = new double[rows];
        for (int i = 0; i < rows; i++) {
            if (dataset[i] == null || dataset[i].length != dataset[0].length) {
                throw new IllegalArgumentException("All rows must have the same number of columns");
            }
            labels[i] = dataset[i][dataset[0].length - 1];
        }
        return labels;
    }
}