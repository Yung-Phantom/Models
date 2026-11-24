package models.ml.Model;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.DatasetHandler.helpers.DatasetSplit;
import models.ml.Preprocessing.Scaling.StandardScaler;

/**
 * Base class for all models (KNN, LinearRegression, LogisticRegression, NaiveBayes, SVM).
 * <p>
 * Provides common infrastructure for dense and sparse data handling, training/test splits via
 * {@link DatasetLoader}, serialization, and incremental learning.
 * <p>
 * <b>Data modes:</b> Models detect mode from constructors or first {@code fit()} call.
 * Dense uses {@code double[][]}; sparse uses {@code List<Map<Integer, Double>>}.
 * <p>
 * <b>Fit contract:</b> All {@code fit()} overloads validate inputs first; if validation fails,
 * previous state is unchanged. If training throws after validation, model is left unfitted.
 * <p>
 * <b>Serialization:</b> {@code save()}/{@code load()} use Java serialization.
 * Warning: version-sensitive, unsafe for untrusted files.
 * <p>
 * Subclasses must implement all abstract methods.
 *
 * @see KNN
 * @see LinearRegression
 * @see LogisticRegression
 * @see NaiveBayes
 * @see SVM
 * @since 1.0
 */
public abstract class Model {

    protected double[][] denseTrainingDataset;
    protected double[][] denseTestDataset;
    protected List<Map<Integer, Double>> sparseTrainingDataset;
    protected List<Map<Integer, Double>> sparseTestDataset;
    protected double[] trainingLabels;
    protected double[] testLabels;
    protected boolean sparse = false;
    protected DatasetLoader loader;
    protected DatasetSplit split;
    protected StandardScaler scaler = new StandardScaler();

    protected Model() {
    }

    protected Model(double[][] trainingDataset) {
        if (trainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        this.denseTrainingDataset = scaler.extractFeatures(trainingDataset);
        this.trainingLabels = scaler.extractLabels(trainingDataset);
        this.sparse = false;
    }

    protected Model(double[][] trainingDataset, double[] trainingLabels) {
        if (trainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        if (trainingLabels == null) {
            throw new IllegalArgumentException("Training labels cannot be null.");
        }
        if (trainingDataset.length != trainingLabels.length) {
            throw new IllegalArgumentException("Training dataset length must match training labels length.");
        }
        this.denseTrainingDataset = trainingDataset;
        this.trainingLabels = trainingLabels;
        this.sparse = false;
    }

    protected Model(double[][] trainingDataset, double[][] testDataset) {
        if (trainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        if (testDataset == null) {
            throw new IllegalArgumentException("Test dataset cannot be null.");
        }
        this.denseTrainingDataset = scaler.extractFeatures(trainingDataset);
        this.trainingLabels = scaler.extractLabels(trainingDataset);
        this.denseTestDataset = scaler.extractFeatures(testDataset);
        this.testLabels = scaler.extractLabels(testDataset);
        this.sparse = false;
    }

    protected Model(double[][] trainingDataset, double[] trainingLabels,
            double[][] testDataset, double[] testLabels) {
        if (trainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        if (trainingLabels == null) {
            throw new IllegalArgumentException("Training labels cannot be null.");
        }
        if (testDataset == null) {
            throw new IllegalArgumentException("Test dataset cannot be null.");
        }
        if (testLabels == null) {
            throw new IllegalArgumentException("Test labels cannot be null.");
        }
        if (trainingDataset.length != trainingLabels.length) {
            throw new IllegalArgumentException("Training dataset length must match training labels length.");
        }
        if (testDataset.length != testLabels.length) {
            throw new IllegalArgumentException("Test dataset length must match test labels length.");
        }
        this.denseTrainingDataset = trainingDataset;
        this.trainingLabels = trainingLabels;
        this.denseTestDataset = testDataset;
        this.testLabels = testLabels;
        this.sparse = false;
    }

    protected Model(List<Map<Integer, Double>> trainingDataset) {
        if (trainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        this.sparseTrainingDataset = trainingDataset;
        this.sparse = true;
    }

    protected Model(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels) {
        if (trainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        if (trainingLabels == null) {
            throw new IllegalArgumentException("Training labels cannot be null.");
        }
        if (trainingDataset.size() != trainingLabels.length) {
            throw new IllegalArgumentException("Training dataset size must match training labels length.");
        }
        this.sparseTrainingDataset = trainingDataset;
        this.trainingLabels = trainingLabels;
        this.sparse = true;
    }

    protected Model(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset) {
        if (trainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        if (testDataset == null) {
            throw new IllegalArgumentException("Test dataset cannot be null.");
        }
        this.sparseTrainingDataset = trainingDataset;
        this.sparseTestDataset = testDataset;
        this.sparse = true;
    }

    protected Model(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels) {
        if (trainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        if (trainingLabels == null) {
            throw new IllegalArgumentException("Training labels cannot be null.");
        }
        if (testDataset == null) {
            throw new IllegalArgumentException("Test dataset cannot be null.");
        }
        if (testLabels == null) {
            throw new IllegalArgumentException("Test labels cannot be null.");
        }
        if (trainingDataset.size() != trainingLabels.length) {
            throw new IllegalArgumentException("Training dataset size must match training labels length.");
        }
        if (testDataset.size() != testLabels.length) {
            throw new IllegalArgumentException("Test dataset size must match test labels length.");
        }
        this.sparseTrainingDataset = trainingDataset;
        this.trainingLabels = trainingLabels;
        this.sparseTestDataset = testDataset;
        this.testLabels = testLabels;
        this.sparse = true;
    }

    protected Model(DatasetLoader loader, int splitPercent, String splitStrategy) {
        if (loader == null) {
            throw new IllegalArgumentException("Loader cannot be null.");
        }
        if (splitStrategy == null) {
            throw new IllegalArgumentException("Split strategy cannot be null.");
        }
        this.loader = loader;
        this.split = loader.split(splitPercent, splitStrategy);
        if (split == null) {
            throw new IllegalArgumentException("Split cannot be null.");
        }
        this.denseTrainingDataset = split.train;
        this.denseTestDataset = split.test;
        if (denseTrainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        if (denseTestDataset == null) {
            throw new IllegalArgumentException("Test dataset cannot be null.");
        }
        this.sparse = false;
    }

    protected Model(DatasetLoader loader, double splitDecimal, String splitStrategy) {
        if (loader == null) {
            throw new IllegalArgumentException("Loader cannot be null.");
        }
        if (splitStrategy == null) {
            throw new IllegalArgumentException("Split strategy cannot be null.");
        }
        this.loader = loader;
        this.split = loader.split(splitDecimal, splitStrategy);
        if (split == null) {
            throw new IllegalArgumentException("Split cannot be null.");
        }
        this.denseTrainingDataset = split.train;
        this.denseTestDataset = split.test;
        if (denseTrainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        if (denseTestDataset == null) {
            throw new IllegalArgumentException("Test dataset cannot be null.");
        }
        this.sparse = false;
    }

    protected Model(DatasetLoader loader, int splitPercent) {
        if (loader == null) {
            throw new IllegalArgumentException("Loader cannot be null.");
        }
        this.loader = loader;
        this.split = loader.split(splitPercent);
        if (split == null) {
            throw new IllegalArgumentException("Split cannot be null.");
        }
        this.denseTrainingDataset = split.train;
        this.denseTestDataset = split.test;
        if (denseTrainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        if (denseTestDataset == null) {
            throw new IllegalArgumentException("Test dataset cannot be null.");
        }
        this.sparse = false;
    }

    protected Model(DatasetLoader loader, double splitDecimal) {
        if (loader == null) {
            throw new IllegalArgumentException("Loader cannot be null.");
        }
        this.loader = loader;
        this.split = loader.split(splitDecimal);
        if (split == null) {
            throw new IllegalArgumentException("Split cannot be null.");
        }
        this.denseTrainingDataset = split.train;
        this.denseTestDataset = split.test;
        if (denseTrainingDataset == null) {
            throw new IllegalArgumentException("Training dataset cannot be null.");
        }
        if (denseTestDataset == null) {
            throw new IllegalArgumentException("Test dataset cannot be null.");
        }
        this.sparse = false;
    }

    /**
     * Trains on dense data (last column = label).
     *
     * @param trainingDataset features + label matrix
     * @throws IllegalArgumentException if null or empty
     */
    public abstract void fit(double[][] trainingDataset);

    /**
     * Trains on dense data with explicit labels.
     *
     * @param trainingDataset feature matrix
     * @param trainingLabels  label vector
     * @throws IllegalArgumentException if null, empty, or length mismatch
     */
    public abstract void fit(double[][] trainingDataset, double[] trainingLabels);

    /**
     * Trains on dense data with test set (last column = label).
     *
     * @param trainingDataset feature+label matrix
     * @param testDataset     test feature+label matrix
     * @throws IllegalArgumentException if null or empty
     */
    public abstract void fit(double[][] trainingDataset, double[][] testDataset);

    /**
     * Trains on dense data with explicit labels and test set.
     *
     * @param trainingDataset feature matrix
     * @param trainingLabels  label vector
     * @param testDataset     test feature matrix
     * @param testLabels      test label vector
     * @throws IllegalArgumentException if null, empty, or length mismatch
     */
    public abstract void fit(double[][] trainingDataset, double[] trainingLabels,
            double[][] testDataset, double[] testLabels);

    /**
     * Trains on sparse data (labels must be set separately or via another overload).
     *
     * @param trainingDataset sparse feature vectors
     * @throws IllegalArgumentException if null or empty
     */
    public abstract void fit(List<Map<Integer, Double>> trainingDataset);

    /**
     * Trains on sparse data with explicit labels.
     *
     * @param trainingDataset sparse feature vectors
     * @param trainingLabels  label vector
     * @throws IllegalArgumentException if null, empty, or size mismatch
     */
    public abstract void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels);

    /**
     * Trains on sparse data with test set.
     *
     * @param trainingDataset sparse training features
     * @param testDataset     sparse test features
     * @throws IllegalArgumentException if null or empty
     */
    public abstract void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset);

    /**
     * Trains on sparse data with explicit labels and test set.
     *
     * @param trainingDataset sparse training features
     * @param trainingLabels  training label vector
     * @param testDataset     sparse test features
     * @param testLabels      test label vector
     * @throws IllegalArgumentException if null, empty, or size mismatch
     */
    public abstract void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels);

    /**
     * Trains using a {@link DatasetLoader} with split percentage and strategy.
     *
     * @param loader        dataset loader
     * @param splitPercent  training percentage (0-100) or fraction (0.0-1.0)
     * @param splitStrategy "random"/"r", "stratified"/"s", "timeseries"/"t"
     * @throws IllegalArgumentException if loader or strategy is null
     */
    public abstract void fit(DatasetLoader loader, int splitPercent, String splitStrategy);

    /**
     * Trains using a {@link DatasetLoader} with split decimal and strategy.
     *
     * @param loader        dataset loader
     * @param splitDecimal  training fraction (0.0-1.0)
     * @param splitStrategy "random"/"r", "stratified"/"s", "timeseries"/"t"
     * @throws IllegalArgumentException if loader or strategy is null
     */
    public abstract void fit(DatasetLoader loader, double splitDecimal, String splitStrategy);

    /**
     * Predicts labels for the internal test set.
     *
     * @return array of predicted label indices
     * @throws IllegalStateException if not fitted
     */
    public abstract double[] predictLabels();

    /**
     * Predicts a single dense instance.
     *
     * @param row feature vector
     * @return predicted label index
     * @throws IllegalStateException       if not fitted
     * @throws IllegalArgumentException    if row is null or wrong dimension
     */
    public abstract double predict(double[] row);

    /**
     * Predicts a single sparse instance.
     *
     * @param row sparse feature map
     * @return predicted label index
     * @throws IllegalStateException       if not fitted
     * @throws IllegalArgumentException    if row is null
     */
    public abstract double predict(Map<Integer, Double> row);

    /**
     * Returns class probabilities for the internal test set.
     *
     * @return list of probability maps (class index → probability)
     * @throws IllegalStateException if not fitted
     */
    public abstract List<Map<Integer, Double>> predictProbabilities();

    /**
     * Multi-label prediction from probabilities.
     *
     * @param threshold probability threshold
     * @return array of label index arrays per instance
     * @throws IllegalStateException if not fitted
     */
    public abstract double[][] predictMultiLabel(double threshold);

    /**
     * Evaluates on external dense data.
     *
     * @param dataset feature matrix
     * @param labels  true label vector
     * @param metrics metric names (model-specific)
     * @return metric name → value map
     * @throws IllegalArgumentException if null, empty, or length mismatch
     * @throws IllegalStateException    if not fitted
     */
    public abstract Map<String, Double> evaluate(double[][] dataset, double[] labels, String[] metrics);

    /**
     * Evaluates on internal test set.
     *
     * @param metrics metric names
     * @return metric name → value map
     * @throws IllegalStateException if not fitted or no test set
     */
    public abstract Map<String, Double> evaluate(String[] metrics);

    /**
     * Evaluates on external sparse data.
     *
     * @param dataset sparse feature vectors
     * @param labels  true label vector
     * @param metrics metric names
     * @return metric name → value map
     * @throws IllegalArgumentException if null, empty, or size mismatch
     * @throws IllegalStateException    if not fitted
     */
    public abstract Map<String, Double> evaluate(List<Map<Integer, Double>> dataset, double[] labels, String[] metrics);

    /**
     * Saves model via Java serialization.
     *
     * @param filename output file path
     * @throws RuntimeException if I/O fails
     */
    public abstract void save(String filename);

    /**
     * Loads model via Java serialization.
     *
     * @param filename input file path
     * @throws RuntimeException if I/O fails or class mismatch
     */
    public abstract void load(String filename);

    /**
     * Returns human-readable model type name.
     *
     * @return model type (e.g., "K-Nearest Neighbors")
     */
    public abstract String getModelType();

    /**
     * Returns model version.
     *
     * @return version string
     */
    public abstract String getVersion();

    /**
     * Rebuilds internal state after hyperparameter changes.
     * Call after modifying public hyperparameter fields.
     *
     * @throws IllegalArgumentException if training data not set
     */
    public abstract void refresh();

    /**
     * Incrementally updates the model with a new dense batch.
     * Merges batch into training data and calls {@link #refresh()}.
     *
     * @param batchDataset feature matrix
     * @param batchLabels  label vector
     * @throws IllegalArgumentException if null, empty, or length mismatch
     */
    public void partialFit(double[][] batchDataset, double[] batchLabels) {
        if (batchDataset == null || batchDataset.length == 0) {
            throw new IllegalArgumentException("batchDataset cannot be empty");
        }
        if (batchLabels == null || batchLabels.length != batchDataset.length) {
            throw new IllegalArgumentException("batchLabels must match batchDataset length");
        }

        // Merge new batch into existing dense dataset
        int oldSize = (denseTrainingDataset != null) ? denseTrainingDataset.length : 0;
        int newSize = oldSize + batchDataset.length;

        double[][] mergedData = new double[newSize][];
        double[] mergedLabels = new double[newSize];

        if (oldSize > 0) {
            System.arraycopy(denseTrainingDataset, 0, mergedData, 0, oldSize);
            System.arraycopy(trainingLabels, 0, mergedLabels, 0, oldSize);
        }
        System.arraycopy(batchDataset, 0, mergedData, oldSize, batchDataset.length);
        System.arraycopy(batchLabels, 0, mergedLabels, oldSize, batchLabels.length);

        this.denseTrainingDataset = mergedData;
        this.trainingLabels = mergedLabels;
        this.sparse = false;

        refresh();
    }

    /**
     * Incrementally updates the model with a new sparse batch.
     * Merges batch into training data and calls {@link #refresh()}.
     *
     * @param batchDataset sparse feature vectors
     * @param batchLabels  label vector
     * @throws IllegalArgumentException if null, empty, or size mismatch
     */
    public void partialFit(List<Map<Integer, Double>> batchDataset, double[] batchLabels) {
        if (batchDataset == null || batchDataset.isEmpty()) {
            throw new IllegalArgumentException("batchDataset cannot be empty");
        }
        if (batchLabels == null || batchLabels.length != batchDataset.size()) {
            throw new IllegalArgumentException("batchLabels must match batchDataset size");
        }

        // Merge new batch into existing sparse dataset
        if (sparseTrainingDataset == null) {
            sparseTrainingDataset = new ArrayList<>();
        }
        sparseTrainingDataset.addAll(batchDataset);

        int oldSize = (trainingLabels != null) ? trainingLabels.length : 0;
        int newSize = oldSize + batchLabels.length;

        double[] mergedLabels = new double[newSize];
        if (oldSize > 0) {
            System.arraycopy(trainingLabels, 0, mergedLabels, 0, oldSize);
        }
        System.arraycopy(batchLabels, 0, mergedLabels, oldSize, batchLabels.length);

        this.trainingLabels = mergedLabels;
        this.sparse = true;

        refresh();
    }

    /**
     * Throws if model is in sparse mode.
     *
     * @throws IllegalStateException if {@code sparse == true}
     */
    protected void requireDense() {
        if (sparse) {
            throw new IllegalStateException(
                    "This operation requires a dense model, but the model was initialized with sparse data.");
        }
    }

    /**
     * Throws if model is in dense mode.
     *
     * @throws IllegalStateException if {@code sparse == false}
     */
    protected void requireSparse() {
        if (!sparse) {
            throw new IllegalStateException(
                    "This operation requires a sparse model, but the model was initialized with dense data.");
        }
    }
}
