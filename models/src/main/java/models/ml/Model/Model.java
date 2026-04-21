package models.ml.Model;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.DatasetHandler.helpers.DatasetSplit;
import models.ml.Preprocessing.Scaling.StandardScaler;

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

    public abstract void fit(double[][] trainingDataset);

    public abstract void fit(double[][] trainingDataset, double[] trainingLabels);

    public abstract void fit(double[][] trainingDataset, double[][] testDataset);

    public abstract void fit(double[][] trainingDataset, double[] trainingLabels,
            double[][] testDataset, double[] testLabels);

    public abstract void fit(List<Map<Integer, Double>> trainingDataset);

    public abstract void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels);

    public abstract void fit(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset);

    public abstract void fit(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels,
            List<Map<Integer, Double>> testDataset, double[] testLabels);

    public abstract void fit(DatasetLoader loader, int splitPercent, String splitStrategy);

    public abstract void fit(DatasetLoader loader, double splitDecimal, String splitStrategy);

    public abstract int[] predictLabels();

    public abstract List<Map<Integer, Double>> predictProbabilities();

    public abstract int[][] predictMultiLabel(double threshold);

    public abstract Map<String, Double> evaluate(double[][] dataset, double[] labels, String[] metrics);

    public abstract Map<String, Double> evaluate(String[] metrics);

    public abstract Map<String, Double> evaluate(List<Map<Integer, Double>> dataset, double[] labels, String[] metrics);

    public abstract void save(String filename);

    public abstract void load(String filename);

    public abstract String getModelType();

    public abstract String getVersion();

    public abstract void refresh();

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

    protected void requireDense() {
        if (sparse) {
            throw new IllegalStateException(
                    "This operation requires a dense model, but the model was initialized with sparse data.");
        }
    }

    protected void requireSparse() {
        if (!sparse) {
            throw new IllegalStateException(
                    "This operation requires a sparse model, but the model was initialized with dense data.");
        }
    }
}
