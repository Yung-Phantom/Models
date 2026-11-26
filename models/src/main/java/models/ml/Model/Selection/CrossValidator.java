package models.ml.Model.Selection;

import models.ml.Model.Model;
import models.ml.Model.Metrics;
import models.ml.Preprocessing.Scaling.Transformer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * K-fold cross-validation utility for {@link Model} instances.
 * <p>
 * Supports both stratified (classification) and random (regression) splitting.
 * Each fold uses a fresh model instance from a {@code Supplier<Model>}.
 * Optional leakage-safe preprocessing via {@link Transformer} fitted on each fold's training data only.
 *
 * @since 1.0
 */
public final class CrossValidator {

    private CrossValidator() {}

    /**
     * Runs k-fold cross-validation on the given data.
     *
     * @param modelFactory   supplier that provides a fresh, unfitted model instance for each fold
     * @param X              feature matrix (n samples × m features), must not be null, empty, or ragged
     * @param y              label vector (n samples), must not be null, empty, or contain NaN
     * @param k              number of folds, must satisfy 2 <= k <= n
     * @param seed           random seed for reproducible fold assignment
     * @param stratified     if true, use stratified splitting (classification only);
     *                       if false, use random shuffle split (regression or classification)
     * @param metrics        metric names to compute (e.g., "accuracy", "mse", "rmse", "mae", "r2",
     *                       "precision", "recall", "f1score"); must not be null or empty
     * @return cross-validation result with per-fold metrics, means, and standard deviations
     * @throws IllegalArgumentException if any input validation fails
     */
    public static CrossValidationResult run(Supplier<Model> modelFactory,
                                             double[][] X, double[] y, int k, long seed,
                                             boolean stratified, String[] metrics) {
        return runInternal(modelFactory, X, y, k, seed, stratified, metrics, null);
    }

    /**
     * Runs k-fold cross-validation with leakage-safe preprocessing.
     * <p>
     * For each fold, a fresh transformer is created from {@code transformerFactory}, fitted on the
     * fold's training rows only, then applied to both training and test rows before model fitting
     * and evaluation.
     *
     * @param modelFactory       supplier that provides a fresh, unfitted model instance for each fold
     * @param X                  feature matrix (n samples × m features)
     * @param y                  label vector (n samples)
     * @param k                  number of folds
     * @param seed               random seed
     * @param stratified         stratified splitting flag
     * @param metrics            metric names to compute
     * @param transformerFactory supplier for a fresh transformer instance per fold
     * @return cross-validation result
     * @throws IllegalArgumentException if any input validation fails
     */
    public static CrossValidationResult run(Supplier<Model> modelFactory,
                                             double[][] X, double[] y, int k, long seed,
                                             boolean stratified, String[] metrics,
                                             Supplier<Transformer> transformerFactory) {
        return runInternal(modelFactory, X, y, k, seed, stratified, metrics, transformerFactory);
    }

    private static CrossValidationResult runInternal(Supplier<Model> modelFactory,
                                                      double[][] X, double[] y, int k, long seed,
                                                      boolean stratified, String[] metrics,
                                                      Supplier<Transformer> transformerFactory) {
        // --- Input validation ---
        validateInputs(X, y, k, stratified, metrics);

        int n = X.length;

        // Determine if labels are continuous (regression) or categorical (classification)
        boolean isRegression = hasContinuousLabels(y);

        // Stratified mode requires classification
        if (stratified && isRegression) {
            throw new IllegalArgumentException(
                    "Stratified cross-validation requires categorical labels; " +
                    "regression labels appear continuous. Use stratified=false.");
        }

        // --- Generate fold indices ---
        List<int[]> testFolds = generateFolds(y, k, seed, stratified, isRegression);

        // --- Run each fold ---
        Map<String, List<Double>> foldMetrics = new LinkedHashMap<>();
        for (String metric : metrics) {
            foldMetrics.put(metric, new ArrayList<>(k));
        }

        for (int fold = 0; fold < k; fold++) {
            int[] testIdx = testFolds.get(fold);
            int[] trainIdx = buildTrainIndices(n, testIdx);

            // Extract train/test data
            double[][] XTrain = extractRows(X, trainIdx);
            double[] yTrain = extractRows(y, trainIdx);
            double[][] XTest = extractRows(X, testIdx);
            double[] yTest = extractRows(y, testIdx);

            // Optional preprocessing
            if (transformerFactory != null) {
                Transformer transformer = transformerFactory.get();
                transformer.fit(XTrain);
                XTrain = transformer.transform(XTrain);
                XTest = transformer.transform(XTest);
            }

            // Fresh model for this fold
            Model model = modelFactory.get();

            // Fit on train, evaluate on test
            model.fit(XTrain, yTrain, XTest, yTest);
            Map<String, Double> results = model.evaluate(metrics);

            // Collect metrics
            for (String metric : metrics) {
                Double value = results.get(metric);
                if (value == null || Double.isNaN(value) || Double.isInfinite(value)) {
                    throw new IllegalStateException(
                            "Metric '" + metric + "' returned null/NaN/Infinite in fold " + (fold + 1));
                }
                foldMetrics.get(metric).add(value);
            }
        }

        return new CrossValidationResult(foldMetrics);
    }

    /**
     * Validates all inputs before running any folds.
     */
    private static void validateInputs(double[][] X, double[] y, int k, boolean stratified, String[] metrics) {
        if (X == null) {
            throw new IllegalArgumentException("X cannot be null");
        }
        if (y == null) {
            throw new IllegalArgumentException("y cannot be null");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("X cannot be empty");
        }
        if (y.length == 0) {
            throw new IllegalArgumentException("y cannot be empty");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X length (" + X.length + ") must match y length (" + y.length + ")");
        }

        // Check ragged rows
        int numFeatures = X[0].length;
        if (numFeatures == 0) {
            throw new IllegalArgumentException("X must have at least one feature column");
        }
        for (int i = 0; i < X.length; i++) {
            if (X[i] == null) {
                throw new IllegalArgumentException("X[" + i + "] cannot be null");
            }
            if (X[i].length != numFeatures) {
                throw new IllegalArgumentException("X[" + i + "] has " + X[i].length +
                        " features, expected " + numFeatures + " (ragged row)");
            }
        }

        // Check NaN labels
        for (int i = 0; i < y.length; i++) {
            if (Double.isNaN(y[i])) {
                throw new IllegalArgumentException("y[" + i + "] is NaN");
            }
        }

        int n = X.length;
        if (k < 2 || k > n) {
            throw new IllegalArgumentException("k must satisfy 2 <= k <= n (n=" + n + ", k=" + k + ")");
        }

        if (metrics == null || metrics.length == 0) {
            throw new IllegalArgumentException("metrics cannot be null or empty");
        }
        for (String m : metrics) {
            if (m == null || m.isBlank()) {
                throw new IllegalArgumentException("metric names cannot be null or blank");
            }
        }

        // Stratified validation: each class must have at least k samples
        if (stratified) {
            Map<Double, Integer> classCounts = new HashMap<>();
            for (double label : y) {
                classCounts.merge(label, 1, Integer::sum);
            }
            for (Map.Entry<Double, Integer> entry : classCounts.entrySet()) {
                if (entry.getValue() < k) {
                    throw new IllegalArgumentException(
                            "Class " + entry.getKey() + " has only " + entry.getValue() +
                            " sample(s), but k=" + k + " folds require at least " + k +
                            " samples per class for stratified splitting");
                }
            }
        }
    }

    /**
     * Checks if labels appear to be continuous (regression) vs categorical (classification).
     * Uses the same logic as DatasetSplitter.hasContinuousLabels.
     */
    private static boolean hasContinuousLabels(double[] y) {
        for (double label : y) {
            if (label != Math.floor(label)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Generates fold indices for k-fold cross-validation.
     *
     * @param y           labels
     * @param k           number of folds
     * @param seed        random seed
     * @param stratified  whether to use stratified splitting
     * @param isRegression whether the problem is regression
     * @return list of test index arrays for each fold
     */
    private static List<int[]> generateFolds(double[] y, int k, long seed, boolean stratified, boolean isRegression) {
        int n = y.length;
        Random rand = new Random(seed);

        if (!stratified || isRegression) {
            // Random shuffle split
            Integer[] indices = IntStream.range(0, n).boxed().toArray(Integer[]::new);
            // Fisher-Yates shuffle
            for (int i = n - 1; i > 0; i--) {
                int j = rand.nextInt(i + 1);
                Integer tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
            return splitIndicesIntoFolds(Arrays.stream(indices).mapToInt(i -> i).toArray(), k);
        } else {
            // Stratified: group by class, shuffle within each class, then distribute
            Map<Double, List<Integer>> byClass = new LinkedHashMap<>();
            for (int i = 0; i < n; i++) {
                byClass.computeIfAbsent(y[i], key -> new ArrayList<>()).add(i);
            }

            // Shuffle within each class
            for (List<Integer> indices : byClass.values()) {
                Collections.shuffle(indices, rand);
            }

            // Distribute each class's indices across folds as evenly as possible
            List<List<Integer>> foldLists = new ArrayList<>(k);
            for (int i = 0; i < k; i++) {
                foldLists.add(new ArrayList<>());
            }

            for (List<Integer> classIndices : byClass.values()) {
                int m = classIndices.size();
                int baseSize = m / k;
                int extra = m % k;
                int start = 0;
                for (int fold = 0; fold < k; fold++) {
                    int size = baseSize + (fold < extra ? 1 : 0);
                    for (int j = 0; j < size; j++) {
                        foldLists.get(fold).add(classIndices.get(start + j));
                    }
                    start += size;
                }
            }

            // Convert to arrays and shuffle fold order for randomness
            List<int[]> folds = new ArrayList<>(k);
            for (List<Integer> foldIndices : foldLists) {
                int[] arr = foldIndices.stream().mapToInt(i -> i).toArray();
                // Shuffle within fold to avoid ordering artifacts
                for (int i = arr.length - 1; i > 0; i--) {
                    int j = rand.nextInt(i + 1);
                    int tmp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = tmp;
                }
                folds.add(arr);
            }

            // Shuffle fold order
            Collections.shuffle(folds, rand);
            return folds;
        }
    }

    /**
     * Splits a shuffled index array into k folds with sizes differing by at most 1.
     */
    private static List<int[]> splitIndicesIntoFolds(int[] shuffledIndices, int k) {
        int n = shuffledIndices.length;
        int baseSize = n / k;
        int extra = n % k;
        List<int[]> folds = new ArrayList<>(k);
        int start = 0;
        for (int fold = 0; fold < k; fold++) {
            int size = baseSize + (fold < extra ? 1 : 0);
            int[] foldIndices = Arrays.copyOfRange(shuffledIndices, start, start + size);
            folds.add(foldIndices);
            start += size;
        }
        return folds;
    }

    /**
     * Builds training indices as complement of test indices.
     */
    private static int[] buildTrainIndices(int n, int[] testIdx) {
        boolean[] isTest = new boolean[n];
        for (int idx : testIdx) {
            isTest[idx] = true;
        }
        int trainCount = n - testIdx.length;
        int[] trainIdx = new int[trainCount];
        int pos = 0;
        for (int i = 0; i < n; i++) {
            if (!isTest[i]) {
                trainIdx[pos++] = i;
            }
        }
        return trainIdx;
    }

    /**
     * Extracts rows from a 2D array.
     */
    private static double[][] extractRows(double[][] data, int[] indices) {
        double[][] result = new double[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            result[i] = Arrays.copyOf(data[indices[i]], data[indices[i]].length);
        }
        return result;
    }

    /**
     * Extracts elements from a 1D array.
     */
    private static double[] extractRows(double[] data, int[] indices) {
        double[] result = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            result[i] = data[indices[i]];
        }
        return result;
    }
}