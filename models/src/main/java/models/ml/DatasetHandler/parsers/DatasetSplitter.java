package models.ml.DatasetHandler.parsers;

import models.ml.DatasetHandler.helpers.Dataset;
import models.ml.DatasetHandler.helpers.DatasetSplit;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * Utility class for splitting datasets into train/test sets.
 * Supports random, stratified, and time-series splitting strategies.
 */
public final class DatasetSplitter {

    private DatasetSplitter() {}

    /**
     * Splits a dataset using the specified strategy.
     *
     * @param dataset     dataset to split
     * @param trainPercent training percentage (0.0-1.0 or 0-100)
     * @param strategy    split strategy: "random"/"r", "stratified"/"s", "timeseries"/"t"
     * @param rand        random number generator for reproducible splits
     * @return dataset split with train and test sets
     */
    public static DatasetSplit split(Dataset dataset, double trainPercent, String strategy, Random rand) {
        if (strategy == null)
            strategy = "random";

        // For stratified, check if we should fall back to random based on data
        if ("stratified".equalsIgnoreCase(strategy) || "s".equalsIgnoreCase(strategy)) {
            if (hasContinuousLabels(dataset)) {
                return randomSplit(dataset, trainPercent, rand);
            }
        }

        switch (strategy.toLowerCase(Locale.ROOT)) {
            case "stratified":
            case "s":
                return stratifiedSplit(dataset, trainPercent, rand);

            case "timeseries":
            case "t":
                return timeSeriesSplit(dataset, trainPercent, rand);

            case "random":
            case "r":
            default:
                return randomSplit(dataset, trainPercent, rand);
        }
    }

    /**
     * Checks if the dataset has continuous (non-integer) labels.
     *
     * @param dataset dataset to check
     * @return true if labels are continuous, false if categorical
     */
    public static boolean hasContinuousLabels(Dataset dataset) {
        // Check original label strings from labelToIndex
        for (String labelStr : dataset.labelToIndex.keySet()) {
            try {
                double labelVal = Double.parseDouble(labelStr);
                if (labelVal != Math.floor(labelVal)) {
                    return true;
                }
            } catch (NumberFormatException e) {
                // Non-numeric label, treat as categorical
            }
        }

        // If no label column was specified, check the last data column
        if (dataset.labelToIndex.isEmpty()) {
            double[][] data = dataset.data;
            if (data.length == 0 || data[0].length == 0) {
                return false;
            }
            int labelIndex = data[0].length - 1;
            for (double[] row : data) {
                double label = row[labelIndex];
                if (label != Math.floor(label)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Performs a random split of the dataset.
     *
     * @param dataset     dataset to split
     * @param trainPercent training percentage
     * @param rand        random number generator
     * @return dataset split
     */
    public static DatasetSplit randomSplit(Dataset dataset, double trainPercent, Random rand) {
        double[][] data = dataset.data;
        int n = data.length;
        int trainSize = (trainPercent <= 1.0) ? (int) (n * trainPercent) : (int) (n * trainPercent / 100.0);

        List<double[]> shuffled = new ArrayList<>(Arrays.asList(data));
        Collections.shuffle(shuffled, rand);

        double[][] train = shuffled.subList(0, trainSize)
                .toArray(new double[0][]);

        double[][] test = shuffled.subList(trainSize, n)
                .toArray(new double[0][]);

        return new DatasetSplit(train, test);
    }

    /**
     * Performs a stratified split of the dataset (preserves class distribution).
     *
     * @param dataset     dataset to split
     * @param trainPercent training percentage
     * @param rand        random number generator
     * @return dataset split
     */
    public static DatasetSplit stratifiedSplit(Dataset dataset, double trainPercent, Random rand) {
        // If continuous labels detected, fall back to random split
        if (hasContinuousLabels(dataset)) {
            return randomSplit(dataset, trainPercent, rand);
        }

        double[][] data = dataset.data;
        int labelIndex = data[0].length - 1;

        Map<Double, List<double[]>> byLabel = new LinkedHashMap<>();

        for (double[] row : data) {
            double label = row[labelIndex];
            byLabel.computeIfAbsent(label, k -> new ArrayList<>()).add(row);
        }

        List<double[]> train = new ArrayList<>();
        List<double[]> test = new ArrayList<>();

        for (List<double[]> rows : byLabel.values()) {
            Collections.shuffle(rows, rand);
            int trainSize = (trainPercent <= 1.0) ? (int) (rows.size() * trainPercent) : (int) (rows.size() * trainPercent / 100.0);

            // Class with 1 row -> goes to train
            // Class with 2+ rows -> at least 1 in train and 1 in test
            if (rows.size() == 1) {
                trainSize = 1;
            } else if (trainSize == 0) {
                trainSize = 1;
            } else if (trainSize >= rows.size()) {
                trainSize = rows.size() - 1;
            }

            train.addAll(rows.subList(0, trainSize));
            test.addAll(rows.subList(trainSize, rows.size()));
        }

        Collections.shuffle(train, rand);
        Collections.shuffle(test, rand);

        return new DatasetSplit(
                train.toArray(new double[0][]),
                test.toArray(new double[0][]));
    }

    /**
     * Performs a time-series split of the dataset (first N% train, rest test).
     *
     * @param dataset     dataset to split
     * @param trainPercent training percentage
     * @param rand        random number generator (unused, kept for API consistency)
     * @return dataset split
     */
    public static DatasetSplit timeSeriesSplit(Dataset dataset, double trainPercent, Random rand) {
        double[][] data = dataset.data;
        int n = data.length;
        int trainSize = (trainPercent <= 1.0) ? (int) (n * trainPercent) : (int) (n * trainPercent / 100.0);

        double[][] train = Arrays.copyOfRange(data, 0, trainSize);
        double[][] test = Arrays.copyOfRange(data, trainSize, n);

        return new DatasetSplit(train, test);
    }
}