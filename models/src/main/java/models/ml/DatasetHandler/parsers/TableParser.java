package models.ml.DatasetHandler.parsers;

import models.ml.DatasetHandler.helpers.Dataset;
import models.ml.DatasetHandler.helpers.DatasetConfig;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Utility class for parsing table data (rows of string arrays) into {@link Dataset} objects.
 * Handles header configuration, ID column skipping, label column mapping, and numeric parsing.
 */
public final class TableParser {

    private TableParser() {}

    /**
     * Parses a list of string rows into a {@link Dataset}.
     *
     * @param rows        list of string arrays, first row is header if config.hasHeader
     * @param config      configuration for header and ID column handling
     * @param labelColumn name of the label column (case-insensitive), or null for no label
     * @param numericLabel whether the label column contains numeric values (regression)
     * @return parsed dataset with data matrix, header, and label mapping
     * @throws IllegalArgumentException if label column not found or non-numeric label with numericLabel=true
     */
    public static Dataset parse(List<String[]> rows, DatasetConfig config, String labelColumn, boolean numericLabel) {

        String[] header;
        if (config.hasHeader) {
            header = rows.get(0);
        } else {
            header = new String[0];
        }

        int labelIdx;
        if (labelColumn == null) {
            labelIdx = -1;
        } else {
            labelIdx = indexOf(header, labelColumn);
        }

        if (labelColumn != null && labelIdx == -1) {
            throw new IllegalArgumentException(
                    "Label column '" + labelColumn + "' not found in header: " +
                            Arrays.toString(header));
        }

        Map<String, Integer> labelMap = new LinkedHashMap<>();
        java.util.List<double[]> data = new java.util.ArrayList<>();

        int r = 0;
        if (config.hasHeader) {
            r = 1;
        }
        for (int i = r; i < rows.size(); i++) {
            String[] row = rows.get(i);
            java.util.List<Double> nums = new java.util.ArrayList<>();

            for (int c = 0; c < row.length; c++) {
                if (config.hasID && c == 0)
                    continue;
                if (c == labelIdx)
                    continue;
                nums.add(parseDouble(row[c]));
            }

            if (labelIdx >= 0) {
                String label = row[labelIdx];
                if (numericLabel) {
                    // Numeric label: keep parsed double value, labelToIndex stays empty
                    try {
                        double labelVal = Double.parseDouble(label.trim());
                        nums.add(labelVal);
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException("Non-numeric label at row " + (i + 1) + ": " + label);
                    }
                } else {
                    // Categorical label: map to index
                    labelMap.putIfAbsent(label, labelMap.size());
                    nums.add(labelMap.get(label).doubleValue());
                }
            }

            data.add(nums.stream().mapToDouble(Double::doubleValue).toArray());
        }

        return new Dataset(
                data.toArray(new double[0][]),
                header,
                labelMap);
    }

    /**
     * Case-insensitive search for a key in a string array.
     *
     * @param arr array to search
     * @param key key to find (case-insensitive)
     * @return index of key, or -1 if not found
     */
    public static int indexOf(String[] arr, String key) {
        for (int i = 0; i < arr.length; i++)
            if (arr[i].equalsIgnoreCase(key))
                return i;
        return -1;
    }

    /**
     * Parses a string to double, returning NaN on failure.
     *
     * @param s string to parse
     * @return parsed double or Double.NaN
     */
    public static double parseDouble(String s) {
        try {
            return Double.parseDouble(s.trim());
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    /**
     * Strips whitespace and surrounding quotes from a string.
     *
     * @param s string to strip
     * @return stripped string
     */
    public static String strip(String s) {
        return s.trim().replace("\"", "");
    }
}