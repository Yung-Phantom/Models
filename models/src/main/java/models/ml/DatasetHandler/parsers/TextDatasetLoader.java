package models.ml.DatasetHandler.parsers;

import models.ml.DatasetHandler.helpers.DatasetConfig;
import models.ml.DatasetHandler.helpers.TextDataset;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Utility class for loading text datasets from CSV, TSV, TXT, or JSON files.
 * Supports label mapping for consistent train/test label encoding.
 */
public final class TextDatasetLoader {

    private TextDatasetLoader() {}

    /**
     * Loads a text dataset from a file.
     *
     * @param path        path to the data file (csv, tsv, txt, or json)
     * @param textColumn  name of the column containing text data
     * @param labelColumn name of the column containing labels
     * @param config      dataset configuration (header/ID handling)
     * @param delimiter   explicit delimiter for CSV/TSV, or null for auto-detection
     * @return text dataset with texts, labels, and label mapping
     * @throws IOException if file cannot be read or format is unsupported
     */
    public static TextDataset loadText(String path,
            String textColumn,
            String labelColumn,
            DatasetConfig config,
            Character delimiter)
            throws IOException {

        return loadText(path, textColumn, labelColumn, config, delimiter, null);
    }

    /**
     * Loads a text dataset from a file with a predefined label mapping.
     *
     * @param path        path to the data file (csv, tsv, txt, or json)
     * @param textColumn  name of the column containing text data
     * @param labelColumn name of the column containing labels
     * @param config      dataset configuration (header/ID handling)
     * @param delimiter   explicit delimiter for CSV/TSV, or null for auto-detection
     * @param labelMap    predefined label to index mapping (for test data)
     * @return text dataset with texts, labels, and label mapping
     * @throws IOException                    if file cannot be read or format is unsupported
     * @throws IllegalArgumentException       if columns not found or unseen label in test data
     */
    public static TextDataset loadText(String path,
            String textColumn,
            String labelColumn,
            DatasetConfig config,
            Character delimiter,
            Map<String, Integer> labelMap)
            throws IOException {

        String ext = getExtension(path);
        List<Map<String, String>> records;

        switch (ext) {
            case "csv":
            case "tsv":
            case "txt":
                records = loadTextFromCsv(path, config, delimiter);
                break;
            case "json":
                records = loadTextFromJson(path);
                break;
            default:
                throw new IOException("Unsupported format for text dataset: " + ext);
        }

        if (records.isEmpty()) {
            return new TextDataset(new ArrayList<>(), new double[0], new LinkedHashMap<>(), new String[0]);
        }

        // Validate columns exist
        Set<String> availableColumns = records.get(0).keySet();
        if (!availableColumns.contains(textColumn)) {
            throw new IllegalArgumentException("Text column '" + textColumn + "' not found in header: " + availableColumns);
        }
        if (!availableColumns.contains(labelColumn)) {
            throw new IllegalArgumentException("Label column '" + labelColumn + "' not found in header: " + availableColumns);
        }

        // Build label mapping
        Map<String, Integer> finalLabelMap;
        if (labelMap != null) {
            finalLabelMap = new LinkedHashMap<>(labelMap);
        } else {
            finalLabelMap = new LinkedHashMap<>();
        }

        List<String> texts = new ArrayList<>();
        List<Double> labelsList = new ArrayList<>();

        for (Map<String, String> record : records) {
            String text = record.get(textColumn);
            String labelStr = record.get(labelColumn);

            // Skip rows with empty text
            if (text == null || text.trim().isEmpty()) {
                continue;
            }

            if (labelStr == null) {
                continue; // Skip rows with missing label
            }

            Integer labelIdx;
            if (labelMap != null) {
                // Reuse training label mapping - unseen labels throw
                labelIdx = finalLabelMap.get(labelStr);
                if (labelIdx == null) {
                    throw new IllegalArgumentException("Unseen label in test data: " + labelStr);
                }
            } else {
                // Build new label mapping
                labelIdx = finalLabelMap.get(labelStr);
                if (labelIdx == null) {
                    labelIdx = finalLabelMap.size();
                    finalLabelMap.put(labelStr, labelIdx);
                }
            }

            texts.add(text);
            labelsList.add(labelIdx.doubleValue());
        }

        double[] labels = labelsList.stream().mapToDouble(Double::doubleValue).toArray();
        String[] header = availableColumns.toArray(new String[0]);

        return new TextDataset(texts, labels, finalLabelMap, header);
    }

    private static List<Map<String, String>> loadTextFromCsv(String path, DatasetConfig config, Character delimiter) throws IOException {
        List<String[]> rows = CsvParser.parse(path, delimiter);
        if (rows.isEmpty()) {
            return Collections.emptyList();
        }

        String[] header;
        int startRow;
        if (config.hasHeader) {
            header = rows.get(0);
            startRow = 1;
        } else {
            header = new String[rows.get(0).length];
            for (int i = 0; i < header.length; i++) {
                header[i] = "col" + i;
            }
            startRow = 0;
        }

        List<Map<String, String>> records = new ArrayList<>();
        for (int i = startRow; i < rows.size(); i++) {
            String[] row = rows.get(i);
            if (row.length != header.length) {
                continue; // Skip malformed rows
            }
            Map<String, String> record = new LinkedHashMap<>();
            for (int j = 0; j < header.length; j++) {
                record.put(header[j], row[j]);
            }
            records.add(record);
        }
        return records;
    }

    private static List<Map<String, String>> loadTextFromJson(String path) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
            JsonParser parser = new JsonParser(reader);
            return parser.parseArray();
        }
    }

    private static String getExtension(String path) {
        int i = path.lastIndexOf('.');
        return (i < 0) ? "" : path.substring(i + 1);
    }
}