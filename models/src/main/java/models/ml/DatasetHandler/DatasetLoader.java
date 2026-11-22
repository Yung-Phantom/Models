package models.ml.DatasetHandler;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.sql.*;
import java.util.*;

import models.ml.DatasetHandler.helpers.Dataset;
import models.ml.DatasetHandler.helpers.DatasetConfig;
import models.ml.DatasetHandler.helpers.DatasetSplit;
import models.ml.DatasetHandler.helpers.TextDataset;
import models.ml.DatasetHandler.parsers.CsvParser;
import models.ml.DatasetHandler.parsers.DatasetSplitter;
import models.ml.DatasetHandler.parsers.JsonParser;
import models.ml.DatasetHandler.parsers.SqliteLoader;
import models.ml.DatasetHandler.parsers.TableParser;
import models.ml.DatasetHandler.parsers.TextDatasetLoader;

public class DatasetLoader {

    public final Dataset dataset;

    public DatasetLoader(String path, DatasetConfig config, String labelColumn, Character delimiter)
            throws IOException, SQLException {
        String ext = getExtension(path);

        switch (ext) {
            case "csv":
            case "tsv":
            case "txt":
                dataset = loadCsv(path, config, labelColumn, delimiter);
                break;

            case "json":
                dataset = loadJson(path, config, labelColumn);
                break;

            case "sqlite":
                dataset = loadSqlite(path, labelColumn);
                break;

            default:
                throw new IOException("Unsupported format: " + ext);
        }
    }

    public DatasetLoader(String path) throws IOException, SQLException {
        this(path, DatasetConfig.DEFAULT, null, null);
    }

    public DatasetLoader(String path, DatasetConfig config)
            throws IOException, SQLException {
        this(path, config, null, null);
    }

    public DatasetLoader(String path, String labelColumn) throws IOException, SQLException {
        this(path, DatasetConfig.DEFAULT, labelColumn, null);
    }

    public DatasetLoader(String path, Character delimiter) throws IOException, SQLException {
        this(path, DatasetConfig.DEFAULT, null, delimiter);
    }

    public DatasetLoader(String path, DatasetConfig config, String labelColumn)
            throws IOException, SQLException {
        this(path, config, labelColumn, null);
    }

    public DatasetLoader(String path, DatasetConfig config, Character delimiter) throws IOException, SQLException {
        this(path, config, null, delimiter);
    }

    public DatasetLoader(String path, String labelColumn, Character delimiter) throws IOException, SQLException {
        this(path, DatasetConfig.DEFAULT, labelColumn, delimiter);
    }

    public DatasetLoader(String path, DatasetConfig config, String labelColumn, Character delimiter, boolean numericLabel)
            throws IOException, SQLException {
        String ext = getExtension(path);

        switch (ext) {
            case "csv":
            case "tsv":
            case "txt":
                dataset = loadCsv(path, config, labelColumn, delimiter, numericLabel);
                break;

            case "json":
                dataset = loadJson(path, config, labelColumn, numericLabel);
                break;

            case "sqlite":
                dataset = loadSqlite(path, labelColumn);
                break;

            default:
                throw new IOException("Unsupported format: " + ext);
        }
    }

    public Dataset getDataset() {
        return dataset;
    }

    // ==================== Split methods ====================

    public DatasetSplit split(double trainPercent) {
        return DatasetSplitter.split(dataset, trainPercent, "s", new Random());
    }

    public DatasetSplit split(double trainPercent, String strategy) {
        return DatasetSplitter.split(dataset, trainPercent, strategy, new Random());
    }

    public DatasetSplit split(double trainPercent, String strategy, long seed) {
        return DatasetSplitter.split(dataset, trainPercent, strategy, new Random(seed));
    }

    public DatasetSplit randomSplit(double trainPercent) {
        return DatasetSplitter.randomSplit(dataset, trainPercent, new Random());
    }

    public DatasetSplit randomSplit(double trainPercent, long seed) {
        return DatasetSplitter.randomSplit(dataset, trainPercent, new Random(seed));
    }

    public DatasetSplit stratifiedSplit(double trainPercent) {
        return DatasetSplitter.stratifiedSplit(dataset, trainPercent, new Random());
    }

    public DatasetSplit stratifiedSplit(double trainPercent, long seed) {
        return DatasetSplitter.stratifiedSplit(dataset, trainPercent, new Random(seed));
    }

    public DatasetSplit timeSeriesSplit(double trainPercent) {
        return DatasetSplitter.timeSeriesSplit(dataset, trainPercent, new Random());
    }

    public DatasetSplit timeSeriesSplit(double trainPercent, long seed) {
        return DatasetSplitter.timeSeriesSplit(dataset, trainPercent, new Random(seed));
    }

    // ==================== CSV Loading ====================

    public static Dataset loadCsv(String path, DatasetConfig config, String labelColumn, Character delimiter)
            throws IOException {

        return loadCsv(path, config, labelColumn, delimiter, false);
    }

    public static Dataset loadCsv(String path, DatasetConfig config, String labelColumn, Character delimiter, boolean numericLabel)
            throws IOException {

        // Use CsvParser for character-level reading (handles quoted fields with delimiters, quotes, newlines)
        List<String[]> rows = CsvParser.parse(path, delimiter);
        if (rows.isEmpty()) {
            return new Dataset(new double[0][0], new String[0], Map.of());
        }

        return TableParser.parse(rows, config, labelColumn, numericLabel);
    }

    // ==================== JSON Loading ====================

    public static Dataset loadJson(String path,
            DatasetConfig config,
            String labelColumn)
            throws IOException {

        return loadJson(path, config, labelColumn, false);
    }

    public static Dataset loadJson(String path,
            DatasetConfig config,
            String labelColumn,
            boolean numericLabel)
            throws IOException {

        try (BufferedReader reader = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
            JsonParser parser = new JsonParser(reader);
            List<Map<String, String>> objects = parser.parseArray();

            if (objects.isEmpty()) {
                return new Dataset(new double[0][0], new String[0], Map.of());
            }

            // Use first object's keys as the header (column order)
            List<String> header = new ArrayList<>(objects.get(0).keySet());

            List<String[]> rows = new ArrayList<>();
            rows.add(header.toArray(new String[0]));

            for (Map<String, String> obj : objects) {
                String[] row = new String[header.size()];
                for (int i = 0; i < header.size(); i++) {
                    String key = header.get(i);
                    row[i] = obj.getOrDefault(key, "");
                }
                rows.add(row);
            }

            return TableParser.parse(rows, config, labelColumn, numericLabel);
        }
    }

    // ==================== Text Dataset Loading ====================

    public static TextDataset loadText(String path,
            String textColumn,
            String labelColumn,
            DatasetConfig config,
            Character delimiter)
            throws IOException {

        return TextDatasetLoader.loadText(path, textColumn, labelColumn, config, delimiter);
    }

    public static TextDataset loadText(String path,
            String textColumn,
            String labelColumn,
            DatasetConfig config,
            Character delimiter,
            Map<String, Integer> labelMap)
            throws IOException {

        return TextDatasetLoader.loadText(path, textColumn, labelColumn, config, delimiter, labelMap);
    }

    // ==================== SQLite Loading ====================

    public static Dataset loadSqlite(String dbPath, String labelColumn) throws SQLException {
        if (labelColumn == null) {
            throw new IllegalStateException(
                    "No label column specified. Supervised models require labels.");
        }
        List<String[]> rows = SqliteLoader.readFirstTable(dbPath);
        return TableParser.parse(rows, DatasetConfig.NO_ID_WITH_HEADER, labelColumn, false);
    }

    // ==================== Table Parsing ====================

    public static Dataset parseTable(List<String[]> rows, DatasetConfig config, String labelColumn) {
        return TableParser.parse(rows, config, labelColumn, false);
    }

    public static Dataset parseTable(List<String[]> rows, DatasetConfig config, String labelColumn, boolean numericLabel) {
        return TableParser.parse(rows, config, labelColumn, numericLabel);
    }

    // ==================== CSV Line Parsing ====================

    public static String[] splitCsvLine(String line, Character delimiter) {
        return CsvParser.splitCsvLine(line, delimiter);
    }

    public static final Character[] COMMON_DELIMITERS = CsvParser.COMMON_DELIMITERS;

    public static Character detectDelimiter(List<String> lines) {
        return CsvParser.detectDelimiter(lines);
    }

    public static int countDelimiterOutsideQuotes(String s, Character delim) {
        return CsvParser.countDelimiterOutsideQuotes(s, delim);
    }

    public static int indexOf(String[] arr, String key) {
        return TableParser.indexOf(arr, key);
    }

    public static double parseDouble(String s) {
        return TableParser.parseDouble(s);
    }

    public static String strip(String s) {
        return TableParser.strip(s);
    }

    public static String getExtension(String path) {
        int i = path.lastIndexOf('.');
        return (i < 0) ? "" : path.substring(i + 1);
    }
}