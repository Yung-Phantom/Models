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

/**
 * Loads tabular datasets from CSV, TSV, TXT, JSON, or SQLite files.
 * <p>
 * Supports automatic format detection by extension:
 * <ul>
 *   <li>{@code .csv}, {@code .tsv}, {@code .txt} → CSV parser (quoted fields, auto-delimiter)</li>
 *   <li>{@code .json} → JSON array of objects</li>
 *   <li>{@code .sqlite} → First table in database (requires {@code labelColumn})</li>
 * </ul>
 * Constructors accept {@link DatasetConfig} for header/ID handling and optional {@code labelColumn}.
 * <p>
 * Example:
 * <pre>{@code
 * DatasetLoader loader = new DatasetLoader("data.csv", DatasetConfig.HAS_ID_WITH_HEADER, "target");
 * DatasetSplit split = loader.split(80, "stratified");
 * }</pre>
 *
 * @see DatasetConfig
 * @see DatasetSplit
 * @see TextDatasetLoader
 * @see ImageDatasetLoader
 * @see AudioDatasetLoader
 * @since 1.0
 */
public class DatasetLoader {

    public final Dataset dataset;

    /**
     * Full constructor with all options.
     *
     * @param path          file path; extension determines loader
     * @param config        header/ID configuration; must not be null
     * @param labelColumn   label column name; null = last column
     * @param delimiter     custom delimiter; null = auto-detect
     * @throws IOException        if file cannot be read or format unsupported
     * @throws SQLException       if SQLite load fails
     */
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

    /**
     * Minimal constructor: default config, last column as label.
     *
     * @param path file path
     * @throws IOException  if file cannot be read
     * @throws SQLException if SQLite load fails
     */
    public DatasetLoader(String path) throws IOException, SQLException {
        this(path, DatasetConfig.DEFAULT, null, null);
    }

    /**
     * Constructor with config only.
     *
     * @param path   file path
     * @param config header/ID configuration
     * @throws IOException  if file cannot be read
     * @throws SQLException if SQLite load fails
     */
    public DatasetLoader(String path, DatasetConfig config)
            throws IOException, SQLException {
        this(path, config, null, null);
    }

    /**
     * Constructor with label column name.
     *
     * @param path         file path
     * @param labelColumn  label column name; null = last column
     * @throws IOException  if file cannot be read
     * @throws SQLException if SQLite load fails
     */
    public DatasetLoader(String path, String labelColumn) throws IOException, SQLException {
        this(path, DatasetConfig.DEFAULT, labelColumn, null);
    }

    /**
     * Constructor with custom delimiter.
     *
     * @param path      file path
     * @param delimiter custom delimiter character
     * @throws IOException  if file cannot be read
     * @throws SQLException if SQLite load fails
     */
    public DatasetLoader(String path, Character delimiter) throws IOException, SQLException {
        this(path, DatasetConfig.DEFAULT, null, delimiter);
    }

    /**
     * Constructor with config and label column.
     *
     * @param path         file path
     * @param config       header/ID configuration
     * @param labelColumn  label column name; null = last column
     * @throws IOException  if file cannot be read
     * @throws SQLException if SQLite load fails
     */
    public DatasetLoader(String path, DatasetConfig config, String labelColumn)
            throws IOException, SQLException {
        this(path, config, labelColumn, null);
    }

    /**
     * Constructor with config and delimiter.
     *
     * @param path      file path
     * @param config    header/ID configuration
     * @param delimiter custom delimiter character
     * @throws IOException  if file cannot be read
     * @throws SQLException if SQLite load fails
     */
    public DatasetLoader(String path, DatasetConfig config, Character delimiter) throws IOException, SQLException {
        this(path, config, null, delimiter);
    }

    /**
     * Constructor with label column and delimiter.
     *
     * @param path         file path
     * @param labelColumn  label column name; null = last column
     * @param delimiter    custom delimiter character
     * @throws IOException  if file cannot be read
     * @throws SQLException if SQLite load fails
     */
    public DatasetLoader(String path, String labelColumn, Character delimiter) throws IOException, SQLException {
        this(path, DatasetConfig.DEFAULT, labelColumn, delimiter);
    }

    /**
     * Full constructor with numeric label option.
     *
     * @param path         file path
     * @param config       header/ID configuration
     * @param labelColumn  label column name; null = last column
     * @param delimiter    custom delimiter character
     * @param numericLabel true = parse labels as double (regression); false = categorical strings
     * @throws IOException        if file cannot be read or format unsupported
     * @throws SQLException       if SQLite load fails
     */
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

    /**
     * Returns the loaded dataset.
     *
     * @return the {@link Dataset} containing data matrix, header, and label map
     */
    public Dataset getDataset() {
        return dataset;
    }

    // ==================== Split methods ====================

    /**
     * Splits dataset with default stratified strategy (80/20 if trainPercent=80).
     *
     * @param trainPercent training percentage (0-100) or fraction (0.0-1.0)
     * @return train/test split
     */
    public DatasetSplit split(double trainPercent) {
        return DatasetSplitter.split(dataset, trainPercent, "s", new Random());
    }

    /**
     * Splits dataset with specified strategy.
     *
     * @param trainPercent training percentage (0-100) or fraction (0.0-1.0)
     * @param strategy     "random"/"r", "stratified"/"s", "timeseries"/"t"
     * @return train/test split
     */
    public DatasetSplit split(double trainPercent, String strategy) {
        return DatasetSplitter.split(dataset, trainPercent, strategy, new Random());
    }

    /**
     * Splits dataset with strategy and seed for reproducibility.
     *
     * @param trainPercent training percentage (0-100) or fraction (0.0-1.0)
     * @param strategy     "random"/"r", "stratified"/"s", "timeseries"/"t"
     * @param seed         random seed
     * @return train/test split
     */
    public DatasetSplit split(double trainPercent, String strategy, long seed) {
        return DatasetSplitter.split(dataset, trainPercent, strategy, new Random(seed));
    }

    /**
     * Random split (no stratification).
     *
     * @param trainPercent training percentage (0-100) or fraction (0.0-1.0)
     * @return train/test split
     */
    public DatasetSplit randomSplit(double trainPercent) {
        return DatasetSplitter.randomSplit(dataset, trainPercent, new Random());
    }

    /**
     * Random split with seed.
     *
     * @param trainPercent training percentage (0-100) or fraction (0.0-1.0)
     * @param seed         random seed
     * @return train/test split
     */
    public DatasetSplit randomSplit(double trainPercent, long seed) {
        return DatasetSplitter.randomSplit(dataset, trainPercent, new Random(seed));
    }

    /**
     * Stratified split (preserves class proportions).
     *
     * @param trainPercent training percentage (0-100) or fraction (0.0-1.0)
     * @return train/test split
     */
    public DatasetSplit stratifiedSplit(double trainPercent) {
        return DatasetSplitter.stratifiedSplit(dataset, trainPercent, new Random());
    }

    /**
     * Stratified split with seed.
     *
     * @param trainPercent training percentage (0-100) or fraction (0.0-1.0)
     * @param seed         random seed
     * @return train/test split
     */
    public DatasetSplit stratifiedSplit(double trainPercent, long seed) {
        return DatasetSplitter.stratifiedSplit(dataset, trainPercent, new Random(seed));
    }

    /**
     * Time-series split (first N% train, rest test, no shuffle).
     *
     * @param trainPercent training percentage (0-100) or fraction (0.0-1.0)
     * @return train/test split
     */
    public DatasetSplit timeSeriesSplit(double trainPercent) {
        return DatasetSplitter.timeSeriesSplit(dataset, trainPercent, new Random());
    }

    /**
     * Time-series split with seed.
     *
     * @param trainPercent training percentage (0-100) or fraction (0.0-1.0)
     * @param seed         random seed (used for any tie-breaking)
     * @return train/test split
     */
    public DatasetSplit timeSeriesSplit(double trainPercent, long seed) {
        return DatasetSplitter.timeSeriesSplit(dataset, trainPercent, new Random(seed));
    }

    // ==================== CSV Loading ====================

    /**
     * Loads a CSV/TSV/TXT file as a {@link Dataset}.
     *
     * @param path         file path
     * @param config       header/ID configuration
     * @param labelColumn  label column name; null = last column
     * @param delimiter    custom delimiter; null = auto-detect
     * @return parsed dataset
     * @throws IOException if file cannot be read
     */
    public static Dataset loadCsv(String path, DatasetConfig config, String labelColumn, Character delimiter)
            throws IOException {

        return loadCsv(path, config, labelColumn, delimiter, false);
    }

    /**
     * Loads a CSV/TSV/TXT file with numeric label option.
     *
     * @param path         file path
     * @param config       header/ID configuration
     * @param labelColumn  label column name; null = last column
     * @param delimiter    custom delimiter; null = auto-detect
     * @param numericLabel true = parse labels as double (regression)
     * @return parsed dataset
     * @throws IOException if file cannot be read
     */
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

    /**
     * Loads a JSON file (array of objects) as a {@link Dataset}.
     *
     * @param path         file path
     * @param config       header/ID configuration
     * @param labelColumn  label column name; null = last column
     * @return parsed dataset
     * @throws IOException if file cannot be read
     */
    public static Dataset loadJson(String path,
            DatasetConfig config,
            String labelColumn)
            throws IOException {

        return loadJson(path, config, labelColumn, false);
    }

    /**
     * Loads a JSON file with numeric label option.
     *
     * @param path         file path
     * @param config       header/ID configuration
     * @param labelColumn  label column name; null = last column
     * @param numericLabel true = parse labels as double (regression)
     * @return parsed dataset
     * @throws IOException if file cannot be read
     */
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

    /**
     * Loads a text classification dataset from CSV/TSV.
     *
     * @param path         file path
     * @param textColumn   name of text column
     * @param labelColumn  name of label column
     * @param config       header/ID configuration
     * @param delimiter    custom delimiter; null = auto-detect
     * @return {@link TextDataset} with texts, numeric labels, and encoder
     * @throws IOException if file cannot be read
     */
    public static TextDataset loadText(String path,
            String textColumn,
            String labelColumn,
            DatasetConfig config,
            Character delimiter)
            throws IOException {

        return TextDatasetLoader.loadText(path, textColumn, labelColumn, config, delimiter);
    }

    /**
     * Loads a text classification dataset with custom label map.
     *
     * @param path      file path
     * @param textColumn  name of text column
     * @param labelColumn name of label column
     * @param config      header/ID configuration
     * @param delimiter   custom delimiter; null = auto-detect
     * @param labelMap    explicit label→index mapping; must cover all labels
     * @return {@link TextDataset}
     * @throws IOException if file cannot be read
     */
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

    /**
     * Loads the first table from a SQLite database.
     * <p>
     * Requires explicit {@code labelColumn} (throws {@link IllegalStateException} if null).
     * Uses {@link DatasetConfig#NO_ID_WITH_HEADER} (no ID column, first row is header).
     *
     * @param dbPath      path to .sqlite file
     * @param labelColumn label column name (required)
     * @return parsed dataset
     * @throws SQLException if database cannot be read
     * @throws IllegalStateException if labelColumn is null
     */
    public static Dataset loadSqlite(String dbPath, String labelColumn) throws SQLException {
        if (labelColumn == null) {
            throw new IllegalStateException(
                    "No label column specified. Supervised models require labels.");
        }
        List<String[]> rows = SqliteLoader.readFirstTable(dbPath);
        return TableParser.parse(rows, DatasetConfig.NO_ID_WITH_HEADER, labelColumn, false);
    }

    // ==================== Table Parsing ====================

    /**
     * Parses pre-split string rows into a {@link Dataset}.
     *
     * @param rows        list of string arrays (first row = header if config has header)
     * @param config      header/ID configuration
     * @param labelColumn label column name; null = last column
     * @return parsed dataset
     */
    public static Dataset parseTable(List<String[]> rows, DatasetConfig config, String labelColumn) {
        return TableParser.parse(rows, config, labelColumn, false);
    }

    /**
     * Parses pre-split string rows with numeric label option.
     *
     * @param rows         list of string arrays
     * @param config       header/ID configuration
     * @param labelColumn  label column name; null = last column
     * @param numericLabel true = parse labels as double
     * @return parsed dataset
     */
    public static Dataset parseTable(List<String[]> rows, DatasetConfig config, String labelColumn, boolean numericLabel) {
        return TableParser.parse(rows, config, labelColumn, numericLabel);
    }

    // ==================== CSV Line Parsing ====================

    /**
     * Splits a CSV line respecting quoted fields.
     *
     * @param line      CSV line
     * @param delimiter delimiter character; null = comma
     * @return array of fields
     */
    public static String[] splitCsvLine(String line, Character delimiter) {
        return CsvParser.splitCsvLine(line, delimiter);
    }

    public static final Character[] COMMON_DELIMITERS = CsvParser.COMMON_DELIMITERS;

    /**
     * Detects the most likely delimiter from sample lines.
     *
     * @param lines sample lines (first few lines of file)
     * @return detected delimiter, or comma if uncertain
     */
    public static Character detectDelimiter(List<String> lines) {
        return CsvParser.detectDelimiter(lines);
    }

    /**
     * Counts delimiter occurrences outside quoted strings.
     *
     * @param s     input string
     * @param delim delimiter character
     * @return count of delimiters outside quotes
     */
    public static int countDelimiterOutsideQuotes(String s, Character delim) {
        return CsvParser.countDelimiterOutsideQuotes(s, delim);
    }

    /**
     * Finds the index of a key in a string array.
     *
     * @param arr array to search
     * @param key key to find
     * @return index, or -1 if not found
     */
    public static int indexOf(String[] arr, String key) {
        return TableParser.indexOf(arr, key);
    }

    /**
     * Parses a string to double, returning 0.0 on failure.
     *
     * @param s string to parse
     * @return parsed double or 0.0
     */
    public static double parseDouble(String s) {
        return TableParser.parseDouble(s);
    }

    /**
     * Strips whitespace and surrounding quotes from a string.
     *
     * @param s string to strip
     * @return stripped string
     */
    public static String strip(String s) {
        return TableParser.strip(s);
    }

    /**
     * Extracts the file extension (substring after last dot).
     *
     * @param path file path
     * @return extension (empty if no dot)
     */
    public static String getExtension(String path) {
        int i = path.lastIndexOf('.');
        return (i < 0) ? "" : path.substring(i + 1);
    }
}