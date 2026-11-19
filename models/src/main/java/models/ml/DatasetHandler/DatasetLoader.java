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
        return split(trainPercent, "s", 0);
    }

    public DatasetSplit split(double trainPercent, String strategy) {
        return split(trainPercent, strategy, 0);
    }

    public DatasetSplit split(double trainPercent, String strategy, long seed) {
        if (strategy == null)
            strategy = "random";

        // For stratified, check if we should fall back to random based on data
        if ("stratified".equalsIgnoreCase(strategy) || "s".equalsIgnoreCase(strategy)) {
            if (hasContinuousLabels()) {
                return randomSplit(trainPercent, seed);
            }
        }

        switch (strategy.toLowerCase()) {
            case "stratified":
            case "s":
                return stratifiedSplit(trainPercent, seed);

            case "timeseries":
            case "t":
                return timeSeriesSplit(trainPercent, seed);

            case "random":
            case "r":
            default:
                return randomSplit(trainPercent, seed);
        }
    }

    /**
     * Checks if the original labels (before mapping to indices) contain non-integer values (continuous targets).
     * Uses labelToIndex keys which are the raw label strings. If no label column was specified,
     * checks the last column of the data matrix.
     */
    private boolean hasContinuousLabels() {
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

    public DatasetSplit randomSplit(double trainPercent) {
        return randomSplit(trainPercent, 0);
    }

    public DatasetSplit randomSplit(double trainPercent, long seed) {
        double[][] data = dataset.data;
        int n = data.length;
        int trainSize = (trainPercent <= 1.0) ? (int) (n * trainPercent) : (int) (n * trainPercent / 100.0);

        List<double[]> shuffled = new ArrayList<>(Arrays.asList(data));
        Random rand = new Random(seed);
        Collections.shuffle(shuffled, rand);

        double[][] train = shuffled.subList(0, trainSize)
                .toArray(new double[0][]);

        double[][] test = shuffled.subList(trainSize, n)
                .toArray(new double[0][]);

        return new DatasetSplit(train, test);
    }

    public DatasetSplit stratifiedSplit(double trainPercent) {
        return stratifiedSplit(trainPercent, 0);
    }

    public DatasetSplit stratifiedSplit(double trainPercent, long seed) {
        // If continuous labels detected, fall back to random split
        if (hasContinuousLabels()) {
            return randomSplit(trainPercent, seed);
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
        Random rand = new Random(seed);

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

    public DatasetSplit timeSeriesSplit(double trainPercent) {
        return timeSeriesSplit(trainPercent, 0);
    }

    public DatasetSplit timeSeriesSplit(double trainPercent, long seed) {
        double[][] data = dataset.data;
        int n = data.length;
        int trainSize = (trainPercent <= 1.0) ? (int) (n * trainPercent) : (int) (n * trainPercent / 100.0);

        double[][] train = Arrays.copyOfRange(data, 0, trainSize);
        double[][] test = Arrays.copyOfRange(data, trainSize, n);

        return new DatasetSplit(train, test);
    }

    // ==================== CSV Loading ====================

    public static Dataset loadCsv(String path, DatasetConfig config, String labelColumn, Character delimiter)
            throws IOException {

        return loadCsv(path, config, labelColumn, delimiter, false);
    }

    public static Dataset loadCsv(String path, DatasetConfig config, String labelColumn, Character delimiter, boolean numericLabel)
            throws IOException {

        // Use character-level reader for large files
        List<String[]> rows = parseCsvStream(path, delimiter);
        if (rows.isEmpty()) {
            return new Dataset(new double[0][0], new String[0], Map.of());
        }

        return parseTable(rows, config, labelColumn, numericLabel);
    }

    // Parse CSV with character-level reader (handles quoted fields with delimiters, quotes, newlines)
    private static List<String[]> parseCsvStream(String path, Character delimiter) throws IOException {
        List<String[]> rows = new ArrayList<>();
        StringBuilder lineBuilder = new StringBuilder();
        List<String> rawLines = new ArrayList<>(); // Store raw lines for re-parsing after delimiter detection

        try (BufferedReader reader = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (lineBuilder.length() > 0) {
                    // Continuing a multi-line quoted field
                    lineBuilder.append('\n');
                }
                lineBuilder.append(line);

                // Check if we have unclosed quotes
                int quoteCount = 0;
                for (int i = 0; i < lineBuilder.length(); i++) {
                    if (lineBuilder.charAt(i) == '"') {
                        // Check for escaped quote
                        if (i + 1 < lineBuilder.length() && lineBuilder.charAt(i + 1) == '"') {
                            i++; // Skip escaped quote
                        } else {
                            quoteCount++;
                        }
                    }
                }

                if (quoteCount % 2 == 0) {
                    // Even number of quotes - field is closed
                    String fullLine = lineBuilder.toString();
                    lineBuilder.setLength(0);

                    if (fullLine.trim().isEmpty()) {
                        continue;
                    }

                    rawLines.add(fullLine);
                }
                // Else: odd number of quotes, continue reading next line
            }
        }

        // If we still have content in lineBuilder (unclosed quote at EOF)
        if (lineBuilder.length() > 0) {
            throw new IOException("Unclosed quote at end of file");
        }

        if (rawLines.isEmpty()) {
            return rows;
        }

        // Determine delimiter
        Character sep;
        if (delimiter != null) {
            sep = delimiter;
        } else {
            // Detect delimiter from first few raw lines
            int sampleSize = Math.min(rawLines.size(), 20);
            sep = detectDelimiterFromRawLines(rawLines, sampleSize);
            if (sep == null) {
                sep = ','; // fallback
            }
        }

        // Parse all lines with detected/provided delimiter
        int expectedCols = -1;
        for (String fullLine : rawLines) {
            String[] parsed = splitCsvLine(fullLine, sep);

            if (expectedCols == -1) {
                expectedCols = parsed.length;
            } else if (parsed.length != expectedCols) {
                throw new IOException(
                        "Inconsistent column count. Expected " + expectedCols +
                                " but found " + parsed.length + " in line:\n" + fullLine);
            }

            rows.add(parsed);
        }

        return rows;
    }

    // Helper to detect delimiter from a single row
    private static Character detectDelimiterFromRow(String row) {
        Character best = ',';
        int bestScore = -1;

        for (Character d : COMMON_DELIMITERS) {
            int count = countDelimiterOutsideQuotes(row, d);
            if (count > bestScore) {
                bestScore = count;
                best = d;
            }
        }

        return bestScore > 0 ? best : null;
    }

    // Helper to detect delimiter from multiple raw lines
    private static Character detectDelimiterFromRawLines(List<String> rawLines, int sampleSize) {
        Character best = ',';
        int bestScore = -1;

        for (Character d : COMMON_DELIMITERS) {
            Integer expected = null;
            boolean consistent = true;
            int score = 0;

            for (int i = 0; i < sampleSize; i++) {
                int count = countDelimiterOutsideQuotes(rawLines.get(i), d);
                if (expected == null)
                    expected = count;
                else if (!expected.equals(count)) {
                    consistent = false;
                    break;
                }
                score += count;
            }

            if (consistent && score > bestScore) {
                bestScore = score;
                best = d;
            }
        }

        return bestScore > 0 ? best : null;
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

            return parseTable(rows, config, labelColumn, numericLabel);
        }
    }

    // ==================== Text Dataset Loading ====================

    public static TextDataset loadText(String path,
            String textColumn,
            String labelColumn,
            DatasetConfig config,
            Character delimiter)
            throws IOException {

        return loadText(path, textColumn, labelColumn, config, delimiter, null);
    }

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
        List<String[]> rows = parseCsvStream(path, delimiter);
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

    // ==================== SQLite Loading ====================

    public static Dataset loadSqlite(String dbPath, String labelColumn) throws SQLException {
        if (labelColumn == null) {
            throw new IllegalStateException(
                    "No label column specified. Supervised models require labels.");
        }
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + dbPath)) {
            Statement stmt = conn.createStatement();
            ResultSet table = stmt.executeQuery("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1");
            if (!table.next()) {
                throw new SQLException("No table found");
            }
            String tableName = table.getString("name");
            String query = "SELECT * FROM " + tableName;

            ResultSet rs = stmt.executeQuery(query);
            ResultSetMetaData md = rs.getMetaData();
            int cols = md.getColumnCount();

            List<String[]> rows = new ArrayList<>();
            String[] header = new String[cols];
            for (int i = 1; i <= cols; i++)
                header[i - 1] = md.getColumnName(i);
            rows.add(header);

            while (rs.next()) {
                String[] row = new String[cols];
                for (int i = 1; i <= cols; i++)
                    row[i - 1] = String.valueOf(rs.getObject(i));
                rows.add(row);
            }

            return parseTable(rows, DatasetConfig.NO_ID_WITH_HEADER, labelColumn);
        } catch (SQLException e) {
            throw e;
        }
    }

    // ==================== Table Parsing ====================

    public static Dataset parseTable(List<String[]> rows, DatasetConfig config, String labelColumn) {
        return parseTable(rows, config, labelColumn, false);
    }

    public static Dataset parseTable(List<String[]> rows, DatasetConfig config, String labelColumn, boolean numericLabel) {

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
        List<double[]> data = new ArrayList<>();

        int r = 0;
        if (config.hasHeader) {
            r = 1;
        }
        for (int i = r; i < rows.size(); i++) {
            String[] row = rows.get(i);
            List<Double> nums = new ArrayList<>();

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

    // ==================== CSV Line Parsing ====================

    public static String[] splitCsvLine(String line, Character delimiter) {
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;

        for (int i = 0; i < line.length(); i++) {
            Character c = line.charAt(i);

            if (c == '"') {
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (c == delimiter && !inQuotes) {
                tokens.add(current.toString().trim());
                current.setLength(0);
            } else {
                current.append(c);
            }
        }

        tokens.add(current.toString().trim());
        return tokens.toArray(new String[0]);
    }

    public static final Character[] COMMON_DELIMITERS = { ',', '\t', ';', '|' };

    public static Character detectDelimiter(List<String> lines) {
        int sampleSize = Math.min(lines.size(), 20);
        Character best = ',';
        int bestScore = -1;

        for (Character d : COMMON_DELIMITERS) {
            Integer expected = null;
            boolean consistent = true;
            int score = 0;

            for (int i = 0; i < sampleSize; i++) {
                int count = countDelimiterOutsideQuotes(lines.get(i), d);
                if (expected == null)
                    expected = count;
                else if (!expected.equals(count)) {
                    consistent = false;
                    break;
                }
                score += count;
            }

            if (consistent && score > bestScore) {
                bestScore = score;
                best = d;
            }
        }

        return best;
    }

    public static int countDelimiterOutsideQuotes(String s, Character delim) {
        boolean inQuotes = false;
        int count = 0;

        for (Character c : s.toCharArray()) {
            if (c == '"')
                inQuotes = !inQuotes;
            else if (c == delim && !inQuotes)
                count++;
        }
        return count;
    }

    public static int indexOf(String[] arr, String key) {
        for (int i = 0; i < arr.length; i++)
            if (arr[i].equalsIgnoreCase(key))
                return i;
        return -1;
    }

    public static double parseDouble(String s) {
        try {
            return Double.parseDouble(s.trim());
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    public static String strip(String s) {
        return s.trim().replace("\"", "");
    }

    public static String getExtension(String path) {
        int i = path.lastIndexOf('.');
        return (i < 0) ? "" : path.substring(i + 1);
    }

    // ==================== JSON Parser (Recursive Descent) ====================

    static class JsonParser {
        private final BufferedReader reader;
        private int ch;
        private long position = 0;

        public JsonParser(BufferedReader reader) throws IOException {
            this.reader = reader;
            nextChar();
        }

        private void nextChar() throws IOException {
            ch = reader.read();
            position++;
        }

        private void skipWhitespace() throws IOException {
            while (ch != -1 && Character.isWhitespace(ch)) {
                nextChar();
            }
        }

        public List<Map<String, String>> parseArray() throws IOException {
            skipWhitespace();
            if (ch != '[') {
                throw new IOException("Expected '[' at position " + position);
            }
            nextChar();

            List<Map<String, String>> array = new ArrayList<>();
            skipWhitespace();

            if (ch == ']') {
                nextChar();
                return array;
            }

            while (true) {
                skipWhitespace();
                if (ch == '}') {
                    throw new IOException("Unexpected '}' at position " + position);
                }
                if (ch != '{') {
                    throw new IOException("Expected '{' at position " + position);
                }
                array.add(parseObject());
                skipWhitespace();
                if (ch == ',') {
                    nextChar();
                    continue;
                }
                if (ch == ']') {
                    nextChar();
                    break;
                }
                throw new IOException("Expected ',' or ']' at position " + position);
            }
            return array;
        }

        private Map<String, String> parseObject() throws IOException {
            Map<String, String> obj = new LinkedHashMap<>();
            nextChar(); // consume '{'
            skipWhitespace();

            if (ch == '}') {
                nextChar();
                return obj;
            }

            while (true) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                if (ch != ':') {
                    throw new IOException("Expected ':' at position " + position);
                }
                nextChar();
                String value = parseValue();
                obj.put(key, value);
                skipWhitespace();
                if (ch == ',') {
                    nextChar();
                    continue;
                }
                if (ch == '}') {
                    nextChar();
                    break;
                }
                throw new IOException("Expected ',' or '}' at position " + position);
            }
            return obj;
        }

        private String parseString() throws IOException {
            if (ch != '"') {
                throw new IOException("Expected string at position " + position);
            }
            nextChar(); // consume opening quote

            StringBuilder sb = new StringBuilder();
            while (ch != -1) {
                if (ch == '"') {
                    nextChar(); // consume closing quote
                    return sb.toString();
                }
                if (ch == '\\') {
                    nextChar();
                    if (ch == -1) throw new IOException("Unexpected end of input in string escape");
                    switch (ch) {
                        case '"': sb.append('"'); break;
                        case '\\': sb.append('\\'); break;
                        case '/': sb.append('/'); break;
                        case 'b': sb.append('\b'); break;
                        case 'f': sb.append('\f'); break;
                        case 'n': sb.append('\n'); break;
                        case 'r': sb.append('\r'); break;
                        case 't': sb.append('\t'); break;
                        case 'u': {
                            // Unicode escape \\uXXXX
                            if (ch == 'u') nextChar();
                            int codePoint = 0;
                            for (int i = 0; i < 4; i++) {
                                if (ch == -1) throw new IOException("Unexpected end of input in unicode escape");
                                int digit = Character.digit((char) ch, 16);
                                if (digit == -1) throw new IOException("Invalid unicode escape at position " + position);
                                codePoint = (codePoint << 4) | digit;
                                nextChar();
                            }
                            sb.append((char) codePoint);
                            continue; // already consumed next char
                        }
                        default:
                            throw new IOException("Invalid escape sequence \\\\" + (char) ch + " at position " + position);
                    }
                } else {
                    sb.append((char) ch);
                }
                nextChar();
            }
            throw new IOException("Unterminated string at position " + position);
        }

        private String parseValue() throws IOException {
            skipWhitespace();
            if (ch == -1) throw new IOException("Unexpected end of input");

            switch (ch) {
                case '"':
                    return parseString();
                case '{':
                    // Nested object - return raw JSON text
                    return parseRawJson();
                case '[':
                    // Nested array - return raw JSON text
                    return parseRawJson();
                case 't':
                    return parseLiteral("true", "true");
                case 'f':
                    return parseLiteral("false", "false");
                case 'n':
                    return parseLiteral("null", "");
                default:
                    // Number
                    return parseNumber();
            }
        }

        private String parseRawJson() throws IOException {
            // Parse nested object/array and return its raw JSON text
            StringBuilder sb = new StringBuilder();
            int depth = 0;
            char startChar = (char) ch;
            char endChar = (startChar == '{') ? '}' : ']';

            while (ch != -1) {
                char c = (char) ch;
                sb.append(c);
                if (c == startChar) {
                    depth++;
                } else if (c == endChar) {
                    depth--;
                    if (depth == 0) {
                        nextChar();
                        return sb.toString();
                    }
                } else if (c == '"') {
                    // Handle strings inside - skip escaped quotes
                    nextChar();
                    while (ch != -1) {
                        char cc = (char) ch;
                        sb.append(cc);
                        if (cc == '"') {
                            // Check if escaped
                            int backslashCount = 0;
                            for (int i = sb.length() - 2; i >= 0 && sb.charAt(i) == '\\'; i--) {
                                backslashCount++;
                            }
                            if (backslashCount % 2 == 0) {
                                nextChar();
                                break;
                            }
                        }
                        nextChar();
                    }
                    continue;
                }
                nextChar();
            }
            throw new IOException("Unterminated nested structure at position " + position);
        }

        private String parseLiteral(String expected, String value) throws IOException {
            for (int i = 0; i < expected.length(); i++) {
                if (ch != expected.charAt(i)) {
                    throw new IOException("Expected '" + expected + "' at position " + position);
                }
                nextChar();
            }
            return value;
        }

        private String parseNumber() throws IOException {
            StringBuilder sb = new StringBuilder();
            boolean hasDigit = false;

            // Optional sign
            if (ch == '-') {
                sb.append('-');
                nextChar();
            }

            // Integer part
            if (ch == '0') {
                sb.append('0');
                nextChar();
                hasDigit = true;
            } else if (ch >= '1' && ch <= '9') {
                while (ch >= '0' && ch <= '9') {
                    sb.append((char) ch);
                    nextChar();
                    hasDigit = true;
                }
            } else {
                throw new IOException("Expected digit at position " + position);
            }

            // Fractional part
            if (ch == '.') {
                sb.append('.');
                nextChar();
                if (ch < '0' || ch > '9') {
                    throw new IOException("Expected digit after decimal point at position " + position);
                }
                while (ch >= '0' && ch <= '9') {
                    sb.append((char) ch);
                    nextChar();
                }
            }

            // Exponent
            if (ch == 'e' || ch == 'E') {
                sb.append((char) ch);
                nextChar();
                if (ch == '+' || ch == '-') {
                    sb.append((char) ch);
                    nextChar();
                }
                if (ch < '0' || ch > '9') {
                    throw new IOException("Expected digit in exponent at position " + position);
                }
                while (ch >= '0' && ch <= '9') {
                    sb.append((char) ch);
                    nextChar();
                }
            }

            if (!hasDigit) {
                throw new IOException("Invalid number at position " + position);
            }
            return sb.toString();
        }
    }
}