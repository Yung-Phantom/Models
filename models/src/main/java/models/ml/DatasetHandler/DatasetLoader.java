package models.ml.DatasetHandler;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.sql.*;
import java.util.*;

import models.ml.DatasetHandler.helpers.Dataset;
import models.ml.DatasetHandler.helpers.DatasetConfig;
import models.ml.DatasetHandler.helpers.DatasetSplit;

public class DatasetLoader {

    private final Dataset dataset;

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

    public Dataset getDataset() {
        return dataset;
    }

    public DatasetSplit split(double trainPercent) {
        return split(trainPercent, "s");
    }

    public DatasetSplit split(double trainPercent, String strategy) {

        if (strategy == null)
            strategy = "random";

        switch (strategy.toLowerCase()) {
            case "stratified":
            case "s":
                return stratifiedSplit(trainPercent);

            case "timeseries":
            case "t":
                return timeSeriesSplit(trainPercent);

            case "random":
            case "r":
            default:
                return randomSplit(trainPercent);
        }
    }

    private DatasetSplit randomSplit(double trainPercent) {

        double[][] data = dataset.data;
        int n = data.length;
        int trainSize = (int) (n * trainPercent / 100.0);

        List<double[]> shuffled = new ArrayList<>(Arrays.asList(data));
        Collections.shuffle(shuffled, new Random());

        double[][] train = shuffled.subList(0, trainSize)
                .toArray(new double[0][]);

        double[][] test = shuffled.subList(trainSize, n)
                .toArray(new double[0][]);

        return new DatasetSplit(train, test);
    }

    private DatasetSplit stratifiedSplit(double trainPercent) {

        double[][] data = dataset.data;
        int labelIndex = data[0].length - 1;

        Map<Double, List<double[]>> byLabel = new LinkedHashMap<>();

        for (double[] row : data) {
            double label = row[labelIndex];
            byLabel.computeIfAbsent(label, k -> new ArrayList<>()).add(row);
        }

        List<double[]> train = new ArrayList<>();
        List<double[]> test = new ArrayList<>();
        Random rand = new Random();

        for (List<double[]> rows : byLabel.values()) {
            Collections.shuffle(rows, rand);
            int trainSize = (int) (rows.size() * trainPercent / 100.0);

            train.addAll(rows.subList(0, trainSize));
            test.addAll(rows.subList(trainSize, rows.size()));
        }

        Collections.shuffle(train, rand);
        Collections.shuffle(test, rand);

        return new DatasetSplit(
                train.toArray(new double[0][]),
                test.toArray(new double[0][]));
    }

    private DatasetSplit timeSeriesSplit(double trainPercent) {

        double[][] data = dataset.data;
        int n = data.length;
        int trainSize = (int) (n * trainPercent / 100.0);

        double[][] train = Arrays.copyOfRange(data, 0, trainSize);
        double[][] test = Arrays.copyOfRange(data, trainSize, n);

        return new DatasetSplit(train, test);
    }

    private static Dataset loadCsv(String path, DatasetConfig config, String labelColumn, Character delimiter)
            throws IOException {

        List<String> lines = Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            return new Dataset(new double[0][0], new String[0], Map.of());
        }
        Character sep;
        if (delimiter != null) {
            sep = delimiter;
        } else {
            sep = detectDelimiter(lines);
        }

        List<String[]> rows = new ArrayList<>();
        int expectedCols = -1;

        for (String line : lines) {
            if (line.trim().isEmpty()) {
                continue;
            }
            String[] parsed = splitCsvLine(line, sep);

            if (expectedCols == -1) {
                expectedCols = parsed.length;
            } else if (parsed.length != expectedCols) {
                throw new IOException(
                        "Inconsistent column count. Expected " + expectedCols +
                                " but found " + parsed.length + " in line:\n" + line);
            }

            rows.add(parsed);
        }

        return parseTable(rows, config, labelColumn);
    }

    private static Dataset loadJson(String path,
            DatasetConfig config,
            String labelColumn)
            throws IOException {

        String content = Files.readString(Paths.get(path)).trim();
        if (!content.startsWith("["))
            throw new IOException("JSON must be an array of objects");

        List<Map<String, String>> objects = new ArrayList<>();
        content = content.substring(1, content.length() - 1);

        for (String obj : content.split("\\},\\s*\\{")) {
            obj = obj.replace("{", "").replace("}", "");
            Map<String, String> map = new LinkedHashMap<>();
            for (String kv : obj.split(",")) {
                String[] pair = kv.split(":", 2);
                map.put(strip(pair[0]), strip(pair[1]));
            }
            objects.add(map);
        }

        List<String[]> rows = new ArrayList<>();
        rows.add(objects.get(0).keySet().toArray(new String[0]));

        for (Map<String, String> obj : objects)
            rows.add(obj.values().toArray(new String[0]));

        return parseTable(rows, config, labelColumn);
    }

    private static Dataset loadSqlite(String dbPath, String labelColumn) throws SQLException {
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

    private static Dataset parseTable(List<String[]> rows, DatasetConfig config, String labelColumn) {

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
                labelMap.putIfAbsent(label, labelMap.size());
                nums.add(labelMap.get(label).doubleValue());
            }

            data.add(nums.stream().mapToDouble(Double::doubleValue).toArray());
        }

        return new Dataset(
                data.toArray(new double[0][]),
                header,
                labelMap);
    }

    private static String[] splitCsvLine(String line, Character delimiter) {
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

    private static final Character[] COMMON_DELIMITERS = { ',', '\t', ';', '|' };

    private static Character detectDelimiter(List<String> lines) {

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

    private static int countDelimiterOutsideQuotes(String s, Character delim) {
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

    private static int indexOf(String[] arr, String key) {
        for (int i = 0; i < arr.length; i++)
            if (arr[i].equalsIgnoreCase(key))
                return i;
        return -1;
    }

    private static double parseDouble(String s) {
        try {
            return Double.parseDouble(s.trim());
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    private static String strip(String s) {
        return s.trim().replace("\"", "");
    }

    private static String getExtension(String path) {
        int i = path.lastIndexOf('.');
        return (i < 0) ? "" : path.substring(i + 1);
    }
}
