package models.ml.DatasetHandler.parsers;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for parsing CSV files with support for quoted fields, custom delimiters,
 * and automatic delimiter detection.
 */
public final class CsvParser {

    private CsvParser() {}

    /**
     * Common delimiter characters for CSV files.
     */
    public static final Character[] COMMON_DELIMITERS = { ',', '\t', ';', '|' };

    /**
     * Parses a CSV file into a list of string arrays.
     *
     * @param path      path to the CSV file
     * @param delimiter explicit delimiter character, or null for auto-detection
     * @return list of rows, each row is an array of cell strings
     * @throws IOException if file cannot be read or has unclosed quotes
     */
    public static List<String[]> parse(String path, Character delimiter) throws IOException {
        List<String[]> rows = new ArrayList<>();
        StringBuilder lineBuilder = new StringBuilder();
        List<String> rawLines = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (lineBuilder.length() > 0) {
                    lineBuilder.append('\n');
                }
                lineBuilder.append(line);

                int quoteCount = 0;
                for (int i = 0; i < lineBuilder.length(); i++) {
                    if (lineBuilder.charAt(i) == '"') {
                        if (i + 1 < lineBuilder.length() && lineBuilder.charAt(i + 1) == '"') {
                            i++;
                        } else {
                            quoteCount++;
                        }
                    }
                }

                if (quoteCount % 2 == 0) {
                    String fullLine = lineBuilder.toString();
                    lineBuilder.setLength(0);

                    if (fullLine.trim().isEmpty()) {
                        continue;
                    }

                    rawLines.add(fullLine);
                }
            }
        }

        if (lineBuilder.length() > 0) {
            throw new IOException("Unclosed quote at end of file");
        }

        if (rawLines.isEmpty()) {
            return rows;
        }

        Character sep;
        if (delimiter != null) {
            sep = delimiter;
        } else {
            int sampleSize = Math.min(rawLines.size(), 20);
            sep = detectDelimiterFromRawLines(rawLines, sampleSize);
            if (sep == null) {
                sep = ',';
            }
        }

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

    /**
     * Splits a single CSV line into fields, respecting quoted fields and escaped quotes.
     *
     * @param line      CSV line to parse
     * @param delimiter delimiter character
     * @return array of field strings (trimmed)
     */
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

    /**
     * Detects the delimiter from a list of lines by scoring common delimiters.
     *
     * @param lines list of CSV lines (without trailing newlines)
     * @return detected delimiter, or ',' if none scores positively
     */
    public static Character detectDelimiter(List<String> lines) {
        return detectDelimiterFromRawLines(lines, Math.min(lines.size(), 20));
    }

    /**
     * Counts occurrences of a delimiter outside quoted strings.
     *
     * @param s     string to search
     * @param delim delimiter character
     * @return count of delimiter outside quotes
     */
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

    // Private helper to detect delimiter from raw lines with consistency checking
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
}