package models.ml.DatasetHandler.parsers;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Recursive-descent JSON parser for arrays of flat objects.
 * Parses JSON arrays of objects into a list of string maps.
 * Nested objects and arrays are returned as raw JSON strings.
 */
public final class JsonParser {

    private final BufferedReader reader;
    private int ch;
    private long position = 0;

    /**
     * Creates a new JSON parser reading from the given reader.
     *
     * @param reader buffered reader to read JSON from
     * @throws IOException if an I/O error occurs reading the first character
     */
    public JsonParser(BufferedReader reader) throws IOException {
        this.reader = reader;
        nextChar();
    }

    /**
     * Parses a JSON file from the given path.
     *
     * @param path path to the JSON file (UTF-8 encoded)
     * @return list of objects as string maps
     * @throws IOException if file cannot be read or JSON is malformed
     */
    public static List<Map<String, String>> parseFile(String path) throws IOException {
        try (BufferedReader reader = java.nio.file.Files.newBufferedReader(
                java.nio.file.Paths.get(path), java.nio.charset.StandardCharsets.UTF_8)) {
            JsonParser parser = new JsonParser(reader);
            return parser.parseArray();
        }
    }

    /**
     * Parses a JSON array from the current position.
     *
     * @return list of parsed objects
     * @throws IOException if JSON is malformed or I/O error occurs
     */
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
                        continue;
                    }
                    default:
                        throw new IOException("Invalid escape sequence \\" + (char) ch + " at position " + position);
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
                return parseRawJson();
            case '[':
                return parseRawJson();
            case 't':
                return parseLiteral("true", "true");
            case 'f':
                return parseLiteral("false", "false");
            case 'n':
                return parseLiteral("null", "");
            default:
                return parseNumber();
        }
    }

    private String parseRawJson() throws IOException {
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
                nextChar();
                while (ch != -1) {
                    char cc = (char) ch;
                    sb.append(cc);
                    if (cc == '"') {
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

        if (ch == '-') {
            sb.append('-');
            nextChar();
        }

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

    private void nextChar() throws IOException {
        ch = reader.read();
        position++;
    }

    private void skipWhitespace() throws IOException {
        while (ch != -1 && Character.isWhitespace(ch)) {
            nextChar();
        }
    }
}