package models.ml.Utils;

import java.util.*;

public class Json {
    private static class Parser {
        private final String s;
        private int pos = 0;

        Parser(String s) {
            this.s = s;
        }

        private void skipWhitespace() {
            while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) {
                pos++;
            }
        }

        Object parse() throws Exception {
            skipWhitespace();
            if (pos >= s.length()) throw new Exception("Unexpected end of input");
            char c = s.charAt(pos);
            if (c == '{') return parseObject();
            if (c == '[') return parseArray();
            if (c == '"') return parseString();
            if (c == 't' || c == 'f') return parseBoolean();
            if (c == 'n') return parseNull();
            return parseNumber();
        }

        private Map<String, Object> parseObject() throws Exception {
            Map<String, Object> map = new LinkedHashMap<>();
            pos++; // skip '{'
            skipWhitespace();
            if (pos < s.length() && s.charAt(pos) == '}') {
                pos++;
                return map;
            }
            while (true) {
                skipWhitespace();
                if (pos >= s.length() || s.charAt(pos) != '"') {
                    throw new Exception("Expected string key at position " + pos);
                }
                String key = parseString();
                skipWhitespace();
                if (pos >= s.length() || s.charAt(pos) != ':') {
                    throw new Exception("Expected ':' at position " + pos);
                }
                pos++;
                Object value = parse();
                map.put(key, value);
                skipWhitespace();
                if (pos >= s.length()) throw new Exception("Unexpected end of input in object");
                char c = s.charAt(pos);
                if (c == '}') {
                    pos++;
                    break;
                }
                if (c != ',') throw new Exception("Expected ',' or '}' at position " + pos);
                pos++;
            }
            return map;
        }

        private List<Object> parseArray() throws Exception {
            List<Object> list = new ArrayList<>();
            pos++; // skip '['
            skipWhitespace();
            if (pos < s.length() && s.charAt(pos) == ']') {
                pos++;
                return list;
            }
            while (true) {
                list.add(parse());
                skipWhitespace();
                if (pos >= s.length()) throw new Exception("Unexpected end of input in array");
                char c = s.charAt(pos);
                if (c == ']') {
                    pos++;
                    break;
                }
                if (c != ',') throw new Exception("Expected ',' or ']' at position " + pos);
                pos++;
            }
            return list;
        }

        private String parseString() throws Exception {
            if (pos >= s.length() || s.charAt(pos) != '"') {
                throw new Exception("Expected '\"' at position " + pos);
            }
            pos++; // skip opening quote
            StringBuilder sb = new StringBuilder();
            while (pos < s.length()) {
                char c = s.charAt(pos);
                if (c == '"') {
                    pos++;
                    return sb.toString();
                }
                if (c == '\\') {
                    pos++;
                    if (pos >= s.length()) throw new Exception("Unexpected end of input in string escape");
                    char next = s.charAt(pos);
                    switch (next) {
                        case '"': sb.append('"'); break;
                        case '\\': sb.append('\\'); break;
                        case '/': sb.append('/'); break;
                        case 'b': sb.append('\b'); break;
                        case 'f': sb.append('\f'); break;
                        case 'n': sb.append('\n'); break;
                        case 'r': sb.append('\r'); break;
                        case 't': sb.append('\t'); break;
                        case 'u': {
                            if (pos + 4 >= s.length()) throw new Exception("Incomplete unicode escape at position " + pos);
                            String hex = s.substring(pos + 1, pos + 5);
                            try {
                                int codePoint = Integer.parseInt(hex, 16);
                                sb.append((char) codePoint);
                            } catch (NumberFormatException e) {
                                throw new Exception("Invalid unicode escape \\u" + hex + " at position " + pos);
                            }
                            pos += 4;
                            break;
                        }
                        default:
                            throw new Exception("Invalid escape sequence \\" + next + " at position " + pos);
                    }
                } else {
                    sb.append(c);
                }
                pos++;
            }
            throw new Exception("Unterminated string at position " + pos);
        }

        private Boolean parseBoolean() throws Exception {
            if (s.startsWith("true", pos)) {
                pos += 4;
                return true;
            }
            if (s.startsWith("false", pos)) {
                pos += 5;
                return false;
            }
            throw new Exception("Expected boolean at position " + pos);
        }

        private Object parseNull() throws Exception {
            if (s.startsWith("null", pos)) {
                pos += 4;
                return null;
            }
            throw new Exception("Expected null at position " + pos);
        }

        private Number parseNumber() throws Exception {
            int start = pos;
            if (s.charAt(pos) == '-') pos++;
            if (pos >= s.length() || !Character.isDigit(s.charAt(pos))) {
                throw new Exception("Expected digit at position " + pos);
            }
            if (s.charAt(pos) == '0') {
                pos++;
            } else {
                while (pos < s.length() && Character.isDigit(s.charAt(pos))) pos++;
            }
            boolean isDouble = false;
            if (pos < s.length() && s.charAt(pos) == '.') {
                isDouble = true;
                pos++;
                if (pos >= s.length() || !Character.isDigit(s.charAt(pos))) {
                    throw new Exception("Expected digit after decimal point at position " + pos);
                }
                while (pos < s.length() && Character.isDigit(s.charAt(pos))) pos++;
            }
            if (pos < s.length() && (s.charAt(pos) == 'e' || s.charAt(pos) == 'E')) {
                isDouble = true;
                pos++;
                if (pos < s.length() && (s.charAt(pos) == '+' || s.charAt(pos) == '-')) pos++;
                if (pos >= s.length() || !Character.isDigit(s.charAt(pos))) {
                    throw new Exception("Expected digit in exponent at position " + pos);
                }
                while (pos < s.length() && Character.isDigit(s.charAt(pos))) pos++;
            }
            String numStr = s.substring(start, pos);
            try {
                return Double.parseDouble(numStr);
            } catch (NumberFormatException e) {
                throw new Exception("Invalid number at position " + start);
            }
        }
    }

    private static class Writer {
        private final StringBuilder sb = new StringBuilder();

        void write(Object obj) {
            if (obj == null) {
                sb.append("null");
            } else if (obj instanceof String) {
                writeString((String) obj);
            } else if (obj instanceof Number) {
                writeNumber((Number) obj);
            } else if (obj instanceof Boolean) {
                sb.append(obj.toString());
            } else if (obj instanceof Map) {
                writeMap((Map<?, ?>) obj);
            } else if (obj instanceof List) {
                writeList((List<?>) obj);
            } else {
                throw new IllegalArgumentException("Unsupported type: " + obj.getClass());
            }
        }

        private void writeString(String s) {
            sb.append('"');
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '"': sb.append("\\\""); break;
                    case '\\': sb.append("\\\\"); break;
                    case '\b': sb.append("\\b"); break;
                    case '\f': sb.append("\\f"); break;
                    case '\n': sb.append("\\n"); break;
                    case '\r': sb.append("\\r"); break;
                    case '\t': sb.append("\\t"); break;
                    default:
                        if (c < 0x20) {
                            sb.append(String.format("\\u%04x", (int) c));
                        } else {
                            sb.append(c);
                        }
                }
            }
            sb.append('"');
        }

        private void writeNumber(Number n) {
            if (n instanceof Double || n instanceof Float) {
                double d = n.doubleValue();
                if (Double.isNaN(d) || Double.isInfinite(d)) {
                    throw new IllegalArgumentException("NaN and Infinity are not valid JSON numbers");
                }
                // Use enough precision to round-trip
                sb.append(String.format("%.17g", d));
            } else if (n instanceof Long) {
                long l = n.longValue();
                if (l >= Integer.MIN_VALUE && l <= Integer.MAX_VALUE) {
                    sb.append(l);
                } else {
                    sb.append(l);
                }
            } else {
                sb.append(n.toString());
            }
        }

        private void writeMap(Map<?, ?> map) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : map.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                Object key = e.getKey();
                if (!(key instanceof String)) {
                    throw new IllegalArgumentException("Map keys must be strings");
                }
                writeString((String) key);
                sb.append(':');
                write(e.getValue());
            }
            sb.append('}');
        }

        private void writeList(List<?> list) {
            sb.append('[');
            boolean first = true;
            for (Object v : list) {
                if (!first) sb.append(',');
                first = false;
                write(v);
            }
            sb.append(']');
        }

        @Override
        public String toString() {
            return sb.toString();
        }
    }

    public static Object parse(String json) {
        if (json == null) throw new IllegalArgumentException("JSON string cannot be null");
        try {
            Parser parser = new Parser(json);
            Object result = parser.parse();
            parser.skipWhitespace();
            if (parser.pos < json.length()) {
                throw new IllegalArgumentException("Trailing garbage at position " + parser.pos);
            }
            return result;
        } catch (Exception e) {
            throw new IllegalArgumentException("Malformed JSON at position " + (e.getMessage().contains("position") ? e.getMessage().split("position ")[1] : "unknown") + ": " + e.getMessage(), e);
        }
    }

    public static String write(Object obj) {
        if (obj == null) return "null";
        Writer writer = new Writer();
        writer.write(obj);
        return writer.toString();
    }

    public static String escape(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\b': sb.append("\\b"); break;
                case '\f': sb.append("\\f"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        return sb.toString();
    }
}