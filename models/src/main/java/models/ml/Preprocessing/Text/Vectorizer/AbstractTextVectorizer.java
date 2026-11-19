package models.ml.Preprocessing.Text.Vectorizer;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public abstract class AbstractTextVectorizer implements TextVectorizer {
    protected final Map<String, Integer> vocabulary = new LinkedHashMap<>();
    protected final Pattern tokenPattern;
    protected final boolean toLowerCase;
    protected final Set<String> stopwords;
    protected final int ngramMin;
    protected final int ngramMax;
    protected int numDocuments = 0;

    protected AbstractTextVectorizer(String tokenPattern, boolean toLowerCase, Set<String> stopwords,
                                     int ngramMin, int ngramMax) {
        this.tokenPattern = Pattern.compile(tokenPattern);
        this.toLowerCase = toLowerCase;
        this.stopwords = stopwords != null ? new HashSet<>(stopwords) : Collections.emptySet();
        this.ngramMin = Math.max(1, ngramMin);
        this.ngramMax = Math.max(this.ngramMin, ngramMax);
    }

    /** Returns tokens after applying lowercase and stopword filtering. */
    protected List<String> tokenize(String doc) {
        if (doc == null || doc.isEmpty()) {
            return Collections.emptyList();
        }
        String processed = toLowerCase ? doc.toLowerCase(Locale.ROOT) : doc;
        Matcher matcher = tokenPattern.matcher(processed);
        List<String> tokens = new ArrayList<>();
        while (matcher.find()) {
            String token = matcher.group();
            if (!stopwords.contains(token)) {
                tokens.add(token);
            }
        }
        return tokens;
    }

    /** Generate n-grams from tokens. Word n-grams are joined by a single space. */
    protected List<String> generateNgrams(List<String> tokens) {
        List<String> ngrams = new ArrayList<>();
        int n = tokens.size();
        for (int len = ngramMin; len <= ngramMax && len <= n; len++) {
            for (int i = 0; i <= n - len; i++) {
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < len; j++) {
                    if (j > 0) sb.append(' ');
                    sb.append(tokens.get(i + j));
                }
                ngrams.add(sb.toString());
            }
        }
        return ngrams;
    }

    /** Build vocabulary from documents. Must be implemented by subclasses. */
    protected abstract void buildVocabulary(List<String> docs);

    @Override
    public void fit(List<String> docs) {
        if (docs == null) {
            return;
        }
        vocabulary.clear();
        numDocuments = 0;
        for (String doc : docs) {
            if (doc != null) {
                numDocuments++;
            }
        }
        buildVocabulary(docs);
    }

    @Override
    public Map<Integer, Double> transform(String doc) {
        if (!isFitted()) {
            throw new IllegalStateException("Vectorizer must be fitted before transform");
        }
        if (doc == null) {
            throw new IllegalArgumentException("Document cannot be null");
        }
        Map<Integer, Double> result = new HashMap<>();
        if (doc.isEmpty() || vocabulary.isEmpty()) {
            return result;
        }
        List<String> tokens = tokenize(doc);
        List<String> ngrams = generateNgrams(tokens);
        addToSparseVector(ngrams, result);
        return result;
    }

    @Override
    public List<Map<Integer, Double>> transform(List<String> docs) {
        if (!isFitted()) {
            throw new IllegalStateException("Vectorizer must be fitted before transform");
        }
        if (docs == null) {
            return Collections.emptyList();
        }
        List<Map<Integer, Double>> result = new ArrayList<>(docs.size());
        for (String doc : docs) {
            result.add(transform(doc));
        }
        return result;
    }

    /** Add n-grams to sparse vector. Implemented by subclasses (count vs TF-IDF). */
    protected abstract void addToSparseVector(List<String> ngrams, Map<Integer, Double> result);

    @Override
    public int getFeatureSize() {
        return vocabulary.size();
    }

    @Override
    public boolean isFitted() {
        return !vocabulary.isEmpty();
    }

    /** Escape JSON string: backslash and double quote. */
    protected static String escape(String s) {
        return s == null ? "" : s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    /** Unescape JSON string. */
    protected static String unescape(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char next = s.charAt(i + 1);
                if (next == '"' || next == '\\') {
                    sb.append(next);
                    i++;
                } else {
                    sb.append(c);
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /** Find matching brace in JSON object. */
    protected static int findMatchingBrace(String s, int start) {
        int depth = 0;
        for (int i = start; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '{') depth++;
            else if (c == '}') {
                depth--;
                if (depth == 0) return i;
            }
        }
        return -1;
    }

    /** Parse vocabulary from JSON body (key->index). Handles escaped quotes and backslashes. */
    protected void parseVocabularyJson(String body) {
        vocabulary.clear();
        int pos = 0;
        while (pos < body.length()) {
            while (pos < body.length() && (Character.isWhitespace(body.charAt(pos)) || body.charAt(pos) == ',')) pos++;
            if (pos >= body.length() || body.charAt(pos) != '"') break;
            pos++; // skip opening quote
            // Parse key with escape handling
            StringBuilder keyBuilder = new StringBuilder();
            while (pos < body.length()) {
                char c = body.charAt(pos);
                if (c == '\\' && pos + 1 < body.length()) {
                    // Escape sequence
                    pos++;
                    char next = body.charAt(pos);
                    if (next == '"' || next == '\\') {
                        keyBuilder.append(next);
                    } else {
                        keyBuilder.append('\\').append(next);
                    }
                    pos++;
                } else if (c == '"') {
                    // End of key
                    pos++;
                    break;
                } else {
                    keyBuilder.append(c);
                    pos++;
                }
            }
            String key = keyBuilder.toString();

            while (pos < body.length() && (Character.isWhitespace(body.charAt(pos)) || body.charAt(pos) == ':')) pos++;
            int numStart = pos;
            while (pos < body.length() && (Character.isDigit(body.charAt(pos)) || body.charAt(pos) == '-')) pos++;
            if (numStart == pos) break;
            try {
                int idx = Integer.parseInt(body.substring(numStart, pos));
                vocabulary.put(key, idx);
            } catch (NumberFormatException ignored) {}
        }
    }
}