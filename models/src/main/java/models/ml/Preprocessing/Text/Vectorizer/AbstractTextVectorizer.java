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
    protected boolean fitted = false;

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
            throw new IllegalArgumentException("Documents list cannot be null");
        }
        vocabulary.clear();
        numDocuments = 0;
        for (String doc : docs) {
            if (doc != null) {
                numDocuments++;
            }
        }
        buildVocabulary(docs);
        fitted = true;
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
        return fitted;
    }

    // ==================== JSON parsing helpers ====================

    protected static Boolean getBoolean(Map<String, Object> map, String key) {
        Object v = map.get(key);
        if (v instanceof Boolean) return (Boolean) v;
        return null;
    }

    protected static Integer getInteger(Map<String, Object> map, String key) {
        Object v = map.get(key);
        if (v instanceof Number) return ((Number) v).intValue();
        return null;
    }

    protected static String getString(Map<String, Object> map, String key) {
        Object v = map.get(key);
        if (v instanceof String) return (String) v;
        return null;
    }

    protected static void requireInteger(Map<String, Object> map, String key) {
        Object v = map.get(key);
        if (!(v instanceof Number)) {
            throw new IllegalArgumentException("Field '" + key + "' must be a whole number");
        }
        double d = ((Number) v).doubleValue();
        if (d != Math.floor(d)) {
            throw new IllegalArgumentException("Field '" + key + "' must be a whole number, got " + d);
        }
    }

    protected static void requireFiniteDouble(Map<String, Object> map, String key) {
        Object v = map.get(key);
        if (!(v instanceof Number)) {
            throw new IllegalArgumentException("Field '" + key + "' must be a number");
        }
        double d = ((Number) v).doubleValue();
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            throw new IllegalArgumentException("Field '" + key + "' must be a finite number, got " + d);
        }
    }
}