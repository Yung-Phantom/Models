package models.ml.Preprocessing.Text.BagOfWords;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AbstractBagOfWords {

    private final Map<String, Integer> vocabulary; // deterministic order
    private final Pattern tokenPattern;
    private final boolean toLowerCase;
    private final Set<String> stopwords;

    // Default constructor
    public AbstractBagOfWords() {
        this("\\b\\w+\\b", true, Collections.emptySet());
    }

    // Configurable constructor
    public AbstractBagOfWords(String tokenPattern, boolean toLowerCase, Set<String> stopwords) {
        this.vocabulary = new LinkedHashMap<>();
        this.tokenPattern = Pattern.compile(tokenPattern);
        this.toLowerCase = toLowerCase;
        this.stopwords = stopwords != null ? stopwords : Collections.emptySet();
    }

    /** -------------------- FIT -------------------- */
    public void fit(List<String> corpus) {
        if (corpus == null)
            return;
        int index = vocabulary.size();

        for (String doc : corpus) {
            if (doc == null)
                continue;
            Matcher matcher = tokenPattern.matcher(preprocess(doc));
            while (matcher.find()) {
                String token = matcher.group();
                if (!stopwords.contains(token) && !vocabulary.containsKey(token)) {
                    vocabulary.put(token, index++);
                }
            }
        }
    }

    /** -------------------- TRANSFORM (SPARSE) -------------------- */
    public Map<Integer, Integer> transform(String doc) {
        Map<Integer, Integer> sparseVector = new HashMap<>();
        if (doc == null || doc.isEmpty() || vocabulary.isEmpty())
            return sparseVector;

        Matcher matcher = tokenPattern.matcher(preprocess(doc));
        while (matcher.find()) {
            String token = matcher.group();
            Integer idx = vocabulary.get(token);
            if (idx != null) {
                sparseVector.put(idx, sparseVector.getOrDefault(idx, 0) + 1);
            }
        }
        return sparseVector;
    }

    public List<Map<Integer, Integer>> transform(List<String> corpus) {
        List<Map<Integer, Integer>> result = new ArrayList<>();
        if (corpus == null)
            return result;
        for (String doc : corpus) {
            result.add(transform(doc));
        }
        return result;
    }

    /** -------------------- TRANSFORM (DENSE) -------------------- */
    public double[] transformDense(String doc) {
        double[] vec = new double[vocabulary.size()];
        if (doc == null || doc.isEmpty() || vocabulary.isEmpty())
            return vec;

        Matcher matcher = tokenPattern.matcher(preprocess(doc));
        while (matcher.find()) {
            String token = matcher.group();
            Integer idx = vocabulary.get(token);
            if (idx != null)
                vec[idx] += 1.0;
        }
        return vec;
    }

    public List<double[]> transformDense(List<String> corpus) {
        List<double[]> result = new ArrayList<>();
        if (corpus == null)
            return result;
        for (String doc : corpus) {
            result.add(transformDense(doc));
        }
        return result;
    }

    /** -------------------- FIT + TRANSFORM -------------------- */
    public Map<Integer, Integer> fitTransform(String doc) {
        fit(Collections.singletonList(doc));
        return transform(doc);
    }

    public List<Map<Integer, Integer>> fitTransform(List<String> corpus) {
        fit(corpus);
        return transform(corpus);
    }

    /** -------------------- VOCABULARY -------------------- */
    public Map<String, Integer> getVocabulary() {
        return Collections.unmodifiableMap(vocabulary);
    }

    public int getFeatureSize() {
        return vocabulary.size();
    }

    /** -------------------- JSON (DE)SERIALIZATION -------------------- */
    public String toJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"vocab\":{");
        int i = 0;
        for (Map.Entry<String, Integer> e : vocabulary.entrySet()) {
            sb.append("\"").append(escape(e.getKey())).append("\":").append(e.getValue());
            if (++i < vocabulary.size())
                sb.append(",");
        }
        sb.append("}}");
        return sb.toString();
    }

    public static AbstractBagOfWords fromJson(String json) {
        AbstractBagOfWords bow = new AbstractBagOfWords();
        if (json == null || json.isEmpty())
            return bow;

        int vocabIdx = json.indexOf("\"vocab\"");
        if (vocabIdx < 0)
            return bow;
        int start = json.indexOf('{', vocabIdx);
        if (start < 0)
            return bow;
        int end = findMatchingBrace(json, start);
        if (end < 0)
            return bow;

        String body = json.substring(start + 1, end).trim();
        int pos = 0;
        while (pos < body.length()) {
            while (pos < body.length() && (Character.isWhitespace(body.charAt(pos)) || body.charAt(pos) == ','))
                pos++;
            if (pos >= body.length() || body.charAt(pos) != '"')
                break;

            int keyStart = pos + 1;
            int keyEnd = body.indexOf('"', keyStart);
            if (keyEnd < 0)
                break;
            String key = unescape(body.substring(keyStart, keyEnd));
            pos = keyEnd + 1;
            while (pos < body.length() && (Character.isWhitespace(body.charAt(pos)) || body.charAt(pos) == ':'))
                pos++;

            int numStart = pos;
            while (pos < body.length() && (Character.isDigit(body.charAt(pos)) || body.charAt(pos) == '-'))
                pos++;
            if (numStart == pos)
                break;

            try {
                int idx = Integer.parseInt(body.substring(numStart, pos));
                bow.vocabulary.put(key, idx);
            } catch (NumberFormatException ignored) {
            }
        }
        return bow;
    }

    /** -------------------- HELPERS -------------------- */
    private String preprocess(String doc) {
        return toLowerCase ? doc.toLowerCase(Locale.ROOT) : doc;
    }

    private static int findMatchingBrace(String s, int start) {
        int depth = 0;
        for (int i = start; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '{')
                depth++;
            else if (c == '}') {
                depth--;
                if (depth == 0)
                    return i;
            }
        }
        return -1;
    }

    private static String escape(String s) {
        return s == null ? "" : s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String unescape(String s) {
        return s == null ? "" : s.replace("\\\"", "\"").replace("\\\\", "\\");
    }

    public Map<Integer, Integer> mergeSparse(List<Map<Integer, Integer>> tokenMaps) {
        Map<Integer, Integer> merged = new HashMap<>();
        for (Map<Integer, Integer> m : tokenMaps) {
            for (Map.Entry<Integer, Integer> e : m.entrySet()) {
                merged.put(e.getKey(), merged.getOrDefault(e.getKey(), 0) + e.getValue());
            }
        }
        return merged;
    }
}