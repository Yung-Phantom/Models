package models.ml.Preprocessing.Text.Vectorizer;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TfidfVectorizer extends AbstractTextVectorizer {
    private final boolean binary;
    private final int minDocFreq;
    private final int maxFeatures;
    private final boolean sublinearTf;
    private final boolean l2Normalize;
    Map<Integer, Integer> documentFrequencies;
    double[] idf;

    public TfidfVectorizer() {
        this("\\b\\w+\\b", true, Collections.emptySet(), false, 1, -1, false, true, 1, 1);
    }

    public TfidfVectorizer(String tokenPattern, boolean toLowerCase, Set<String> stopwords,
                           boolean binary, int minDocFreq, int maxFeatures,
                           boolean sublinearTf, boolean l2Normalize,
                           int ngramMin, int ngramMax) {
        super(tokenPattern, toLowerCase, stopwords, ngramMin, ngramMax);
        this.binary = binary;
        this.minDocFreq = Math.max(1, minDocFreq);
        this.maxFeatures = maxFeatures > 0 ? maxFeatures : Integer.MAX_VALUE;
        this.sublinearTf = sublinearTf;
        this.l2Normalize = l2Normalize;
        this.documentFrequencies = new HashMap<>();
    }

    @Override
    protected void buildVocabulary(List<String> docs) {
        if (docs == null) return;
        Map<String, Integer> docFreqMap = new HashMap<>();

        for (String doc : docs) {
            if (doc == null) continue;
            List<String> tokens = tokenize(doc);
            List<String> ngrams = generateNgrams(tokens);
            Set<String> seenInDoc = new HashSet<>();
            for (String ngram : ngrams) {
                if (seenInDoc.add(ngram)) {
                    docFreqMap.put(ngram, docFreqMap.getOrDefault(ngram, 0) + 1);
                }
            }
        }

        this.documentFrequencies = new HashMap<>();
        int index = 0;

        List<Map.Entry<String, Integer>> candidates = new ArrayList<>();
        for (Map.Entry<String, Integer> e : docFreqMap.entrySet()) {
            if (e.getValue() >= minDocFreq) {
                candidates.add(e);
            }
        }

        candidates.sort((a, b) -> {
            int cmp = Integer.compare(b.getValue(), a.getValue());
            if (cmp != 0) return cmp;
            return a.getKey().compareTo(b.getKey());
        });

        for (int i = 0; i < Math.min(candidates.size(), maxFeatures); i++) {
            String term = candidates.get(i).getKey();
            vocabulary.put(term, index++);
            documentFrequencies.put(index - 1, candidates.get(i).getValue());
        }

        // Precompute IDF values: idf = ln((N+1)/(df+1)) + 1
        idf = new double[vocabulary.size()];
        for (Map.Entry<Integer, Integer> e : documentFrequencies.entrySet()) {
            int df = e.getValue();
            idf[e.getKey()] = Math.log((numDocuments + 1.0) / (df + 1.0)) + 1.0;
        }
    }

    @Override
    protected void addToSparseVector(List<String> ngrams, Map<Integer, Double> result) {
        // First pass: compute raw tf (or binary)
        Map<Integer, Double> tf = new HashMap<>();
        for (String ngram : ngrams) {
            Integer idx = vocabulary.get(ngram);
            if (idx != null) {
                if (binary) {
                    tf.put(idx, 1.0);
                } else {
                    tf.put(idx, tf.getOrDefault(idx, 0.0) + 1.0);
                }
            }
        }

        // Second pass: apply TF-IDF
        for (Map.Entry<Integer, Double> e : tf.entrySet()) {
            int idx = e.getKey();
            double tfVal = e.getValue();
            if (sublinearTf && tfVal > 0) {
                tfVal = 1.0 + Math.log(tfVal);
            }
            double tfidfVal = tfVal * idf[idx];
            result.put(idx, tfidfVal);
        }

        // L2 normalization
        if (l2Normalize && !result.isEmpty()) {
            double norm = 0.0;
            for (double v : result.values()) {
                norm += v * v;
            }
            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (Map.Entry<Integer, Double> e : result.entrySet()) {
                    e.setValue(e.getValue() / norm);
                }
            }
        }
    }

    @Override
    public String toJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"binary\":").append(binary)
          .append(",\"minDocFreq\":").append(minDocFreq)
          .append(",\"maxFeatures\":").append(maxFeatures)
          .append(",\"sublinearTf\":").append(sublinearTf)
          .append(",\"l2Normalize\":").append(l2Normalize)
          .append(",\"toLowerCase\":").append(toLowerCase)
          .append(",\"ngramMin\":").append(ngramMin)
          .append(",\"ngramMax\":").append(ngramMax)
          .append(",\"tokenPattern\":\"").append(escape(tokenPattern.pattern())).append("\"")
          .append(",\"stopwords\":[");
        boolean first = true;
        for (String sw : stopwords) {
            if (!first) sb.append(",");
            sb.append("\"").append(escape(sw)).append("\"");
            first = false;
        }
        sb.append("]");
        sb.append(",\"idf\":[");
        first = true;
        for (double v : idf) {
            if (!first) sb.append(",");
            sb.append(v);
            first = false;
        }
        sb.append("]");
        sb.append(",\"vocab\":{");
        int i = 0;
        for (Map.Entry<String, Integer> e : vocabulary.entrySet()) {
            sb.append("\"").append(escape(e.getKey())).append("\":").append(e.getValue());
            if (++i < vocabulary.size()) sb.append(",");
        }
        sb.append("}}");
        return sb.toString();
    }

    public static TfidfVectorizer fromJson(String json) {
        if (json == null || json.isEmpty()) {
            return new TfidfVectorizer();
        }

        // Parse fields
        boolean binary = parseBooleanField(json, "binary", false);
        int minDocFreq = parseIntField(json, "minDocFreq", 1);
        int maxFeatures = parseIntField(json, "maxFeatures", -1);
        boolean sublinearTf = parseBooleanField(json, "sublinearTf", false);
        boolean l2Normalize = parseBooleanField(json, "l2Normalize", true);
        boolean toLowerCase = parseBooleanField(json, "toLowerCase", true);
        int ngramMin = parseIntField(json, "ngramMin", 1);
        int ngramMax = parseIntField(json, "ngramMax", 1);
        String tokenPattern = parseStringField(json, "tokenPattern", "\\b\\w+\\b");

        // Parse stopwords
        Set<String> stopwords = new HashSet<>();
        int swStart = json.indexOf("\"stopwords\":[");
        if (swStart >= 0) {
            int swEnd = json.indexOf("]", swStart);
            if (swEnd > swStart) {
                String swBody = json.substring(swStart + 12, swEnd).trim();
                if (!swBody.isEmpty()) {
                    String[] parts = swBody.split(",");
                    for (String p : parts) {
                        String sw = p.trim();
                        if (sw.startsWith("\"") && sw.endsWith("\"")) {
                            sw = sw.substring(1, sw.length() - 1);
                        }
                        stopwords.add(unescape(sw));
                    }
                }
            }
        }

        TfidfVectorizer vec = new TfidfVectorizer(tokenPattern, toLowerCase, stopwords,
                binary, minDocFreq, maxFeatures, sublinearTf, l2Normalize, ngramMin, ngramMax);

        // Parse IDF array
        int idfStart = json.indexOf("\"idf\":[");
        if (idfStart < 0) {
            throw new IllegalArgumentException("Missing 'idf' field in JSON");
        }
        int idfEnd = json.indexOf("]", idfStart);
        if (idfEnd <= idfStart) {
            throw new IllegalArgumentException("Malformed 'idf' field in JSON");
        }
        String idfBody = json.substring(idfStart + 7, idfEnd).trim();
        if (idfBody.isEmpty()) {
            throw new IllegalArgumentException("'idf' array is empty");
        }
        String[] parts = idfBody.split(",");
        vec.idf = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                vec.idf[i] = Double.parseDouble(parts[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid number in 'idf': " + parts[i].trim());
            }
        }

        // Parse vocabulary - required field
        int vocabIdx = json.indexOf("\"vocab\":{");
        if (vocabIdx < 0) {
            throw new IllegalArgumentException("Missing 'vocab' field in JSON");
        }
        int start = json.indexOf('{', vocabIdx);
        int end = findMatchingBrace(json, start);
        if (start < 0 || end <= start) {
            throw new IllegalArgumentException("Malformed 'vocab' field in JSON");
        }
        String body = json.substring(start + 1, end).trim();
        vec.parseVocabularyJson(body);
        if (vec.vocabulary.isEmpty()) {
            throw new IllegalArgumentException("Vocabulary is empty after parsing JSON");
        }
        // Rebuild documentFrequencies from IDF if needed
        vec.documentFrequencies = new HashMap<>();
        for (Map.Entry<String, Integer> e : vec.vocabulary.entrySet()) {
            vec.documentFrequencies.put(e.getValue(), 1); // placeholder, not used after fit
        }

        return vec;
    }

    private static boolean parseBooleanField(String json, String field, boolean defaultValue) {
        int idx = json.indexOf("\"" + field + "\":");
        if (idx < 0) return defaultValue;
        int start = idx + field.length() + 3;
        if (start + 4 <= json.length() && json.substring(start, start + 4).equals("true")) return true;
        if (start + 5 <= json.length() && json.substring(start, start + 5).equals("false")) return false;
        return defaultValue;
    }

    private static int parseIntField(String json, String field, int defaultValue) {
        int idx = json.indexOf("\"" + field + "\":");
        if (idx < 0) return defaultValue;
        int start = idx + field.length() + 3;
        int end = start;
        while (end < json.length() && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '-')) end++;
        if (end <= start) return defaultValue;
        try {
            return Integer.parseInt(json.substring(start, end));
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private static String parseStringField(String json, String field, String defaultValue) {
        int idx = json.indexOf("\"" + field + "\":\"");
        if (idx < 0) return defaultValue;
        int start = idx + field.length() + 4;
        int end = json.indexOf('"', start);
        if (end < start) return defaultValue;
        return unescape(json.substring(start, end));
    }
}