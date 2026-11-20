package models.ml.Preprocessing.Text.Vectorizer;

import models.ml.Utils.Json;
import java.util.*;
import java.util.regex.Pattern;

public class BagOfWordsVectorizer extends AbstractTextVectorizer {
    private final boolean binary;
    private final int minDocFreq;
    private final int maxFeatures;
    private Map<Integer, Integer> documentFrequencies;

    public BagOfWordsVectorizer() {
        this("\\b\\w+\\b", true, Collections.emptySet(), false, 1, -1, 1, 1);
    }

    public BagOfWordsVectorizer(String tokenPattern, boolean toLowerCase, Set<String> stopwords,
                                boolean binary, int minDocFreq, int maxFeatures,
                                int ngramMin, int ngramMax) {
        super(tokenPattern, toLowerCase, stopwords, ngramMin, ngramMax);
        this.binary = binary;
        this.minDocFreq = Math.max(1, minDocFreq);
        this.maxFeatures = maxFeatures > 0 ? maxFeatures : Integer.MAX_VALUE;
        this.documentFrequencies = new HashMap<>();
    }

    @Override
    protected void buildVocabulary(List<String> docs) {
        if (docs == null) {
            throw new IllegalArgumentException("Documents list cannot be null");
        }
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

        // Filter by minDocFreq and collect candidates
        List<Map.Entry<String, Integer>> candidates = new ArrayList<>();
        for (Map.Entry<String, Integer> e : docFreqMap.entrySet()) {
            if (e.getValue() >= minDocFreq) {
                candidates.add(e);
            }
        }

        // Sort by document frequency descending, then alphabetically for ties
        candidates.sort((a, b) -> {
            int cmp = Integer.compare(b.getValue(), a.getValue()); // descending by df
            if (cmp != 0) return cmp;
            return a.getKey().compareTo(b.getKey()); // alphabetical tie-break
        });

        // Keep top maxFeatures
        for (int i = 0; i < Math.min(candidates.size(), maxFeatures); i++) {
            vocabulary.put(candidates.get(i).getKey(), index++);
            documentFrequencies.put(index - 1, candidates.get(i).getValue());
        }

        fitted = true;
    }

    @Override
    protected void addToSparseVector(List<String> ngrams, Map<Integer, Double> result) {
        if (!fitted) {
            throw new IllegalStateException("Vectorizer must be fitted before transform");
        }
        for (String ngram : ngrams) {
            Integer idx = vocabulary.get(ngram);
            if (idx != null) {
                if (binary) {
                    result.put(idx, 1.0);
                } else {
                    result.put(idx, result.getOrDefault(idx, 0.0) + 1.0);
                }
            }
        }
    }

    @Override
    public String toJson() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("binary", binary);
        map.put("minDocFreq", minDocFreq);
        map.put("maxFeatures", maxFeatures);
        map.put("toLowerCase", toLowerCase);
        map.put("ngramMin", ngramMin);
        map.put("ngramMax", ngramMax);
        map.put("tokenPattern", tokenPattern.pattern());
        List<String> stopwordsList = new ArrayList<>(stopwords);
        map.put("stopwords", stopwordsList);
        map.put("vocab", vocabulary);
        map.put("fitted", fitted);
        return Json.write(map);
    }

    public static BagOfWordsVectorizer fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }

        Object parsed = Json.parse(json);
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("JSON root must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> map = (Map<String, Object>) parsed;

        // Required fields
        Boolean binaryObj = getBoolean(map, "binary");
        Object minDocFreqObj = map.get("minDocFreq");
        Object maxFeaturesObj = map.get("maxFeatures");
        Boolean toLowerCaseObj = getBoolean(map, "toLowerCase");
        Object ngramMinObj = map.get("ngramMin");
        Object ngramMaxObj = map.get("ngramMax");
        String tokenPattern = getString(map, "tokenPattern");
        Object stopwordsObj = map.get("stopwords");
        Object vocabObj = map.get("vocab");
        Boolean fittedObj = getBoolean(map, "fitted");

        if (binaryObj == null) throw new IllegalArgumentException("Missing required field: binary");
        if (minDocFreqObj == null) throw new IllegalArgumentException("Missing required field: minDocFreq");
        if (maxFeaturesObj == null) throw new IllegalArgumentException("Missing required field: maxFeatures");
        if (toLowerCaseObj == null) throw new IllegalArgumentException("Missing required field: toLowerCase");
        if (ngramMinObj == null) throw new IllegalArgumentException("Missing required field: ngramMin");
        if (ngramMaxObj == null) throw new IllegalArgumentException("Missing required field: ngramMax");
        if (tokenPattern == null) throw new IllegalArgumentException("Missing required field: tokenPattern");
        if (vocabObj == null) throw new IllegalArgumentException("Missing required field: vocab");

        // Validate whole numbers for integer fields
        requireInteger(map, "minDocFreq");
        requireInteger(map, "maxFeatures");
        requireInteger(map, "ngramMin");
        requireInteger(map, "ngramMax");

        int minDocFreq = ((Number) minDocFreqObj).intValue();
        int maxFeatures = ((Number) maxFeaturesObj).intValue();
        int ngramMin = ((Number) ngramMinObj).intValue();
        int ngramMax = ((Number) ngramMaxObj).intValue();

        if (ngramMin < 1) throw new IllegalArgumentException("ngramMin must be >= 1");
        if (ngramMax < ngramMin) throw new IllegalArgumentException("ngramMax must be >= ngramMin");

        // Validate tokenPattern is a valid regex
        try {
            Pattern.compile(tokenPattern);
        } catch (Exception e) {
            throw new IllegalArgumentException("Invalid tokenPattern regex: " + e.getMessage());
        }

        // Parse stopwords - must be a list of strings
        Set<String> stopwords = new HashSet<>();
        if (stopwordsObj instanceof List) {
            for (Object sw : (List<?>) stopwordsObj) {
                if (!(sw instanceof String)) {
                    throw new IllegalArgumentException("stopwords must be a list of strings");
                }
                stopwords.add((String) sw);
            }
        } else if (stopwordsObj != null) {
            throw new IllegalArgumentException("stopwords must be a list");
        }

        // Parse vocabulary - must be object with whole number values
        Map<String, Integer> vocab = new LinkedHashMap<>();
        if (vocabObj instanceof Map) {
            for (Map.Entry<?, ?> e : ((Map<?, ?>) vocabObj).entrySet()) {
                if (!(e.getKey() instanceof String)) {
                    throw new IllegalArgumentException("Vocabulary key must be string");
                }
                Object val = e.getValue();
                if (!(val instanceof Number)) {
                    throw new IllegalArgumentException("Vocabulary value for key '" + e.getKey() + "' must be a whole number");
                }
                double d = ((Number) val).doubleValue();
                if (d != Math.floor(d)) {
                    throw new IllegalArgumentException("Vocabulary value for key '" + e.getKey() + "' must be a whole number, got " + d);
                }
                vocab.put((String) e.getKey(), ((Number) val).intValue());
            }
        } else {
            throw new IllegalArgumentException("vocab must be an object");
        }

        // Validate vocabulary indices are exactly 0..n-1 with no duplicates
        if (!vocab.isEmpty()) {
            int n = vocab.size();
            boolean[] seen = new boolean[n];
            for (int idx : vocab.values()) {
                if (idx < 0 || idx >= n) {
                    throw new IllegalArgumentException("Vocabulary index out of range: " + idx + " (size=" + n + ")");
                }
                if (seen[idx]) {
                    throw new IllegalArgumentException("Duplicate vocabulary index: " + idx);
                }
                seen[idx] = true;
            }
        }

        BagOfWordsVectorizer vec = new BagOfWordsVectorizer(tokenPattern, getBoolean(map, "toLowerCase"), stopwords,
                binaryObj, minDocFreq, maxFeatures, ngramMin, ngramMax);
        vec.vocabulary.putAll(vocab);
        // If "fitted" field is missing, default to true if vocab is not empty (backward compatibility)
        vec.fitted = fittedObj != null ? fittedObj : !vocab.isEmpty();

        return vec;
    }
}