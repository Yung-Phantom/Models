package models.ml.Preprocessing.Text.TFIDF;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AbstractTFIDF {

    private final Map<String, Integer> vocabulary;
    private final Map<Integer, Integer> documentFrequencies;
    private final Pattern tokenPattern;
    private int numDocuments = 0;

    public AbstractTFIDF() {
        this("\\b\\w+\\b");
    }

    public AbstractTFIDF(String tokenPattern) {
        this.vocabulary = new HashMap<>();
        this.documentFrequencies = new HashMap<>();
        this.tokenPattern = Pattern.compile(tokenPattern);
    }

    /**
     * Build vocabulary and document frequencies
     */
    public void fit(List<String> corpus) {
        int index = 0;
        numDocuments = corpus.size();

        for (String doc : corpus) {
            Set<String> seen = new HashSet<>();
            Matcher matcher = tokenPattern.matcher(doc.toLowerCase());

            while (matcher.find()) {
                String token = matcher.group();

                if (!vocabulary.containsKey(token)) {
                    vocabulary.put(token, index++);
                }

                if (seen.add(token)) {
                    int idx = vocabulary.get(token);
                    documentFrequencies.put(idx,
                            documentFrequencies.getOrDefault(idx, 0) + 1);
                }
            }
        }
    }

    /**
     * Transform corpus into TF-IDF sparse vectors
     */
    public List<Map<Integer, Double>> transform(List<String> corpus) {
        List<Map<Integer, Double>> result = new ArrayList<>();

        for (String doc : corpus) {
            Map<Integer, Integer> tf = new HashMap<>();
            Matcher matcher = tokenPattern.matcher(doc.toLowerCase());

            while (matcher.find()) {
                String token = matcher.group();
                Integer idx = vocabulary.get(token);
                if (idx != null) {
                    tf.put(idx, tf.getOrDefault(idx, 0) + 1);
                }
            }

            Map<Integer, Double> tfidf = new HashMap<>();
            for (Map.Entry<Integer, Integer> e : tf.entrySet()) {
                int idx = e.getKey();
                int termFreq = e.getValue();
                int df = documentFrequencies.getOrDefault(idx, 1);

                double idf = Math.log((numDocuments + 1.0) / (df + 1.0)) + 1.0;
                tfidf.put(idx, termFreq * idf);
            }

            result.add(tfidf);
        }

        return result;
    }

    public List<Map<Integer, Double>> fitTransform(List<String> corpus) {
        fit(corpus);
        return transform(corpus);
    }

    public Map<String, Integer> getVocabulary() {
        return Collections.unmodifiableMap(vocabulary);
    }
}
