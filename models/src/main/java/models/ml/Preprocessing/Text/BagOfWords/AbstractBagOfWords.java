package models.ml.Preprocessing.Text.BagOfWords;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AbstractBagOfWords {

    private final Map<String, Integer> vocabulary;
    private final Pattern tokenPattern;

    // Default constructor (sklearn-like default)
    public AbstractBagOfWords() {
        this("\\b\\w+\\b");
    }

    // Configurable token pattern
    public AbstractBagOfWords(String tokenPattern) {
        this.vocabulary = new HashMap<>();
        this.tokenPattern = Pattern.compile(tokenPattern);
    }

    /**
     * Build vocabulary from corpus
     */
    public void fit(List<String> corpus) {
        int index = 0;

        for (String doc : corpus) {
            Matcher matcher = tokenPattern.matcher(doc.toLowerCase());

            while (matcher.find()) {
                String token = matcher.group();
                if (!vocabulary.containsKey(token)) {
                    vocabulary.put(token, index++);
                }
            }
        }
    }

    /**
     * Transform corpus into sparse vectors
     */
    public List<Map<Integer, Integer>> transform(List<String> corpus) {
        List<Map<Integer, Integer>> result = new ArrayList<>();

        for (String doc : corpus) {
            Map<Integer, Integer> sparseVector = new HashMap<>();
            Matcher matcher = tokenPattern.matcher(doc.toLowerCase());

            while (matcher.find()) {
                String token = matcher.group();
                Integer index = vocabulary.get(token);
                if (index != null) {
                    sparseVector.put(index, sparseVector.getOrDefault(index, 0) + 1);
                }
            }
            result.add(sparseVector);
        }
        return result;
    }

    /**
     * Fit + transform
     */
    public List<Map<Integer, Integer>> fitTransform(List<String> corpus) {
        fit(corpus);
        return transform(corpus);
    }

    /**
     * Expose vocabulary
     */
    public Map<String, Integer> getVocabulary() {
        return Collections.unmodifiableMap(vocabulary);
    }
}
