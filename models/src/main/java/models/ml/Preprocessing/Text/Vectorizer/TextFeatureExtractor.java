package models.ml.Preprocessing.Text.Vectorizer;

import models.ml.Preprocessing.Features.FeatureExtractor;

import java.util.List;
import java.util.Map;

/**
 * Adapter that wraps a {@link TextVectorizer} as a generic {@link FeatureExtractor<String>}.
 * <p>
 * Pure delegation: every method forwards to the underlying {@code TextVectorizer}.
 * This allows text vectorizers to be used with the generic {@link models.ml.Pipeline.FeaturePipeline}.
 * <p>
 * Not thread-safe.
 */
public final class TextFeatureExtractor implements FeatureExtractor<String> {

    private final TextVectorizer vectorizer;

    /**
     * Creates an adapter for the given vectorizer.
     *
     * @param vectorizer the text vectorizer to wrap; must not be null
     * @throws IllegalArgumentException if vectorizer is null
     */
    public TextFeatureExtractor(TextVectorizer vectorizer) {
        if (vectorizer == null) {
            throw new IllegalArgumentException("vectorizer cannot be null");
        }
        this.vectorizer = vectorizer;
    }

    /**
     * Returns the wrapped vectorizer.
     *
     * @return the underlying TextVectorizer
     */
    public TextVectorizer getVectorizer() {
        return vectorizer;
    }

    @Override
    public void fit(List<String> inputs) {
        vectorizer.fit(inputs);
    }

    @Override
    public Map<Integer, Double> transform(String input) {
        return vectorizer.transform(input);
    }

    @Override
    public List<Map<Integer, Double>> transform(List<String> inputs) {
        return vectorizer.transform(inputs);
    }

    @Override
    public int getFeatureSize() {
        return vectorizer.getFeatureSize();
    }

    @Override
    public boolean isFitted() {
        return vectorizer.isFitted();
    }

    @Override
    public String toJson() {
        return vectorizer.toJson();
    }

    /**
     * Reconstructs a TextFeatureExtractor from its JSON representation.
     *
     * @param json JSON string produced by {@link #toJson()}; must not be null or empty
     * @return new TextFeatureExtractor instance with the same configuration and fitted state
     * @throws IllegalArgumentException if JSON is malformed, missing fields, or has invalid values
     */
    public static TextFeatureExtractor fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }
        Object parsed = models.ml.Utils.Json.parse(json);
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("JSON root must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> map = (Map<String, Object>) parsed;

        // Detect vectorizer type - currently TF-IDF and BagOfWords are supported
        // Check for TF-IDF specific fields (has idf)
        if (map.containsKey("idf") && map.containsKey("vocab") && map.containsKey("binary")) {
            TfidfVectorizer vec = TfidfVectorizer.fromJson(json);
            return new TextFeatureExtractor(vec);
        }
        // Check for BagOfWords specific fields (no idf, has vocab)
        if (map.containsKey("vocab") && map.containsKey("binary") && !map.containsKey("idf")) {
            BagOfWordsVectorizer vec = BagOfWordsVectorizer.fromJson(json);
            return new TextFeatureExtractor(vec);
        }

        throw new IllegalArgumentException("Unknown or unsupported vectorizer type");
    }
}