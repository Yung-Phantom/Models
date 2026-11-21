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
}