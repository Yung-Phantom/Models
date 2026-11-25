package models.ml.Preprocessing.Features;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Abstract base class for dense feature extractors.
 * <p>
 * Provides shared implementation for dense feature extractors including
 * fit validation, fitted flag management, and dense/sparse transform methods.
 * Concrete subclasses must implement {@link #extractDense(Object)} and {@link #getFeatureSize()}.
 * <p>
 * Not thread-safe.
 *
 * @param <T> the input type
 */
public abstract class AbstractDenseFeatureExtractor<T> implements FeatureExtractor<T> {

    private boolean fitted = false;

    @Override
    public void fit(List<T> inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("inputs list cannot be null");
        }
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs.get(i) == null) {
                throw new IllegalArgumentException("input at index " + i + " cannot be null");
            }
        }
        // Empty list is allowed; extractors with no trainable parameters (e.g., stateless)
        // remain fitted. Subclasses requiring data should override and validate.
        fitted = true;
    }

    /**
     * Marks the extractor as fitted without running validation.
     * Intended for use by {@code fromJson} methods to restore fitted state.
     */
    protected void markFitted() {
        this.fitted = true;
    }

    /**
     * Extracts dense features from a single input.
     *
     * @param input input data; must not be null
     * @return dense feature vector of length {@link #getFeatureSize()}
     */
    protected abstract double[] extractDense(T input);

    /**
     * Returns the feature dimensionality.
     * Available before and after fit.
     */
    public abstract int getFeatureSize();

    @Override
    public boolean isFitted() {
        return fitted;
    }

    private void requireFitted() {
        if (!fitted) {
            throw new IllegalStateException("Extractor must be fitted before transform");
        }
    }

    private void validateInputList(List<T> inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("inputs list cannot be null");
        }
    }

    @Override
    public double[] transformDense(T input) {
        requireFitted();
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        return extractDense(input);
    }

    @Override
    public double[][] transformDense(List<T> inputs) {
        requireFitted();
        validateInputList(inputs);
        double[][] result = new double[inputs.size()][];
        for (int i = 0; i < inputs.size(); i++) {
            T input = inputs.get(i);
            if (input == null) {
                throw new IllegalArgumentException("input at index " + i + " cannot be null");
            }
            result[i] = extractDense(input);
        }
        return result;
    }

    @Override
    public Map<Integer, Double> transform(T input) {
        requireFitted();
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        return FeatureVectors.toSparse(extractDense(input));
    }

    @Override
    public List<Map<Integer, Double>> transform(List<T> inputs) {
        requireFitted();
        validateInputList(inputs);
        List<Map<Integer, Double>> result = new ArrayList<>(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            T input = inputs.get(i);
            if (input == null) {
                throw new IllegalArgumentException("input at index " + i + " cannot be null");
            }
            result.add(FeatureVectors.toSparse(extractDense(input)));
        }
        return result;
    }

    @Override
    public String toJson() {
        if (!fitted) {
            throw new IllegalStateException("Extractor must be fitted before toJson");
        }
        return doToJson();
    }

    /**
     * Subclass-specific JSON serialization.
     * Called only after fit; implementation must include a "fitted": true field.
     */
    protected abstract String doToJson();
}