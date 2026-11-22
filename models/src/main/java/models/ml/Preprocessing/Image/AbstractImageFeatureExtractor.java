package models.ml.Preprocessing.Image;

import models.ml.Preprocessing.Features.FeatureExtractor;
import models.ml.Preprocessing.Features.FeatureVectors;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Abstract base class for image feature extractors.
 * <p>
 * Concrete subclasses must implement {@link #extractDense(ImageData)} and {@link #getFeatureSize()}.
 * The {@code fit} method only validates input and marks the extractor as fitted; no learning occurs.
 * <p>
 * Not thread-safe.
 */
public abstract class AbstractImageFeatureExtractor implements FeatureExtractor<ImageData> {

    private boolean fitted = false;

    @Override
    public void fit(List<ImageData> inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("inputs list cannot be null");
        }
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs.get(i) == null) {
                throw new IllegalArgumentException("input at index " + i + " cannot be null");
            }
        }
        // Empty list is allowed (matches VectorizerTest precedent for fit)
        fitted = true;
    }

    /**
     * Marks the extractor as fitted without running validation.
     * Intended for use by {@code fromJson} methods to restore fitted state.
     */
    void markFitted() {
        this.fitted = true;
    }

    /**
     * Extracts dense features from a single image.
     *
     * @param image input image; must not be null
     * @return dense feature vector of length {@link #getFeatureSize()}
     */
    protected abstract double[] extractDense(ImageData image);

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

    private void validateInputList(List<ImageData> inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("inputs list cannot be null");
        }
    }

    @Override
    public double[] transformDense(ImageData input) {
        requireFitted();
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        return extractDense(input);
    }

    @Override
    public double[][] transformDense(List<ImageData> inputs) {
        requireFitted();
        validateInputList(inputs);
        double[][] result = new double[inputs.size()][];
        for (int i = 0; i < inputs.size(); i++) {
            ImageData img = inputs.get(i);
            if (img == null) {
                throw new IllegalArgumentException("input at index " + i + " cannot be null");
            }
            result[i] = extractDense(img);
        }
        return result;
    }

    @Override
    public Map<Integer, Double> transform(ImageData input) {
        requireFitted();
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        return FeatureVectors.toSparse(extractDense(input));
    }

    @Override
    public List<Map<Integer, Double>> transform(List<ImageData> inputs) {
        requireFitted();
        validateInputList(inputs);
        List<Map<Integer, Double>> result = new ArrayList<>(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            ImageData img = inputs.get(i);
            if (img == null) {
                throw new IllegalArgumentException("input at index " + i + " cannot be null");
            }
            result.add(FeatureVectors.toSparse(extractDense(img)));
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