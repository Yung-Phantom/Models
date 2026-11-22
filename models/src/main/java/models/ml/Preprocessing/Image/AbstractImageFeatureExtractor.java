package models.ml.Preprocessing.Image;

import models.ml.Preprocessing.Features.AbstractDenseFeatureExtractor;
import models.ml.Preprocessing.Features.FeatureVectors;

import java.util.Map;

/**
 * Abstract base class for image feature extractors.
 * <p>
 * Concrete subclasses must implement {@link #extractDense(ImageData)} and {@link #getFeatureSize()}.
 * The {@code fit} method only validates input and marks the extractor as fitted; no learning occurs.
 * <p>
 * Not thread-safe.
 */
public abstract class AbstractImageFeatureExtractor extends AbstractDenseFeatureExtractor<ImageData> {

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
    public Map<Integer, Double> transform(ImageData input) {
        return FeatureVectors.toSparse(transformDense(input));
    }

    /**
     * Subclass-specific JSON serialization.
     * Called only after fit; implementation must include a "fitted": true field.
     */
    protected abstract String doToJson();
}