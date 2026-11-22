package models.ml.Preprocessing.Audio;

import models.ml.Preprocessing.Features.AbstractDenseFeatureExtractor;
import models.ml.Preprocessing.Features.FeatureVectors;

import java.util.Map;

/**
 * Abstract base class for audio feature extractors.
 * <p>
 * Concrete subclasses must implement {@link #extractDense(AudioData)} and {@link #getFeatureSize()}.
 * The {@code fit} method only validates input and marks the extractor as fitted; no learning occurs.
 * <p>
 * Not thread-safe.
 */
public abstract class AbstractAudioFeatureExtractor extends AbstractDenseFeatureExtractor<AudioData> {

    /**
     * Extracts dense features from a single audio clip.
     *
     * @param audio input audio; must not be null
     * @return dense feature vector of length {@link #getFeatureSize()}
     */
    protected abstract double[] extractDense(AudioData audio);

    /**
     * Returns the feature dimensionality.
     * Available before and after fit.
     */
    public abstract int getFeatureSize();

    @Override
    public Map<Integer, Double> transform(AudioData input) {
        return FeatureVectors.toSparse(transformDense(input));
    }

    /**
     * Subclass-specific JSON serialization.
     * Called only after fit; implementation must include a "fitted": true field.
     */
    protected abstract String doToJson();
}