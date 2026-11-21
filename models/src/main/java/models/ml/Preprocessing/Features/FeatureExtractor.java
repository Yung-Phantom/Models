package models.ml.Preprocessing.Features;

import java.util.List;
import java.util.Map;

/**
 * Generic feature extractor interface for converting inputs of type T into sparse feature vectors.
 * <p>
 * Contract:
 * <ul>
 *   <li>Call {@link #fit(List)} before any {@code transform} method.</li>
 *   <li>{@code transform} methods must not mutate fitted state.</li>
 *   <li>Returned sparse map keys (feature indices) must be in the range {@code [0, getFeatureSize())}.</li>
 *   <li>Not thread-safe.</li>
 * </ul>
 *
 * @param <T> the input type (e.g., String for text, double[] for dense vectors, etc.)
 */
public interface FeatureExtractor<T> {

    /**
     * Fits the extractor on the given inputs, building any internal state (vocabulary, statistics, etc.).
     *
     * @param inputs the training inputs; must not be null
     * @throws IllegalArgumentException if inputs is null
     */
    void fit(List<T> inputs);

    /**
     * Transforms a single input into a sparse feature vector.
     *
     * @param input the input to transform; must not be null
     * @return a map from feature index to non-zero value; empty map if no features
     * @throws IllegalStateException  if not fitted
     * @throws IllegalArgumentException if input is null
     */
    Map<Integer, Double> transform(T input);

    /**
     * Transforms a batch of inputs into sparse feature vectors.
     *
     * @param inputs the inputs to transform; must not be null
     * @return a list of sparse feature vectors, one per input
     * @throws IllegalStateException  if not fitted
     * @throws IllegalArgumentException if inputs is null
     */
    List<Map<Integer, Double>> transform(List<T> inputs);

    /**
     * Returns the feature dimensionality (size of the feature space).
     *
     * @return the number of features; must be non-negative
     */
    int getFeatureSize();

    /**
     * Returns whether the extractor has been fitted.
     *
     * @return true if fitted, false otherwise
     */
    boolean isFitted();

    /**
     * Returns a JSON representation of the fitted extractor.
     *
     * @return JSON string
     * @throws IllegalStateException if not fitted
     */
    String toJson();

    /**
     * Convenience method: fits on inputs and returns their transformations.
     *
     * @param inputs the training inputs
     * @return the transformed inputs
     */
    default List<Map<Integer, Double>> fitTransform(List<T> inputs) {
        fit(inputs);
        return transform(inputs);
    }

    /**
     * Transforms a single input into a dense vector of length {@link #getFeatureSize()}.
     * Default implementation converts the sparse map; subclasses may override for efficiency.
     *
     * @param input the input to transform
     * @return dense feature vector
     * @throws IllegalStateException  if not fitted
     * @throws IllegalArgumentException if input is null
     */
    default double[] transformDense(T input) {
        return FeatureVectors.toDense(transform(input), getFeatureSize());
    }

    /**
     * Transforms a batch of inputs into dense vectors.
     * Default implementation converts each sparse map; subclasses may override for efficiency.
     *
     * @param inputs the inputs to transform
     * @return dense feature matrix (one row per input)
     * @throws IllegalStateException  if not fitted
     * @throws IllegalArgumentException if inputs is null
     */
    default double[][] transformDense(List<T> inputs) {
        List<Map<Integer, Double>> sparse = transform(inputs);
        return FeatureVectors.toDense(sparse, getFeatureSize());
    }
}