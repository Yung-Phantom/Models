package models.ml.Preprocessing.Features;

import models.ml.Preprocessing.Scaling.Transformer;
import models.ml.Utils.Json;
import models.ml.Utils.JsonFields;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Feature extractor that wraps another extractor and applies a scaler to its dense output.
 * <p>
 * The scaler is fitted on the training data's dense features during {@code fit()}.
 * The child extractor is also fitted on the same data.
 * <p>
 * Not thread-safe.
 *
 * @param <T> the input type
 */
public final class ScalingFeatureExtractor<T> implements FeatureExtractor<T> {

    private final FeatureExtractor<T> child;
    private final Transformer scaler;
    private boolean fitted = false;

    /**
     * Creates a scaling feature extractor.
     *
     * @param child  the child feature extractor; must not be null
     * @param scaler the scaler to apply to the child's dense output; must not be null
     * @throws IllegalArgumentException if child or scaler is null
     */
    public ScalingFeatureExtractor(FeatureExtractor<T> child, Transformer scaler) {
        if (child == null) {
            throw new IllegalArgumentException("child extractor cannot be null");
        }
        if (scaler == null) {
            throw new IllegalArgumentException("scaler cannot be null");
        }
        this.child = child;
        this.scaler = scaler;
    }

    @Override
    public void fit(List<T> inputs) {
        // Validate inputs first
        if (inputs == null) {
            throw new IllegalArgumentException("inputs list cannot be null");
        }
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("inputs list cannot be empty");
        }
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs.get(i) == null) {
                throw new IllegalArgumentException("input at index " + i + " cannot be null");
            }
        }

        // Set fitted = false BEFORE fitting child/scaler
        // If any subsequent step throws, fitted remains false
        fitted = false;

        // Fit the child extractor first
        child.fit(inputs);

        // Get dense features from child and fit the scaler on them
        // Scaler will validate non-empty data
        double[][] denseFeatures = child.transformDense(inputs);
        scaler.fit(denseFeatures);

        // Mark as fitted only if both child and scaler succeeded
        fitted = true;
    }

    @Override
    public int getFeatureSize() {
        return child.getFeatureSize();
    }

    @Override
    public boolean isFitted() {
        return fitted && child.isFitted();
    }

    @Override
    public double[] transformDense(T input) {
        if (!isFitted()) {
            throw new IllegalStateException("Extractor must be fitted before transformDense");
        }
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        double[] childDense = child.transformDense(input);
        return scaler.transform(childDense);
    }

    @Override
    public double[][] transformDense(List<T> inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("inputs list cannot be null");
        }
        if (!isFitted()) {
            throw new IllegalStateException("Extractor must be fitted before transformDense");
        }
        double[][] childDense = child.transformDense(inputs);
        return scaler.transform(childDense);
    }

    @Override
    public Map<Integer, Double> transform(T input) {
        return FeatureVectors.toSparse(transformDense(input));
    }

    @Override
    public List<Map<Integer, Double>> transform(List<T> inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("inputs list cannot be null");
        }
        if (!isFitted()) {
            throw new IllegalStateException("Extractor must be fitted before transform");
        }
        List<Map<Integer, Double>> result = new ArrayList<>(inputs.size());
        for (T input : inputs) {
            result.add(transform(input));
        }
        return result;
    }

    @Override
    public String toJson() {
        if (!isFitted()) {
            throw new IllegalStateException("Extractor must be fitted before toJson");
        }
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("type", "scaled");

        // Parse child's JSON and embed as object
        Object childParsed = Json.parse(child.toJson());
        if (!(childParsed instanceof Map)) {
            throw new IllegalStateException("Child toJson must return an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> childMap = (Map<String, Object>) childParsed;
        map.put("extractor", childMap);

        // Parse scaler's JSON and embed as object
        Object scalerParsed = Json.parse(scaler.toJson());
        if (!(scalerParsed instanceof Map)) {
            throw new IllegalStateException("Scaler toJson must return an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> scalerMap = (Map<String, Object>) scalerParsed;
        map.put("scaler", scalerMap);

        return Json.write(map);
    }

    /**
     * Reconstructs a ScalingFeatureExtractor from its JSON representation.
     *
     * @param json JSON string produced by {@link #toJson()}; must not be null or empty
     * @param <T>  the input type
     * @return new ScalingFeatureExtractor instance with the same configuration and fitted state
     * @throws IllegalArgumentException if JSON is malformed, missing fields, or has invalid values
     */
    @SuppressWarnings("unchecked")
    public static <T> ScalingFeatureExtractor<T> fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }
        Object parsed = Json.parse(json);
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("JSON root must be an object");
        }
        Map<String, Object> map = (Map<String, Object>) parsed;

        String type = JsonFields.requireString(map, "type");
        if (!"scaled".equals(type)) {
            throw new IllegalArgumentException("Expected type 'scaled', got '" + type + "'");
        }

        if (!map.containsKey("extractor")) {
            throw new IllegalArgumentException("Missing required field: extractor");
        }
        if (!map.containsKey("scaler")) {
            throw new IllegalArgumentException("Missing required field: scaler");
        }

        Object extractorObj = map.get("extractor");
        Object scalerObj = map.get("scaler");

        if (!(extractorObj instanceof Map)) {
            throw new IllegalArgumentException("'extractor' must be an object");
        }
        if (!(scalerObj instanceof Map)) {
            throw new IllegalArgumentException("'scaler' must be an object");
        }

        String extractorJson = Json.write(extractorObj);
        String scalerJson = Json.write(scalerObj);

        FeatureExtractor<T> child = (FeatureExtractor<T>) FeatureExtractorRegistry.fromJson(extractorJson);
        Transformer scaler = ScalerRegistry.fromJson(scalerJson);

        ScalingFeatureExtractor<T> result = new ScalingFeatureExtractor<>(child, scaler);
        if (child.isFitted()) {
            result.markFitted();
        }
        return result;
    }

    /**
     * Marks the extractor as fitted without running validation.
     * Intended for use by {@code fromJson} methods to restore fitted state.
     */
    protected void markFitted() {
        this.fitted = true;
    }
}