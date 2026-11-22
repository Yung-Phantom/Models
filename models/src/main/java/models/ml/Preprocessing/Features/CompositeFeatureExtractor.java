package models.ml.Preprocessing.Features;

import models.ml.Utils.Json;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Feature extractor that concatenates outputs from multiple child extractors.
 * <p>
 * Children are fitted independently; concatenation uses index offsets.
 * <p>
 * Not thread-safe.
 *
 * @param <T> input type
 */
public final class CompositeFeatureExtractor<T> implements FeatureExtractor<T> {

    private final List<FeatureExtractor<T>> children;
    private final int featureSize;
    private final int[] offsets;

    /**
     * Creates a composite feature extractor.
     *
     * @param children list of child extractors; must not be null or empty, no null elements
     * @throws IllegalArgumentException if children is null, empty, or contains null
     */
    public CompositeFeatureExtractor(List<FeatureExtractor<T>> children) {
        if (children == null) {
            throw new IllegalArgumentException("children list cannot be null");
        }
        if (children.isEmpty()) {
            throw new IllegalArgumentException("children list cannot be empty");
        }
        List<FeatureExtractor<T>> copy = new ArrayList<>(children.size());
        int totalSize = 0;
        int[] off = new int[children.size()];
        for (int i = 0; i < children.size(); i++) {
            FeatureExtractor<T> child = children.get(i);
            if (child == null) {
                throw new IllegalArgumentException("child at index " + i + " cannot be null");
            }
            off[i] = totalSize;
            totalSize += child.getFeatureSize();
            copy.add(child);
        }
        this.children = Collections.unmodifiableList(copy);
        this.featureSize = totalSize;
        this.offsets = off;
    }

    /**
     * Returns the child extractors (unmodifiable view).
     */
    public List<FeatureExtractor<T>> getChildren() {
        return children;
    }

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
        for (FeatureExtractor<T> child : children) {
            child.fit(inputs);
        }
    }

    @Override
    public int getFeatureSize() {
        return featureSize;
    }

    @Override
    public boolean isFitted() {
        for (FeatureExtractor<T> child : children) {
            if (!child.isFitted()) {
                return false;
            }
        }
        return true;
    }

    @Override
    public double[] transformDense(T input) {
        if (!isFitted()) {
            throw new IllegalStateException("Extractor must be fitted before transformDense");
        }
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        double[] result = new double[featureSize];
        for (int i = 0; i < children.size(); i++) {
            double[] childDense = children.get(i).transformDense(input);
            int off = offsets[i];
            System.arraycopy(childDense, 0, result, off, childDense.length);
        }
        return result;
    }

    @Override
    public double[][] transformDense(List<T> inputs) {
        if (!isFitted()) {
            throw new IllegalStateException("Extractor must be fitted before transformDense");
        }
        if (inputs == null) {
            throw new IllegalArgumentException("inputs list cannot be null");
        }
        double[][] result = new double[inputs.size()][featureSize];
        for (int i = 0; i < inputs.size(); i++) {
            result[i] = transformDense(inputs.get(i));
        }
        return result;
    }

    @Override
    public Map<Integer, Double> transform(T input) {
        return FeatureVectors.toSparse(transformDense(input));
    }

    @Override
    public List<Map<Integer, Double>> transform(List<T> inputs) {
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
        map.put("type", "composite");
        List<Map<String, Object>> childrenJson = new ArrayList<>(children.size());
        for (FeatureExtractor<?> child : children) {
            // Parse child's JSON to include as object
            Object parsed = Json.parse(child.toJson());
            if (!(parsed instanceof Map)) {
                throw new IllegalStateException("Child toJson must return an object");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> childMap = (Map<String, Object>) parsed;
            childrenJson.add(childMap);
        }
        map.put("children", childrenJson);
        return Json.write(map);
    }

    // NOTE: fromJson is not implemented in this task. The JSON format includes
    // the full child configurations, so a fromJson would need a registry of extractor types.
}