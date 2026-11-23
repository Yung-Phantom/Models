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
 * Feature size and offsets are computed at call time to support children
 * whose feature size changes after fit.
 * <p>
 * Not thread-safe.
 *
 * @param <T> input type
 */
public final class CompositeFeatureExtractor<T> implements FeatureExtractor<T> {

    private final List<FeatureExtractor<T>> children;

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
        for (int i = 0; i < children.size(); i++) {
            FeatureExtractor<T> child = children.get(i);
            if (child == null) {
                throw new IllegalArgumentException("child at index " + i + " cannot be null");
            }
            copy.add(child);
        }
        this.children = Collections.unmodifiableList(copy);
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

    /**
     * Computes the total feature size by summing child feature sizes.
     * Called at call time to reflect any changes in child feature sizes after fit.
     */
    @Override
    public int getFeatureSize() {
        int total = 0;
        for (FeatureExtractor<T> child : children) {
            total += child.getFeatureSize();
        }
        return total;
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

    /**
     * Shared guard: checks fitted state and validates input list.
     * Throws IllegalStateException if not fitted, IllegalArgumentException if inputs is null.
     */
    private void requireFittedAndNonNullList(List<T> inputs) {
        if (!isFitted()) {
            throw new IllegalStateException("Extractor must be fitted before transform");
        }
        if (inputs == null) {
            throw new IllegalArgumentException("inputs list cannot be null");
        }
    }

    @Override
    public double[] transformDense(T input) {
        if (!isFitted()) {
            throw new IllegalStateException("Extractor must be fitted before transformDense");
        }
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        List<double[]> childArrays = new ArrayList<>(children.size());
        int totalSize = 0;
        for (FeatureExtractor<T> child : children) {
            double[] childDense = child.transformDense(input);
            childArrays.add(childDense);
            totalSize += childDense.length;
        }
        double[] result = new double[totalSize];
        int off = 0;
        for (double[] childDense : childArrays) {
            System.arraycopy(childDense, 0, result, off, childDense.length);
            off += childDense.length;
        }
        return result;
    }

    @Override
    public double[][] transformDense(List<T> inputs) {
        requireFittedAndNonNullList(inputs);
        List<double[]> rows = new ArrayList<>(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            T input = inputs.get(i);
            if (input == null) {
                throw new IllegalArgumentException("input at index " + i + " cannot be null");
            }
            rows.add(transformDense(input));
        }
        return rows.toArray(new double[0][]);
    }

    @Override
    public Map<Integer, Double> transform(T input) {
        return FeatureVectors.toSparse(transformDense(input));
    }

    @Override
    public List<Map<Integer, Double>> transform(List<T> inputs) {
        requireFittedAndNonNullList(inputs);
        List<Map<Integer, Double>> result = new ArrayList<>(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            T input = inputs.get(i);
            if (input == null) {
                throw new IllegalArgumentException("input at index " + i + " cannot be null");
            }
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