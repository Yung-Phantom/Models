package models.ml.Preprocessing.Labels;

import models.ml.Utils.Json;
import java.util.*;
import java.util.stream.Collectors;

public class LabelEncoder {
    private final List<String> classes = new ArrayList<>();
    private final Map<String, Integer> classToIndex = new HashMap<>();
    private boolean fitted = false;

    public LabelEncoder() {}

    public void fit(List<String> labels) {
        if (labels == null) {
            throw new IllegalArgumentException("Labels list cannot be null");
        }
        if (labels.isEmpty()) {
            throw new IllegalArgumentException("Labels list cannot be empty");
        }
        for (String label : labels) {
            if (label == null) {
                throw new IllegalArgumentException("Labels list cannot contain null");
            }
        }

        // Create a copy to avoid mutation issues
        List<String> labelsCopy = new ArrayList<>(labels);

        // Get distinct labels and sort them
        List<String> distinctSorted = labelsCopy.stream()
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        // Build new mappings
        List<String> newClasses = new ArrayList<>(distinctSorted);
        Map<String, Integer> newClassToIndex = new HashMap<>();
        for (int i = 0; i < newClasses.size(); i++) {
            newClassToIndex.put(newClasses.get(i), i);
        }

        // Only update state after successful validation
        this.classes.clear();
        this.classes.addAll(newClasses);
        this.classToIndex.clear();
        this.classToIndex.putAll(newClassToIndex);
        this.fitted = true;
    }

    public double transform(String label) {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before transform");
        }
        if (label == null) {
            throw new IllegalArgumentException("Label cannot be null");
        }
        Integer idx = classToIndex.get(label);
        if (idx == null) {
            throw new IllegalArgumentException("Unknown label: " + label);
        }
        return (double) idx;
    }

    public double[] transform(List<String> labels) {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before transform");
        }
        if (labels == null) {
            throw new IllegalArgumentException("Labels list cannot be null");
        }
        double[] result = new double[labels.size()];
        for (int i = 0; i < labels.size(); i++) {
            String label = labels.get(i);
            if (label == null) {
                throw new IllegalArgumentException("Label at index " + i + " cannot be null");
            }
            result[i] = transform(label);
        }
        return result;
    }

    public String inverseTransform(double index) {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before inverseTransform");
        }
        if (Double.isNaN(index) || Double.isInfinite(index)) {
            throw new IllegalArgumentException("Index must be a finite number");
        }
        if (index != Math.floor(index)) {
            throw new IllegalArgumentException("Index must be a whole number");
        }
        int idx = (int) index;
        if (idx < 0 || idx >= classes.size()) {
            throw new IllegalArgumentException("Index out of range: " + idx);
        }
        return classes.get(idx);
    }

    public String[] inverseTransform(double[] indices) {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before inverseTransform");
        }
        if (indices == null) {
            throw new IllegalArgumentException("Indices array cannot be null");
        }
        String[] result = new String[indices.length];
        for (int i = 0; i < indices.length; i++) {
            result[i] = inverseTransform(indices[i]);
        }
        return result;
    }

    public List<String> getClasses() {
        return Collections.unmodifiableList(new ArrayList<>(classes));
    }

    public int size() {
        return classes.size();
    }

    public boolean isFitted() {
        return fitted;
    }

    public String toJson() {
        if (!fitted) {
            throw new IllegalStateException("Encoder must be fitted before toJson");
        }
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("classes", new ArrayList<>(classes));
        map.put("fitted", fitted);
        return Json.write(map);
    }

    public static LabelEncoder fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }

        Object parsed = Json.parse(json);
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("JSON root must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> map = (Map<String, Object>) parsed;

        if (!map.containsKey("classes")) {
            throw new IllegalArgumentException("Missing required field: classes");
        }
        if (!map.containsKey("fitted")) {
            throw new IllegalArgumentException("Missing required field: fitted");
        }

        Object classesObj = map.get("classes");
        Object fittedObj = map.get("fitted");

        if (!(classesObj instanceof List)) {
            throw new IllegalArgumentException("classes must be an array");
        }
        if (!(fittedObj instanceof Boolean)) {
            throw new IllegalArgumentException("fitted must be a boolean");
        }

        @SuppressWarnings("unchecked")
        List<Object> classesList = (List<Object>) classesObj;
        boolean fitted = (Boolean) fittedObj;

        List<String> classes = new ArrayList<>();
        for (Object obj : classesList) {
            if (!(obj instanceof String)) {
                throw new IllegalArgumentException("classes array must contain only strings");
            }
            classes.add((String) obj);
        }

        // Check for duplicates
        if (new HashSet<>(classes).size() != classes.size()) {
            throw new IllegalArgumentException("Duplicate classes found");
        }

        if (fitted && classes.isEmpty()) {
            throw new IllegalArgumentException("fitted=true with empty classes is invalid");
        }
        if (!fitted && !classes.isEmpty()) {
            throw new IllegalArgumentException("fitted=false with non-empty classes is invalid");
        }

        LabelEncoder encoder = new LabelEncoder();
        encoder.classes.addAll(classes);
        encoder.fitted = fitted;
        for (int i = 0; i < classes.size(); i++) {
            encoder.classToIndex.put(classes.get(i), i);
        }
        return encoder;
    }

    public static LabelEncoder fromMap(Map<String, Integer> classToIndex) {
        if (classToIndex == null) {
            throw new IllegalArgumentException("classToIndex map cannot be null");
        }
        if (classToIndex.isEmpty()) {
            throw new IllegalArgumentException("classToIndex map cannot be empty");
        }

        int n = classToIndex.size();
        int[] indexArray = new int[n];
        Arrays.fill(indexArray, -1);

        for (Map.Entry<String, Integer> entry : classToIndex.entrySet()) {
            String key = entry.getKey();
            Integer idx = entry.getValue();
            if (key == null) {
                throw new IllegalArgumentException("classToIndex map contains null key");
            }
            if (idx == null) {
                throw new IllegalArgumentException("classToIndex map contains null value for key: " + key);
            }
            if (idx < 0) {
                throw new IllegalArgumentException("Index cannot be negative: " + idx);
            }
            if (idx >= n) {
                throw new IllegalArgumentException("Index out of range: " + idx);
            }
            if (indexArray[idx] != -1) {
                throw new IllegalArgumentException("Duplicate index: " + idx);
            }
            indexArray[idx] = 1;
        }

        // Verify all indices 0..n-1 are present
        for (int i = 0; i < n; i++) {
            if (indexArray[i] == -1) {
                throw new IllegalArgumentException("Missing index: " + i);
            }
        }

        LabelEncoder encoder = new LabelEncoder();
        // Build classes list in index order
        List<String> classesInOrder = new ArrayList<>(Collections.nCopies(n, null));
        for (Map.Entry<String, Integer> entry : classToIndex.entrySet()) {
            classesInOrder.set(entry.getValue(), entry.getKey());
        }
        encoder.classes.addAll(classesInOrder);
        encoder.classToIndex.putAll(classToIndex);
        encoder.fitted = true;
        return encoder;
    }
}