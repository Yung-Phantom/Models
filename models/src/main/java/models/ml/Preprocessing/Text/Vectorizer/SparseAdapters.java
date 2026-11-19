package models.ml.Preprocessing.Text.Vectorizer;

import java.util.*;

public class SparseAdapters {
    private SparseAdapters() {}

    public static Map<Integer, Double> toDouble(Map<Integer, Integer> source) {
        if (source == null) return Collections.emptyMap();
        Map<Integer, Double> result = new HashMap<>(source.size());
        for (Map.Entry<Integer, Integer> e : source.entrySet()) {
            result.put(e.getKey(), e.getValue().doubleValue());
        }
        return result;
    }

    public static List<Map<Integer, Double>> toDouble(List<Map<Integer, Integer>> source) {
        if (source == null) return Collections.emptyList();
        List<Map<Integer, Double>> result = new ArrayList<>(source.size());
        for (Map<Integer, Integer> m : source) {
            result.add(toDouble(m));
        }
        return result;
    }
}