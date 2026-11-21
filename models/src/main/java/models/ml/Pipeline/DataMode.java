package models.ml.Pipeline;

import java.util.List;
import java.util.Map;

/**
 * Data representation mode for feature vectors in a pipeline.
 */
public enum DataMode {
    /** Sparse representation: {@code List<Map<Integer, Double>>} */
    SPARSE,
    /** Dense representation: {@code double[][]} */
    DENSE
}