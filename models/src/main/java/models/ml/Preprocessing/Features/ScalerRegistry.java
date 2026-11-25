package models.ml.Preprocessing.Features;

import models.ml.Preprocessing.Scaling.MaxAbsScaler;
import models.ml.Preprocessing.Scaling.MinMaxScaler;
import models.ml.Preprocessing.Scaling.SimpleImputer;
import models.ml.Preprocessing.Scaling.Transformer;
import models.ml.Preprocessing.Scaling.ZScoreScaler;
import models.ml.Utils.Json;

import java.util.Map;

/**
 * Registry for deserializing Transformer (scaler) instances from JSON.
 * Supports all built-in scaler types.
 */
public final class ScalerRegistry {

    private ScalerRegistry() {}

    public static Transformer fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }
        Object parsed = Json.parse(json);
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("JSON root must be an object");
        }
        Map<String, Object> map = (Map<String, Object>) parsed;

        // Detect scaler type by fields
        if (map.containsKey("means") && map.containsKey("stds") && map.containsKey("numFeatures")) {
            return ZScoreScaler.fromJson(json);
        }
        if (map.containsKey("mins") && map.containsKey("maxs") && map.containsKey("numFeatures")) {
            return MinMaxScaler.fromJson(json);
        }
        if (map.containsKey("maxAbs") && map.containsKey("numFeatures")) {
            return MaxAbsScaler.fromJson(json);
        }
        if (map.containsKey("strategy") && map.containsKey("fillValues") && map.containsKey("numFeatures")) {
            return SimpleImputer.fromJson(json);
        }

        throw new IllegalArgumentException("Unknown scaler type or missing required fields");
    }
}