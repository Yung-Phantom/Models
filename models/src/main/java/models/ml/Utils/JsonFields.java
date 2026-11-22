package models.ml.Utils;

import java.util.Map;

/**
 * Shared helpers for reading validated fields from a {@code Map<String, Object>}
 * produced by {@link Json#parse}.
 * <p>
 * Not instantiable; all methods are static.
 * Follows the validation conventions used in {@link models.ml.Preprocessing.Text.Vectorizer.TfidfVectorizer#fromJson}.
 */
public final class JsonFields {

    private JsonFields() {}

    /**
     * Retrieves a required integer field from the map.
     *
     * @param map   parsed JSON object; must not be null
     * @param field field name; must not be null
     * @return the integer value
     * @throws IllegalArgumentException if field is missing, not a Number, not a whole number,
     *         or outside the int range
     */
    public static int requireInt(Map<String, Object> map, String field) {
        if (map == null) {
            throw new IllegalArgumentException("map cannot be null");
        }
        if (field == null) {
            throw new IllegalArgumentException("field name cannot be null");
        }
        Object v = map.get(field);
        if (v == null) {
            throw new IllegalArgumentException("Missing required field: " + field);
        }
        if (!(v instanceof Number)) {
            throw new IllegalArgumentException("Field '" + field + "' must be a number");
        }
        double d = ((Number) v).doubleValue();
        if (d != Math.floor(d)) {
            throw new IllegalArgumentException("Field '" + field + "' must be a whole number, got " + d);
        }
        if (d < Integer.MIN_VALUE || d > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Field '" + field + "' value " + d + " outside int range [" + Integer.MIN_VALUE + ", " + Integer.MAX_VALUE + "]");
        }
        return ((Number) v).intValue();
    }

    /**
     * Retrieves a required boolean field from the map.
     *
     * @param map   parsed JSON object; must not be null
     * @param field field name; must not be null
     * @return the boolean value
     * @throws IllegalArgumentException if field is missing or not a Boolean
     */
    public static boolean requireBoolean(Map<String, Object> map, String field) {
        if (map == null) {
            throw new IllegalArgumentException("map cannot be null");
        }
        if (field == null) {
            throw new IllegalArgumentException("field name cannot be null");
        }
        Object v = map.get(field);
        if (v == null) {
            throw new IllegalArgumentException("Missing required field: " + field);
        }
        if (!(v instanceof Boolean)) {
            throw new IllegalArgumentException("Field '" + field + "' must be a boolean");
        }
        return (Boolean) v;
    }

    /**
     * Retrieves a required string field from the map.
     *
     * @param map   parsed JSON object; must not be null
     * @param field field name; must not be null
     * @return the string value
     * @throws IllegalArgumentException if field is missing or not a String
     */
    public static String requireString(Map<String, Object> map, String field) {
        if (map == null) {
            throw new IllegalArgumentException("map cannot be null");
        }
        if (field == null) {
            throw new IllegalArgumentException("field name cannot be null");
        }
        Object v = map.get(field);
        if (v == null) {
            throw new IllegalArgumentException("Missing required field: " + field);
        }
        if (!(v instanceof String)) {
            throw new IllegalArgumentException("Field '" + field + "' must be a string");
        }
        return (String) v;
    }
}