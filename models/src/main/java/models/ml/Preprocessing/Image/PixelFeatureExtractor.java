package models.ml.Preprocessing.Image;

import models.ml.Preprocessing.Features.AbstractDenseFeatureExtractor;
import models.ml.Preprocessing.Features.FeatureVectors;
import models.ml.Utils.Json;
import models.ml.Utils.JsonFields;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Feature extractor that resizes images to a fixed size and flattens pixels.
 * <p>
 * Configuration is immutable after construction.
 * Not thread-safe.
 */
public final class PixelFeatureExtractor extends AbstractDenseFeatureExtractor<ImageData> {

    private final int targetWidth;
    private final int targetHeight;
    private final ChannelMode mode;

    /**
     * Creates a pixel feature extractor.
     *
     * @param targetWidth  output width; must be > 0
     * @param targetHeight output height; must be > 0
     * @param mode         channel mode; must not be null
     * @throws IllegalArgumentException if any parameter is invalid
     */
    public PixelFeatureExtractor(int targetWidth, int targetHeight, ChannelMode mode) {
        if (targetWidth <= 0) {
            throw new IllegalArgumentException("targetWidth must be > 0, got " + targetWidth);
        }
        if (targetHeight <= 0) {
            throw new IllegalArgumentException("targetHeight must be > 0, got " + targetHeight);
        }
        if (mode == null) {
            throw new IllegalArgumentException("mode cannot be null");
        }
        this.targetWidth = targetWidth;
        this.targetHeight = targetHeight;
        this.mode = mode;
    }

    /**
     * Returns the target width.
     */
    public int getTargetWidth() {
        return targetWidth;
    }

    /**
     * Returns the target height.
     */
    public int getTargetHeight() {
        return targetHeight;
    }

    /**
     * Returns the channel mode.
     */
    public ChannelMode getMode() {
        return mode;
    }

    @Override
    public int getFeatureSize() {
        return targetWidth * targetHeight * (mode == ChannelMode.GRAY ? 1 : 3);
    }

    @Override
    protected double[] extractDense(ImageData image) {
        ImageData converted = ImageOps.convert(image, mode);
        ImageData resized = ImageOps.resize(converted, targetWidth, targetHeight);
        return resized.getPixels();
    }

    @Override
    protected String doToJson() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("type", "pixel");
        map.put("width", targetWidth);
        map.put("height", targetHeight);
        map.put("mode", mode.name());
        map.put("fitted", true);
        return Json.write(map);
    }

    /**
     * Reconstructs a PixelFeatureExtractor from its JSON representation.
     *
     * @param json JSON string produced by {@link #toJson()}; must not be null or empty
     * @return new PixelFeatureExtractor instance with the same configuration
     * @throws IllegalArgumentException if JSON is malformed, missing fields, or has invalid values
     */
    public static PixelFeatureExtractor fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }
        Object parsed = Json.parse(json);
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("JSON root must be an object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> map = (Map<String, Object>) parsed;

        String type = JsonFields.requireString(map, "type");
        if (!"pixel".equals(type)) {
            throw new IllegalArgumentException("Expected type 'pixel', got '" + type + "'");
        }

        int width = JsonFields.requireInt(map, "width");
        int height = JsonFields.requireInt(map, "height");
        String modeStr = JsonFields.requireString(map, "mode");
        ChannelMode mode;
        try {
            mode = ChannelMode.valueOf(modeStr);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid mode: " + modeStr + " (expected GRAY or RGB)");
        }
        boolean fitted = JsonFields.requireBoolean(map, "fitted");

        if (width <= 0) {
            throw new IllegalArgumentException("width must be > 0, got " + width);
        }
        if (height <= 0) {
            throw new IllegalArgumentException("height must be > 0, got " + height);
        }

        PixelFeatureExtractor extractor = new PixelFeatureExtractor(width, height, mode);
        if (fitted) {
            extractor.markFitted();
        }
        return extractor;
    }
}