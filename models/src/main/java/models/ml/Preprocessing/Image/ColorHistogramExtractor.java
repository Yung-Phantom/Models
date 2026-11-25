package models.ml.Preprocessing.Image;

import models.ml.Preprocessing.Features.AbstractDenseFeatureExtractor;
import models.ml.Utils.Json;
import models.ml.Utils.JsonFields;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Feature extractor that computes normalized color histograms.
 * <p>
 * Size-independent: works on images of any resolution without resizing.
 * Each channel's histogram bins sum to 1.0 (normalized by pixel count).
 * Not thread-safe.
 */
public final class ColorHistogramExtractor extends AbstractDenseFeatureExtractor<ImageData> {

    private final int bins;
    private final ChannelMode mode;

    /**
     * Creates a color histogram feature extractor.
     *
     * @param bins number of bins per channel; must be >= 2
     * @param mode channel mode; must not be null
     * @throws IllegalArgumentException if bins {@code <} 2 or mode is null
     */
    public ColorHistogramExtractor(int bins, ChannelMode mode) {
        if (bins < 2) {
            throw new IllegalArgumentException("bins must be >= 2, got " + bins);
        }
        if (mode == null) {
            throw new IllegalArgumentException("mode cannot be null");
        }
        this.bins = bins;
        this.mode = mode;
    }

    /**
     * Returns the number of bins per channel.
     */
    public int getBins() {
        return bins;
    }

    /**
     * Returns the channel mode.
     */
    public ChannelMode getMode() {
        return mode;
    }

    @Override
    public int getFeatureSize() {
        return bins * (mode == ChannelMode.GRAY ? 1 : 3);
    }

    @Override
    protected double[] extractDense(ImageData image) {
        ImageData converted = ImageOps.convert(image, mode);
        int w = converted.getWidth();
        int h = converted.getHeight();
        int c = converted.getChannels();
        double[] pixels = converted.getPixels();

        int channels = c;
        double[] histogram = new double[bins * channels];
        int pixelCount = w * h;

        for (int i = 0; i < pixelCount; i++) {
            for (int ch = 0; ch < channels; ch++) {
                double v = pixels[i * channels + ch];
                // Clamp to [0, 1) to handle edge case of 1.0
                // If v is 1.0 or greater, map to last bin (bins - 1)
                int bin;
                if (v >= 1.0) {
                    bin = bins - 1;
                } else {
                    bin = (int) (v * bins);
                    if (bin >= bins) {
                        bin = bins - 1;
                    }
                }
                histogram[ch * bins + bin] += 1.0;
            }
        }

        // Normalize each channel's histogram to sum to 1.0
        for (int ch = 0; ch < channels; ch++) {
            double sum = 0.0;
            for (int b = 0; b < bins; b++) {
                sum += histogram[ch * bins + b];
            }
            if (sum > 0) {
                for (int b = 0; b < bins; b++) {
                    histogram[ch * bins + b] /= sum;
                }
            }
        }

        return histogram;
    }

    @Override
    protected String doToJson() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("type", "colorHistogram");
        map.put("bins", bins);
        map.put("mode", mode.name());
        map.put("fitted", true);
        return Json.write(map);
    }

    /**
     * Reconstructs a ColorHistogramExtractor from its JSON representation.
     *
     * @param json JSON string produced by {@link #toJson()}; must not be null or
     *             empty
     * @return new ColorHistogramExtractor instance with the same configuration
     * @throws IllegalArgumentException if JSON is malformed, missing fields, or has
     *                                  invalid values
     */
    public static ColorHistogramExtractor fromJson(String json) {
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
        if (!"colorHistogram".equals(type)) {
            throw new IllegalArgumentException("Expected type 'colorHistogram', got '" + type + "'");
        }

        int bins = JsonFields.requireInt(map, "bins");
        String modeStr = JsonFields.requireString(map, "mode");
        ChannelMode mode;
        try {
            mode = ChannelMode.valueOf(modeStr);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid mode: " + modeStr + " (expected GRAY or RGB)");
        }
        boolean fitted = JsonFields.requireBoolean(map, "fitted");

        if (bins < 2) {
            throw new IllegalArgumentException("bins must be >= 2, got " + bins);
        }

        ColorHistogramExtractor extractor = new ColorHistogramExtractor(bins, mode);
        if (fitted) {
            extractor.markFitted();
        }
        return extractor;
    }
}