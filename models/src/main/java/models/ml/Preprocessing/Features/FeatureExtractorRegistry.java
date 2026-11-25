package models.ml.Preprocessing.Features;

import models.ml.Preprocessing.Audio.MfccExtractor;
import models.ml.Preprocessing.Audio.SpectralStatsExtractor;
import models.ml.Preprocessing.Image.ColorHistogramExtractor;
import models.ml.Preprocessing.Image.PixelFeatureExtractor;
import models.ml.Preprocessing.Text.Vectorizer.BagOfWordsVectorizer;
import models.ml.Preprocessing.Text.Vectorizer.TextFeatureExtractor;
import models.ml.Preprocessing.Text.Vectorizer.TfidfVectorizer;
import models.ml.Utils.Json;
import models.ml.Utils.JsonFields;

import java.util.Map;

/**
 * Registry for deserializing FeatureExtractor instances from JSON.
 * Supports all built-in extractor types.
 */
public final class FeatureExtractorRegistry {

    private FeatureExtractorRegistry() {}

    @SuppressWarnings("unchecked")
    public static <T> FeatureExtractor<T> fromJson(String json) {
        if (json == null || json.isEmpty()) {
            throw new IllegalArgumentException("JSON string cannot be null or empty");
        }
        Object parsed = Json.parse(json);
        if (!(parsed instanceof Map)) {
            throw new IllegalArgumentException("JSON root must be an object");
        }
        Map<String, Object> map = (Map<String, Object>) parsed;

        // Detect type: extractors with "type" field, or text vectorizers by field presence
        String type = null;
        if (map.containsKey("type")) {
            type = JsonFields.requireString(map, "type");
        } else if (map.containsKey("idf") && map.containsKey("vocab") && map.containsKey("binary")) {
            // TF-IDF vectorizer (has idf field)
            type = "tfidf";
        } else if (map.containsKey("vocab") && map.containsKey("binary") && !map.containsKey("idf")) {
            // BagOfWords vectorizer (no idf field)
            type = "bow";
        }

        if (type == null) {
            throw new IllegalArgumentException("Cannot determine extractor type: missing 'type' field and not a recognized vectorizer format");
        }

        switch (type) {
            case "pixel":
                return (FeatureExtractor<T>) PixelFeatureExtractor.fromJson(json);
            case "colorHistogram":
                return (FeatureExtractor<T>) ColorHistogramExtractor.fromJson(json);
            case "spectralStats":
                return (FeatureExtractor<T>) SpectralStatsExtractor.fromJson(json);
            case "mfcc":
                return (FeatureExtractor<T>) MfccExtractor.fromJson(json);
            case "scaled":
                return (FeatureExtractor<T>) ScalingFeatureExtractor.fromJson(json);
            case "composite":
                return (FeatureExtractor<T>) CompositeFeatureExtractor.fromJson(json);
            case "tfidf": {
                TextFeatureExtractor tf = new TextFeatureExtractor(TfidfVectorizer.fromJson(json));
                return (FeatureExtractor<T>) tf;
            }
            case "bow": {
                TextFeatureExtractor tf = new TextFeatureExtractor(BagOfWordsVectorizer.fromJson(json));
                return (FeatureExtractor<T>) tf;
            }
            default:
                throw new IllegalArgumentException("Unknown extractor type: " + type);
        }
    }
}