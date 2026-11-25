package models.ml.Pipeline;

import models.ml.Model.Model;
import models.ml.Preprocessing.Features.FeatureExtractor;
import models.ml.Preprocessing.Labels.LabelEncoder;
import models.ml.Preprocessing.Text.Vectorizer.TextVectorizer;
import models.ml.Utils.Json;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Utility class for saving and loading fitted pipelines.
 * <p>
 * Supports both {@link FeaturePipeline} and {@link TextPipeline}.
 * Saves extractor/encoder as JSON and model via Java serialization.
 * <p>
 * <b>Security Warning:</b> Java deserialization is used to load the model component.
 * Never load pipelines from untrusted sources, as deserialization can execute arbitrary code.
 * <p>
 * Not thread-safe.
 */
public final class PipelineIO {

    private static final String METADATA_FILE = "pipeline.json";
    private static final String EXTRACTOR_FILE = "extractor.json";
    private static final String ENCODER_FILE = "encoder.json";
    private static final String MODEL_FILE = "model.ser";

    private PipelineIO() {}

    /**
     * Saves a fitted {@link FeaturePipeline} to the specified directory.
     *
     * @param pipeline the fitted pipeline to save; must not be null
     * @param dir      the target directory; must not be null, will be created if it doesn't exist
     * @param <T>      the input type
     * @throws IllegalStateException       if pipeline is not fitted
     * @throws IllegalArgumentException    if pipeline or dir is null
     * @throws RuntimeException            if I/O fails
     */
    public static <T> void save(FeaturePipeline<T> pipeline, Path dir) {
        if (pipeline == null) {
            throw new IllegalArgumentException("pipeline cannot be null");
        }
        if (dir == null) {
            throw new IllegalArgumentException("dir cannot be null");
        }
        if (!pipeline.isFitted()) {
            throw new IllegalStateException("Pipeline must be fitted before save");
        }

        try {
            Files.createDirectories(dir);

            // Save extractor
            String extractorJson = pipeline.getExtractor().toJson();
            Files.writeString(dir.resolve(EXTRACTOR_FILE), extractorJson);

            // Save encoder
            String encoderJson = pipeline.getLabelEncoder().toJson();
            Files.writeString(dir.resolve(ENCODER_FILE), encoderJson);

            // Save model via Java serialization
            Path modelPath = dir.resolve(MODEL_FILE);
            pipeline.getModel().save(modelPath.toString());

            // Save metadata
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("type", "featurePipeline");
            metadata.put("dataMode", pipeline.getDataMode().name());
            metadata.put("fitted", pipeline.isFitted());
            Files.writeString(dir.resolve(METADATA_FILE), Json.write(metadata));

        } catch (IOException e) {
            throw new RuntimeException("Error saving pipeline: " + e.getMessage(), e);
        }
    }

    /**
     * Saves a fitted {@link TextPipeline} to the specified directory.
     *
     * @param pipeline the fitted pipeline to save; must not be null
     * @param dir      the target directory; must not be null, will be created if it doesn't exist
     * @throws IllegalStateException       if pipeline is not fitted
     * @throws IllegalArgumentException    if pipeline or dir is null
     * @throws RuntimeException            if I/O fails
     */
    public static void save(TextPipeline pipeline, Path dir) {
        if (pipeline == null) {
            throw new IllegalArgumentException("pipeline cannot be null");
        }
        if (dir == null) {
            throw new IllegalArgumentException("dir cannot be null");
        }
        if (!pipeline.isFitted()) {
            throw new IllegalStateException("Pipeline must be fitted before save");
        }

        try {
            Files.createDirectories(dir);

            // Save vectorizer (as extractor)
            String extractorJson = pipeline.getVectorizer().toJson();
            Files.writeString(dir.resolve(EXTRACTOR_FILE), extractorJson);

            // Save encoder
            String encoderJson = pipeline.getLabelEncoder().toJson();
            Files.writeString(dir.resolve(ENCODER_FILE), encoderJson);

            // Save model via Java serialization
            Path modelPath = dir.resolve(MODEL_FILE);
            pipeline.getModel().save(modelPath.toString());

            // Save metadata
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("type", "textPipeline");
            metadata.put("fitted", pipeline.isFitted());
            Files.writeString(dir.resolve(METADATA_FILE), Json.write(metadata));

        } catch (IOException e) {
            throw new RuntimeException("Error saving pipeline: " + e.getMessage(), e);
        }
    }

    /**
     * Loads a {@link FeaturePipeline} from the specified directory.
     *
     * @param dir       the source directory; must not be null
     * @param extractorFactory factory to create a new extractor instance (for type parameter)
     * @param modelFactory factory to create a new model instance
     * @param <T>       the input type
     * @return a new fitted FeaturePipeline instance
     * @throws IllegalArgumentException if dir is null, metadata is invalid, or required files are missing,
     *         or if any file is corrupt, truncated, or of wrong type
     * @throws RuntimeException         for unexpected errors not related to input validation
     */
    public static <T> FeaturePipeline<T> loadFeaturePipeline(
            Path dir,
            FeatureExtractorFactory<T> extractorFactory,
            ModelFactory modelFactory) {
        if (dir == null) {
            throw new IllegalArgumentException("dir cannot be null");
        }
        if (extractorFactory == null) {
            throw new IllegalArgumentException("extractorFactory cannot be null");
        }
        if (modelFactory == null) {
            throw new IllegalArgumentException("modelFactory cannot be null");
        }

        try {
            // Load metadata
            Path metadataPath = dir.resolve(METADATA_FILE);
            if (!Files.exists(metadataPath)) {
                throw new IllegalArgumentException("Missing metadata file: " + METADATA_FILE);
            }
            String metadataJson = Files.readString(metadataPath);
            Object parsed = Json.parse(metadataJson);
            if (!(parsed instanceof Map)) {
                throw new IllegalArgumentException("Metadata JSON root must be an object");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> metadata = (Map<String, Object>) parsed;

            String type = (String) metadata.get("type");
            if (!"featurePipeline".equals(type)) {
                throw new IllegalArgumentException("Expected type 'featurePipeline', got '" + type + "'");
            }

            // Load extractor
            Path extractorPath = dir.resolve(EXTRACTOR_FILE);
            if (!Files.exists(extractorPath)) {
                throw new IllegalArgumentException("Missing extractor file: " + EXTRACTOR_FILE);
            }
            String extractorJson = Files.readString(extractorPath);
            FeatureExtractor<T> extractor = extractorFactory.create(extractorJson);

            // Load encoder
            Path encoderPath = dir.resolve(ENCODER_FILE);
            if (!Files.exists(encoderPath)) {
                throw new IllegalArgumentException("Missing encoder file: " + ENCODER_FILE);
            }
            String encoderJson = Files.readString(encoderPath);
            LabelEncoder encoder = LabelEncoder.fromJson(encoderJson);

            // Load model
            Path modelPath = dir.resolve(MODEL_FILE);
            if (!Files.exists(modelPath)) {
                throw new IllegalArgumentException("Missing model file: " + MODEL_FILE);
            }
            Model model = modelFactory.create();
            model.load(modelPath.toString());

            // Parse dataMode
            String dataModeStr = (String) metadata.get("dataMode");
            DataMode dataMode = DataMode.valueOf(dataModeStr);

            // Create pipeline with loaded components
            FeaturePipeline<T> pipeline = new FeaturePipeline<>(extractor, encoder, model, dataMode);
            // Mark as fitted since all components are already fitted
            pipeline.markFitted();
            return pipeline;

        } catch (IllegalArgumentException e) {
            // Check if it's from Json.parse (which wraps ParseException as cause)
            if (e.getCause() != null && e.getCause() instanceof RuntimeException) {
                throw new IllegalArgumentException("Corrupt or invalid pipeline file: " + e.getMessage(), e);
            }
            throw e;
        } catch (ClassCastException | IllegalStateException e) {
            throw new IllegalArgumentException("Corrupt or invalid pipeline file: " + e.getMessage(), e);
        } catch (IOException e) {
            throw new IllegalArgumentException("Corrupt or invalid pipeline file: " + e.getMessage(), e);
        } catch (RuntimeException e) {
            // Check if caused by ClassNotFoundException or IOException from deserialization
            Throwable cause = e.getCause();
            if (cause instanceof ClassNotFoundException || cause instanceof IOException) {
                throw new IllegalArgumentException("Corrupt or invalid pipeline file: " + cause.getMessage(), cause);
            }
            // Other RuntimeExceptions (e.g., JSON parse errors) are also corrupt file errors
            throw new IllegalArgumentException("Corrupt or invalid pipeline file: " + e.getMessage(), e);
        }
    }

    /**
     * Loads a {@link TextPipeline} from the specified directory.
     *
     * @param dir            the source directory; must not be null
     * @param vectorizerFactory factory to create a new vectorizer instance
     * @param modelFactory   factory to create a new model instance
     * @return a new fitted TextPipeline instance
     * @throws IllegalArgumentException if dir is null, metadata is invalid, or required files are missing,
     *         or if any file is corrupt, truncated, or of wrong type
     * @throws RuntimeException         for unexpected errors not related to input validation
     */
    public static TextPipeline loadTextPipeline(
            Path dir,
            TextVectorizerFactory vectorizerFactory,
            ModelFactory modelFactory) {
        if (dir == null) {
            throw new IllegalArgumentException("dir cannot be null");
        }
        if (vectorizerFactory == null) {
            throw new IllegalArgumentException("vectorizerFactory cannot be null");
        }
        if (modelFactory == null) {
            throw new IllegalArgumentException("modelFactory cannot be null");
        }

        try {
            // Load metadata
            Path metadataPath = dir.resolve(METADATA_FILE);
            if (!Files.exists(metadataPath)) {
                throw new IllegalArgumentException("Missing metadata file: " + METADATA_FILE);
            }
            String metadataJson = Files.readString(metadataPath);
            Object parsed = Json.parse(metadataJson);
            if (!(parsed instanceof Map)) {
                throw new IllegalArgumentException("Metadata JSON root must be an object");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> metadata = (Map<String, Object>) parsed;

            String type = (String) metadata.get("type");
            if (!"textPipeline".equals(type)) {
                throw new IllegalArgumentException("Expected type 'textPipeline', got '" + type + "'");
            }

            // Load vectorizer
            Path extractorPath = dir.resolve(EXTRACTOR_FILE);
            if (!Files.exists(extractorPath)) {
                throw new IllegalArgumentException("Missing extractor file: " + EXTRACTOR_FILE);
            }
            String vectorizerJson = Files.readString(extractorPath);
            TextVectorizer vectorizer = vectorizerFactory.create(vectorizerJson);

            // Load encoder
            Path encoderPath = dir.resolve(ENCODER_FILE);
            if (!Files.exists(encoderPath)) {
                throw new IllegalArgumentException("Missing encoder file: " + ENCODER_FILE);
            }
            String encoderJson = Files.readString(encoderPath);
            LabelEncoder encoder = LabelEncoder.fromJson(encoderJson);

            // Load model
            Path modelPath = dir.resolve(MODEL_FILE);
            if (!Files.exists(modelPath)) {
                throw new IllegalArgumentException("Missing model file: " + MODEL_FILE);
            }
            Model model = modelFactory.create();
            model.load(modelPath.toString());

            // Create pipeline with loaded components
            TextPipeline pipeline = new TextPipeline(vectorizer, encoder, model);
            pipeline.markFitted();
            return pipeline;

        } catch (IllegalArgumentException e) {
            // Check if it's from Json.parse (which wraps ParseException as cause)
            if (e.getCause() != null && e.getCause() instanceof RuntimeException) {
                throw new IllegalArgumentException("Corrupt or invalid pipeline file: " + e.getMessage(), e);
            }
            throw e;
        } catch (ClassCastException | IllegalStateException e) {
            throw new IllegalArgumentException("Corrupt or invalid pipeline file: " + e.getMessage(), e);
        } catch (IOException e) {
            throw new IllegalArgumentException("Corrupt or invalid pipeline file: " + e.getMessage(), e);
        } catch (RuntimeException e) {
            // Check if caused by ClassNotFoundException or IOException from deserialization
            Throwable cause = e.getCause();
            if (cause instanceof ClassNotFoundException || cause instanceof IOException) {
                throw new IllegalArgumentException("Corrupt or invalid pipeline file: " + cause.getMessage(), cause);
            }
            // Other RuntimeExceptions (e.g., JSON parse errors) are also corrupt file errors
            throw new IllegalArgumentException("Corrupt or invalid pipeline file: " + e.getMessage(), e);
        }
    }

    /**
     * Functional interface for creating FeatureExtractor instances from JSON.
     *
     * @param <T> the input type
     */
    @FunctionalInterface
    public interface FeatureExtractorFactory<T> {
        FeatureExtractor<T> create(String json);
    }

    /**
     * Functional interface for creating TextVectorizer instances from JSON.
     */
    @FunctionalInterface
    public interface TextVectorizerFactory {
        TextVectorizer create(String json);
    }

    /**
     * Functional interface for creating Model instances.
     */
    @FunctionalInterface
    public interface ModelFactory {
        Model create();
    }
}