package models.ml.Pipeline;

import models.ml.Model.Model;
import models.ml.Preprocessing.Features.FeatureExtractor;
import models.ml.Preprocessing.Labels.LabelEncoder;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Generic feature extraction and model training pipeline.
 * <p>
 * Mirrors {@link TextPipeline} but works with any input type {@code T} via a {@link FeatureExtractor}.
 * Supports both sparse and dense data modes.
 * <p>
 * Contract (same as TextPipeline):
 * <ul>
 *   <li>Validation is performed first; if validation fails, the previous fit is unchanged.</li>
 *   <li>If training fails after validation, the pipeline is left unfitted and the exception propagates unchanged.</li>
 *   <li>Not thread-safe.</li>
 * </ul>
 *
 * @param <T> the input type
 */
public class FeaturePipeline<T> {
    private final FeatureExtractor<T> extractor;
    private final LabelEncoder encoder;
    private final Model model;
    private final DataMode dataMode;
    private boolean fitted = false;

    /**
     * Creates a pipeline with a fresh LabelEncoder and SPARSE data mode.
     *
     * @param extractor the feature extractor; must not be null
     * @param model     the model to train; must not be null
     * @throws IllegalArgumentException if any argument is null
     */
    public FeaturePipeline(FeatureExtractor<T> extractor, Model model) {
        this(extractor, new LabelEncoder(), model, DataMode.SPARSE);
    }

    /**
     * Creates a pipeline with the given encoder and SPARSE data mode.
     *
     * @param extractor the feature extractor; must not be null
     * @param encoder   the label encoder; must not be null
     * @param model     the model to train; must not be null
     * @throws IllegalArgumentException if any argument is null
     */
    public FeaturePipeline(FeatureExtractor<T> extractor, LabelEncoder encoder, Model model) {
        this(extractor, encoder, model, DataMode.SPARSE);
    }

    /**
     * Creates a pipeline with full configuration.
     *
     * @param extractor the feature extractor; must not be null
     * @param encoder   the label encoder; must not be null
     * @param model     the model to train; must not be null
     * @param dataMode  the data mode (sparse or dense); must not be null
     * @throws IllegalArgumentException if any argument is null
     */
    public FeaturePipeline(FeatureExtractor<T> extractor, LabelEncoder encoder, Model model, DataMode dataMode) {
        if (extractor == null) {
            throw new IllegalArgumentException("extractor cannot be null");
        }
        if (encoder == null) {
            throw new IllegalArgumentException("encoder cannot be null");
        }
        if (model == null) {
            throw new IllegalArgumentException("model cannot be null");
        }
        if (dataMode == null) {
            throw new IllegalArgumentException("dataMode cannot be null");
        }
        this.extractor = extractor;
        this.encoder = encoder;
        this.model = model;
        this.dataMode = dataMode;
    }

    /**
     * Fits the pipeline on the given inputs and labels.
     * <p>
     * Validation is performed first. If validation fails, the previous fit is unchanged.
     * If training fails after validation, the pipeline is left unfitted and the exception propagates unchanged.
     *
     * @param inputs the training inputs; must not be null, empty, or contain null elements
     * @param labels the training labels; must not be null, empty, or contain null elements
     * @throws IllegalArgumentException if validation fails
     * @throws RuntimeException         if training fails (pipeline left unfitted)
     */
    public void fit(List<T> inputs, List<String> labels) {
        requireValidPairs(inputs, labels);

        // Mark as unfitted before training; will be set to true only if all steps succeed
        fitted = false;

        // Fit the extractor and encoder
        extractor.fit(inputs);
        encoder.fit(labels);
        double[] y = encoder.transform(labels);

        // Train the model using the appropriate mode
        if (dataMode == DataMode.SPARSE) {
            List<Map<Integer, Double>> xSparse = extractor.transform(inputs);
            model.fit(xSparse, y);
        } else {
            double[][] xDense = extractor.transformDense(inputs);
            model.fit(xDense, y);
        }

        // Only mark as fitted after all training steps succeed
        fitted = true;
    }

    /**
     * Validates input/label pairs for fit and score.
     * Throws IllegalArgumentException with the same messages as before.
     */
    private void requireValidPairs(List<T> inputs, List<String> labels) {
        if (inputs == null) {
            throw new IllegalArgumentException("inputs list cannot be null");
        }
        if (labels == null) {
            throw new IllegalArgumentException("labels list cannot be null");
        }
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("inputs list cannot be empty");
        }
        if (labels.isEmpty()) {
            throw new IllegalArgumentException("labels list cannot be empty");
        }
        if (inputs.size() != labels.size()) {
            throw new IllegalArgumentException("inputs and labels must have the same size");
        }
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs.get(i) == null) {
                throw new IllegalArgumentException("input at index " + i + " cannot be null");
            }
        }
        for (int i = 0; i < labels.size(); i++) {
            if (labels.get(i) == null) {
                throw new IllegalArgumentException("label at index " + i + " cannot be null");
            }
        }
    }

    /**
     * Predicts the label for a single input.
     *
     * @param input the input to predict; must not be null
     * @return the predicted label
     * @throws IllegalStateException  if not fitted
     * @throws IllegalArgumentException if input is null
     */
    public String predict(T input) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline must be fitted before predict");
        }
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        double prediction = predictInternal(input);
        return encoder.inverseTransform(prediction);
    }

    private double predictInternal(T input) {
        if (dataMode == DataMode.SPARSE) {
            Map<Integer, Double> x = extractor.transform(input);
            return model.predict(x);
        } else {
            double[] x = extractor.transformDense(input);
            return model.predict(x);
        }
    }

    /**
     * Predicts labels for a batch of inputs.
     *
     * @param inputs the inputs to predict; must not be null or contain null elements
     * @return list of predicted labels
     * @throws IllegalStateException  if not fitted
     * @throws IllegalArgumentException if inputs is null or contains null elements
     */
    public List<String> predictAll(List<T> inputs) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline must be fitted before predictAll");
        }
        if (inputs == null) {
            throw new IllegalArgumentException("inputs list cannot be null");
        }
        List<String> result = new ArrayList<>(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            T input = inputs.get(i);
            if (input == null) {
                throw new IllegalArgumentException("input at index " + i + " cannot be null");
            }
            result.add(predict(input));
        }
        return result;
    }

    /**
     * Computes the accuracy score on the given inputs and true labels.
     * <p>
     * An unknown true label (not seen during fit) counts as wrong.
     *
     * @param inputs the test inputs; must not be null, empty, or contain null elements
     * @param labels the true labels; must not be null, empty, or contain null elements
     * @return accuracy in [0, 1]
     * @throws IllegalStateException  if not fitted
     * @throws IllegalArgumentException if validation fails
     */
    public double score(List<T> inputs, List<String> labels) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline must be fitted before score");
        }
        requireValidPairs(inputs, labels);

        int correct = 0;
        for (int i = 0; i < inputs.size(); i++) {
            String predicted = predict(inputs.get(i));
            if (predicted.equals(labels.get(i))) {
                correct++;
            }
        }
        return (double) correct / inputs.size();
    }

    /**
     * Returns whether the pipeline has been fitted.
     *
     * @return true if fitted, false otherwise
     */
    public boolean isFitted() {
        return fitted;
    }

    /**
     * Returns the feature extractor.
     *
     * @return the extractor
     */
    public FeatureExtractor<T> getExtractor() {
        return extractor;
    }

    /**
     * Returns the label encoder.
     *
     * @return the encoder
     */
    public LabelEncoder getLabelEncoder() {
        return encoder;
    }

    /**
     * Returns the model.
     *
     * @return the model
     */
    public Model getModel() {
        return model;
    }

    /**
     * Returns the data mode.
     *
     * @return the data mode
     */
    public DataMode getDataMode() {
        return dataMode;
    }

    /**
     * Marks the pipeline as fitted without running validation.
     * Intended for use by {@link PipelineIO} when loading a pre-fitted pipeline.
     */
    void markFitted() {
        this.fitted = true;
    }
}