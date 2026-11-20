package models.ml.Pipeline;

import models.ml.Model.Model;
import models.ml.Preprocessing.Labels.LabelEncoder;
import models.ml.Preprocessing.Text.Vectorizer.TextVectorizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class TextPipeline {
    private final TextVectorizer vectorizer;
    private final LabelEncoder encoder;
    private final Model model;
    private boolean fitted = false;

    public TextPipeline(TextVectorizer vectorizer, Model model) {
        if (vectorizer == null) {
            throw new IllegalArgumentException("vectorizer cannot be null");
        }
        if (model == null) {
            throw new IllegalArgumentException("model cannot be null");
        }
        this.vectorizer = vectorizer;
        this.model = model;
        this.encoder = new LabelEncoder();
    }

    public TextPipeline(TextVectorizer vectorizer, LabelEncoder encoder, Model model) {
        if (vectorizer == null) {
            throw new IllegalArgumentException("vectorizer cannot be null");
        }
        if (encoder == null) {
            throw new IllegalArgumentException("encoder cannot be null");
        }
        if (model == null) {
            throw new IllegalArgumentException("model cannot be null");
        }
        this.vectorizer = vectorizer;
        this.encoder = encoder;
        this.model = model;
    }

    public void fit(List<String> texts, List<String> labels) {
        // Validate everything BEFORE touching the vectorizer, encoder or model
        if (texts == null) {
            throw new IllegalArgumentException("texts list cannot be null");
        }
        if (labels == null) {
            throw new IllegalArgumentException("labels list cannot be null");
        }
        if (texts.isEmpty()) {
            throw new IllegalArgumentException("texts list cannot be empty");
        }
        if (labels.isEmpty()) {
            throw new IllegalArgumentException("labels list cannot be empty");
        }
        if (texts.size() != labels.size()) {
            throw new IllegalArgumentException("texts and labels must have the same size");
        }
        for (int i = 0; i < texts.size(); i++) {
            if (texts.get(i) == null) {
                throw new IllegalArgumentException("text at index " + i + " cannot be null");
            }
        }
        for (int i = 0; i < labels.size(); i++) {
            if (labels.get(i) == null) {
                throw new IllegalArgumentException("label at index " + i + " cannot be null");
            }
        }

        // Fit the vectorizer and encoder
        vectorizer.fit(texts);
        List<Map<Integer, Double>> x = vectorizer.transform(texts);
        encoder.fit(labels);
        double[] y = encoder.transform(labels);

        // Train the model using sparse fit
        model.fit(x, y);

        fitted = true;
    }

    public String predict(String text) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline must be fitted before predict");
        }
        if (text == null) {
            throw new IllegalArgumentException("text cannot be null");
        }
        Map<Integer, Double> x = vectorizer.transform(text);
        double prediction = model.predict(x);
        return encoder.inverseTransform(prediction);
    }

    public List<String> predictAll(List<String> texts) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline must be fitted before predictAll");
        }
        if (texts == null) {
            throw new IllegalArgumentException("texts list cannot be null");
        }
        List<String> result = new ArrayList<>(texts.size());
        for (int i = 0; i < texts.size(); i++) {
            String text = texts.get(i);
            if (text == null) {
                throw new IllegalArgumentException("text at index " + i + " cannot be null");
            }
            result.add(predict(text));
        }
        return result;
    }

    public double score(List<String> texts, List<String> labels) {
        if (!fitted) {
            throw new IllegalStateException("Pipeline must be fitted before score");
        }
        if (texts == null) {
            throw new IllegalArgumentException("texts list cannot be null");
        }
        if (labels == null) {
            throw new IllegalArgumentException("labels list cannot be null");
        }
        if (texts.isEmpty()) {
            throw new IllegalArgumentException("texts list cannot be empty");
        }
        if (labels.isEmpty()) {
            throw new IllegalArgumentException("labels list cannot be empty");
        }
        if (texts.size() != labels.size()) {
            throw new IllegalArgumentException("texts and labels must have the same size");
        }
        for (int i = 0; i < texts.size(); i++) {
            if (texts.get(i) == null) {
                throw new IllegalArgumentException("text at index " + i + " cannot be null");
            }
            if (labels.get(i) == null) {
                throw new IllegalArgumentException("label at index " + i + " cannot be null");
            }
        }

        int correct = 0;
        for (int i = 0; i < texts.size(); i++) {
            String predicted = predict(texts.get(i));
            if (predicted.equals(labels.get(i))) {
                correct++;
            }
        }
        return (double) correct / texts.size();
    }

    public boolean isFitted() {
        return fitted;
    }

    public TextVectorizer getVectorizer() {
        return vectorizer;
    }

    public LabelEncoder getLabelEncoder() {
        return encoder;
    }

    public Model getModel() {
        return model;
    }
}