package models.ml.Preprocessing.Text.DocToVec;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AbstractDoc2Vec {

    private final int vectorSize;
    private final double learningRate;
    private final int epochs;
    private final Pattern tokenPattern;

    private final Map<String, double[]> wordVectors = new HashMap<>();
    private final List<double[]> documentVectors = new ArrayList<>();
    private final Random random = new Random();

    public AbstractDoc2Vec(int vectorSize, double learningRate, int epochs) {
        this(vectorSize, learningRate, epochs, "\\b\\w+\\b");
    }

    public AbstractDoc2Vec(int vectorSize, double learningRate, int epochs, String tokenPattern) {
        this.vectorSize = vectorSize;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.tokenPattern = Pattern.compile(tokenPattern);
    }

    /**
     * Train document vectors
     */
    public void fit(List<String> corpus) {
        initialize(corpus);

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int docId = 0; docId < corpus.size(); docId++) {
                List<String> tokens = tokenize(corpus.get(docId));
                double[] docVector = documentVectors.get(docId);

                for (String token : tokens) {
                    double[] wordVector = wordVectors.get(token);
                    if (wordVector == null)
                        continue;

                    // simple gradient update
                    for (int i = 0; i < vectorSize; i++) {
                        double error = wordVector[i] - docVector[i];
                        docVector[i] += learningRate * error;
                        wordVector[i] -= learningRate * error;
                    }
                }
            }
        }
    }

    /**
     * Return dense document embeddings
     */
    public List<double[]> transform() {
        return documentVectors;
    }

    public List<double[]> fitTransform(List<String> corpus) {
        fit(corpus);
        return transform();
    }

    private void initialize(List<String> corpus) {
        documentVectors.clear();
        wordVectors.clear();

        for (int i = 0; i < corpus.size(); i++) {
            documentVectors.add(randomVector());
        }

        for (String doc : corpus) {
            for (String token : tokenize(doc)) {
                wordVectors.putIfAbsent(token, randomVector());
            }
        }
    }

    private List<String> tokenize(String document) {
        List<String> tokens = new ArrayList<>();
        Matcher matcher = tokenPattern.matcher(document.toLowerCase());
        while (matcher.find()) {
            tokens.add(matcher.group());
        }
        return tokens;
    }

    private double[] randomVector() {
        double[] vec = new double[vectorSize];
        for (int i = 0; i < vectorSize; i++) {
            vec[i] = (random.nextDouble() - 0.5) / vectorSize;
        }
        return vec;
    }

    public int getVectorSize() {
        return vectorSize;
    }
}
