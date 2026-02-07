package models.ml.Preprocessing.Text.WordToVec;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AbstractWord2Vec {

    private final int vectorSize;
    private final int windowSize;
    private final double learningRate;
    private final int epochs;
    private final Pattern tokenPattern;

    private final Map<String, double[]> wordVectors = new HashMap<>();
    private final Random random = new Random();

    public AbstractWord2Vec(int vectorSize, int windowSize, double learningRate, int epochs) {
        this(vectorSize, windowSize, learningRate, epochs, "\\b\\w+\\b");
    }

    public AbstractWord2Vec(
            int vectorSize,
            int windowSize,
            double learningRate,
            int epochs,
            String tokenPattern) {
        this.vectorSize = vectorSize;
        this.windowSize = windowSize;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.tokenPattern = Pattern.compile(tokenPattern);
    }

    /**
     * Train word embeddings (CBOW)
     */
    public void fit(List<String> corpus) {
        initialize(corpus);

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (String doc : corpus) {
                List<String> tokens = tokenize(doc);

                for (int i = 0; i < tokens.size(); i++) {
                    String target = tokens.get(i);
                    double[] targetVec = wordVectors.get(target);

                    double[] contextVec = new double[vectorSize];
                    int count = 0;

                    for (int j = Math.max(0, i - windowSize); j <= Math.min(tokens.size() - 1, i + windowSize); j++) {

                        if (j == i)
                            continue;
                        double[] vec = wordVectors.get(tokens.get(j));
                        if (vec != null) {
                            for (int k = 0; k < vectorSize; k++) {
                                contextVec[k] += vec[k];
                            }
                            count++;
                        }
                    }

                    if (count == 0)
                        continue;

                    for (int k = 0; k < vectorSize; k++) {
                        contextVec[k] /= count;
                        double error = targetVec[k] - contextVec[k];
                        targetVec[k] -= learningRate * error;
                    }
                }
            }
        }
    }

    public Map<String, double[]> transform() {
        return Collections.unmodifiableMap(wordVectors);
    }

    public Map<String, double[]> fitTransform(List<String> corpus) {
        fit(corpus);
        return transform();
    }

    private void initialize(List<String> corpus) {
        wordVectors.clear();
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
