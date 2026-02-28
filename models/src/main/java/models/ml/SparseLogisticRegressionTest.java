package models.ml;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.SQLException;
import java.util.*;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.Preprocessing.Text.TFIDF.TFIDF;
import models.ml.Preprocessing.Text.BagOfWords.BagOfWords;
import models.ml.LogisticRegression.LogisticRegression;

public class SparseLogisticRegressionTest {
    public static void main(String[] args) throws IOException, SQLException {
        String datasetPath = "C:/Users/Kotei Justice/Documents/Models/Datasets/emotion-labels-test.csv";

        List<String> lines = Files.readAllLines(Paths.get(datasetPath), StandardCharsets.UTF_8);

        List<String> texts = new ArrayList<>();
        List<Double> labels = new ArrayList<>();
        Map<String, Integer> labelMap = new LinkedHashMap<>();

        // Parse CSV (skip header)
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i);
            int lastCommaIndex = line.lastIndexOf(',');
            if (lastCommaIndex > 0) {
                String text = line.substring(0, lastCommaIndex);
                String label = line.substring(lastCommaIndex + 1).trim();

                if (text.startsWith("\"")) {
                    text = text.substring(1);
                }
                if (text.endsWith("\"")) {
                    text = text.substring(0, text.length() - 1);
                }

                texts.add(text);
                labelMap.putIfAbsent(label, labelMap.size());
                labels.add((double) labelMap.get(label));
            }
        }

        System.out.println("Loaded " + texts.size() + " texts with " + labelMap.size() + " emotion labels");
        System.out.println("Label mapping: " + labelMap);
        System.out.println();

        // Stratified split 80/20
        Map<Integer, List<Integer>> labelIndices = new LinkedHashMap<>();
        for (int i = 0; i < labels.size(); i++) {
            int label = (int) labels.get(i).doubleValue();
            labelIndices.computeIfAbsent(label, k -> new ArrayList<>()).add(i);
        }

        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> testIndices = new ArrayList<>();
        Random rand = new Random(42);

        for (List<Integer> indices : labelIndices.values()) {
            Collections.shuffle(indices, rand);
            int splitPoint = (int) (indices.size() * 0.8);
            trainIndices.addAll(indices.subList(0, splitPoint));
            testIndices.addAll(indices.subList(splitPoint, indices.size()));
        }

        Collections.shuffle(trainIndices, rand);
        Collections.shuffle(testIndices, rand);

        List<String> trainTexts = new ArrayList<>();
        List<String> testTexts = new ArrayList<>();
        double[] trainLabels = new double[trainIndices.size()];
        double[] testLabels = new double[testIndices.size()];

        for (int i = 0; i < trainIndices.size(); i++) {
            int idx = trainIndices.get(i);
            trainTexts.add(texts.get(idx));
            trainLabels[i] = labels.get(idx);
        }

        for (int i = 0; i < testIndices.size(); i++) {
            int idx = testIndices.get(i);
            testTexts.add(texts.get(idx));
            testLabels[i] = labels.get(idx);
        }

        System.out.println("Train size: " + trainTexts.size() + " Test size: " + testTexts.size());

        // Feature extractors: TF-IDF and BagOfWords
        // 1) TF-IDF
        System.out.println("\n---- TF-IDF experiments ----");
        TFIDF tfidf = new TFIDF();
        tfidf.fit(trainTexts);
        List<Map<Integer, Double>> trainTfidf = tfidf.transform(trainTexts);
        List<Map<Integer, Double>> testTfidf = tfidf.transform(testTexts);
        int vocabSizeTfidf = tfidf.getVocabulary().size();
        System.out.println("TF-IDF vocabulary size: " + vocabSizeTfidf);

        runLogisticExperiments(trainTfidf, testTfidf, trainLabels, testLabels, vocabSizeTfidf);

        // 2) BagOfWords
        System.out.println("\n---- BagOfWords experiments ----");
        BagOfWords bow = new BagOfWords();
        bow.fit(trainTexts);
        List<Map<Integer, Integer>> trainBowRaw = bow.transform(trainTexts);
        List<Map<Integer, Integer>> testBowRaw = bow.transform(testTexts);

        List<Map<Integer, Double>> trainBow = new ArrayList<>();
        List<Map<Integer, Double>> testBow = new ArrayList<>();

        for (Map<Integer, Integer> row : trainBowRaw) {
            Map<Integer, Double> converted = new LinkedHashMap<>();
            for (Map.Entry<Integer, Integer> e : row.entrySet()) {
                converted.put(e.getKey(), e.getValue().doubleValue());
            }
            trainBow.add(converted);
        }

        for (Map<Integer, Integer> row : testBowRaw) {
            Map<Integer, Double> converted = new LinkedHashMap<>();
            for (Map.Entry<Integer, Integer> e : row.entrySet()) {
                converted.put(e.getKey(), e.getValue().doubleValue());
            }
            testBow.add(converted);
        }

        int vocabSizeBow = bow.getVocabulary().size();
        System.out.println("BagOfWords vocabulary size: " + vocabSizeBow);

        runLogisticExperiments(trainBow, testBow, trainLabels, testLabels, vocabSizeBow);

        System.out.println("\nSparse Logistic Regression tests completed.");
    }

    private static void runLogisticExperiments(List<Map<Integer, Double>> trainSparse,
            List<Map<Integer, Double>> testSparse,
            double[] trainLabels,
            double[] testLabels,
            int numFeatures) {

        double[] learningRates = {0.1, 0.01};
        int[] epochsList = {200, 500};

        for (double lr : learningRates) {
            for (int epochs : epochsList) {
                System.out.println("Training LogisticRegression with lr=" + lr + " epochs=" + epochs);

                LogisticRegression log = new LogisticRegression(trainSparse, trainLabels, testSparse, testLabels,
                        numFeatures, "m", lr, epochs);

                double acc = log.accuracy();
                System.out.println("Accuracy: " + String.format("%.4f", acc));

                int[] preds = log.predictAll();
                System.out.println("First 10 predictions: " + Arrays.toString(Arrays.copyOf(preds, Math.min(10, preds.length))));
                System.out.println("First 10 actuals: " + Arrays.toString(Arrays.copyOf(testLabels, Math.min(10, testLabels.length))));
                System.out.println();
            }
        }
    }
}
