package models.ml;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Map;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.DatasetHandler.helpers.DatasetConfig;
import models.ml.DatasetHandler.helpers.DatasetSplit;
import models.ml.Model.LogisticRegression.LogisticRegression;
import models.ml.Model.LinearRegression.LinearRegression;
import models.ml.Model.NaiveBayes.NaiveBayes;
import models.ml.Model.KNN.KNN;
import models.ml.Model.SVM.SVM;
import models.ml.Preprocessing.Scaling.StandardScaler;

public class App {
    public static void main(String[] args) throws IOException, SQLException {
        // Load the dataset (Iris or your medical dataset)
        DatasetLoader dl = new DatasetLoader(
                "C:/Users/Kotei Justice/Documents/Models/Datasets/SVMTrain.csv",
                DatasetConfig.HAS_ID_WITH_HEADER,
                "Survived");

        // Define evaluation metrics
        String[] metrics = { "accuracy", "precision", "recall" };

        // --- Naive Bayes ---
        NaiveBayes nb = new NaiveBayes();
        nb.fit(dl, 80, "t");
        Map<String, Double> nbResults = nb.evaluate(metrics);
        System.out.println("--- Naive Bayes Evaluation Results ---");
        nbResults.forEach((metric, value) -> System.out.println(metric + ": " + value));
        System.out.println("NB Predictions (first 5): " +
                java.util.Arrays.toString(java.util.Arrays.copyOfRange(nb.predictLabels(), 0, 5)));

        // --- KNN ---
        KNN knn = new KNN();
        knn.fit(dl, 80, "t");
        Map<String, Double> knnResults = knn.evaluate(metrics);
        System.out.println("\n--- KNN Evaluation Results ---");
        knnResults.forEach((metric, value) -> System.out.println(metric + ": " + value));
        System.out.println("KNN Predictions (first 5): " +
                java.util.Arrays.toString(java.util.Arrays.copyOfRange(knn.predictLabels(), 0, 5)));

        // --- SVM tuned ---
        SVM svm = new SVM();
        System.out.println("\nFitting SVM with RBF kernel, C=10, learningRate=0.01, epochs=5000...");
        svm.fit(dl, 80, "t");
        // Override params before refresh
        svm.kernel = "rbf";
        svm.C = 10.0;
        svm.learningRate = 0.01;
        svm.epochs = 3;
        svm.gamma = 0.1;
        svm.degree = 3;
        svm.coef0 = 1.0;
        svm.method = "svc";
        svm.refresh();

        Map<String, Double> svmResults = svm.evaluate(metrics);
        System.out.println("--- Tuned SVM Evaluation Results ---");
        svmResults.forEach((metric, value) -> System.out.println(metric + ": " + value));
        System.out.println("SVM Predictions (first 5): " +
                java.util.Arrays.toString(java.util.Arrays.copyOfRange(svm.predictLabels(), 0, 5)));

        // --- Logistic Regression ---
        System.out.println("\n--- Logistic Regression (Binary) ---");
        LogisticRegression lr = new LogisticRegression();
        lr.fit(dl, 80, "t");
        lr.method = "binary";
        lr.learningRate = 0.01;
        lr.epochs = 1000;
        lr.refresh();

        Map<String, Double> lrResults = lr.evaluate(metrics);
        lrResults.forEach((metric, value) -> System.out.println(metric + ": " + value));
        System.out.println("LR Predictions (first 5): " +
                java.util.Arrays.toString(java.util.Arrays.copyOfRange(lr.predictLabels(), 0, 5)));

        // Logistic Regression probabilities
        System.out.println("\n--- Logistic Regression Probabilities (first 3) ---");
        var probs = lr.predictProbabilities();
        for (int i = 0; i < Math.min(3, probs.size()); i++) {
            System.out.println("Sample " + i + ": " + probs.get(i));
        }

        // Logistic Regression multinomial example (using same binary dataset for demo)
        System.out.println("\n--- Logistic Regression (Multinomial) ---");
        LogisticRegression lrMulti = new LogisticRegression();
        lrMulti.fit(dl, 80, "t");
        lrMulti.method = "multinomial";
        lrMulti.learningRate = 0.01;
        lrMulti.epochs = 1000;
        lrMulti.refresh();

        Map<String, Double> lrMultiResults = lrMulti.evaluate(metrics);
        lrMultiResults.forEach((metric, value) -> System.out.println(metric + ": " + value));
        System.out.println("LR Multi Predictions (first 5): " +
                java.util.Arrays.toString(java.util.Arrays.copyOfRange(lrMulti.predictLabels(), 0, 5)));

        // Save/Load demo
        System.out.println("\n--- Logistic Regression Save/Load ---");
        lr.save("lr_model.ser");
        LogisticRegression lrLoaded = new LogisticRegression();
        lrLoaded.load("lr_model.ser");
        Map<String, Double> lrLoadedResults = lrLoaded.evaluate(metrics);
        lrLoadedResults.forEach((metric, value) -> System.out.println(metric + ": " + value));
        System.out.println("Loaded LR Predictions (first 5): " +
                java.util.Arrays.toString(java.util.Arrays.copyOfRange(lrLoaded.predictLabels(), 0, 5)));

        // --- Linear Regression (wine quality prediction) ---
        System.out.println("\n--- Linear Regression (Wine Quality) ---");
        DatasetLoader wineLoader = new DatasetLoader(
                "C:/Users/Kotei Justice/Documents/Models/Datasets/winequality-red.csv",
                DatasetConfig.NO_ID_WITH_HEADER,
                "quality");

        String[] regMetrics = { "mse", "rmse", "mae", "r2" };
        LinearRegression linReg = new LinearRegression();
        linReg.fit(wineLoader, 80, "t");
        linReg.method = "normal";
        linReg.refresh();

        Map<String, Double> linRegResults = linReg.evaluate(regMetrics);
        linRegResults.forEach((metric, value) -> System.out.println(metric + ": " + value));
        System.out.println("Linear Regression Predictions (first 5): " +
                java.util.Arrays.toString(java.util.Arrays.copyOfRange(linReg.predictLabels(), 0, 5)));

        // Linear Regression with gradient descent
        System.out.println("\n--- Linear Regression (Gradient Descent) ---");
        LinearRegression linRegGD = new LinearRegression();
        linRegGD.fit(wineLoader, 80, "t");
        linRegGD.method = "gd";
        linRegGD.learningRate = 0.01;
        linRegGD.epochs = 1000;
        linRegGD.refresh();

        Map<String, Double> linRegGDResults = linRegGD.evaluate(regMetrics);
        linRegGDResults.forEach((metric, value) -> System.out.println(metric + ": " + value));
        System.out.println("Linear Regression GD Predictions (first 5): " +
                java.util.Arrays.toString(java.util.Arrays.copyOfRange(linRegGD.predictLabels(), 0, 5)));

        // Save/Load demo
        System.out.println("\n--- Linear Regression Save/Load ---");
        linReg.save("linreg_model.ser");
        LinearRegression linRegLoaded = new LinearRegression();
        linRegLoaded.load("linreg_model.ser");
        Map<String, Double> linRegLoadedResults = linRegLoaded.evaluate(regMetrics);
        linRegLoadedResults.forEach((metric, value) -> System.out.println(metric + ": " + value));
        System.out.println("Loaded Linear Regression Predictions (first 5): " +
                java.util.Arrays.toString(java.util.Arrays.copyOfRange(linRegLoaded.predictLabels(), 0, 5)));

        DatasetSplit split = dl.split(80, "t");
        double[][] train = new StandardScaler().extractFeatures(split.train);
        double[][] test = new StandardScaler().extractFeatures(split.test);
        double[] trainLabels = new StandardScaler().extractLabels(split.train);
        double[] testLabels = new StandardScaler().extractLabels(split.test);
    }
}