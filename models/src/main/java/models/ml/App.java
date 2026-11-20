
package models.ml;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Arrays;
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
import models.ml.Preprocessing.Scaling.ZScoreScaler;
import models.ml.Preprocessing.Scaling.MinMaxScaler;
import models.ml.Preprocessing.Scaling.MaxAbsScaler;
import models.ml.Preprocessing.Scaling.SimpleImputer;
import models.ml.Preprocessing.Scaling.Transformer;

/**
 * Demonstration application for the Models ML library.
 * <p>
 * This class showcases the capabilities of the library by running various
 * classification and regression models on sample datasets, demonstrating
 * preprocessing pipelines, and illustrating model persistence.
 * </p>
 * <p>
 * The demo uses two datasets:
 * <ul>
 *   <li>SVMtrain.csv - Classification dataset (Titanic survival)</li>
 *   <li>winequality-red.csv - Regression dataset (wine quality prediction)</li>
 * </ul>
 * </p>
 * <p>
 * Run with: {@code cd models && mvn exec:java -Dexec.mainClass=models.ml.App}
 * </p>
 *
 * @author Justice
 * @version 1.0
 */
public class App {

        private static final String DATASET_PATH = "Datasets/SVMtrain.csv";

        private static final String WINE_PATH = "Datasets/winequality-red.csv";

        private static final String[] CLASSIFICATION_METRICS = {
                        "accuracy", "precision", "recall"
        };

        private static final String[] REGRESSION_METRICS = {
                        "mse", "rmse", "mae", "r2"
        };

        /**
         * Main entry point demonstrating the ML library.
         * <p>
         * Executes the following experiments:
         * </p>
         * <ul>
         *   <li>Classification models on SVMtrain.csv: Naive Bayes, KNN, SVM, Logistic Regression (binary & multinomial)</li>
         *   <li>Preprocessing experiments with ZScore, MinMax, and MaxAbs scalers</li>
         *   <li>Regression models on winequality-red.csv: Linear Regression (normal equation & gradient descent)</li>
         *   <li>Model save/load verification</li>
         * </ul>
         *
         * @param args command-line arguments (unused)
         * @throws IOException if dataset files cannot be read
         * @throws SQLException if SQLite database access fails
         */
        public static void main(String[] args)
                        throws IOException, SQLException {

                // ==========================================
                // LOAD CLASSIFICATION DATASET
                // ==========================================

                DatasetLoader dl = new DatasetLoader(
                                DATASET_PATH,
                                DatasetConfig.HAS_ID_WITH_HEADER,
                                "Survived");

                // ==========================================
                // NAIVE BAYES
                // ==========================================

                System.out.println("\n========== NAIVE BAYES ==========");

                NaiveBayes nb = new NaiveBayes();

                nb.fit(dl, 80, "t");

                printClassificationResults(
                                nb.evaluate(CLASSIFICATION_METRICS));

                printPredictions(nb.predictLabels());

                // ==========================================
                // KNN
                // ==========================================

                System.out.println("\n========== KNN ==========");

                KNN knn = new KNN();

                knn.fit(dl, 80, "t");

                printClassificationResults(
                                knn.evaluate(CLASSIFICATION_METRICS));

                printPredictions(knn.predictLabels());

                // ==========================================
                // SVM
                // ==========================================

                System.out.println("\n========== TUNED SVM ==========");

                SVM svm = new SVM();

                svm.kernel = "rbf";
                svm.C = 20.0;
                svm.learningRate = 0.001;
                svm.epochs = 5000;
                svm.gamma = 0.01;
                svm.degree = 3;
                svm.coef0 = 1.0;
                svm.method = "svc";

                svm.fit(dl, 80, "t");

                printClassificationResults(
                                svm.evaluate(CLASSIFICATION_METRICS));

                printPredictions(svm.predictLabels());

                // ==========================================
                // BINARY LOGISTIC REGRESSION
                // ==========================================

                System.out.println(
                                "\n========== BINARY LOGISTIC REGRESSION ==========");

                LogisticRegression lr = new LogisticRegression();

                lr.method = "binary";
                lr.learningRate = 0.0005;
                lr.epochs = 5000;

                lr.fit(dl, 80, "t");
                lr.refresh();

                printClassificationResults(
                                lr.evaluate(CLASSIFICATION_METRICS));

                printPredictions(lr.predictLabels());

                System.out.println("\nLogistic Regression Probabilities:");

                var probabilities = lr.predictProbabilities();

                for (int i = 0; i < Math.min(3, probabilities.size()); i++) {
                        System.out.println(
                                        "Sample " + i + ": " + probabilities.get(i));
                }

                // Save and load
                System.out.println("\nBinary Logistic Regression Save/Load:");

                lr.save("lr_model.ser");

                LogisticRegression lrLoaded = new LogisticRegression();
                lrLoaded.load("lr_model.ser");

                printClassificationResults(
                                lrLoaded.evaluate(CLASSIFICATION_METRICS));

                // ==========================================
                // MULTINOMIAL LOGISTIC REGRESSION
                // ==========================================

                System.out.println(
                                "\n========== MULTINOMIAL LOGISTIC REGRESSION ==========");

                LogisticRegression lrMulti = new LogisticRegression();

                lrMulti.method = "multinomial";
                lrMulti.learningRate = 0.0005;
                lrMulti.epochs = 5000;

                lrMulti.fit(dl, 80, "t");
                lrMulti.refresh();

                printClassificationResults(
                                lrMulti.evaluate(CLASSIFICATION_METRICS));

                printPredictions(lrMulti.predictLabels());

                // ==========================================
                // CLASSIFICATION PREPROCESSING EXPERIMENTS
                // ==========================================

                System.out.println(
                                "\n========== CLASSIFICATION PREPROCESSING ==========");

                DatasetSplit classificationSplit = dl.split(80, "t");

                runPreprocessingExperiment(
                                "CLASSIFICATION - Z SCORE",
                                classificationSplit,
                                new ZScoreScaler());

                runPreprocessingExperiment(
                                "CLASSIFICATION - MIN MAX",
                                classificationSplit,
                                new MinMaxScaler());

                runPreprocessingExperiment(
                                "CLASSIFICATION - MAX ABS",
                                classificationSplit,
                                new MaxAbsScaler());

                // ==========================================
                // LOAD WINE DATASET
                // ==========================================

                DatasetLoader wineLoader = new DatasetLoader(
                                WINE_PATH,
                                DatasetConfig.NO_ID_WITH_HEADER,
                                "quality");

                // ==========================================
                // LINEAR REGRESSION - NORMAL
                // ==========================================

                System.out.println(
                                "\n========== LINEAR REGRESSION (NORMAL) ==========");

                LinearRegression linReg = new LinearRegression();

                linReg.method = "normal";

                linReg.fit(wineLoader, 80, "t");
                linReg.refresh();

                printRegressionResults(
                                linReg.evaluate(REGRESSION_METRICS));

                printPredictions(linReg.predictLabels());

                System.out.println("\nLinear Regression Save/Load:");

                linReg.save("linreg_model.ser");

                LinearRegression linRegLoaded = new LinearRegression();
                linRegLoaded.load("linreg_model.ser");

                printRegressionResults(
                                linRegLoaded.evaluate(REGRESSION_METRICS));

                // ==========================================
                // LINEAR REGRESSION - GRADIENT DESCENT
                // ==========================================

                System.out.println(
                                "\n========== LINEAR REGRESSION (GRADIENT DESCENT) ==========");

                LinearRegression linRegGD = new LinearRegression();

                linRegGD.method = "gd";
                linRegGD.learningRate = 0.0001;
                linRegGD.epochs = 10000;

                linRegGD.fit(wineLoader, 80, "t");
                linRegGD.refresh();

                printRegressionResults(
                                linRegGD.evaluate(REGRESSION_METRICS));

                printPredictions(linRegGD.predictLabels());

                System.out.println("\nGradient Descent Save/Load:");

                linRegGD.save("linreg_gd_model.ser");

                LinearRegression linRegGDLoaded = new LinearRegression();
                linRegGDLoaded.load("linreg_gd_model.ser");

                printRegressionResults(
                                linRegGDLoaded.evaluate(REGRESSION_METRICS));

                // ==========================================
                // REGRESSION PREPROCESSING EXPERIMENTS
                // ==========================================

                System.out.println(
                                "\n========== WINE PREPROCESSING ==========");

                DatasetSplit wineSplit = wineLoader.split(80, "t");

                runPreprocessingExperiment(
                                "WINE - Z SCORE",
                                wineSplit,
                                new ZScoreScaler());

                runPreprocessingExperiment(
                                "WINE - MIN MAX",
                                wineSplit,
                                new MinMaxScaler());

                runPreprocessingExperiment(
                                "WINE - MAX ABS",
                                wineSplit,
                                new MaxAbsScaler());

                // ==========================================
                // COMPLETED
                // ==========================================

                System.out.println(
                                "\n========== ALL EXPERIMENTS COMPLETED ==========");
        }

        // ==========================================
        // PREPROCESSING EXPERIMENT
        // ==========================================

        /**
         * Runs a preprocessing experiment with the given scaler.
         * <p>
         * Extracts features and labels from the split, imputes missing values
         * using the mean strategy, applies the scaler, and prints statistics.
         * </p>
         *
         * @param name   descriptive name for the experiment
         * @param split  train/test split containing features and labels
         * @param scaler the transformer to apply (ZScore, MinMax, MaxAbs, etc.)
         */
        private static void runPreprocessingExperiment(
                        String name,
                        DatasetSplit split,
                        Transformer scaler) {

                System.out.println("\n--- " + name + " ---");

                StandardScaler extractor = new StandardScaler();

                // Extract features
                double[][] trainFeatures = extractor.extractFeatures(split.train);

                double[][] testFeatures = extractor.extractFeatures(split.test);

                // Extract labels without scaling
                double[] trainLabels = extractor.extractLabelsRegression(split.train);

                double[] testLabels = extractor.extractLabelsRegression(split.test);

                // Impute missing values using training statistics
                SimpleImputer imputer = new SimpleImputer(
                                SimpleImputer.Strategy.MEAN);

                double[][] imputedTrain = imputer.fitTransform(trainFeatures);

                double[][] imputedTest = imputer.transform(testFeatures);

                // Fit scaler using training data only
                scaler.fit(imputedTrain);

                double[][] scaledTrain = scaler.transform(imputedTrain);

                double[][] scaledTest = scaler.transform(imputedTest);

                // Report results
                System.out.println(
                                "Training samples: " + scaledTrain.length);

                System.out.println(
                                "Testing samples: " + scaledTest.length);

                System.out.println(
                                "Number of features: " + scaledTrain[0].length);

                System.out.println(
                                "Training labels: " + trainLabels.length);

                System.out.println(
                                "Testing labels: " + testLabels.length);

                System.out.println(
                                "First scaled training sample: " +
                                                Arrays.toString(scaledTrain[0]));

                System.out.println(
                                "First scaled testing sample: " +
                                                Arrays.toString(scaledTest[0]));

                System.out.println(
                                "Scaler configuration: " + scaler.toJson());

                System.out.println(
                                "Imputer configuration: " + imputer.toJson());
        }

        /**
         * Prints classification evaluation metrics.
         *
         * @param results map of metric names to values (accuracy, precision, recall, etc.)
         */
        private static void printClassificationResults(
                        Map<String, Double> results) {

                results.forEach(
                                (metric, value) -> System.out.println(metric + ": " + value));
        }

        /**
         * Prints regression evaluation metrics.
         *
         * @param results map of metric names to values (mse, rmse, mae, r2, etc.)
         */
        private static void printRegressionResults(
                        Map<String, Double> results) {

                results.forEach(
                                (metric, value) -> System.out.println(metric + ": " + value));
        }

        /**
         * Prints the first 5 predictions (or fewer if less available).
         *
         * @param predictions array of predicted values
         */
        private static void printPredictions(double[] predictions) {

                System.out.println(
                                "Predictions: " +
                                                Arrays.toString(
                                                                Arrays.copyOfRange(
                                                                                predictions,
                                                                                0,
                                                                                Math.min(5, predictions.length))));
        }
}
