package models.ml;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Map;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.DatasetHandler.helpers.DatasetConfig;
import models.ml.Model.KNN.KNN;

public class App {
    public static void main(String[] args) throws IOException, SQLException {
        DatasetLoader dl = new DatasetLoader(
                "C:/Users/Kotei Justice/Documents/Models/Datasets/Iris.csv",
                DatasetConfig.HAS_ID_WITH_HEADER,
                "Species");

        KNN knn = new KNN();
        System.out.println("Fitting KNN using automated DatasetLoader bridge...");
        knn.fit(dl, 70, "stratified");
        String[] metrics = { "accuracy", "precision", "recall" };
        Map<String, Double> results = knn.evaluate(metrics);

        System.out.println("--- KNN Evaluation Results ---");
        results.forEach((metric, value) -> System.out.println(metric + ": " + value));

        int[] predictions = knn.predictLabels();
        System.out.println("First 5 predictions: " +
                java.util.Arrays.toString(java.util.Arrays.copyOfRange(predictions, 0, 5)));
    }
}