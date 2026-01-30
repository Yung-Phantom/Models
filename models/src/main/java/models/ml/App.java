package models.ml;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.DatasetHandler.helpers.DatasetSplit;
import models.ml.KNN.KNN;
import models.ml.LinearRegression.LinearRegression;
import models.ml.NaiveBayes.NaiveBayes;

public class App {

    public static void main(String[] args) throws Exception {
        DatasetLoader datasetLoader = new DatasetLoader(
                "C:/Users/Kotei Justice/Documents/Models/Datasets/database.sqlite", "Species");
        DatasetSplit split = datasetLoader.split(80);
        double[][] train = split.train;
        double[][] test = split.test;

        LinearRegression lr = new LinearRegression(train, test, "normal", 0.1, 1000);
        double[] predictedLabel = lr.predictAll();
        for (double i : predictedLabel) {
            System.out.println("predicted label: " + i);
        }
        System.out.println("R2: " + lr.r2() + "\nMse:" + lr.mse());

        NaiveBayes nb = new NaiveBayes(train, test, "multinomial");
        int[] predict = nb.predictAll();
        for (int i : predict) {
        System.out.println("predicted label: " + i);
        }
        System.out.println("Accuracy: " + nb.accuracy());

        KNN knn = new KNN(train, test, "minkowski", 10);
        int[] prediction = knn.predictAllMajority();
        for (int i : prediction) {
        System.out.println("predicted label: " + i);
        }
        System.out.println("Accuracy: " + knn.accuracy());
    }

}
