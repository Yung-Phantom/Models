package models.ml;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.DatasetHandler.helpers.DatasetConfig;
import models.ml.DatasetHandler.helpers.DatasetSplit;
import models.ml.KNN.KNN;
import models.ml.LinearRegression.LinearRegression;
import models.ml.LogisticRegression.LogisticRegression;
import models.ml.NaiveBayes.NaiveBayes;
import models.ml.SVM.SVM;

public class App {

    public static void main(String[] args) throws Exception {
        DatasetLoader datasetLoader = new DatasetLoader("C:/Users/Kotei Justice/Documents/Models/Datasets/Iris.csv");
        DatasetSplit split = datasetLoader.split(80);
        double[][] train = split.train;
        double[][] test = split.test;

        // SVM Example
        SVM svm = new SVM(train, test);
        int[] svmPredictions = svm.predictAll();
        for (int p : svmPredictions) {
            System.out.println("SVM predicted label: " + p);
        }
        System.out.println("SVM Accuracy: " + svm.accuracy());

    }

}
