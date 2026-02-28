package models.ml;

import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.DatasetHandler.helpers.DatasetConfig;
import models.ml.DatasetHandler.helpers.DatasetSplit;
import models.ml.LogisticRegression.LogisticRegression;
import models.ml.NaiveBayes.NaiveBayes;
import models.ml.Preprocessing.Scaling.StandardScaler;
import models.ml.SVM.SVM;

public class App {
    public static void main(String[] args) throws IOException, SQLException {
        DatasetLoader dl = new DatasetLoader(
                "C:/Users/Kotei Justice/Documents/Models/Datasets/Iris.csv", DatasetConfig.HAS_ID_WITH_HEADER,
                "Species");

        DatasetSplit split = dl.split(0.02, "n");
        System.out.println("Train size: " + split.train.length);
        System.out.println("Test size: " + split.test.length);
        StandardScaler scaler = new StandardScaler();
        double[][] train = scaler.extractFeatures(split.train);
        double[] trainLabels = scaler.extractLabels(split.train);
        double[][] test = scaler.extractFeatures(split.test);
        double[] testLabels = scaler.extractLabels(split.test);
        
        // Initialize SVM
        SVM svm = new SVM(
                train, trainLabels,
                test, testLabels,
                1.0,        // C
                0.01,       // learning rate
                1000,       // epochs
                "linearKernel",
                "linearsvc"
        );

        // Compute accuracy
        double accuracy = svm.accuracy();
        System.out.println("SVM Accuracy: " + accuracy);
    }
}
