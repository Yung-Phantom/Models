package models.ml;

import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.DatasetHandler.helpers.DatasetSplit;
import models.ml.KNN.KNN;
import models.ml.NaiveBayes.NaiveBayes;
import models.ml.Preprocessing.Text.TFIDF.TFIDF;

public class App {
    public static void main(String[] args) throws IOException, SQLException {

        List<String> corpus = new ArrayList<>();
        corpus.add("The quick brown fox jumps over the lazy dog.");
        corpus.add("Lorem ipsum dolor sit amet consectetur adipiscing elit.");

        // simple labels
        double[] labels = {0, 1};

        // --- TF-IDF ---
        TFIDF tfidf = new TFIDF();
        List<Map<Integer, Double>> features = tfidf.fitTransform(corpus);


        DatasetLoader dl = new DatasetLoader("C:/Users/Kotei Justice/Documents/Models/Datasets/Iris.csv", "Species");
        DatasetSplit dataset = dl.split(6, "random");
        // --- KNN ---
        NaiveBayes nb = new NaiveBayes(dataset.train, dataset.test);
        System.out.println(nb.accuracy());
    }
}
