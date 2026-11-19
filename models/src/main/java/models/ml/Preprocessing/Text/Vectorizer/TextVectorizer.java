package models.ml.Preprocessing.Text.Vectorizer;

import java.util.List;
import java.util.Map;

public interface TextVectorizer {
    void fit(List<String> docs);
    Map<Integer, Double> transform(String doc);
    List<Map<Integer, Double>> transform(List<String> docs);
    int getFeatureSize();
    boolean isFitted();
    String toJson();
}