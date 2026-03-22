package models.ml.Preprocessing.Text.BagOfWords;

import java.util.List;
import java.util.Map;

public class BagOfWords {

    private AbstractBagOfWords bow;

    public BagOfWords() {
        this.bow = new AbstractBagOfWords();
    }

    public BagOfWords(String tokenPattern, boolean toLowerCase, java.util.Set<String> stopwords) {
        this.bow = new AbstractBagOfWords(tokenPattern, toLowerCase, stopwords);
    }

    public void fit(List<String> corpus) {
        bow.fit(corpus);
    }

    public Map<Integer, Integer> transform(String doc) {
        return bow.transform(doc);
    }

    public List<Map<Integer, Integer>> transform(List<String> corpus) {
        return bow.transform(corpus);
    }

    public double[] transformDense(String doc) {
        return bow.transformDense(doc);
    }

    public List<double[]> transformDense(List<String> corpus) {
        return bow.transformDense(corpus);
    }

    public Map<Integer, Integer> fitTransform(String doc) {
        return bow.fitTransform(doc);
    }

    public List<Map<Integer, Integer>> fitTransform(List<String> corpus) {
        return bow.fitTransform(corpus);
    }

    public Map<String, Integer> getVocabulary() {
        return bow.getVocabulary();
    }

    public int getFeatureSize() {
        return bow.getFeatureSize();
    }

    public Map<Integer, Integer> mergefitTransformSparse(List<String> ngrams) {
        List<Map<Integer, Integer>> tokenMaps = bow.fitTransform(ngrams);
        return bow.mergeSparse(tokenMaps);
    }

    public String toJson() {
        return bow.toJson();
    }

    public static BagOfWords fromJson(String json) {
        AbstractBagOfWords abow = AbstractBagOfWords.fromJson(json);
        BagOfWords bw = new BagOfWords();
        bw.bow = abow;
        return bw;
    }
}