package models.ml.Preprocessing.Text.TFIDF;

import java.util.List;
import java.util.Map;

public class TFIDF {

    private final AbstractTFIDF abstractTFIDF;

    public TFIDF() {
        this.abstractTFIDF = new AbstractTFIDF();
    }

    public TFIDF(String tokenPattern) {
        this.abstractTFIDF = new AbstractTFIDF(tokenPattern);
    }

    public void fit(List<String> corpus) {
        abstractTFIDF.fit(corpus);
    }

    public List<Map<Integer, Double>> transform(List<String> corpus) {
        return abstractTFIDF.transform(corpus);
    }

    public List<Map<Integer, Double>> fitTransform(List<String> corpus) {
        return abstractTFIDF.fitTransform(corpus);
    }

    public Map<String, Integer> getVocabulary() {
        return abstractTFIDF.getVocabulary();
    }
}
