package models.ml.Preprocessing.Text.BagOfWords;

import java.util.List;
import java.util.Map;

public class BagOfWords {

    private final AbstractBagOfWords abstractBagOfWords;

    public BagOfWords() {
        this.abstractBagOfWords = new AbstractBagOfWords();
    }

    public BagOfWords(String tokenPattern) {
        this.abstractBagOfWords = new AbstractBagOfWords(tokenPattern);
    }

    public void fit(List<String> corpus) {
        abstractBagOfWords.fit(corpus);
    }

    public List<Map<Integer, Integer>> transform(List<String> corpus) {
        return abstractBagOfWords.transform(corpus);
    }

    public List<Map<Integer, Integer>> fitTransform(List<String> corpus) {
        return abstractBagOfWords.fitTransform(corpus);
    }

    public Map<String, Integer> getVocabulary() {
        return abstractBagOfWords.getVocabulary();
    }
}
