package models.ml.Preprocessing.Text.WordToVec;

import java.util.List;
import java.util.Map;

public class Word2Vec {

    private final AbstractWord2Vec abstractWord2Vec;

    public Word2Vec(int vectorSize, int windowSize, double learningRate, int epochs) {
        this.abstractWord2Vec = new AbstractWord2Vec(vectorSize, windowSize, learningRate, epochs);
    }

    public Word2Vec(
            int vectorSize,
            int windowSize,
            double learningRate,
            int epochs,
            String tokenPattern) {
        this.abstractWord2Vec = new AbstractWord2Vec(vectorSize, windowSize, learningRate, epochs, tokenPattern);
    }

    public void fit(List<String> corpus) {
        abstractWord2Vec.fit(corpus);
    }

    public Map<String, double[]> transform() {
        return abstractWord2Vec.transform();
    }

    public Map<String, double[]> fitTransform(List<String> corpus) {
        return abstractWord2Vec.fitTransform(corpus);
    }

    public int getVectorSize() {
        return abstractWord2Vec.getVectorSize();
    }
}
