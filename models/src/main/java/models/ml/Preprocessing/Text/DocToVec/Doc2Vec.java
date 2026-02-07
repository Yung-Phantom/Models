package models.ml.Preprocessing.Text.DocToVec;

import java.util.List;

public class Doc2Vec {

    private final AbstractDoc2Vec abstractDoc2Vec;

    public Doc2Vec(int vectorSize, double learningRate, int epochs) {
        this.abstractDoc2Vec = new AbstractDoc2Vec(vectorSize, learningRate, epochs);
    }

    public Doc2Vec(int vectorSize, double learningRate, int epochs, String tokenPattern) {
        this.abstractDoc2Vec = new AbstractDoc2Vec(vectorSize, learningRate, epochs, tokenPattern);
    }

    public void fit(List<String> corpus) {
        abstractDoc2Vec.fit(corpus);
    }

    public List<double[]> transform() {
        return abstractDoc2Vec.transform();
    }

    public List<double[]> fitTransform(List<String> corpus) {
        return abstractDoc2Vec.fitTransform(corpus);
    }

    public int getVectorSize() {
        return abstractDoc2Vec.getVectorSize();
    }
}
