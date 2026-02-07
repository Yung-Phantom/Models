package models.ml.Preprocessing.Text.NgramGenerator;

import java.util.List;

public class NgramGenerator {

    private final AbstractNgramGenerator abstractNgramGenerator;

    public NgramGenerator(int n) {
        this.abstractNgramGenerator = new AbstractNgramGenerator(n);
    }

    public NgramGenerator(int n, String tokenPattern) {
        this.abstractNgramGenerator = new AbstractNgramGenerator(n, tokenPattern);
    }

    public List<String> generate(String document) {
        return abstractNgramGenerator.generate(document);
    }

    public List<List<String>> generate(List<String> corpus) {
        return abstractNgramGenerator.generate(corpus);
    }

    public int getN() {
        return abstractNgramGenerator.getN();
    }
}
