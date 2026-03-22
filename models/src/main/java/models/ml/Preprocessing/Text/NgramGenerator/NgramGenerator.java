package models.ml.Preprocessing.Text.NgramGenerator;

import java.util.List;
import java.util.Set;

public class NgramGenerator {

    private AbstractNgramGenerator generator;

    // ---------------- Constructors ----------------
    public NgramGenerator(int n) {
        this.generator = new AbstractNgramGenerator(n);
    }

    public NgramGenerator(int minN, int maxN) {
        this.generator = new AbstractNgramGenerator(minN, maxN);
    }

    public NgramGenerator(int minN, int maxN, String tokenPattern, boolean toLowerCase, Set<String> stopwords) {
        this.generator = new AbstractNgramGenerator(minN, maxN, tokenPattern, toLowerCase, stopwords);
    }

    // ---------------- Single doc ----------------
    public List<String> generate(String document) {
        return generator.generate(document);
    }

    // ---------------- Corpus ----------------
    public List<List<String>> generate(List<String> corpus) {
        return generator.generate(corpus);
    }

    // ---------------- Getters ----------------
    public int getMinN() {
        return generator.getMinN();
    }

    public int getMaxN() {
        return generator.getMaxN();
    }

    public boolean isToLowerCase() {
        return generator.isToLowerCase();
    }

    public Set<String> getStopwords() {
        return generator.getStopwords();
    }
}