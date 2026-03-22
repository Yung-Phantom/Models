package models.ml.Preprocessing.Text.TFIDF;

import java.util.*;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TFIDF {

    private final AbstractTFIDF abstractTFIDF;
    private double[] idf;
    private Function<String, List<String>> tokenizer;

    public TFIDF() {
        this.abstractTFIDF = new AbstractTFIDF();
        this.tokenizer = null;
    }

    public TFIDF(String tokenPattern) {
        this.abstractTFIDF = new AbstractTFIDF(tokenPattern);
        this.tokenizer = null;
    }

    public void setTokenizer(Function<String, List<String>> tokenizer) {
        this.tokenizer = tokenizer;
    }

    public void fit(List<String> corpus) {
        abstractTFIDF.fit(corpus);
        computeIDF(corpus);
    }

    public List<Map<Integer, Double>> transformSparse(List<String> corpus) {
        List<Map<Integer, Double>> tfMaps = new ArrayList<>();

        for (String doc : corpus) {
            List<String> tokens = tokenizer != null ? tokenizer.apply(doc) : defaultTokenize(doc);
            Map<Integer, Integer> tf = new HashMap<>();

            for (String token : tokens) {
                Integer idx = abstractTFIDF.getVocabulary().get(token);
                if (idx != null)
                    tf.put(idx, tf.getOrDefault(idx, 0) + 1);
            }

            Map<Integer, Double> tfidf = new HashMap<>();
            for (Map.Entry<Integer, Integer> e : tf.entrySet()) {
                tfidf.put(e.getKey(), e.getValue() * idf[e.getKey()]);
            }

            tfMaps.add(tfidf);
        }

        return tfMaps;
    }

    public double[][] transformDense(List<String> corpus) {
        List<Map<Integer, Double>> sparse = transformSparse(corpus);
        int V = abstractTFIDF.getVocabulary().size();
        double[][] dense = new double[corpus.size()][V];

        for (int i = 0; i < sparse.size(); i++) {
            for (Map.Entry<Integer, Double> e : sparse.get(i).entrySet()) {
                dense[i][e.getKey()] = e.getValue();
            }
        }

        return dense;
    }

    public List<Map<Integer, Double>> fitTransform(List<String> corpus) {
        fit(corpus);
        return transformSparse(corpus);
    }

    private void computeIDF(List<String> corpus) {
        int N = corpus.size();
        int V = abstractTFIDF.getVocabulary().size();
        idf = new double[V];

        Map<Integer, Integer> df = new HashMap<>();
        for (String doc : corpus) {
            Set<Integer> seen = new HashSet<>();
            List<String> tokens = tokenizer != null ? tokenizer.apply(doc) : defaultTokenize(doc);

            for (String token : tokens) {
                Integer idx = abstractTFIDF.getVocabulary().get(token);
                if (idx != null)
                    seen.add(idx);
            }

            for (Integer idx : seen)
                df.put(idx, df.getOrDefault(idx, 0) + 1);
        }

        for (int i = 0; i < V; i++) {
            int docFreq = df.getOrDefault(i, 0);
            idf[i] = Math.log((N + 1.0) / (docFreq + 1.0)) + 1.0; 
        }
    }

    private List<String> defaultTokenize(String doc) {
        List<String> tokens = new ArrayList<>();
        Pattern pattern = Pattern.compile("\\b\\w+\\b");
        Matcher matcher = pattern.matcher(doc.toLowerCase());
        while (matcher.find()) {
            tokens.add(matcher.group());
        }
        return tokens;
    }

    public Map<String, Integer> getVocabulary() {
        return abstractTFIDF.getVocabulary();
    }

    public double[] getIdf() {
        return idf != null ? Arrays.copyOf(idf, idf.length) : new double[0];
    }
}