package models.ml.Preprocessing.Text.NgramGenerator;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AbstractNgramGenerator {

    private final int n;
    private final Pattern tokenPattern;

    public AbstractNgramGenerator(int n) {
        this(n, "\\b\\w+\\b");
    }

    public AbstractNgramGenerator(int n, String tokenPattern) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be >= 1");
        }
        this.n = n;
        this.tokenPattern = Pattern.compile(tokenPattern);
    }

    /**
     * Generate n-grams from a single document
     */
    public List<String> generate(String document) {
        List<String> tokens = tokenize(document);
        List<String> ngrams = new ArrayList<>();

        if (tokens.size() < n) {
            return ngrams;
        }

        for (int i = 0; i <= tokens.size() - n; i++) {
            StringBuilder ngram = new StringBuilder(tokens.get(i));
            for (int j = 1; j < n; j++) {
                ngram.append("_").append(tokens.get(i + j));
            }
            ngrams.add(ngram.toString());
        }

        return ngrams;
    }

    /**
     * Generate n-grams for a corpus
     */
    public List<List<String>> generate(List<String> corpus) {
        List<List<String>> result = new ArrayList<>();
        for (String doc : corpus) {
            result.add(generate(doc));
        }
        return result;
    }

    private List<String> tokenize(String document) {
        List<String> tokens = new ArrayList<>();
        Matcher matcher = tokenPattern.matcher(document.toLowerCase());
        while (matcher.find()) {
            tokens.add(matcher.group());
        }
        return tokens;
    }

    public int getN() {
        return n;
    }
}
