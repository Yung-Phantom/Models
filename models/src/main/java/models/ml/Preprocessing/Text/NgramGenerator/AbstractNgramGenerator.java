package models.ml.Preprocessing.Text.NgramGenerator;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AbstractNgramGenerator {

    private final int minN;
    private final int maxN;
    private final Pattern tokenPattern;
    private final Set<String> stopwords;
    private final boolean toLowerCase;

    // ---------------- Constructors ----------------
    public AbstractNgramGenerator(int n) {
        this(n, n, "\\b\\w+\\b", true, Collections.emptySet());
    }

    public AbstractNgramGenerator(int minN, int maxN) {
        this(minN, maxN, "\\b\\w+\\b", true, Collections.emptySet());
    }

    public AbstractNgramGenerator(int minN, int maxN, String tokenPattern, boolean toLowerCase, Set<String> stopwords) {
        if (minN <= 0 || maxN < minN) {
            throw new IllegalArgumentException("Invalid n-gram range");
        }
        this.minN = minN;
        this.maxN = maxN;
        this.tokenPattern = Pattern.compile(tokenPattern);
        this.toLowerCase = toLowerCase;
        this.stopwords = (stopwords != null) ? stopwords : Collections.emptySet();
    }

    // ---------------- Tokenization ----------------
    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        if (text == null || text.isEmpty())
            return tokens;

        String doc = toLowerCase ? text.toLowerCase() : text;
        Matcher matcher = tokenPattern.matcher(doc);

        while (matcher.find()) {
            String token = matcher.group();
            if (!stopwords.contains(token)) {
                tokens.add(token);
            }
        }
        return tokens;
    }

    // ---------------- N-gram generation ----------------
    public List<String> generate(String document) {
        List<String> tokens = tokenize(document);
        List<String> ngrams = new ArrayList<>();

        for (int n = minN; n <= maxN; n++) {
            for (int i = 0; i <= tokens.size() - n; i++) {
                StringBuilder sb = new StringBuilder(tokens.get(i));
                for (int j = 1; j < n; j++) {
                    sb.append("_").append(tokens.get(i + j));
                }
                ngrams.add(sb.toString());
            }
        }
        return ngrams;
    }

    public List<List<String>> generate(List<String> corpus) {
        List<List<String>> result = new ArrayList<>();
        for (String doc : corpus) {
            result.add(generate(doc));
        }
        return result;
    }

    // ---------------- Getters ----------------
    public int getMinN() {
        return minN;
    }

    public int getMaxN() {
        return maxN;
    }

    public boolean isToLowerCase() {
        return toLowerCase;
    }

    public Set<String> getStopwords() {
        return Collections.unmodifiableSet(stopwords);
    }
}