package models.ml.DatasetHandler;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;

/**
 * DocToVec
 *
 * Convert documents (from files or in-memory strings) into numeric vectors.
 * Supports:
 * - TF-IDF vectorization (default)
 * - Average word embeddings (requires loading pretrained embeddings)
 *
 * File loaders:
 * - Plain text (.txt)
 * - CSV (.csv) - assumes one document per row or a specified column index
 * - TSV (.tsv) - same as CSV but tab-separated
 * - JSON (.json) - supports array of strings or array of objects with a text
 * field
 *
 * Usage:
 * DocToVec dv = new DocToVec();
 * dv.addFile("data/docs.csv", "csv", 0); // column 0 contains text
 * dv.fit(); // builds vocabulary and IDF
 * double[][] vectors = dv.transformAll();
 *
 * Author: Kotei Justice (style matched to your project)
 * Version: 1.0
 */
public class DocToVec {

    public enum Mode {
        TFIDF, AVG_EMBEDDING
    }

    // Public configuration
    public Mode mode = Mode.TFIDF;
    public boolean lowercase = true;
    public boolean removeStopwords = true;
    public int ngram = 1; // currently only 1 supported (unigrams)
    public Set<String> stopwords = defaultStopwords();
    public double minDf = 1; // minimum document frequency to keep term
    public int maxVocabSize = Integer.MAX_VALUE;

    // Internal storage
    private final List<String> rawDocuments = new ArrayList<>();
    private final List<String[]> tokenizedDocuments = new ArrayList<>();
    private final Map<String, Integer> vocab = new LinkedHashMap<>(); // token -> index
    private final Map<Integer, Integer> docFreq = new HashMap<>(); // tokenIndex -> df
    private double[] idf = null; // idf per vocab index

    // For embedding mode
    private final Map<String, double[]> wordEmbeddings = new HashMap<>();
    private int embeddingDim = 0;

    // Regex tokenizer (simple)
    private static final Pattern TOKEN_PATTERN = Pattern.compile("[\\p{L}\\p{N}]+");

    public DocToVec() {
    }

    /* -------------------- Data loading -------------------- */

    /**
     * Add a single file. type: "txt", "csv", "tsv", "json".
     * For CSV/TSV, specify column index (0-based) that contains text; if -1, use
     * first column.
     */
    public void addFile(String path, String type, int textColumnIndex) throws IOException {
        String lower = type == null ? "" : type.toLowerCase();
        switch (lower) {
            case "txt":
                addPlainTextFile(path);
                break;
            case "csv":
                addDelimitedFile(path, ',', textColumnIndex);
                break;
            case "tsv":
                addDelimitedFile(path, '\t', textColumnIndex);
                break;
            case "json":
                addJsonFile(path);
                break;
            default:
                // try to infer by extension
                String ext = getExtension(path).toLowerCase();
                if ("txt".equals(ext))
                    addPlainTextFile(path);
                else if ("csv".equals(ext))
                    addDelimitedFile(path, ',', textColumnIndex);
                else if ("tsv".equals(ext))
                    addDelimitedFile(path, '\t', textColumnIndex);
                else if ("json".equals(ext))
                    addJsonFile(path);
                else
                    throw new IllegalArgumentException("Unsupported file type: " + type);
        }
    }

    private void addPlainTextFile(String path) throws IOException {
        List<String> lines = Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);
        StringBuilder sb = new StringBuilder();
        for (String line : lines) {
            if (line.trim().isEmpty()) {
                if (sb.length() > 0) {
                    addDocument(sb.toString());
                    sb.setLength(0);
                }
            } else {
                if (sb.length() > 0)
                    sb.append(' ');
                sb.append(line.trim());
            }
        }
        if (sb.length() > 0)
            addDocument(sb.toString());
    }

    private void addDelimitedFile(String path, char sep, int textColumnIndex) throws IOException {
        try (BufferedReader br = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = splitLine(line, sep);
                int idx = textColumnIndex >= 0 && textColumnIndex < parts.length ? textColumnIndex : 0;
                addDocument(parts[idx]);
            }
        }
    }

    private void addJsonFile(String path) throws IOException {
        // Very small JSON support: either ["doc1","doc2",...] or [{"text":"..."} , ...]
        String content = new String(Files.readAllBytes(Paths.get(path)), StandardCharsets.UTF_8).trim();
        if (content.startsWith("[")) {
            // naive parse: split top-level strings or objects
            // This is intentionally simple: for robust parsing use a JSON library.
            // We'll handle array of strings and array of objects with "text" field.
            int i = 0;
            // remove outer brackets
            String inner = content.substring(1, content.length() - 1).trim();
            // split by top-level commas (naive)
            List<String> items = splitTopLevel(inner);
            for (String item : items) {
                item = item.trim();
                if (item.startsWith("\"") && item.endsWith("\"")) {
                    String s = unquote(item);
                    addDocument(s);
                } else if (item.startsWith("{") && item.endsWith("}")) {
                    // find "text": "..."
                    String text = extractJsonField(item, "text");
                    if (text != null)
                        addDocument(text);
                }
            }
        } else {
            throw new IllegalArgumentException("Unsupported JSON structure in " + path);
        }
    }

    private static List<String> splitTopLevel(String s) {
        List<String> out = new ArrayList<>();
        int depth = 0;
        int start = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '{' || c == '[')
                depth++;
            else if (c == '}' || c == ']')
                depth--;
            else if (c == ',' && depth == 0) {
                out.add(s.substring(start, i));
                start = i + 1;
            }
        }
        if (start < s.length())
            out.add(s.substring(start));
        return out;
    }

    private static String extractJsonField(String obj, String field) {
        // naive: find "field"\s*:\s*"value"
        Pattern p = Pattern.compile("\"" + Pattern.quote(field) + "\"\\s*:\\s*\"([^\"]*)\"");
        Matcher m = p.matcher(obj);
        if (m.find())
            return m.group(1);
        return null;
    }

    private static String unquote(String s) {
        if (s.length() >= 2 && s.charAt(0) == '"' && s.charAt(s.length() - 1) == '"')
            return s.substring(1, s.length() - 1).replace("\\\"", "\"");
        return s;
    }

    private static String[] splitLine(String line, char sep) {
        // naive CSV/TSV split (no quoted fields)
        return line.split(Pattern.quote(String.valueOf(sep)));
    }

    private static String getExtension(String path) {
        int i = path.lastIndexOf('.');
        if (i < 0)
            return "";
        return path.substring(i + 1);
    }

    public void addDocument(String text) {
        if (text == null)
            return;
        rawDocuments.add(text);
    }

    public void addDocuments(List<String> docs) {
        for (String d : docs)
            addDocument(d);
    }

    /* -------------------- Tokenization & preprocessing -------------------- */

    private String[] tokenize(String text) {
        if (text == null)
            return new String[0];
        String t = lowercase ? text.toLowerCase(Locale.ROOT) : text;
        List<String> tokens = new ArrayList<>();
        Matcher m = TOKEN_PATTERN.matcher(t);
        while (m.find()) {
            String tok = m.group();
            if (removeStopwords && stopwords.contains(tok))
                continue;
            tokens.add(tok);
        }
        return tokens.toArray(new String[0]);
    }

    /* -------------------- Fit / Transform API -------------------- */

    /**
     * Build vocabulary and IDF (for TF-IDF mode).
     * For AVG_EMBEDDING mode, no vocabulary is required.
     */
    public void fit() {
        tokenizedDocuments.clear();
        vocab.clear();
        docFreq.clear();

        for (String doc : rawDocuments) {
            String[] toks = tokenize(doc);
            tokenizedDocuments.add(toks);
        }

        if (mode == Mode.TFIDF) {
            // compute document frequencies
            for (String[] toks : tokenizedDocuments) {
                Set<String> seen = new HashSet<>();
                for (String t : toks) {
                    if (seen.add(t)) {
                        // increment df for token
                        // we'll assign vocab indices later
                        // use a temporary map from token->df via vocab map keys
                        // but we need token->df; use a local map
                    }
                }
            }
            // build token->df map
            Map<String, Integer> tokenDf = new HashMap<>();
            for (String[] toks : tokenizedDocuments) {
                Set<String> seen = new HashSet<>();
                for (String t : toks) {
                    if (seen.add(t))
                        tokenDf.put(t, tokenDf.getOrDefault(t, 0) + 1);
                }
            }

            // filter by minDf and sort by df desc to limit vocab size
            List<Map.Entry<String, Integer>> entries = new ArrayList<>(tokenDf.entrySet());
            entries.removeIf(e -> e.getValue() < Math.max(1, (int) minDf));
            entries.sort((a, b) -> Integer.compare(b.getValue(), a.getValue()));

            int idx = 0;
            for (Map.Entry<String, Integer> e : entries) {
                if (idx >= maxVocabSize)
                    break;
                vocab.put(e.getKey(), idx);
                docFreq.put(idx, e.getValue());
                idx++;
            }

            // compute idf
            int D = tokenizedDocuments.size();
            idf = new double[vocab.size()];
            for (Map.Entry<String, Integer> e : vocab.entrySet()) {
                int vi = e.getValue();
                int df = docFreq.getOrDefault(vi, 1);
                idf[vi] = Math.log((double) (D + 1) / (df + 1)) + 1.0; // smoothed idf
            }
        } else {
            // AVG_EMBEDDING: nothing to do here
        }
    }

    /**
     * Transform all added documents into vectors.
     * Returns double[numDocs][vectorDim]
     */
    public double[][] transformAll() {
        if (mode == Mode.TFIDF)
            return transformAllTfidf();
        else
            return transformAllAvgEmbedding();
    }

    private double[][] transformAllTfidf() {
        int D = tokenizedDocuments.size();
        int V = vocab.size();
        double[][] out = new double[D][V];
        for (int i = 0; i < D; i++) {
            String[] toks = tokenizedDocuments.get(i);
            double[] tf = new double[V];
            for (String t : toks) {
                Integer vi = vocab.get(t);
                if (vi != null)
                    tf[vi] += 1.0;
            }
            // convert to tf-idf (use log tf)
            for (int v = 0; v < V; v++) {
                if (tf[v] > 0)
                    tf[v] = (1.0 + Math.log(tf[v])) * idf[v];
                else
                    tf[v] = 0.0;
            }
            // L2 normalize
            double norm = 0.0;
            for (int v = 0; v < V; v++)
                norm += tf[v] * tf[v];
            norm = Math.sqrt(Math.max(norm, 1e-12));
            for (int v = 0; v < V; v++)
                out[i][v] = tf[v] / norm;
        }
        return out;
    }

    private double[][] transformAllAvgEmbedding() {
        int D = tokenizedDocuments.size();
        int dim = embeddingDim;
        double[][] out = new double[D][dim];
        for (int i = 0; i < D; i++) {
            String[] toks = tokenizedDocuments.get(i);
            double[] sum = new double[dim];
            int count = 0;
            for (String t : toks) {
                double[] vec = wordEmbeddings.get(t);
                if (vec != null) {
                    for (int d = 0; d < dim; d++)
                        sum[d] += vec[d];
                    count++;
                }
            }
            if (count == 0) {
                // leave zero vector
            } else {
                for (int d = 0; d < dim; d++)
                    out[i][d] = sum[d] / count;
                // optional L2 normalize
                double norm = 0.0;
                for (int d = 0; d < dim; d++)
                    norm += out[i][d] * out[i][d];
                norm = Math.sqrt(Math.max(norm, 1e-12));
                for (int d = 0; d < dim; d++)
                    out[i][d] /= norm;
            }
        }
        return out;
    }

    /**
     * Fit then transform all documents.
     */
    public double[][] fitTransform() {
        fit();
        return transformAll();
    }

    /* -------------------- Embedding utilities -------------------- */

    /**
     * Load pretrained word embeddings from a text file (word followed by dims).
     * Example line: "the 0.418 0.24968 -0.41242 ..."
     */
    public void loadEmbeddings(String path, int maxWords) throws IOException {
        try (BufferedReader br = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8)) {
            String line;
            int loaded = 0;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty())
                    continue;
                String[] parts = line.split("\\s+");
                if (parts.length < 2)
                    continue;
                String word = parts[0];
                int dim = parts.length - 1;
                if (embeddingDim == 0)
                    embeddingDim = dim;
                if (dim != embeddingDim)
                    continue; // skip inconsistent
                double[] vec = new double[dim];
                for (int i = 0; i < dim; i++)
                    vec[i] = Double.parseDouble(parts[i + 1]);
                wordEmbeddings.put(word, vec);
                loaded++;
                if (maxWords > 0 && loaded >= maxWords)
                    break;
            }
        }
    }

    /* -------------------- Persistence -------------------- */

    /**
     * Save vectors to CSV (rows = documents, columns = vector dims).
     */
    public static void saveVectorsCsv(double[][] vectors, String path) throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(path), StandardCharsets.UTF_8)) {
            for (double[] row : vectors) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < row.length; i++) {
                    if (i > 0)
                        sb.append(',');
                    sb.append(row[i]);
                }
                bw.write(sb.toString());
                bw.newLine();
            }
        }
    }

    /**
     * Save vocabulary to a simple two-column file: token, index
     */
    public void saveVocab(String path) throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(Paths.get(path), StandardCharsets.UTF_8)) {
            for (Map.Entry<String, Integer> e : vocab.entrySet()) {
                bw.write(e.getKey() + "\t" + e.getValue());
                bw.newLine();
            }
        }
    }

    /* -------------------- Utilities -------------------- */

    private static Set<String> defaultStopwords() {
        String[] arr = new String[] {
                "a", "an", "the", "and", "or", "but", "if", "while", "is", "are", "was", "were", "be", "to", "of", "in",
                "on", "for", "with", "as", "by", "at", "from"
        };
        return new HashSet<>(Arrays.asList(arr));
    }

    /* -------------------- Example usage -------------------- */

    /**
     * Example main showing how to use DocToVec.
     */
    public static void main(String[] args) throws Exception {
        // Example: load a CSV with text in column 0, compute TF-IDF vectors and save
        // them.
        DocToVec dv = new DocToVec();
        dv.mode = Mode.TFIDF;
        dv.addFile("data/docs.csv", "csv", 0);
        dv.minDf = 1;
        dv.fit();
        double[][] vecs = dv.transformAll();
        saveVectorsCsv(vecs, "data/doc_vectors.csv");
        dv.saveVocab("data/vocab.txt");
        System.out.println("Saved " + vecs.length + " vectors, dim=" + (vecs.length > 0 ? vecs[0].length : 0));
    }
}
