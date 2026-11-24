# Text Processing

This page covers `BagOfWordsVectorizer`, `TfidfVectorizer`, `TextPipeline`, and notes on Word2Vec/Doc2Vec.

## AbstractTextVectorizer (Base)

Both vectorizers extend `AbstractTextVectorizer` with shared configuration:

<!-- no-compile -->
```java
new BagOfWordsVectorizer(
    tokenPattern,      // regex, default "\\b\\w+\\b"
    toLowerCase,       // default true
    stopwords,         // Set<String>, default empty
    binary,            // boolean, default false
    minDocFreq,        // int, default 1
    maxFeatures,       // int, default -1 (unlimited)
    ngramMin,          // int, default 1
    ngramMax           // int, default 1
);
```

### Tokenization

- `tokenPattern`: Java regex for token extraction. Default matches word boundaries.
- `toLowerCase`: Apply before tokenization.
- `stopwords`: Removed after tokenization.

### N-grams

- `ngramMin`, `ngramMax`: Inclusive range. Default 1,1 (unigrams only).
- Example: `ngramMin=1, ngramMax=2` → unigrams + bigrams.
- N-grams are joined with `_` (e.g., "new_york").

### Vocabulary Building

`buildVocabulary(List<String> docs)`:
1. Tokenize each doc, generate n-grams.
2. Count document frequency (DF) per n-gram.
3. Filter by `minDocFreq` (default 1).
4. Sort by DF descending, then alphabetically.
5. Keep top `maxFeatures` (default unlimited).
6. Assign indices 0..n-1.

### Common Methods

| Method | Description |
|--------|-------------|
| `fit(List<String>)` | Build vocabulary |
| `transform(String)` | Single doc → sparse map |
| `transform(List<String>)` | Batch → List of sparse maps |
| `transformDense(String)` | Single doc → dense vector |
| `transformDense(List<String>)` | Batch → dense matrix |
| `toJson()` / `fromJson(String)` | JSON round-trip |

---

## BagOfWordsVectorizer

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `binary` | boolean | false | If true, count = 1 per doc (presence only) |
| `minDocFreq` | int | 1 | Minimum document frequency |
| `maxFeatures` | int | -1 (unlimited) | Vocabulary size cap |

### Example

<!-- no-compile -->
```java
BagOfWordsVectorizer vec = new BagOfWordsVectorizer(
    "\\b\\w+\\b", true, Set.of("the","a"),  // tokenPattern, lowercase, stopwords
    false, 1, 1000, 1, 2                    // binary, minDF, maxFeatures, ngramMin, ngramMax
);
vec.fit(List.of("hello world", "world peace"));
Map<Integer, Double> sparse = vec.transform("hello world");
// {0: 1.0, 1: 1.0}  (indices depend on DF sorting)
```

### JSON Format

```json
{
  "binary": false,
  "minDocFreq": 1,
  "maxFeatures": 1000,
  "toLowerCase": true,
  "ngramMin": 1,
  "ngramMax": 2,
  "tokenPattern": "\\b\\w+\\b",
  "stopwords": ["the","a"],
  "vocab": {"hello": 0, "world": 1, "peace": 2},
  "fitted": true
}
```

---

## TfidfVectorizer

### Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `binary` | boolean | false | TF = 1 per doc if true |
| `minDocFreq` | int | 1 | Minimum document frequency |
| `maxFeatures` | int | -1 (unlimited) | Vocabulary size cap |
| `sublinearTf` | boolean | false | TF = 1 + log(tf) if true |
| `l2Normalize` | boolean | true | L2-normalize output vectors |

### TF-IDF Formula

```
tf = count (or 1 if binary)
if sublinearTf and tf > 0: tf = 1 + ln(tf)
idf = ln((N + 1) / (df + 1)) + 1   // N = num docs, df = doc frequency
tfidf = tf * idf
if l2Normalize: divide vector by sqrt(sum(tfidf²))
```

### Example

<!-- no-compile -->
```java
TfidfVectorizer vec = new TfidfVectorizer(
    "\\b\\w+\\b", true, Set.of("the","a"),
    false, 1, 1000, false, true, 1, 1
);
vec.fit(List.of("hello world", "world peace"));
double[] dense = vec.transformDense("hello world");
// [0.707..., 0.707..., 0.0]  (L2 normalized)
```

### JSON Format

```json
{
  "binary": false,
  "minDocFreq": 1,
  "maxFeatures": 1000,
  "sublinearTf": false,
  "l2Normalize": true,
  "toLowerCase": true,
  "ngramMin": 1,
  "ngramMax": 1,
  "tokenPattern": "\\b\\w+\\b",
  "stopwords": ["the","a"],
  "idf": [1.405..., 1.0, 1.405...],
  "vocab": {"hello": 0, "world": 1, "peace": 2},
  "fitted": true
}
```

---

## TextPipeline

End-to-end text classification pipeline.

<!-- no-compile -->
```java
TextPipeline pipeline = new TextPipeline(
    new TfidfVectorizer(),  // or BagOfWordsVectorizer
    new LabelEncoder(),
    new NaiveBayes(),       // MultinomialNB for sparse
    DataMode.SPARSE
);
pipeline.fit(List.of("great movie", "terrible film"), List.of("pos", "neg"));
String pred = pipeline.predict("wonderful film");  // "pos"
double acc = pipeline.score(testTexts, testLabels);
```

### Constructors

<!-- no-compile -->
```java
// Vectorizer + model (new LabelEncoder, SPARSE)
new TextPipeline(vectorizer, model)

// + encoder
new TextPipeline(vectorizer, encoder, model)

// Full control
new TextPipeline(vectorizer, encoder, model, DataMode.SPARSE)
```

### Methods

| Method | Description |
|--------|-------------|
| `fit(List<String>, List<String>)` | Train vectorizer + encoder + model |
| `predict(String)` | Single prediction |
| `predictAll(List<String>)` | Batch predictions |
| `score(List<String>, List<String>)` | Accuracy on test set |
| `isFitted()` | Check if fitted |

### Failure Semantics

- Validation first (non-null, non-empty, matching sizes).
- If validation fails → **previous fit unchanged**.
- If training throws → pipeline left **unfitted**, exception propagates.

---

## Word2Vec / Doc2Vec (Simplified Implementations)

⚠️ **Warning**: The `Word2Vec` and `Doc2Vec` classes in this library are **simplified implementations** for educational/demo purposes. They use basic SGD with negative sampling but lack:

- Hierarchical softmax
- Subword information
- Proper learning rate scheduling
- Multi-threading
- Production-grade quality

For production embeddings, use dedicated libraries (e.g., Deeplearning4j, Gensim via Python interop).

### Quick Reference

<!-- no-compile -->
```java
// Word2Vec
Word2Vec w2v = new Word2Vec(vectorSize, windowSize, minCount, epochs, learningRate);
w2v.fit(sentences);  // List<List<String>>
double[] vec = w2v.getVector("king");

// Doc2Vec (PV-DM)
Doc2Vec d2v = new Doc2Vec(vectorSize, windowSize, minCount, epochs, learningRate);
d2v.fit(taggedDocuments);  // List<TaggedDocument>
double[] docVec = d2v.inferVector(tokens);
```

---

## See Also

- [data-loading.md](data-loading.md) — Load text data
- [models.md](models.md) — Naive Bayes, Logistic Regression for text
- [pipelines.md](pipelines.md) — TextPipeline and FeaturePipeline
- [evaluation.md](evaluation.md) — Metrics