# Persistence

This page covers model serialization (Java), JSON for vectorizers/scalers/extractors, and what cannot be restored.

## Model.save() / Model.load()

All models implement Java serialization via `ObjectOutputStream` / `ObjectInputStream`.

### Usage

<!-- no-compile -->
```java
// Save
model.save("model.ser");

// Load (into same instance or new)
Model loaded = new KNN();  // or LogisticRegression, etc.
loaded.load("model.ser");
```

### What's Saved

| Component | Saved |
|-----------|-------|
| Hyperparameters (k, method, learningRate, etc.) | ✓ |
| Training data (dense or sparse) | ✓ |
| Test data (if provided during fit) | ✓ |
| Labels (training + test) | ✓ |
| Fitted state | ✓ |

### What's NOT Saved

- `StandardScaler` internal state (models recreate it on `refresh()`)
- Any transient fields marked `transient`

### Version Compatibility

⚠️ **Warning**: Java serialization is fragile across versions.
- Changing class structure (adding/removing fields) breaks deserialization.
- Different JDK versions may have issues.
- **Do not deserialize untrusted .ser files** — arbitrary code execution risk via gadget chains.

### Format

Files are standard Java serialization (binary). No human-readable format.

---

## JSON for Vectorizers, Scalers, Extractors

These implement `toJson()` / `fromJson(String)` for interoperability.

### Vectorizers

| Class | toJson() Fields | fromJson() Validates |
|-------|-----------------|----------------------|
| `BagOfWordsVectorizer` | binary, minDocFreq, maxFeatures, toLowerCase, ngramMin, ngramMax, tokenPattern, stopwords, vocab, fitted | Required fields, whole-number indices 0..n-1, valid regex |
| `TfidfVectorizer` | + sublinearTf, l2Normalize, idf array | + idf length = vocab size, finite values |

### Scalers

| Class | toJson() Fields | fromJson() Validates |
|-------|-----------------|----------------------|
| `ZScoreScaler` | means[], stds[], numFeatures | Lengths match, whole numbers |
| `MinMaxScaler` | mins[], maxs[], numFeatures | Lengths match |
| `MaxAbsScaler` | maxAbs[], numFeatures | Length match |

### Feature Extractors

| Class | toJson() Fields | fromJson() Validates |
|-------|-----------------|----------------------|
| `PixelFeatureExtractor` | type, width, height, mode, fitted | width>0, height>0, mode∈{GRAY,RGB} |
| `ColorHistogramExtractor` | type, bins, mode, fitted | bins≥2, mode∈{GRAY,RGB} |
| `SpectralStatsExtractor` | type, frameSize, hop, fitted | Constructor validates (power of 2, hop>0) |
| `MfccExtractor` | type, targetSampleRate, frameSize, hop, numFilters, numCoeffs, fitted | Constructor validates all |

### LabelEncoder

```json
{"classes":["neg","pos"],"fitted":true}
```

Validates: no duplicates, fitted↔non-empty consistency.

### JSON Round-Trip Example

<!-- no-compile -->
```java
// Vectorizer
TfidfVectorizer vec = new TfidfVectorizer();
vec.fit(docs);
String json = vec.toJson();
TfidfVectorizer loaded = TfidfVectorizer.fromJson(json);
assert loaded.transform(doc).equals(vec.transform(doc));

// Scaler
ZScoreScaler scaler = new ZScoreScaler();
scaler.fit(X);
String json = scaler.toJson();
ZScoreScaler loaded = ZScoreScaler.fromJson(json);
assert Arrays.equals(loaded.transform(row), scaler.transform(row));

// Extractor
PixelFeatureExtractor ext = new PixelFeatureExtractor(28, 28, ChannelMode.GRAY);
ext.fit(images);
String json = ext.toJson();
PixelFeatureExtractor loaded = PixelFeatureExtractor.fromJson(json);
assert Arrays.equals(loaded.transformDense(img), ext.transformDense(img));
```

---

## What Cannot Be Restored from JSON

### CompositeFeatureExtractor

- `toJson()` implemented: embeds full child JSON configs
- `fromJson()` **NOT implemented**
- **Reason**: Requires a registry of extractor types to deserialize children

### ScalingFeatureExtractor

- `toJson()` implemented: embeds child extractor JSON + scaler JSON
- `fromJson()` **NOT implemented**
- **Reason**: Requires extractor registry + scaler registry

### Workaround

Reconstruct programmatically:

<!-- no-compile -->
```java
// For ScalingFeatureExtractor
String childJson = extractChildJson(compositeJson);  // parse manually
PixelFeatureExtractor child = PixelFeatureExtractor.fromJson(childJson);
ZScoreScaler scaler = ZScoreScaler.fromJson(extractScalerJson(compositeJson));
ScalingFeatureExtractor<ImageData> restored = new ScalingFeatureExtractor<>(child, scaler);
restored.fit(trainingData);  // Must refit

// For CompositeFeatureExtractor
List<FeatureExtractor<ImageData>> children = parseChildren(compositeJson);
CompositeFeatureExtractor<ImageData> restored = new CompositeFeatureExtractor<>(children);
restored.fit(trainingData);
```

---

## TextPipeline / FeaturePipeline

- **No built-in save/load**
- **Workaround**: Save components separately:
<!-- no-compile -->
  ```java
  // Save
  vectorizer.toJson() → file
  encoder.toJson() → file
  model.save("model.ser")

  // Load
  Vectorizer vec = Vectorizer.fromJson(readFile(...));
  LabelEncoder enc = LabelEncoder.fromJson(readFile(...));
  Model model = new NaiveBayes();
  model.load("model.ser");
  FeaturePipeline<String> pipeline = new FeaturePipeline<>(vec, enc, model, DataMode.SPARSE);
  // Pipeline is NOT fitted — call pipeline.fit() if needed, or use components directly
  ```

---

## Best Practices

1. **Prefer JSON** for vectorizers/scalers/extractors — human-readable, version-tolerant.
2. **Use Java serialization only for models** — and only within same codebase version.
3. **Never deserialize untrusted .ser files**.
4. **Refit after JSON restore** for Composite/Scaling extractors.
5. **Test round-trips** in CI: `assert loaded.transform(x).equals(original.transform(x))`.

---

## See Also

- [models.md](models.md) — Model.save/load details
- [text.md](text.md) — Vectorizer JSON
- [preprocessing.md](preprocessing.md) — Scaler JSON
- [images.md](images.md) / [audio.md](audio.md) — Extractor JSON
- [pipelines.md](pipelines.md) — Pipeline components