# Pipelines

This page covers `FeaturePipeline`, `ScalingFeatureExtractor`, `CompositeFeatureExtractor`, and writing a custom `FeatureExtractor`.

## FeaturePipeline<T>

Generic pipeline: extractor → label encoder → model. Supports both `SPARSE` and `DENSE` modes.

### Constructors

<!-- no-compile -->
```java
// Minimal: new LabelEncoder, SPARSE
new FeaturePipeline<>(extractor, model)

// With encoder
new FeaturePipeline<>(extractor, encoder, model)

// Full control
new FeaturePipeline<>(extractor, encoder, model, DataMode.DENSE)
```

### DataMode

| Mode | Extractor Output | Model Input |
|------|------------------|-------------|
| `SPARSE` | `List<Map<Integer, Double>>` | Sparse `fit(List<Map<Integer, Double>>, double[])` |
| `DENSE` | `double[][]` | Dense `fit(double[][], double[])` |

### Methods

| Method | Description |
|--------|-------------|
| `fit(List<T>, List<String>)` | Train extractor + encoder + model |
| `predict(T)` | Single prediction → label string |
| `predictAll(List<T>)` | Batch predictions |
| `score(List<T>, List<String>)` | Accuracy on test set |
| `isFitted()` | Boolean |
| `getExtractor()` / `getLabelEncoder()` / `getModel()` / `getDataMode()` | Access components |

### Fit Contract

1. **Validate first**: Non-null, non-empty, matching sizes, no null elements.
2. **Mark unfitted**: `fitted = false` before training.
3. **Train**: `extractor.fit(inputs)`, `encoder.fit(labels)`, `model.fit(features, encodedLabels)`.
4. **Mark fitted**: Only if all steps succeed.

**Failure semantics:**
- Validation failure → previous fit **unchanged**.
- Training failure → pipeline left **unfitted**, exception propagates unchanged.

### Example: Image Classification

<!-- no-compile -->
```java
PixelFeatureExtractor extractor = new PixelFeatureExtractor(32, 32, ChannelMode.GRAY);
ScalingFeatureExtractor<ImageData> scaled = new ScalingFeatureExtractor<>(extractor, new ZScoreScaler());

FeaturePipeline<ImageData> pipeline = new FeaturePipeline<>(
    scaled, new LabelEncoder(), new KNN(), DataMode.DENSE
);

pipeline.fit(trainImages, trainLabels);
String pred = pipeline.predict(testImage);
double acc = pipeline.score(testImages, testLabels);
```

### Example: Text Classification

<!-- no-compile -->
```java
TfidfVectorizer vec = new TfidfVectorizer();
FeaturePipeline<String> pipeline = new FeaturePipeline<>(
    vec, new LabelEncoder(), new NaiveBayes(), DataMode.SPARSE
);

pipeline.fit(trainTexts, trainLabels);
```

---

## ScalingFeatureExtractor<T>

Wraps a child extractor + scaler. Fits both on training data.

<!-- no-compile -->
```java
ScalingFeatureExtractor<ImageData> scaled = new ScalingFeatureExtractor<>(
    new PixelFeatureExtractor(32, 32, ChannelMode.GRAY),
    new ZScoreScaler()
);
```

### Fit Behavior

1. Validate inputs (non-null, non-empty, no null elements).
2. Set `fitted = false` **before** fitting child/scaler.
3. `child.fit(inputs)`
4. `scaler.fit(child.transformDense(inputs))`
5. Set `fitted = true` only if both succeed.

**Failure handling:**
- If `child.fit()` throws → `fitted` remains false, exception propagates.
- If `scaler.fit()` throws → `fitted` remains false, exception propagates.
- `fit(null)` or `fit(empty)` **after successful fit** → throws `IllegalArgumentException`, `isFitted()` stays **true** (validation before mutation).

### Methods

| Method | Description |
|--------|-------------|
| `transform(T)` / `transformDense(T)` | Extract → scale |
| `transform(List<T>)` / `transformDense(List<T>)` | Batch |
| `toJson()` | JSON with embedded child + scaler |
| `isFitted()` | `fitted && child.isFitted()` |

### JSON Format

```json
{
  "type": "scaled",
  "extractor": { "type": "pixel", "width": 32, ... },
  "scaler": { "means": [...], "stds": [...], "numFeatures": 1024 }
}
```

**No `fromJson` implemented** — requires extractor/scaler registries.

---

## CompositeFeatureExtractor<T>

Concatenates multiple child extractors.

<!-- no-compile -->
```java
CompositeFeatureExtractor<ImageData> composite = new CompositeFeatureExtractor<>(
    List.of(scaledPixel, scaledHistogram)
);
```

### Constructor Validation

- `children` not null, not empty
- No null elements

### Feature Size

- Computed at call time: `sum(child.getFeatureSize())`
- Supports children whose size changes after fit

### Fit

<!-- no-compile -->
```java
composite.fit(inputs);  // Calls fit() on all children
```

### Transform

<!-- no-compile -->
```java
double[] dense = composite.transformDense(image);  // Concatenates child outputs
double[][] batch = composite.transformDense(images);
```

### Guard Method

Shared private guard used by `transform(List)`, `transformDense(List)`, `transform(List)` (sparse):

<!-- no-compile -->
```java
private void requireFittedAndNonNullList(List<T> inputs) {
    if (!isFitted()) throw new IllegalStateException("Extractor must be fitted before transform");
    if (inputs == null) throw new IllegalArgumentException("inputs list cannot be null");
}
```

### JSON

```json
{
  "type": "composite",
  "children": [
    { "type": "scaled", "extractor": {...}, "scaler": {...} },
    { "type": "colorHistogram", "bins": 16, "mode": "RGB", "fitted": true }
  ]
}
```

**No `fromJson` implemented** — requires extractor registry.

---

## Writing a Custom FeatureExtractor<T>

Extend `AbstractDenseFeatureExtractor<T>` for dense extractors.

### Template

<!-- no-compile -->
```java
public final class MyExtractor extends AbstractDenseFeatureExtractor<MyInputType> {

    private final int param1;
    private final String param2;

    public MyExtractor(int param1, String param2) {
        if (param1 <= 0) throw new IllegalArgumentException("param1 must be > 0");
        if (param2 == null) throw new IllegalArgumentException("param2 cannot be null");
        this.param1 = param1;
        this.param2 = param2;
    }

    @Override
    public int getFeatureSize() {
        return param1;  // Available before and after fit
    }

    @Override
    protected double[] extractDense(MyInputType input) {
        // Convert input to feature vector of length getFeatureSize()
        double[] features = new double[param1];
        // ... compute features ...
        return features;
    }

    @Override
    protected String doToJson() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("type", "myExtractor");
        map.put("param1", param1);
        map.put("param2", param2);
        map.put("fitted", true);
        return Json.write(map);
    }

    public static MyExtractor fromJson(String json) {
        // Parse json, validate, construct, markFitted() if fitted
        MyExtractor ext = new MyExtractor(param1, param2);
        if (fitted) ext.markFitted();
        return ext;
    }
}
```

### Required Methods

| Method | When Called |
|--------|-------------|
| `getFeatureSize()` | Before and after fit |
| `extractDense(T)` | After fit, per input |
| `doToJson()` | After fit, for `toJson()` |

### Inherited from AbstractDenseFeatureExtractor

| Method | Behavior |
|--------|----------|
| `fit(List<T>)` | Validates non-null, non-null elements; marks fitted |
| `markFitted()` | Protected; for `fromJson` |
| `isFitted()` | Returns `fitted` flag |
| `transform(T)` | `FeatureVectors.toSparse(extractDense(input))` |
| `transform(List<T>)` | Validates, loops `transform()` |
| `transformDense(T)` | Calls `extractDense()` |
| `transformDense(List<T>)` | Validates, loops `extractDense()` |
| `toJson()` | Checks fitted, calls `doToJson()` |

### Sparse Extractor

For sparse output, implement `FeatureExtractor<T>` directly (like `BagOfWordsVectorizer` extends `AbstractTextVectorizer`).

---

## See Also

- [images.md](images.md) — PixelFeatureExtractor, ColorHistogramExtractor
- [audio.md](audio.md) — SpectralStatsExtractor, MfccExtractor
- [text.md](text.md) — BagOfWordsVectorizer, TfidfVectorizer
- [preprocessing.md](preprocessing.md) — Scalers for ScalingFeatureExtractor