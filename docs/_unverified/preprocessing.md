# Preprocessing

This page covers the scaling transformers, imputer, label encoder, and a warning about `StandardScaler`.

## Scaling Transformers

All scalers implement `Transformer` with `fit()`, `transform()`, `fitTransform()`, and `toJson()`/`fromJson()`.

### ZScoreScaler

Standardizes features to zero mean and unit variance (sample standard deviation, `n-1` denominator).

<!-- no-compile -->
```java
ZScoreScaler scaler = new ZScoreScaler();
scaler.fit(X);                    // X: double[][]
double[][] scaled = scaler.transform(X);
double[] rowScaled = scaler.transform(row);
```

**Behavior:**
- NaN values in training are skipped for mean/std computation.
- NaN in transform passes through as NaN.
- Constant features (std = 0) are scaled with std = 1 (no change).
- **Dense only** — no sparse support.

**JSON:**
```json
{"means":[0.0,1.5],"stds":[1.0,0.5],"numFeatures":2}
```

### MinMaxScaler

Scales features to [0, 1] range: `(x - min) / (max - min)`.

<!-- no-compile -->
```java
MinMaxScaler scaler = new MinMaxScaler();
scaler.fit(X);
double[][] scaled = scaler.transform(X);
```

**Behavior:**
- NaN values skipped during fit.
- NaN in transform passes through as NaN.
- Constant features: `max = min + 1` so all map to 0.
- **Dense only** — no sparse support.

**JSON:**
```json
{"mins":[0.0,1.0],"maxs":[1.0,3.0],"numFeatures":2}
```

### MaxAbsScaler

Scales by maximum absolute value: `x / max(|x|)`. Preserves sparsity.

<!-- no-compile -->
```java
MaxAbsScaler scaler = new MaxAbsScaler();
scaler.fit(X);                    // dense
scaler.fitSparse(sparseX);        // sparse
double[][] scaled = scaler.transform(X);
List<Map<Integer, Double>> scaledSparse = scaler.transformSparse(sparseX);
```

**Behavior:**
- Dense: NaN skipped during fit; NaN passes through in transform.
- Sparse: `fitSparse()` computes maxAbs from non-zero entries; unseen features dropped in transform.
- Constant zero features: `maxAbs = 1` (no scaling).
- **Supports both dense and sparse**.

**JSON:**
```json
{"maxAbs":[2.0,5.0],"numFeatures":2}
```

### Comparison Table

| Scaler | Dense | Sparse | NaN Handling | Zero-Var Handling |
|--------|-------|--------|--------------|-------------------|
| `ZScoreScaler` | ✓ | ✗ | Skip fit, pass through | std = 1 |
| `MinMaxScaler` | ✓ | ✗ | Skip fit, pass through | max = min + 1 |
| `MaxAbsScaler` | ✓ | ✓ | Skip fit, pass through | maxAbs = 1 |

## SimpleImputer

Fills missing (NaN) values using a strategy.

<!-- no-compile -->
```java
SimpleImputer imputer = new SimpleImputer(SimpleImputer.Strategy.MEAN);
imputer.fit(X);
double[][] filled = imputer.transform(X);
```

### Strategies

| Strategy | Description | Fill Value |
|----------|-------------|------------|
| `MEAN` | Mean of non-NaN values | Column mean |
| `MEDIAN` | Median of non-NaN values | Column median |
| `MOST_FREQUENT` | Most frequent non-NaN value | Mode (ties: smaller value wins) |
| `CONSTANT` | Fixed value | `constantValue` (default 0.0) |

**Behavior:**
- All-NaN column: falls back to 0 (or `constantValue` for CONSTANT).
- **Dense only** — no sparse support.

**JSON:**
```json
{"strategy":"MEAN","constantValue":0.0,"fillValues":[1.5,2.0],"numFeatures":2}
```

## LabelEncoder

Converts string labels ↔ numeric indices (0, 1, 2...). Classes sorted alphabetically.

<!-- no-compile -->
```java
LabelEncoder encoder = new LabelEncoder();
encoder.fit(List.of("cat", "dog", "cat")); // classes: ["cat", "dog"]
double idx = encoder.transform("dog");     // 1.0
String label = encoder.inverseTransform(1.0); // "dog"
```

**Methods:**
- `fit(List<String>)` — throws if null, empty, or contains null
- `transform(String)` / `transform(List<String>)` — throws if not fitted or unknown label
- `inverseTransform(double)` / `inverseTransform(double[])` — throws if not fitted, NaN, non-integer, or out of range
- `getClasses()` — unmodifiable list of class names
- `size()` — number of classes
- `isFitted()` — boolean
- `toJson()` / `fromJson(String)` — JSON round-trip
- `fromMap(Map<String, Integer>)` — build from explicit mapping (validates 0..n-1 contiguous)

## ⚠️ StandardScaler Warning

The class `models.ml.Preprocessing.Scaling.StandardScaler` **does not perform feature scaling**. It only splits a `double[][]` matrix into features and labels:

<!-- no-compile -->
```java
StandardScaler ss = new StandardScaler();
double[][] features = ss.extractFeatures(dataset);  // all columns except last
double[] labels = ss.extractLabels(dataset);        // last column (must be integers)
double[] regLabels = ss.extractLabelsRegression(dataset); // last column (any double)
```

**Use `ZScoreScaler` for actual z-score standardization.**

## See Also

- [data-loading.md](data-loading.md) — Load data before preprocessing
- [models.md](models.md) — Models that use preprocessed data
- [pipelines.md](pipelines.md) — FeaturePipeline with scalers
- [evaluation.md](evaluation.md) — Metrics on scaled data