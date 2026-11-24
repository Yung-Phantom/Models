# Getting Started

This guide introduces the core concepts of the Models library and helps you choose the next page to read.

## Core Concepts

### Dense vs Sparse Data

The library represents data in two modes:

- **Dense** (`double[][]`): A rectangular matrix where every feature has a value (even if zero). Used for numeric tabular data, images (pixels), and audio (MFCC coefficients).
- **Sparse** (`List<Map<Integer, Double>>`): A list of feature-index-to-value maps containing only non-zero entries. Used for text vectorizer output (Bag-of-Words, TF-IDF).

Most models accept both. The `Model` base class constructors detect the mode automatically, or you can specify it explicitly via `DataMode.SPARSE` / `DataMode.DENSE` in `FeaturePipeline`.

### Labels

- Classification labels are strings (e.g., "pos", "neg", "cat", "dog").
- Regression targets are numeric `double[]`.
- `LabelEncoder` converts string labels to numeric indices (`0, 1, 2...`) for model training and back to strings for predictions.
- The positive class for binary metrics (precision, recall, F1) is always index `1` by default.

### Fit / Predict / Evaluate Lifecycle

<!-- no-compile -->
```java
// 1. Load and split data
DatasetLoader loader = new DatasetLoader("data.csv", DatasetConfig.HAS_ID_WITH_HEADER, "target");
DatasetSplit split = loader.split(0.8, "stratified");

// 2. Train (fit)
KNN model = new KNN();
model.fit(split.train, split.test);  // or model.fit(loader, 80, "stratified")

// 3. Predict on new data
double prediction = model.predict(new double[]{1.0, 2.0, 3.0});

// 4. Evaluate on test set
Map<String, Double> metrics = model.evaluate(new String[]{"accuracy", "f1score"});
```

Key rules:
- Call `fit()` before `predict()` or `evaluate()`.
- `fit()` validates inputs first; if validation fails, the previous fitted state is unchanged.
- If training throws after validation, the model is left **unfitted** and the exception propagates unchanged.
- Use `refresh()` after modifying hyperparameters to rebuild internal state.

## Glossary

| Term | Meaning |
|------|---------|
| `Dataset` | In-memory table (`double[][]`) with header and label-to-index map |
| `DatasetSplit` | Pair of `train` and `test` matrices from `DatasetLoader.split()` |
| `DataMode` | Enum: `SPARSE` or `DENSE` for `FeaturePipeline` |
| `Transformer` | Interface for scalers/imputers with `fit()`, `transform()`, `fitTransform()`, `toJson()` |
| `FeatureExtractor<T>` | Interface for converting `T` (text, image, audio) to sparse/dense features |

## Next Steps

| Goal | Read |
|------|------|
| Load CSV/JSON/SQLite, split data | [data-loading.md](data-loading.md) |
| Scale, impute, encode labels | [preprocessing.md](preprocessing.md) |
| Train KNN, LR, NB, SVM | [models.md](models.md) |
| Text classification pipelines | [text.md](text.md) |
| Image feature extraction | [images.md](images.md) |
| Audio feature extraction | [audio.md](audio.md) |
| Build custom pipelines | [pipelines.md](pipelines.md) |
| Metrics formulas and edge cases | [evaluation.md](evaluation.md) |
| Save/load models and JSON | [persistence.md](persistence.md) |
| Common errors and fixes | [faq-troubleshooting.md](faq-troubleshooting.md) |

## See Also

- [data-loading.md](data-loading.md) — Data loading and splitting
- [preprocessing.md](preprocessing.md) — Preprocessing transformers
- [models.md](models.md) — Model reference