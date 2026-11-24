# Models

This page documents all five models: KNN, Naive Bayes, SVM, Linear Regression, Logistic Regression.

## Common Model API

All models extend `models.ml.Model.Model` and share these constructors and methods.

### Constructors (Dense)

<!-- no-compile -->
```java
// Training data only (last column = label)
new KNN(double[][] trainingDataset)

// Explicit features + labels
new KNN(double[][] trainingDataset, double[] trainingLabels)

// With test data (last column = label)
new KNN(double[][] trainingDataset, double[][] testDataset)

// Explicit features + labels for both
new KNN(double[][] trainingDataset, double[] trainingLabels,
        double[][] testDataset, double[] testLabels)

// With hyperparameters
new KNN(double[][] trainingDataset, int k, int p, String method)
```

### Constructors (Sparse)

<!-- no-compile -->
```java
// Sparse training data + explicit labels
new KNN(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels)

// With test data
new KNN(List<Map<Integer, Double>> trainingDataset,
        List<Map<Integer, Double>> testDataset,
        double[] trainingLabels, double[] testLabels)

// With hyperparameters
new KNN(List<Map<Integer, Double>> trainingDataset, int k, int p, String method)
```

### Constructors (DatasetLoader)

<!-- no-compile -->
```java
new KNN(DatasetLoader loader, int splitPercent, String splitStrategy)
new KNN(DatasetLoader loader, double splitDecimal, String splitStrategy)
new KNN(DatasetLoader loader, int splitPercent, String splitStrategy, int k, int p, String method)
```

### Core Methods

| Method | Description |
|--------|-------------|
| `fit(double[][])` / `fit(List<Map<Integer, Double>>)` | Train on data (last col = label) |
| `fit(double[][], double[])` | Train with explicit labels |
| `fit(DatasetLoader, int, String)` | Train with loader + split |
| `predict(double[])` / `predict(Map<Integer, Double>)` | Single prediction |
| `predictLabels()` | Predict on internal test set |
| `predictAll(List<T>)` | Batch predict (FeaturePipeline) |
| `predictProbabilities()` | Class probabilities (where supported) |
| `predictMultiLabel(double threshold)` | Multi-label from probabilities |
| `evaluate(String[])` | Evaluate on internal test set |
| `evaluate(double[][], double[], String[])` | Evaluate on external data |
| `save(String)` / `load(String)` | Java serialization |
| `refresh()` | Rebuild internal state after hyperparameter change |
| `partialFit(batch, labels)` | Incremental update |
| `getModelType()` / `getVersion()` | Metadata |

### Dense/Sparse Support

| Model | Dense | Sparse |
|-------|-------|--------|
| KNN | ✓ | ✓ |
| Naive Bayes (Gaussian) | ✓ | ✗ |
| Naive Bayes (Multinomial/Bernoulli) | ✗ | ✓ |
| SVM (LinearSVC) | ✓ | ✓ |
| SVM (SVC with kernels) | ✓ | ✗ |
| Linear Regression | ✓ | ✓ |
| Logistic Regression | ✓ | ✓ |

---

## KNN (K-Nearest Neighbors)

### Purpose
Instance-based classifier. Stores all training data; predicts by majority vote of k nearest neighbors.

### Hyperparameters

| Parameter | Type | Default (Dense) | Default (Sparse) | Validation |
|-----------|------|-----------------|------------------|------------|
| `k` | int | 3 | 3 | Must be > 0 |
| `p` | int | 2 | 2 | Minkowski power; must be > 0 |
| `method` | String | "euclidean" | "cosine" | See below |

### Distance Methods

| Method String | Dense | Sparse | Formula |
|---------------|-------|--------|---------|
| `euclidean` | ✓ | ✗ | √Σ(xᵢ - yᵢ)² |
| `manhattan` | ✓ | ✗ | Σ\|xᵢ - yᵢ\| |
| `minkowski` | ✓ | ✗ | (Σ\|xᵢ - yᵢ\|ᵖ)^(1/p) |
| `cosine` | ✓ | ✓ | 1 - (x·y)/(\|x\|\|y\|) |

### Common Exceptions

| Message | Cause |
|---------|-------|
| `Model must be fitted before prediction.` | Called `predict()` before `fit()` |
| `Row must have exactly N features` | Input dimension mismatch |
| `trainingDataset cannot be empty` | Empty array passed to `fit()` |
| `This operation requires a sparse model...` | Calling sparse method on dense model or vice versa |

---

## Naive Bayes

### Purpose
Probabilistic classifier with three variants.

### Hyperparameters

| Parameter | Type | Default (Dense) | Default (Sparse) | Validation |
|-----------|------|-----------------|------------------|------------|
| `method` | String | "gaussian" | "multinomial" | "gaussian", "multinomial", "bernoulli" |
| `alpha` | double | 1.0 | 1.0 | Must be > 0 (Laplace smoothing) |

### Method Details

| Method | Data Mode | Assumptions |
|--------|-----------|-------------|
| `gaussian` | Dense only | Features ~ Normal(μ, σ²) per class |
| `multinomial` | Sparse only | Feature counts ~ Multinomial; for TF-IDF/BOW |
| `bernoulli` | Sparse only | Binary features (presence/absence) |

**Dense + multinomial/bernoulli → throws**  
**Sparse + gaussian → throws**

### Common Exceptions

| Message | Cause |
|---------|-------|
| `Sparse data not allowed in dense mode` | Gaussian with sparse input |
| `Dense data not allowed in sparse mode` | Multinomial/Bernoulli with dense input |
| `Model must be fitted before prediction.` | Called `predict()` before `fit()` |

---

## SVM (Support Vector Machine)

### Purpose
Binary classifier with linear and kernel methods.

### Hyperparameters

| Parameter | Type | Default | Validation |
|-----------|------|---------|------------|
| `method` | String | "linearsvc" | "linearsvc", "svc" |
| `kernel` | String | "linearkernel" | "linearkernel", "rbfkernel", "polynomialkernel", "sigmoidkernel" |
| `C` | double | 1.0 | Must be > 0 |
| `learningRate` | double | 0.01 | Must be > 0 |
| `epochs` | int | 1000 | Must be > 0 |
| `gamma` | double | 1.0 | Must be > 0 (RBF, poly, sigmoid) |
| `degree` | int | 3 | Must be > 0 (poly) |
| `coef0` | double | 0.0 | Must be ≥ 0 (poly, sigmoid) |

### Method + Kernel Combinations

| Method | Kernel | Dense | Sparse | Notes |
|--------|--------|-------|--------|-------|
| `linearsvc` | `linearkernel` | ✓ | ✓ | Primal SGD, fast |
| `svc` | `linearkernel` | ✓ | ✗ | Dual form |
| `svc` | `rbfkernel` | ✓ | ✗ | gamma required |
| `svc` | `polynomialkernel` | ✓ | ✗ | degree, gamma, coef0 |
| `svc` | `sigmoidkernel` | ✓ | ✗ | gamma, coef0 |

**SVC with non-linear kernels is dense-only.**

### Common Exceptions

| Message | Cause |
|---------|-------|
| `Gamma must be positive` | gamma ≤ 0 for RBF/poly/sigmoid |
| `Degree must be positive` | degree ≤ 0 for polynomial |
| `Coef0 must be non-negative` | coef0 < 0 for poly/sigmoid |
| `Sparse data not allowed in dense mode` | Non-linear kernel with sparse input |
| `Model must be fitted before prediction.` | Called `predict()` before `fit()` |

---

## Linear Regression

### Purpose
Regression via normal equation or gradient descent.

### Hyperparameters

| Parameter | Type | Default (Dense) | Default (Sparse) | Validation |
|-----------|------|-----------------|------------------|------------|
| `method` | String | "normal" | "gd" | "normal", "gd", "gradientdescent" |
| `learningRate` | double | 0.01 | 0.01 | Must be in (0, 1] |
| `epochs` | int | 1000 | 1000 | Must be > 0 |

### Methods

| Method | Dense | Sparse | Notes |
|--------|-------|--------|-------|
| `normal` | ✓ | ✗ | Closed-form (XᵀX)⁻¹Xᵀy; requires invertible XᵀX |
| `gd` / `gradientdescent` | ✓ | ✓ | Iterative; works best on scaled features |

**Normal equation is dense-only.**

### Supported Metrics

`mse`, `rmse`, `mae`, `r2` — classification metrics throw `IllegalArgumentException`.

### Common Exceptions

| Message | Cause |
|---------|-------|
| `Classification metric 'accuracy' not supported for regression` | Wrong metric type |
| `trainingLabels must be set before calling fit() without labels for sparse data` | Sparse fit without labels |
| `Model must be fitted before prediction.` | Called `predict()` before `fit()` |

---

## Logistic Regression

### Purpose
Binary, multinomial, or ordinal classification via gradient descent.

### Hyperparameters

| Parameter | Type | Default | Validation |
|-----------|------|---------|------------|
| `method` | String | "binary" | "binary", "multinomial", "ordinal" |
| `learningRate` | double | 0.01 | Must be in (0, 1] |
| `epochs` | int | 1000 | Must be > 0 |

### Methods

| Method | Classes | Dense | Sparse | Notes |
|--------|---------|-------|--------|-------|
| `binary` | 2 | ✓ | ✓ | Sigmoid |
| `multinomial` | >2 | ✓ | ✓ | Softmax |
| `ordinal` | >2 | ✓ | ✗ | Ordered categories |

**Ordinal is dense-only.**

### Supported Metrics

`accuracy`, `precision`, `recall`, `mse`, `rmse`, `mae` — others throw.

### Common Exceptions

| Message | Cause |
|---------|-------|
| `Ordinal classification only supports dense data` | Ordinal with sparse |
| `Model must be fitted before prediction.` | Called `predict()` before `fit()` |

---

## Model Base Class Overload Table

| Overload | Dense | Sparse | Loader |
|----------|-------|--------|--------|
| `fit(double[][])` | ✓ | | |
| `fit(double[][], double[])` | ✓ | | |
| `fit(double[][], double[][])` | ✓ | | |
| `fit(double[][], double[], double[][], double[])` | ✓ | | |
| `fit(List<Map<Integer, Double>>)` | | ✓ | |
| `fit(List<Map<Integer, Double>>, double[])` | | ✓ | |
| `fit(List<Map<Integer, Double>>, List<Map<Integer, Double>>)` | | ✓ | |
| `fit(List<Map<Integer, Double>>, double[], List<Map<Integer, Double>>, double[])` | | ✓ | |
| `fit(DatasetLoader, int, String)` | ✓ | | ✓ |
| `fit(DatasetLoader, double, String)` | ✓ | | ✓ |

Plus hyperparameter variants for each (e.g., `fit(double[][], int k, int p)` for KNN).

## See Also

- [preprocessing.md](preprocessing.md) — Scale data before training
- [pipelines.md](pipelines.md) — FeaturePipeline for end-to-end
- [evaluation.md](evaluation.md) — Metrics formulas
- [persistence.md](persistence.md) — Save/load models