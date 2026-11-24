# Evaluation

This page documents every `Metrics` method with formulas, `positiveClass` meaning, edge cases, and the metric names accepted by `Model.evaluate()`.

## Metrics Class

All methods are static. Classification metrics take `int[]` predictions and labels; regression metrics take `double[]`.

### Classification Metrics

#### accuracy(predictions, labels)

```
TP + TN / (TP + TN + FP + FN)
```

<!-- no-compile -->
```java
Metrics.accuracy(new int[]{1,0,1,1}, new int[]{1,0,0,1}); // 0.75
```

**Edge cases:**
- Empty arrays: throws `ArrayIndexOutOfBoundsException` (no explicit check)
- All correct → 1.0, all wrong → 0.0

#### precision(predictions, labels, positiveClass)

```
TP / (TP + FP)  where TP = predicted=positiveClass AND label=positiveClass
```

<!-- no-compile -->
```java
Metrics.precision(new int[]{1,0,1,1}, new int[]{1,0,0,1}, 1); // 2/3 ≈ 0.6667
```

**Edge cases:**
- `TP + FP == 0` (no positive predictions) → **returns 0.0** (not NaN)
- `positiveClass` not present in predictions → 0.0

#### recall(predictions, labels, positiveClass)

```
TP / (TP + FN)  where FN = label=positiveClass AND predicted≠positiveClass
```

<!-- no-compile -->
```java
Metrics.recall(new int[]{1,0,1,1}, new int[]{1,0,0,1}, 1); // 1.0
```

**Edge cases:**
- `TP + FN == 0` (no positive labels) → **returns 0.0**

#### specificity(predictions, labels, positiveClass)

```
TN / (TN + FP)  where negative = not positiveClass
```

<!-- no-compile -->
```java
Metrics.specificity(new int[]{1,0,1,1}, new int[]{1,0,0,1}, 1); // 0.5
```

**Edge cases:**
- `TN + FP == 0` → **returns 0.0**

#### f1Score(predictions, labels, positiveClass)

```
2 * precision * recall / (precision + recall)
```

<!-- no-compile -->
```java
Metrics.f1Score(new int[]{1,0,1,1}, new int[]{1,0,0,1}, 1); // 0.8
```

**Edge cases:**
- `precision + recall == 0` → **returns 0.0**

#### balancedAccuracy(predictions, labels, positiveClass)

```
(recall(positive) + recall(negative)) / 2
= (TP/(TP+FN) + TN/(TN+FP)) / 2
```

<!-- no-compile -->
```java
Metrics.balancedAccuracy(new int[]{1,0,1,1}, new int[]{1,0,0,1}, 1); // 0.75
```

**Edge cases:**
- Both recalls 0 → 0.0

#### mcc(predictions, labels, positiveClass) — Matthews Correlation Coefficient

```
(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

<!-- no-compile -->
```java
Metrics.mcc(new int[]{1,0,1,1}, new int[]{1,0,0,1}, 1); // ~0.577
```

**Edge cases:**
- Denominator 0 → **returns 0.0**

### Regression Metrics

#### mse(predictions, labels)

```
Σ(yᵢ - ŷᵢ)² / n
```

<!-- no-compile -->
```java
Metrics.mse(new double[]{1,2,3}, new double[]{1,2,5}); // 4/3 ≈ 1.333
```

#### rmse(predictions, labels)

```
sqrt(mse)
```

<!-- no-compile -->
```java
Metrics.rmse(new double[]{1,2,3}, new double[]{1,2,5}); // sqrt(4/3) ≈ 1.155
```

#### mae(predictions, labels)

```
Σ|yᵢ - ŷᵢ| / n
```

<!-- no-compile -->
```java
Metrics.mae(new double[]{1,2,3}, new double[]{1,2,5}); // 2/3 ≈ 0.667
```

#### r2(predictions, labels)

```
1 - SSres / SStot
SSres = Σ(yᵢ - ŷᵢ)²
SStot = Σ(yᵢ - ȳ)²
```

<!-- no-compile -->
```java
Metrics.r2(new double[]{1,2,3}, new double[]{1,2,5}); // 7/13 ≈ 0.538
```

**Edge cases:**
- `predictions.length != labels.length` or empty → throws `IllegalArgumentException`
- `SStot == 0` (all labels identical) → **returns 0.0**

---

## Model.evaluate() Accepted Metric Names

Each model's `evaluate(String[] metrics)` accepts a subset. Case-insensitive.

### KNN / Naive Bayes / SVM / LogisticRegression (Classification)

| Metric String | Metrics Method | positiveClass |
|---------------|----------------|---------------|
| `accuracy` | `accuracy()` | — |
| `precision` | `precision(pred, label, 1)` | 1 |
| `recall` | `recall(pred, label, 1)` | 1 |
| `mse` | `mse()` | — |
| `rmse` | `rmse()` | — |
| `mae` | `mae()` | — |

**Note:** `specificity`, `f1score`, `balancedaccuracy`, `mcc` are **not** exposed via `Model.evaluate()` (only available via `Metrics` directly).

### LinearRegression (Regression)

| Metric String | Metrics Method |
|---------------|----------------|
| `mse` | `mse()` |
| `rmse` | `rmse()` |
| `mae` | `mae()` |
| `r2` | `r2()` |

Classification metrics throw: `Classification metric 'accuracy' not supported for regression. Use: mse, rmse, mae, r2`

### LogisticRegression (Classification)

Same as KNN plus regression metrics (since it also computes `mse`/`rmse`/`mae` on encoded labels).

---

## positiveClass Meaning

- **Always index `1`** for binary classification metrics in `Model.evaluate()`.
- `LabelEncoder` sorts classes alphabetically, so `positiveClass=1` corresponds to the **alphabetically later class**.
- Example: classes `["neg", "pos"]` → `neg=0`, `pos=1` → `positiveClass=1` = "pos".
- For multi-class, precision/recall only evaluate class `1` vs rest (one-vs-all).

---

## Edge Case Summary

| Scenario | Classification | Regression |
|----------|----------------|------------|
| Empty arrays | ArrayIndexOutOfBounds | IllegalArgumentException |
| Division by zero (denom=0) | Returns 0.0 | Returns 0.0 (R²) or NaN (not handled) |
| NaN in input | Not checked | Not checked |
| Wrong metric type | IllegalArgumentException | IllegalArgumentException |

---

## Spec Values for Doc Tests

### Classification

| Predictions | Labels | positiveClass | Accuracy | Precision | Recall | F1 |
|-------------|--------|---------------|----------|-----------|--------|-----|
| [1,0,1,1] | [1,0,0,1] | 1 | 0.75 | 2/3 ≈ 0.6667 | 1.0 | 0.8 |

### Regression

| Predictions | Labels | MSE | MAE | RMSE | R² |
|-------------|--------|-----|-----|------|-----|
| [1,2,3] | [1,2,5] | 4/3 ≈ 1.333 | 2/3 ≈ 0.667 | sqrt(4/3) ≈ 1.155 | 7/13 ≈ 0.538 |

---

## See Also

- [models.md](models.md) — Model-specific evaluate() overloads
- [pipelines.md](pipelines.md) — FeaturePipeline.score()
- [persistence.md](persistence.md) — Evaluate after load