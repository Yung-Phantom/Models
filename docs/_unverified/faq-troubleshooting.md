# FAQ & Troubleshooting

The 15 most likely errors with exact messages from source, cause, and fix.

---

## 1. `Unsupported format: xyz`

**Source:** `DatasetLoader` constructor  
**Cause:** File extension not in {csv, tsv, txt, json, sqlite}  
**Fix:** Use supported extension or rename file. Parquet not supported.

---

## 2. `No label column specified. Supervised models require labels.`

**Source:** `DatasetLoader.loadSqlite()`  
**Cause:** SQLite load without explicit `labelColumn` parameter  
**Fix:** Pass label column name: `new DatasetLoader("db.sqlite", "target_column")`

---

## 3. `testFraction must be in (0, 1)`

**Source:** `LabeledDataset.split()`, `DatasetLoader.split()`  
**Cause:** Split fraction ≤ 0 or ≥ 1 (e.g., 0, 1.0, 100)  
**Fix:** Use decimal (0.8) or percent (80) strictly between 0 and 100.

---

## 4. `inputs and labels must have the same size`

**Source:** `FeaturePipeline.requireValidPairs()`, `Model.fit()` overloads  
**Cause:** Feature/label array length mismatch  
**Fix:** Ensure `features.size() == labels.size()` before `fit()`.

---

## 5. `input at index N cannot be null` / `label at index N cannot be null`

**Source:** `FeaturePipeline.requireValidPairs()`, `LabeledDataset` constructor  
**Cause:** Null element in inputs or labels list  
**Fix:** Filter nulls: `list.stream().filter(Objects::nonNull).collect(toList())`

---

## 6. `images, labels, and sources must have the same size`

**Source:** `LabeledDataset` constructor (via `ImageDataset`/`AudioDataset`)  
**Cause:** Three lists passed to constructor have different lengths  
**Fix:** Verify all three lists same length before constructing dataset.

---

## 7. `Model must be fitted before prediction.` / `...before transform`

**Source:** `Model.predict()`, `FeatureExtractor.transform()`, `Scaler.transform()`  
**Cause:** Called prediction/transform before `fit()`  
**Fix:** Call `fit()` first. Check `isFitted()` if uncertain.

---

## 8. `This operation requires a dense model, but the model was initialized with sparse data.`

**Source:** `Model.requireDense()`  
**Cause:** Called dense method (e.g., `predict(double[])`) on sparse model  
**Fix:** Use matching mode: `predict(Map<Integer, Double>)` for sparse, or re-fit with dense data.

---

## 9. `This operation requires a sparse model, but the model was initialized with dense data.`

**Source:** `Model.requireSparse()`  
**Cause:** Called sparse method on dense model  
**Fix:** Use `predict(double[])` for dense, or re-fit with sparse data.

---

## 10. `Row must have exactly N features`

**Source:** `KNN.predict()`, `SVM.predict()`, `LogisticRegression.predict()`, `LinearRegression.predict()`  
**Cause:** Input array length ≠ model's feature count  
**Fix:** Ensure input matches training feature dimension. Check `getFeatureSize()` on extractor.

---

## 11. `trainingDataset cannot be empty`

**Source:** All `Model.fit()` overloads, `FeatureExtractor.fit()`, `Scaler.fit()`  
**Cause:** Empty array/list passed to `fit()`  
**Fix:** Check `data.length > 0` / `list.size() > 0` before calling.

---

## 12. `Gamma must be positive` / `Degree must be positive` / `Coef0 must be non-negative`

**Source:** `SVM.initDenseParams()`  
**Cause:** Invalid kernel hyperparameters for RBF/polynomial/sigmoid  
**Fix:** 
- `gamma > 0` for RBF, poly, sigmoid
- `degree > 0` for polynomial
- `coef0 >= 0` for poly, sigmoid

---

## 13. `Sparse data not allowed in dense mode` / `Dense data not allowed in sparse mode`

**Source:** `NaiveBayes`, `SVM`, `LogisticRegression`, `LinearRegression`  
**Cause:** Method/mode mismatch:
- GaussianNB + sparse input
- MultinomialNB + dense input
- SVC (non-linear kernel) + sparse input
- Ordinal LR + sparse input
- Normal equation LR + sparse input

**Fix:** Match data mode to method (see [models.md](models.md) table).

---

## 14. `Classification metric 'accuracy' not supported for regression. Use: mse, rmse, mae, r2`

**Source:** `LinearRegression.calculateMetrics()`  
**Cause:** Passing classification metric to regression model's `evaluate()`  
**Fix:** Use `mse`, `rmse`, `mae`, `r2` for `LinearRegression`.

---

## 15. `file is not a valid WAV file` / `unsupported encoding` / `unsupported channel count`

**Source:** `AudioDecoder.decode()`  
**Cause:** 
- Not a WAV file (wrong magic bytes)
- Encoding not PCM_SIGNED 16-bit LE or PCM_UNSIGNED 8-bit
- > 2 channels

**Fix:** Convert audio to supported format (e.g., `ffmpeg -i in.mp3 -ac 1 -ar 16000 -sample_fmt s16 out.wav`).

---

## Additional Common Issues

### `IllegalStateException: Scaler must be fitted before transform`

**Cause:** Called `scaler.transform()` before `scaler.fit()`  
**Fix:** Call `fit()` or `fitTransform()` first.

### `IllegalArgumentException: Vocabulary index out of range`

**Source:** `BagOfWordsVectorizer.fromJson()`, `TfidfVectorizer.fromJson()`  
**Cause:** Corrupted JSON with index ≥ vocab size or duplicates  
**Fix:** Re-generate JSON from fitted vectorizer; don't hand-edit.

### `IllegalArgumentException: frameSize must be a power of two`

**Source:** `SpectralStatsExtractor`, `MfccExtractor`, `Fft.magnitudeSpectrum()`  
**Cause:** Frame size not power of 2 (e.g., 200, 300)  
**Fix:** Use 256, 512, 1024, 2048, etc.

### `IllegalArgumentException: fftSize must be >= frame.length`

**Source:** `Fft.magnitudeSpectrum()`  
**Cause:** FFT size smaller than input frame  
**Fix:** Use `fftSize >= frame.length` (power of 2).

### `ArrayIndexOutOfBoundsException` in metrics

**Cause:** Empty prediction/label arrays passed to `Metrics.accuracy()` etc.  
**Fix:** Check `predictions.length > 0` before calling metrics.

### `NullPointerException` in `StandardScaler.extractFeatures()`

**Cause:** Dataset has < 2 columns (need at least 1 feature + 1 label)  
**Fix:** Ensure data has ≥ 2 columns.

---

## Debugging Tips

1. **Enable stack traces**: All exceptions include descriptive messages.
2. **Check `isFitted()`** before predict/transform.
3. **Verify data modes**: Dense vs sparse must match model+extractor.
4. **Test JSON round-trips**: `fromJson(toJson())` should produce identical output.
5. **Use seeded splits**: `split(0.8, "stratified", 42L)` for reproducibility.

---

## See Also

- [models.md](models.md) — Method/mode compatibility tables
- [preprocessing.md](preprocessing.md) — Scaler validation
- [data-loading.md](data-loading.md) — DatasetLoader exceptions
- [audio.md](audio.md) — AudioDecoder constraints