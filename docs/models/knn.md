# KNN (K-Nearest Neighbors)

Instance-based classifier. Stores all training data; predicts by majority vote of k nearest neighbors. Supports both dense (`double[][]`) and sparse (`List<Map<Integer, Double>>`) data with multiple distance metrics.

## Constructors

| Constructor | Description | <!-- src: KNN.java "DEFAULT_K" --> |
|-------------|-------------|---------------------------------------------------|
| `new KNN()` | No-arg constructor; call `fit()` to train | |
| `new KNN(double[][] trainingDataset, int k, int p, String method)` | Dense: training data (last col = label) + hyperparams | <!-- src: KNN.java "DEFAULT_P" --> |
| `new KNN(double[][] trainingDataset, double[] trainingLabels, int k, int p, String method)` | Dense: explicit features + labels + hyperparams | <!-- src: KNN.java "DEFAULT_DENSE_METHOD" --> |
| `new KNN(double[][] trainingDataset, double[][] testDataset, int k, int p, String method)` | Dense: train + test (last col = label) + hyperparams | <!-- src: KNN.java "DEFAULT_SPARSE_METHOD" --> |
| `new KNN(double[][] trainingDataset, double[] trainingLabels, double[][] testDataset, double[] testLabels, int k, int p, String method)` | Dense: explicit train/test + hyperparams | <!-- src: KNN.java "initDenseParams" --> |
| `new KNN(List<Map<Integer, Double>> trainingDataset, int k, int p, String method)` | Sparse: training data + hyperparams | <!-- src: KNN.java "initSparseParams" --> |
| `new KNN(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, int k, int p, String method)` | Sparse: explicit labels + hyperparams | <!-- src: KNN.java "initSparseKNN" --> |
| `new KNN(List<Map<Integer, Double>> trainingDataset, List<Map<Integer, Double>> testDataset, int k, int p, String method)` | Sparse: train + test + hyperparams | <!-- src: AbstractKNN.java "euclidean" --> |
| `new KNN(List<Map<Integer, Double>> trainingDataset, double[] trainingLabels, List<Map<Integer, Double>> testDataset, double[] testLabels, int k, int p, String method)` | Sparse: explicit train/test + hyperparams | <!-- src: AbstractKNN.java "cosine" --> |
| `new KNN(DatasetLoader loader, int splitPercent, String splitStrategy)` | From loader with split % (0-100) | |
| `new KNN(DatasetLoader loader, double splitDecimal, String splitStrategy)` | From loader with split fraction (0.0-1.0) | |

**Defaults:** `k=3`, `p=2`, dense method=`euclidean`, sparse method=`cosine`.  
Hyperparameters are clamped: `k > 0` (else 3), `p > 0` (else 2), method trimmed/lowercased (else default).

## Hyperparameters

| Parameter | Default | Constraint | <!-- src: KNN.java "DEFAULT_K" --> |
|-----------|---------|------------|---------------------------------------------------|
| `k` | 3 | Must be > 0 (clamped to 3) | <!-- src: KNN.java "DEFAULT_P" --> |
| `p` | 2 | Must be > 0 (clamped to 2); only used for `minkowski` | <!-- src: KNN.java "DEFAULT_DENSE_METHOD" --> |
| `method` | dense: `euclidean`, sparse: `cosine` | See Distance Methods table | <!-- src: KNN.java "DEFAULT_SPARSE_METHOD" --> |

## Distance Methods

| Method String | Dense | Sparse | Formula | <!-- src: AbstractKNN.java "euclidean" --> |
|---------------|-------|--------|---------|---------------------------------------------------|
| `euclidean` | ✓ | ✓ | √Σ(xᵢ - yᵢ)² | <!-- src: AbstractKNN.java "manhattan" --> |
| `manhattan` | ✓ | ✓ | Σ\|xᵢ - yᵢ\| | <!-- src: AbstractKNN.java "minkowski" --> |
| `minkowski` | ✓ | ✓ | (Σ\|xᵢ - yᵢ\|ᵖ)^(1/p), p > 0 | <!-- src: AbstractKNN.java "cosine" --> |
| `cosine` | ✓ | ✓ | 1 - (x·y)/(\|x\|\|y\|) | |

**Note:** All four methods are implemented for both dense and sparse modes in `AbstractKNN`. The Javadoc "Sparse: cosine (only)" refers to the default method, not a capability restriction.

**Minkowski p parameter:** Must be > 0. Default p=2 (Euclidean). The public API clamps p to ≥ 1; the safety check in `AbstractKNN.minkowski()` validates p > 0 but is not reachable via normal constructors.

## Fit / Predict / Evaluate / Save / Load

<!-- snippet: KnnExamples#fit-methods -->
```java
public static void fitMethods() {
    KNN knn = new KNN();
    double[][] train = {{1,2,0},{3,4,1}};
    double[][] test = {{5,6,0}};
    double[] labels = {0,1};

    knn.fit(train);                              // last col = label
    knn.fit(train, labels);                      // explicit labels
    knn.fit(train, 5, 2);                        // with k, p
    knn.fit(train, "manhattan");                 // with method
    knn.fit(train, 5, 2, "manhattan");           // all hyperparams
    knn.fit(train, test);                        // with test set
    knn.fit(train, test, 5, 2);                  // test + k,p
    knn.fit(train, test, "manhattan");           // test + method
    knn.fit(train, test, 5, 2, "manhattan");     // test + all
    knn.fit(train, labels, test, labels);        // explicit train+test
    // Loader variants:
    // knn.fit(loader, 80, "stratified");
    // knn.fit(loader, 0.8, "stratified");
}
```

<!-- snippet: KnnExamples#predict-evaluate -->
```java
public static void predictEvaluate() {
    KNN knn = new KNN();
    double[][] train = {{0,0,0},{0,1,0},{1,0,0},{10,10,1},{10,11,1},{11,10,1}};
    knn.fit(train);

    // Single predictions
    double pred1 = knn.predict(new double[]{0.5, 0.5});           // dense
    double pred2 = knn.predict(Map.of(0, 0.5, 1, 0.5));           // sparse

    // Batch predictions
    double[] preds = knn.predictLabels();
    List<Map<Integer, Double>> probs = knn.predictProbabilities();
    double[][] multi = knn.predictMultiLabel(0.5);

    // Evaluate on internal test set
    Map<String, Double> results = knn.evaluate(new String[]{"accuracy", "precision", "recall", "mse", "rmse", "mae"});

    // Evaluate on external data
    Map<String, Double> ext = knn.evaluate(
        new double[][]{{0.1,0.1},{10.1,10.1}}, new double[]{0,1}, new String[]{"accuracy"}
    );
}
```

<!-- snippet: KnnExamples#save-load -->
```java
public static void saveLoad() throws Exception {
    KNN knn = new KNN();
    double[][] train = {{0,0,0},{10,10,1}};
    knn.fit(train);

    String tempFile = System.getProperty("java.io.tmpdir") + "/knn_save_test.ser";
    knn.save(tempFile);

    KNN loaded = new KNN();
    loaded.load(tempFile);

    // Predictions should be identical
    assert knn.predict(new double[]{0.1, 0.1}) == loaded.predict(new double[]{0.1, 0.1});
    assert knn.predict(new double[]{9.9, 9.9}) == loaded.predict(new double[]{9.9, 9.9});
}
```

## Dense vs Sparse

| Model | Dense Methods | Sparse Methods |
|-------|---------------|----------------|
| KNN | euclidean, manhattan, minkowski, cosine | euclidean, manhattan, minkowski, cosine |

Both modes support all four distance methods. Use `cosine` as default for sparse data. Dense predict throws `IllegalStateException` on sparse model and vice versa.

## Common Errors

| Message Fragment | Cause | Fix | <!-- src: KNN.java "Model must be fitted before prediction" --> |
|------------------|-------|-----|---------------------------------------------------|
| `Model must be fitted before prediction.` | Called `predict()` before `fit()` | Call `fit()` first | <!-- src: KNN.java "Row must have exactly" --> |
| `Row must have exactly N features` | Input dimension mismatch | Match training feature count | <!-- src: KNN.java "trainingDataset cannot be empty" --> |
| `trainingDataset cannot be empty` | Empty array passed to `fit()` | Provide non-empty data | <!-- src: KNN.java "trainingLabels cannot be empty" --> |
| `trainingLabels cannot be empty` | Labels array empty | Provide non-empty labels | <!-- src: KNN.java "testDataset cannot be empty" --> |
| `testDataset cannot be empty` | Test array empty | Provide non-empty test data | <!-- src: KNN.java "testLabels cannot be empty" --> |
| `testLabels cannot be empty` | Test labels empty | Provide non-empty test labels | <!-- src: KNN.java "DatasetLoader or underlying dataset cannot be null" --> |
| `DatasetLoader or underlying dataset cannot be null` | Null loader or dataset | Provide valid loader | <!-- src: Model.java "This operation requires a dense model" --> |
| `This operation requires a dense model` | Calling dense method on sparse model | Use correct data mode | <!-- src: Model.java "This operation requires a sparse model" --> |
| `This operation requires a sparse model` | Calling sparse method on dense model | Use correct data mode | <!-- src: AbstractKNN.java "Unsupported method" --> |
| `Unsupported method: <name>` | Invalid distance method string | Use: euclidean, manhattan, minkowski, cosine | <!-- src: AbstractKNN.java "Dataset contains NaN or Infinity" --> |
| `Dataset contains NaN or Infinity` | NaN/Inf in training data | Clean data before fitting | <!-- src: AbstractKNN.java "Sparse dataset contains negative feature index" --> |
| `Sparse dataset contains negative feature index` | Negative key in sparse map | Use non-negative indices | <!-- src: AbstractKNN.java "Dense dataset contains null row" --> |
| `Dense dataset contains null row` | Null row in dense array | Remove null rows | <!-- src: AbstractKNN.java "Sparse dataset contains null row" --> |
| `Sparse dataset contains null row` | Null row in sparse list | Remove null rows | <!-- src: AbstractKNN.java "Inconsistent row length" --> |
| `Inconsistent row length` | Dense rows have different lengths | Ensure consistent dimensions | <!-- src: AbstractKNN.java "Row cannot be null" --> |
| `Row cannot be null` | Null input to predict | Pass valid row | <!-- src: AbstractKNN.java "Vectors cannot be null" --> |
| `Vectors cannot be null` | Null vector in distance calc | Pass valid vectors | <!-- src: AbstractKNN.java "Vector dimension mismatch" --> |
| `Vector dimension mismatch` | Query dimension ≠ training | Match feature count | |

## Example: Separable Classification

<!-- snippet: KnnExamples#spec-knn-classification -->
```java
public static void specKnnClassification() {
    // Points (0,0),(0,1),(1,0) label 0 and (10,10),(10,11),(11,10) label 1
    double[][] trainData = {
        {0, 0, 0},   // (0,0) -> label 0
        {0, 1, 0},   // (0,1) -> label 0
        {1, 0, 0},   // (1,0) -> label 0
        {10, 10, 1}, // (10,10) -> label 1
        {10, 11, 1}, // (10,11) -> label 1
        {11, 10, 1}  // (11,10) -> label 1
    };

    KNN knn = new KNN(trainData, 3, 2, "euclidean");
    knn.fit(trainData);

    double[] query1 = {0.5, 0.5};   // Should predict 0
    double[] query2 = {10.5, 10.5}; // Should predict 1

    double pred1 = knn.predict(query1);
    double pred2 = knn.predict(query2);

    assert pred1 == 0.0 : "Expected 0, got " + pred1;
    assert pred2 == 1.0 : "Expected 1, got " + pred2;

    // Test with explicit labels
    double[][] features = {{0,0},{0,1},{1,0},{10,10},{10,11},{11,10}};
    double[] labels = {0,0,0,1,1,1};

    KNN knn2 = new KNN();
    knn2.fit(features, labels);

    double pred3 = knn2.predict(new double[]{0.5, 0.5});
    double pred4 = knn2.predict(new double[]{10.5, 10.5});

    assert pred3 == 0.0 : "Expected 0, got " + pred3;
    assert pred4 == 1.0 : "Expected 1, got " + pred4;

    // Test evaluate
    Map<String, Double> results = knn2.evaluate(new String[]{"accuracy"});
    assert results.get("accuracy") >= 1.0 : "Expected perfect accuracy";

    // Test save/load
    String tempFile = System.getProperty("java.io.tmpdir") + "/knn_test.ser";
    knn2.save(tempFile);
    KNN loaded = new KNN();
    loaded.load(tempFile);
    double pred5 = loaded.predict(new double[]{0.5, 0.5});
    double pred6 = loaded.predict(new double[]{10.5, 10.5});
    assert pred5 == 0.0 && pred6 == 1.0 : "Save/load failed";

    // Test sparse cosine
    List<Map<Integer, Double>> sparseTrain = new ArrayList<>();
    sparseTrain.add(Map.of(0, 0.0, 1, 0.0));
    sparseTrain.add(Map.of(0, 0.0, 1, 1.0));
    sparseTrain.add(Map.of(0, 1.0, 1, 0.0));
    sparseTrain.add(Map.of(0, 10.0, 1, 10.0));
    sparseTrain.add(Map.of(0, 10.0, 1, 11.0));
    sparseTrain.add(Map.of(0, 11.0, 1, 10.0));

    KNN sparseKnn = new KNN(sparseTrain, labels, 3, 2, "cosine");
    double sparsePred1 = sparseKnn.predict(Map.of(0, 0.5, 1, 0.5));
    double sparsePred2 = sparseKnn.predict(Map.of(0, 10.5, 1, 10.5));
    assert sparsePred1 == 0.0 : "Sparse expected 0, got " + sparsePred1;
    assert sparsePred2 == 1.0 : "Sparse expected 1, got " + sparsePred2;
}
```

## See Also

*(Other documentation pages are under development)*