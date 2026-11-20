# Models

A Java 17 machine-learning library with implementations for classification and
regression, dataset loading, numeric preprocessing, and text classification.
The Maven project is in `models/`; sample datasets are in the repository-level
`Datasets/` directory.

## Features

- **Models:** K-nearest neighbors (KNN), linear regression, logistic regression,
  Naive Bayes, and support vector machines (SVM).
- **Data:** Load CSV, TSV, TXT, JSON, or SQLite datasets. Select a target column
  by name or use the final column. Split data randomly, stratified, or as a
  time series.
- **Preprocessing:** Standard, z-score, min-max, and max-absolute scaling;
  simple imputation; and label encoding.
- **Text:** Bag-of-words and TF-IDF vectorizers, n-gram generation, word and
  document vectorization, and a text-classification pipeline.
- **Evaluation:** Classification metrics include accuracy, precision, recall,
  specificity, F1, balanced accuracy, and MCC. Regression metrics include MSE,
  RMSE, MAE, and R².
- **Model persistence:** Models support saving and loading serialized files.

## Build and test

Requires JDK 17 and Maven.

```shell
cd models
mvn test
```

The Maven module is `models/pom.xml`. Tests are under `models/src/test/java`.

## Quick start

This example loads the bundled Iris CSV, keeps its `Species` target, makes a
stratified 80/20 split, trains KNN, and evaluates accuracy:

```java
import java.util.Map;

import models.ml.DatasetHandler.DatasetLoader;
import models.ml.DatasetHandler.helpers.DatasetConfig;
import models.ml.Model.KNN.KNN;

public class Example {
    public static void main(String[] args) throws Exception {
        DatasetLoader loader = new DatasetLoader(
            "../Datasets/Iris.csv",
            DatasetConfig.HAS_ID_WITH_HEADER,
            "Species"
        );

        KNN model = new KNN();
        model.fit(loader, 80, "stratified");

        Map<String, Double> results =
            model.evaluate(new String[] { "accuracy" });
        System.out.println(results);
    }
}
```

`DatasetConfig` describes whether the source has a header and an ID column.
The split strategy accepts `random`, `stratified`, or `timeseries` (and their
short aliases `r`, `s`, or `t`). A seeded overload is available on
`DatasetLoader.split` for reproducible splits.

## Models and options

All models can be trained with dense numeric data or, where supported, sparse
text-vectorizer output. The common `Model` API provides fitting, prediction,
evaluation, refresh, and save/load operations.

| Model | Purpose | Options |
|---|---|---|
| `KNN` | Classification | Euclidean, Manhattan, Minkowski, or cosine distance; configurable `k` |
| `LinearRegression` | Regression | Normal equation (`normal`) or gradient descent (`gd` / `gradientdescent`) |
| `LogisticRegression` | Classification | Binary, multinomial, or ordinal |
| `NaiveBayes` | Classification | Gaussian, multinomial, or Bernoulli |
| `SVM` | Classification | SVC and configurable kernels |

Model classes are in `models.ml.Model`; numeric preprocessing is in
`models.ml.Preprocessing.Scaling`. Text vectorizers and `TextPipeline` are in
`models.ml.Preprocessing.Text.Vectorizer` and `models.ml.Pipeline`.

## Dataset notes

The repository includes sample CSV and JSON files, plus a SQLite database, in
`Datasets/`. `DatasetLoader` currently supports CSV, TSV, TXT, JSON, and SQLite;
the included Parquet file is not supported by this loader.

The `models.ml.App` class is an experiment/demo entry point. Its dataset paths
are absolute paths from the development machine, so update them for your
environment before running it.
