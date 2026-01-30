# Machine Learning Models Library

A Java-based machine learning library implementing supervised learning algorithms for classification and regression with support for multiple dataset formats.

## Project Structure

```
models/
├── src/main/java/models/ml/
│   ├── App.java                                    # Main entry point and examples
│   ├── DatasetHandler/
│   │   ├── DatasetLoader.java                     # Load CSV, JSON, SQLite datasets
│   │   ├── DocToVec.java                          # Document vectorization utilities
│   │   └── helpers/
│   │       ├── DatasetSplit.java                  # Train/test split container
│   │       ├── Dataset.java                       # Dataset wrapper class
│   │       └── DatasetConfig.java                 # Configuration enum
│   ├── KNN/
│   │   ├── KNN.java
│   │   └── AbstractKNN.java
│   ├── LinearRegression/
│   │   ├── LinearRegression.java
│   │   └── AbstractLinearRegression.java
│   ├── NaiveBayes/
│   │   ├── NaiveBayes.java
│   │   └── AbstractNaiveBayes.java
│   └── LogisticRegression/
│       ├── LogisticRegression.java
│       └── AbstractLogisticRegression.java
├── Datasets/                                       # Sample datasets
│   ├── Iris.csv
│   ├── Social_Network_Ads.csv
│   ├── sentiment_data.csv
│   ├── winequality-red.csv
│   └── ... (20+ datasets)
├── pom.xml                                        # Maven configuration
└── README.md
```

## Available Models

### 1. **K-Nearest Neighbors (KNN)**
Distance-based classification supporting Euclidean and Minkowski metrics.
- Supports configurable k value
- Majority voting for predictions
- Calculates accuracy on test set

```java
KNN knn = new KNN(trainData, testData, "minkowski", 5);
int[] predictions = knn.predictAllMajority();
double accuracy = knn.accuracy();
```

### 2. **Linear Regression**
Regression using Normal Equations or Gradient Descent.
- Methods: "normal" (closed-form) or "gradientDescent"
- Configurable learning rate and epochs
- Evaluation: MSE, R² score

```java
LinearRegression lr = new LinearRegression(trainData, testData, "normal", 0.1, 1000);
double[] predictions = lr.predictAll();
double mse = lr.mse();
double r2 = lr.r2();
```

### 3. **Naive Bayes**
Probabilistic classifier with multiple variants.
- Methods: "gaussian", "multinomial", "bernoulli"
- Alpha smoothing parameter support
- Returns normalized probability distributions

```java
NaiveBayes nb = new NaiveBayes(trainData, testData, "multinomial");
int[] predictions = nb.predictAll();
double accuracy = nb.accuracy();
Map<Integer, Double> probs = nb.predictProbability(0);
```

### 4. **Logistic Regression**
Classification using sigmoid/softmax functions.
- Methods: "binary", "multinomial", "ordinal"
- Configurable learning rate and epochs
- Probability estimates and class predictions

```java
LogisticRegression logReg = new LogisticRegression(trainData, testData, "multinomial");
int[] predictions = logReg.predictAll();
double probability = logReg.predictProbability(0);
double accuracy = logReg.accuracy();
```

## DatasetLoader & DatasetSplit

### Loading Datasets

```java
// Basic load with automatic header and ID detection
DatasetLoader loader = new DatasetLoader("path/to/data.csv", "label_column");
DatasetSplit split = loader.split(80); // 80% train, 20% test

double[][] trainData = split.train;
double[][] testData = split.test;
```

### Supported Formats
- **CSV/TSV/TXT**: Automatic delimiter detection
- **JSON**: Array of objects or newline-delimited
- **SQLite**: Database query support

### Custom Configuration

```java
DatasetLoader loader = new DatasetLoader(
    "path/to/data.csv",
    true,           // hasID
    true,           // hasHeader  
    "label_column", // labelColumn
    ','             // delimiter
);
```

## Quick Start Example

```java
// Load Social Network Ads dataset
DatasetLoader loader = new DatasetLoader(
    "Datasets/Social_Network_Ads.csv", 
    "Purchased"
);
DatasetSplit split = loader.split(80);

// Train logistic regression
LogisticRegression model = new LogisticRegression(
    split.train, 
    split.test, 
    "multinomial"
);

// Predict and evaluate
int[] predictions = model.predictAll();
System.out.println("Accuracy: " + model.accuracy());

// Check probability for specific sample
double prob = model.predictProbability(0);
System.out.println("Probability: " + prob);
```

## Building & Running

```bash
# Build with Maven
cd models
mvn clean compile

# Run main application
mvn exec:java -Dexec.mainClass="models.ml.App"

# Run tests
mvn test
```

## Evaluation Metrics

- **Accuracy**: Fraction of correct predictions (Classification)
- **MSE** (Mean Squared Error): Average squared errors (Regression)
- **R² Score**: Coefficient of determination (Regression)
- **Probability**: Estimated class probability (Logistic Regression)

## Sample Datasets Included

| Dataset | Type | Classes | Features | Records |
|---------|------|---------|----------|---------|
| Iris.csv | Classification | 3 | 4 | 150 |
| Social_Network_Ads.csv | Classification | 2 | multiple | varies |
| sentiment_data.csv | Classification | 3 | text | varies |
| winequality-red.csv | Regression | - | 11 | 1599 |

## Notes

- All models use index-based prediction: `predict(int queryIndex)`
- Datasets are shuffled during train/test split
- Last column in dataset is treated as the label/target variable
- NaN values are handled during numeric conversion
