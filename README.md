# Models

A Java 17 machine-learning library providing classification and regression models, dataset loading, numeric preprocessing, text classification, image and audio feature extraction, and composable pipelines.

## Requirements

- **Java 17+** (compiled for Java 17 bytecode)
- **Runtime Dependencies**: Pure Java with zero mandatory third-party dependencies. Optional `org.xerial:sqlite-jdbc:3.49.1.0` is required only when loading SQLite databases via `DatasetLoader`. Image and audio I/O use the standard JDK `java.desktop` module.

## Install

### 1. Maven Dependency
```xml
<dependency>
    <groupId>models.ml</groupId>
    <artifactId>models</artifactId>
    <version>1.0-SNAPSHOT</version>
</dependency>
```

### 2. Plain JAR on Classpath
Build with `mvn package`. The output is `target/models-1.0-SNAPSHOT.jar` (plus `models-1.0-SNAPSHOT-sources.jar` and `models-1.0-SNAPSHOT-javadoc.jar`). For SQLite loading, include `sqlite-jdbc-3.49.1.0.jar`.

Copy the main JAR and required dependencies to your project's `lib/` folder, then compile and run:

```bash
# Linux / macOS
mkdir -p lib
cp ~/.m2/repository/models/ml/models/1.0-SNAPSHOT/models-1.0-SNAPSHOT.jar lib/
# For SQLite support:
cp ~/.m2/repository/org/xerial/sqlite-jdbc/3.49.1.0/sqlite-jdbc-3.49.1.0.jar lib/

javac -cp "lib/*" MyApp.java
java -cp ".:lib/*" MyApp
```

```powershell
# Windows (PowerShell)
mkdir lib
cp ~/.m2/repository/models/ml/models/1.0-SNAPSHOT/models-1.0-SNAPSHOT.jar lib/
# For SQLite support:
cp ~/.m2/repository/org/xerial/sqlite-jdbc/3.49.1.0/sqlite-jdbc-3.49.1.0.jar lib/

javac -cp "lib/*" MyApp.java
java -cp ".;lib/*" MyApp
```

### 3. Gradle
```groovy
repositories { mavenLocal(); mavenCentral() }
dependencies {
    implementation 'models.ml:models:1.0-SNAPSHOT'
    runtimeOnly 'org.xerial:sqlite-jdbc:3.49.1.0' // only for SQLite
}
```

## Quickstarts

### 1. Numeric CSV Classification
<!-- snippet: QuickstartExamples#numeric-csv -->
```java
DatasetLoader loader = new DatasetLoader(
    csvPath,
    DatasetConfig.HAS_ID_WITH_HEADER,
    "Species"
);

KNN model = new KNN();
model.fit(loader, 80, "stratified");

Map<String, Double> results = model.evaluate(new String[]{"accuracy"});
System.out.println(results);
```

### 2. Text Classification
<!-- snippet: QuickstartExamples#text-classification -->
```java
List<String> texts = List.of(
    "great movie loved it",
    "wonderful acting great",
    "loved wonderful film",
    "terrible movie hated it",
    "awful acting terrible",
    "hated awful film"
);
List<String> labels = List.of("pos", "pos", "pos", "neg", "neg", "neg");

TfidfVectorizer vec = new TfidfVectorizer();
TextPipeline pipeline = new TextPipeline(
    vec, new LabelEncoder(), new NaiveBayes()
);

pipeline.fit(texts, labels);
String pred = pipeline.predict("great wonderful loved");
String pred2 = pipeline.predict("terrible awful hated");
System.out.println("pred pos: " + pred);
System.out.println("pred neg: " + pred2);
```

### 2b. Text Classification with FeaturePipeline
<!-- snippet: QuickstartExamples#text-classification-feature-pipeline -->
```java
List<String> texts = List.of(
    "great movie loved it",
    "wonderful acting great",
    "loved wonderful film",
    "terrible movie hated it",
    "awful acting terrible",
    "hated awful film"
);
List<String> labels = List.of("pos", "pos", "pos", "neg", "neg", "neg");

TfidfVectorizer vec = new TfidfVectorizer();
TextFeatureExtractor extractor = new TextFeatureExtractor(vec);
FeaturePipeline<String> pipeline = new FeaturePipeline<>(
    extractor, new LabelEncoder(), new NaiveBayes(), DataMode.SPARSE
);

pipeline.fit(texts, labels);
String pred = pipeline.predict("great wonderful loved");
String pred2 = pipeline.predict("terrible awful hated");
System.out.println("pred pos: " + pred);
System.out.println("pred neg: " + pred2);
```

### 3. Image Classification
<!-- snippet: QuickstartExamples#image-classification -->
```java
ImageDataset ds = ImageDatasetLoader.load(Paths.get(imagePath));
ImageDataset[] split = ds.split(0.2, 42L);

PixelFeatureExtractor base = new PixelFeatureExtractor(16, 16, ChannelMode.GRAY);
ScalingFeatureExtractor<ImageData> extractor = new ScalingFeatureExtractor<>(base, new ZScoreScaler());

FeaturePipeline<ImageData> pipeline = new FeaturePipeline<>(
    extractor, new LabelEncoder(), new KNN(), DataMode.DENSE
);

pipeline.fit(split[0].getImages(), split[0].getLabels());
double acc = pipeline.score(split[1].getImages(), split[1].getLabels());
System.out.println("Accuracy: " + acc);
```

### 4. Audio Classification
<!-- snippet: QuickstartExamples#audio-classification -->
```java
LabeledDataset<AudioData> ds = AudioDatasetLoader.load(Paths.get(audioPath));
LabeledDataset<AudioData>[] split = ds.split(0.2, 42L);

MfccExtractor base = new MfccExtractor(16000, 512, 160, 40, 13);
ScalingFeatureExtractor<AudioData> extractor = new ScalingFeatureExtractor<>(base, new ZScoreScaler());

FeaturePipeline<AudioData> pipeline = new FeaturePipeline<>(
    extractor, new LabelEncoder(), new LogisticRegression(), DataMode.DENSE
);

pipeline.fit(split[0].getItems(), split[0].getLabels());
double acc = pipeline.score(split[1].getItems(), split[1].getLabels());
System.out.println("Accuracy: " + acc);
```

### 5. Saving and Loading Pipelines
<!-- snippet: PipelinePersistenceExamples#pipeline-io-save-load -->
```java
@Test
public void pipelineIOSaveLoadExample(@TempDir Path tmp) throws Exception {
    // Train and save a text pipeline
    List<String> texts = List.of("great movie", "loved it", "awesome film", "terrible movie", "hated it", "awful film");
    List<String> labels = List.of("pos", "pos", "pos", "neg", "neg", "neg");

    TextPipeline pipeline = new TextPipeline(new TfidfVectorizer(), new NaiveBayes());
    pipeline.fit(texts, labels);

    Path saveDir = tmp.resolve("saved_pipeline");
    PipelineIO.save(pipeline, saveDir);

    // Later: load and predict
    TextPipeline loaded = PipelineIO.loadTextPipeline(
        saveDir,
        json -> new models.ml.Preprocessing.Text.Vectorizer.TextFeatureExtractor(
            models.ml.Preprocessing.Text.Vectorizer.TfidfVectorizer.fromJson(json)).getVectorizer(),
        NaiveBayes::new
    );

    String prediction = loaded.predict("great movie");
    assertEquals("pos", prediction);
}
```

> **Security Note**: The model component inside a saved pipeline is Java-serialized. Never load pipeline files from untrusted sources, as Java deserialization can execute arbitrary code.

## Feature Matrix

### Models
| Model | Supported Methods | Dense | Sparse |
|---|---|:---:|:---:|
| `KNN` | `euclidean`, `manhattan`, `minkowski`, `cosine` | Yes | Yes (cosine only) |
| `NaiveBayes` | `gaussian`, `multinomial`, `bernoulli` | Yes (gaussian) | Yes (multinomial, bernoulli) |
| `SVM` | `linearsvc`, `svc` (`linearkernel`, `rbfkernel`, `polynomialkernel`, `sigmoidkernel`) | Yes | Yes (`linearsvc` only) |
| `LinearRegression` | `normal`, `gd`, `gradientdescent` | Yes | Yes (`gd`/`gradientdescent` only) |
| `LogisticRegression` | `binary`, `multinomial`, `ordinal` | Yes | Yes (`binary`, `multinomial` only) |

### Preprocessing
| Transformer | Purpose | Dense | Sparse | JSON |
|---|---|:---:|:---:|:---:|
| `ZScoreScaler` | Zero-mean, unit-variance standardization | Yes | No | Yes |
| `MinMaxScaler` | Range [0, 1] scaling | Yes | No | Yes |
| `MaxAbsScaler` | Maximum absolute value scaling | Yes | Yes | Yes |
| `SimpleImputer` | Fill missing (NaN) values (MEAN, MEDIAN, MOST_FREQUENT, CONSTANT) | Yes | No | Yes |
| `LabelEncoder` | Alphabetical string label encoding to 0..K-1 | Yes | Yes | Yes |
| `StandardScaler` | Splits features from labels (**does not scale**) | Yes | No | No |

### Loaders
| Loader | Supported Formats | Split Strategies |
|---|---|---|
| `DatasetLoader` | CSV, TSV, TXT, JSON, SQLite | random (`r`), stratified (`s`), timeseries (`t`) |
| `ImageDatasetLoader` | Folder per class: PNG, JPG, JPEG, BMP, GIF | `LabeledDataset.split(fraction, seed)` |
| `AudioDatasetLoader` | Folder per class: WAV PCM 8/16-bit mono/stereo | `LabeledDataset.split(fraction, seed)` |

### Feature Extractors
| Extractor | Input Domain | Output Features | Mode | JSON |
|---|---|---|:---:|:---:|
| `PixelFeatureExtractor` | `ImageData` | Raw resized pixels (width * height * channels) | Dense | Yes |
| `ColorHistogramExtractor` | `ImageData` | Normalized color histogram (bins * channels) | Dense | Yes |
| `SpectralStatsExtractor` | `AudioData` | 8 spectral stats (RMS, ZCR, Centroid, Rolloff mean/std) | Dense | Yes |
| `MfccExtractor` | `AudioData` | 2 * numCoeffs (mean and std per MFCC coefficient) | Dense | Yes |
| `BagOfWordsVectorizer` | Text `String` | Token occurrence / count sparse vectors | Sparse | Yes |
| `TfidfVectorizer` | Text `String` | Sublinear/L2-normalized TF-IDF sparse vectors | Sparse | Yes |
| `ScalingFeatureExtractor` | Extractor + Scaler | Scaled dense features from base extractor | Dense | Yes |
| `CompositeFeatureExtractor`| Multi-extractor | Concatenated features from child extractors | Dense | Yes |

## Package Map

| Package | Purpose | Guide |
|---|---|---|
| `models.ml.DatasetHandler` | Tabular, image, and audio dataset loaders | |
| `models.ml.DatasetHandler.helpers` | Dataset configurations and split containers | |
| `models.ml.Model` | Abstract base class for all models; provides fit/predict/evaluate/save/load | [models/knn.md](docs/models/knn.md) |
| `models.ml.Model.KNN` | K-Nearest Neighbors classifier | [models/knn.md](docs/models/knn.md) |
| `models.ml.Model.NaiveBayes` | Gaussian, Multinomial, and Bernoulli Naive Bayes | |
| `models.ml.Model.SVM` | Support Vector Machines (Linear and Kernel SVC) | |
| `models.ml.Model.LinearRegression` | Ordinary Least Squares and SGD Linear Regression | |
| `models.ml.Model.LogisticRegression` | Binary, multinomial, and ordinal logistic regression | |
| `models.ml.Preprocessing.Scaling` | Feature normalization and missing value imputation | |
| `models.ml.Preprocessing.Labels` | Categorical label encoding and decoding | |
| `models.ml.Preprocessing.Text.Vectorizer` | Bag-of-words and TF-IDF text vectorizers | |
| `models.ml.Preprocessing.Image` | Image decoding, resizing, pixel and color extractors | |
| `models.ml.Preprocessing.Audio` | WAV decoding, FFT, spectral stats, and MFCC extraction | |
| `models.ml.Preprocessing.Features` | Composable and scaled feature extractors | |
| `models.ml.Pipeline` | End-to-end classification pipelines | |

## Documentation Index

- [models/knn.md](docs/models/knn.md): KNN guide with constructors, distance methods, fit/predict/evaluate/save/load, and common errors.

*(Other documentation pages are under development in `docs/_unverified/`)*

## Known Limitations

- **SVM is binary only**: Multi-class classification requires external One-vs-Rest orchestration.
- **SVC is dense-only**: Sparse input is only supported by `linearsvc` with `linearkernel`.
- **Gaussian NB is dense-only**: sparse inputs only for multinomial and bernoulli.
- **Normal equation is dense-only**: LinearRegression normal equation requires dense matrices; gradient descent supports both.
- **No regularization or early stopping**: L2 regularization is available for LinearRegression and LogisticRegression; gradient descent runs for a fixed number of epochs and performs best on scaled features. Early stopping is not implemented.
- **KNN is brute force**: Exact linear scan over stored instances without KD-tree or ball-tree index structures.
- **DatasetLoader extension matching**: Matches file extensions strictly in lowercase (`Locale.ROOT`). SQLite requires an explicit label column.
- **Audio decoding constraints**: AudioDecoder supports only uncompressed WAV PCM 8-bit unsigned and 16-bit signed LE mono/stereo.
- **Resampling filter**: Linear interpolation resampling has no anti-aliasing low-pass filter; downsampling may produce aliasing.
- **Java serialization caveats**: Standard Java serialization is sensitive to class evolution and vulnerable on untrusted inputs.
- **Thread safety**: Model and transformer instances are mutable and not thread-safe.

## License

License: not specified yet
