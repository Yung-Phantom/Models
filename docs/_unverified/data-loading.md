# Data Loading

This page covers `DatasetLoader`, `DatasetConfig`, split strategies, `TextDataset`, `ImageDatasetLoader`, `AudioDatasetLoader`, and `LabeledDataset.split`.

## DatasetLoader

`DatasetLoader` loads tabular data from CSV, TSV, TXT, JSON, or SQLite files. It returns a `Dataset` object containing a `double[][]` matrix, column header, and label-to-index map.

### Supported Formats

| Extension | Loader | Notes |
|-----------|--------|-------|
| `.csv` | `CsvParser` | Comma-separated, auto-detects delimiter if not specified |
| `.tsv` | `CsvParser` | Tab-separated |
| `.txt` | `CsvParser` | Treated as CSV/TSV |
| `.json` | `JsonParser` | Array of objects; first object's keys become header |
| `.sqlite` | `SqliteLoader` | Reads first table; requires explicit `labelColumn` |

### Constructors

<!-- no-compile -->
```java
// Minimal: uses DEFAULT config (no ID, has header), last column as label
new DatasetLoader("data.csv")

// With config
new DatasetLoader("data.csv", DatasetConfig.HAS_ID_WITH_HEADER)

// With label column name
new DatasetLoader("data.csv", "Species")

// With delimiter
new DatasetLoader("data.tsv", '\t')

// Full control
new DatasetLoader("data.csv", DatasetConfig.HAS_ID_WITH_HEADER, "target", ',', false)
```

### DatasetConfig Options

| Constant | `hasID` | `hasHeader` | Use Case |
|----------|---------|-------------|----------|
| `HAS_ID_WITH_HEADER` | true | true | First column is row ID, first row is header |
| `HAS_ID_NO_HEADER` | true | false | First column is row ID, no header |
| `NO_ID_WITH_HEADER` | false | true | No ID column, first row is header |
| `NO_ID_NO_HEADER` | false | false | No ID column, no header |
| `DEFAULT` | false | true | Same as `NO_ID_WITH_HEADER` |

### Label Column Handling

- If `labelColumn` is `null`, the **last column** is used as the label.
- If `numericLabel=true`, label values are parsed as `double` (for regression). Default is `false` (categorical strings).
- SQLite **requires** an explicit `labelColumn`; throws `IllegalStateException` otherwise.

### Split Methods

All split methods return `DatasetSplit { double[][] train, double[][] test }`.

<!-- no-compile -->
```java
// Percent (0-100) or decimal (0.0-1.0) — both accepted
split(80)              // 80% train, default strategy "stratified"
split(0.8)             // Same as above
split(80, "random")    // Strategy: "random"/"r", "stratified"/"s", "timeseries"/"t"
split(0.8, "stratified", 42L) // With seed for reproducibility

// Explicit strategy methods
randomSplit(80)
stratifiedSplit(0.8, 123L)
timeSeriesSplit(80)
```

### Split Strategies

| Strategy | Aliases | Behavior |
|----------|---------|----------|
| `random` | `r` | Random shuffle, no class balancing |
| `stratified` | `s` | Preserves class proportions in train/test |
| `timeseries` | `t` | First N% train (chronological), rest test |

**Notes:**
- `stratified` falls back to `random` if labels are continuous (non-integer).
- Classes with only 1 sample go entirely to train.
- Percent values > 1 are treated as percentages (e.g., `80` = 80%). Values ≤ 1 are decimals (e.g., `0.8` = 80%).

### Common Exceptions

| Message | Cause |
|---------|-------|
| `Unsupported format: xyz` | File extension not in {csv, tsv, txt, json, sqlite} |
| `No label column specified. Supervised models require labels.` | SQLite load without `labelColumn` |
| `testFraction must be in (0, 1)` | Split fraction ≤ 0 or ≥ 1 |
| `inputs and labels must have the same size` | Mismatched array lengths in `fit()` |

## TextDataset

For text classification, `DatasetLoader.loadText()` returns a `TextDataset`:

<!-- no-compile -->
```java
TextDataset td = DatasetLoader.loadText(
    "data.csv",      // path
    "text",          // text column name
    "label",         // label column name
    DatasetConfig.HAS_ID_WITH_HEADER,
    ','              // delimiter
);
```

The `TextDataset` contains:
- `List<String> texts` — raw text documents
- `double[] labels` — numeric label indices (from `LabelEncoder`)
- `LabelEncoder encoder` — for inverse transform

## ImageDatasetLoader / AudioDatasetLoader

Both load folder-per-class layouts:

```
root/
  class_a/
    img1.png
    img2.png
  class_b/
    img3.png
```

<!-- no-compile -->
```java
ImageDataset ds = ImageDatasetLoader.load(Paths.get("images"));
AudioDataset ds = AudioDatasetLoader.load(Paths.get("audio"));
```

### Supported File Types

| Loader | Extensions |
|--------|------------|
| `ImageDatasetLoader` | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif` (case-insensitive) |
| `AudioDatasetLoader` | `.wav` (PCM 8/16-bit mono/stereo only) |

### Returned Objects

- `ImageDataset` wraps `LabeledDataset<ImageData>` with `getImages()`, `getLabels()`, `getSources()`, `split()`.
- `AudioDataset` wraps `LabeledDataset<AudioData>` similarly.

### LabeledDataset.split()

Both datasets delegate to `LabeledDataset.split(testFraction, seed)` for stratified splits:

<!-- no-compile -->
```java
ImageDataset[] split = ds.split(0.3, 42L); // 30% test, seed 42
ImageDataset train = split[0];
ImageDataset test = split[1];
```

Validation:
- `testFraction` must be in (0, 1).
- Null items/labels/sources throw `IllegalArgumentException` with index.
- Size mismatch throws with exact message: `"images, labels, and sources must have the same size"`.

## See Also

- [preprocessing.md](preprocessing.md) — Scaling and encoding after loading
- [models.md](models.md) — Training models on loaded data
- [pipelines.md](pipelines.md) — End-to-end pipelines