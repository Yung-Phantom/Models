# Image Processing

This page covers `ImageData`, `ChannelMode`, `ImageOps`, `PixelFeatureExtractor`, `ColorHistogramExtractor`.

## ImageData

Immutable container for pixel data in [0, 1] range, row-major channel-interleaved.

<!-- no-compile -->
```java
// Grayscale (1 channel)
ImageData img = ImageData.of(28, 28, 1, pixels);  // pixels.length = 784

// RGB (3 channels)
ImageData img = ImageData.of(64, 64, 3, pixels);  // pixels.length = 12288
```

### Layout

```
index = (y * width + x) * channels + c
```

- `channels`: 1 (grayscale) or 3 (RGB)
- Pixel values: finite, in [0, 1]
- Defensive copy on construction

### Methods

| Method | Description |
|--------|-------------|
| `getWidth()` / `getHeight()` / `getChannels()` | Dimensions |
| `getPixels()` | Copy of pixel array |
| `getPixel(x, y, c)` | Single pixel value |

---

## ChannelMode

Enum for target channel conversion:

| Value | Meaning |
|-------|---------|
| `GRAY` | 1 channel (grayscale) |
| `RGB` | 3 channels (RGB) |

---

## ImageOps

Stateless pure operations on `ImageData`.

### convert(image, mode)

Converts between grayscale and RGB.

<!-- no-compile -->
```java
ImageData gray = ImageOps.convert(rgbImage, ChannelMode.GRAY);
ImageData rgb = ImageOps.convert(grayImage, ChannelMode.RGB);
```

**Grayscale formula**: `0.299*R + 0.587*G + 0.114*B` (standard luminance)

**RGB from gray**: Replicates single channel to all three.

### resize(image, newWidth, newHeight)

Bilinear interpolation, preserves channel count.

<!-- no-compile -->
```java
ImageData resized = ImageOps.resize(image, 224, 224);
```

**Algorithm:**
- Pixel-center mapping: `src = (dst + 0.5) * srcSize / dstSize - 0.5`
- Source coordinates clamped to `[0, w-1]` and `[0, h-1]` before interpolation
- Weights always in [0, 1]
- Result clamped to [0, 1] to handle interpolation overshoot
- Same-size returns original; constant images remain constant

---

## PixelFeatureExtractor

Resizes to fixed size and flattens pixels. Extends `AbstractDenseFeatureExtractor<ImageData>`.

<!-- no-compile -->
```java
PixelFeatureExtractor extractor = new PixelFeatureExtractor(28, 28, ChannelMode.GRAY);
extractor.fit(images);  // No-op, marks fitted
double[] features = extractor.transformDense(image);  // length = 28*28*1 = 784
```

### Configuration

| Parameter | Type | Validation |
|-----------|------|------------|
| `targetWidth` | int | > 0 |
| `targetHeight` | int | > 0 |
| `mode` | ChannelMode | Not null |

### Feature Layout

- `GRAY`: `width * height` features (row-major)
- `RGB`: `width * height * 3` features (R, G, B interleaved per pixel)

### JSON

```json
{"type":"pixel","width":28,"height":28,"mode":"GRAY","fitted":true}
```

### fromJson

<!-- no-compile -->
```java
PixelFeatureExtractor loaded = PixelFeatureExtractor.fromJson(json);
// Validates width > 0, height > 0, mode in {GRAY, RGB}
```

---

## ColorHistogramExtractor

Normalized color histograms. Size-independent (no resizing). Extends `AbstractDenseFeatureExtractor<ImageData>`.

<!-- no-compile -->
```java
ColorHistogramExtractor extractor = new ColorHistogramExtractor(16, ChannelMode.RGB);
extractor.fit(images);
double[] features = extractor.transformDense(image);  // length = 16 * 3 = 48
```

### Configuration

| Parameter | Type | Validation |
|-----------|------|------------|
| `bins` | int | ≥ 2 |
| `mode` | ChannelMode | Not null |

### Algorithm

1. Convert image to target mode (`GRAY` or `RGB`)
2. For each pixel and channel:
   - Bin index: `min(bins - 1, (int)(value * bins))` (clamp 1.0 to last bin)
   - Increment bin count
3. Normalize each channel's histogram to sum = 1.0

### Feature Layout

- `GRAY`: `bins` features (normalized histogram)
- `RGB`: `bins * 3` features (channel 0 bins, then channel 1, then channel 2)

### JSON

```json
{"type":"colorHistogram","bins":16,"mode":"RGB","fitted":true}
```

### fromJson

<!-- no-compile -->
```java
ColorHistogramExtractor loaded = ColorHistogramExtractor.fromJson(json);
// Validates bins >= 2, mode in {GRAY, RGB}
```

---

## Decoding Behavior

`ImageDatasetLoader.load(path)` uses `ImageIO.read()`:

- **Always returns RGB** (3 channels) — grayscale PNGs become 3-channel with identical R=G=B
- **Alpha channel**: Composited on white background (alpha blended)
- **Supported formats**: PNG, JPEG, BMP, GIF (case-insensitive extension matching via `Locale.ROOT`)
- **Hidden files/dirs**: Skipped (names starting with `.`)
- **Class ordering**: Alphabetical by directory name (`String.compareTo`)
- **File ordering**: Alphabetical by filename within class

### Example

<!-- no-compile -->
```java
ImageDataset ds = ImageDatasetLoader.load(Paths.get("data/images"));
// ds.getImages() → List<ImageData> (all 3-channel RGB)
// ds.getLabels() → List<String> (class names)
// ds.getSources() → List<String> (file paths)
ImageDataset[] split = ds.split(0.2, 42L);
```

---

## See Also

- [data-loading.md](data-loading.md) — ImageDatasetLoader
- [pipelines.md](pipelines.md) — FeaturePipeline with image extractors
- [models.md](models.md) — KNN, SVM for image classification