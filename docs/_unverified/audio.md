# Audio Processing

This page covers `AudioData`, `AudioDecoder`, `SpectralStatsExtractor`, `MfccExtractor`, `Fft`, `Windows`, `AudioOps`, and the resampling caveat.

## AudioData

Immutable container for audio samples in [-1.0, 1.0] range.

<!-- no-compile -->
```java
AudioData audio = AudioData.of(16000, samples);  // samples.length > 0
```

### Validation

- `sampleRate` > 0
- `samples` not null, not empty
- All values finite and in [-1.0, 1.0]
- Defensive copy on construction

### Methods

| Method | Description |
|--------|-------------|
| `getSampleRate()` | Sample rate in Hz |
| `getSamples()` | Copy of samples array |
| `length()` | Number of samples |
| `getDurationSeconds()` | `length / sampleRate` |

---

## AudioDecoder

Decodes WAV files to `AudioData` using `AudioSystem`.

### Supported Formats

| Encoding | Bit Depth | Channels |
|----------|-----------|----------|
| PCM_SIGNED | 16-bit little-endian | 1 (mono) or 2 (stereo) |
| PCM_UNSIGNED | 8-bit | 1 (mono) or 2 (stereo) |

### Behavior

- Stereo → mono by averaging channels
- Samples scaled to [-1.0, 1.0]:
  - 16-bit: `sample / 32768.0`
  - 8-bit: `(sample - 128.0) / 128.0`
- Throws `IllegalArgumentException` for:
  - Null path, non-regular file, non-WAV file
  - Unsupported encoding (float, 24-bit, etc.)
  - > 2 channels
  - Invalid WAV structure

### Usage

<!-- no-compile -->
```java
AudioData audio = AudioDecoder.decode(Paths.get("audio.wav"));
// audio.getSampleRate() → e.g., 16000
// audio.getSamples() → double[] in [-1, 1]
```

---

## SpectralStatsExtractor

Computes 8 spectral statistics per audio clip. Extends `AbstractDenseFeatureExtractor<AudioData>`.

<!-- no-compile -->
```java
SpectralStatsExtractor extractor = new SpectralStatsExtractor(256, 128);  // frameSize, hop
extractor.fit(audioClips);
double[] features = extractor.transformDense(audioClip);  // length = 8
```

### Parameters

| Parameter | Type | Validation |
|-----------|------|------------|
| `frameSize` | int | > 0, power of 2 |
| `hop` | int | > 0 |

### Feature Vector (8 features, in order)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `mean_rms` | Mean RMS energy across frames |
| 1 | `std_rms` | Population std of RMS |
| 2 | `mean_zcr` | Mean zero-crossing rate |
| 3 | `std_zcr` | Population std of ZCR |
| 4 | `mean_centroid` | Mean spectral centroid (normalized 0-1) |
| 5 | `std_centroid` | Population std of centroid |
| 6 | `mean_rolloff` | Mean spectral rolloff (85%, normalized 0-1) |
| 7 | `std_rolloff` | Population std of rolloff |

### Algorithm

1. Frame audio: `Framing.frames(samples, frameSize, hop)`
2. Per frame: apply Hann window, compute FFT magnitude
3. RMS: `sqrt(mean(x²))`
4. ZCR: `count(sign changes) / (frameSize - 1)`
5. Centroid: `Σ(k * mag[k]) / Σ(mag[k]) / (N - 1)` (N = num bins)
6. Rolloff: Smallest k where cumulative magnitude ≥ 0.85 * total
7. Aggregate: Population mean/std across frames using `FrameStats`

### JSON

```json
{"type":"spectralStats","frameSize":256,"hop":128,"fitted":true}
```

### fromJson

<!-- no-compile -->
```java
SpectralStatsExtractor loaded = SpectralStatsExtractor.fromJson(json);
// Relies on constructor for validation (frameSize power of 2, hop > 0)
```

---

## MfccExtractor

Mel-Frequency Cepstral Coefficients. Extends `AbstractDenseFeatureExtractor<AudioData>`.

<!-- no-compile -->
```java
MfccExtractor extractor = new MfccExtractor(
    16000,    // targetSampleRate
    512,      // frameSize
    160,      // hop
    40,       // numFilters
    13        // numCoeffs
);
extractor.fit(audioClips);
double[] features = extractor.transformDense(audioClip);  // length = 2 * 13 = 26
```

### Parameters

| Parameter | Type | Validation |
|-----------|------|------------|
| `targetSampleRate` | int | > 0 |
| `frameSize` | int | > 0, power of 2 |
| `hop` | int | > 0 |
| `numFilters` | int | ≥ 1 |
| `numCoeffs` | int | 1..numFilters |

### Pipeline

1. Resample to `targetSampleRate` (linear interpolation)
2. Frame → Hann window → FFT → power spectrum
3. Mel filterbank (precomputed) → log energies
4. DCT-II → keep first `numCoeffs` coefficients
5. Per coefficient: population mean + std across frames
6. Output: **Coefficient-major** `[mean_0, std_0, mean_1, std_1, ...]`
7. Feature size = `2 * numCoeffs`

### Resampling Caveat

- `AudioOps.resample()` uses **linear interpolation only**
- **No anti-aliasing filter** applied before downsampling
- Aliasing may occur when downsampling (e.g., 44.1kHz → 16kHz)
- For production: apply low-pass filter before resampling

### JSON

```json
{"type":"mfcc","targetSampleRate":16000,"frameSize":512,"hop":160,"numFilters":40,"numCoeffs":13,"fitted":true}
```

### fromJson

<!-- no-compile -->
```java
MfccExtractor loaded = MfccExtractor.fromJson(json);
// Relies on constructor for all validation
```

---

## Fft

Radix-2 Cooley-Tukey FFT for magnitude spectrum.

<!-- no-compile -->
```java
double[] mag = Fft.magnitudeSpectrum(frame, 256);  // frame.length ≤ 256, power of 2
// Returns length = fftSize/2 + 1 (bins 0..Nyquist)
```

### Validation

- `frame` not null
- `fftSize` > 0, power of 2, ≥ frame.length
- Zero-pads frame to fftSize

---

## Windows

Hann window generation and application.

<!-- no-compile -->
```java
double[] window = Windows.hann(256);  // length 256
Windows.apply(frame, window);         // in-place multiplication
```

**Hann formula**: `0.5 * (1 - cos(2π * n / (N - 1)))`

---

## AudioOps

### resample(audio, targetRate)

Linear interpolation resampling (see caveat above).

<!-- no-compile -->
```java
AudioData resampled = AudioOps.resample(audio, 16000);
```

### fixLength(array, len)

Truncate or zero-pad.

<!-- no-compile -->
```java
double[] fixed = AudioOps.fixLength(samples, 16000);
```

---

## FrameStats

Utility for population mean/std (used by both audio extractors).

<!-- no-compile -->
```java
double mean = FrameStats.mean(values);
double std = FrameStats.std(values);  // Population: sqrt(Σ(x-μ)² / n)
```

Throws `IllegalArgumentException` for null or empty array.

---

## See Also

- [data-loading.md](data-loading.md) — AudioDatasetLoader
- [pipelines.md](pipelines.md) — FeaturePipeline with audio extractors
- [models.md](models.md) — Models for audio classification
- [preprocessing.md](preprocessing.md) — ScalingFeatureExtractor for MFCC