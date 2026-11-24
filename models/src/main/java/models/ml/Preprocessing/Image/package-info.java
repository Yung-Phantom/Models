/**
 * Image data representation and feature extraction.
 * <p>
 * <ul>
 *   <li>{@link ImageData} — Immutable pixels in [0,1], row-major channel-interleaved</li>
 *   <li>{@link ChannelMode} — GRAY (1ch) or RGB (3ch)</li>
 *   <li>{@link ImageOps} — Grayscale/RGB conversion, bilinear resize</li>
 *   <li>{@link PixelFeatureExtractor} — Resize + flatten (dense)</li>
 *   <li>{@link ColorHistogramExtractor} — Normalized histograms, size-independent (dense)</li>
 * </ul>
 * Decoding: {@link ImageDatasetLoader} uses {@code ImageIO} → always RGB, alpha composited on white.
 * See <a href="../../../../../../../docs/images.html">Image Processing Guide</a>.
 *
 * @since 1.0
 */
package models.ml.Preprocessing.Image;