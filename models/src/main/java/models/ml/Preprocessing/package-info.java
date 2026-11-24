/**
 * Feature extraction, scaling, and label encoding.
 * <p>
 * Subpackages:
 * <ul>
 *   <li>{@link models.ml.Preprocessing.Scaling} — ZScore, MinMax, MaxAbs, SimpleImputer, StandardScaler</li>
 *   <li>{@link models.ml.Preprocessing.Labels} — LabelEncoder</li>
 *   <li>{@link models.ml.Preprocessing.Text.Vectorizer} — BagOfWords, TF-IDF, TextPipeline</li>
 *   <li>{@link models.ml.Preprocessing.Image} — ImageData, ImageOps, Pixel/Histogram extractors</li>
 *   <li>{@link models.ml.Preprocessing.Audio} — AudioData, MFCC, SpectralStats, FFT</li>
 *   <li>{@link models.ml.Preprocessing.Features} — FeatureExtractor, Composite, Scaling, FeatureVectors</li>
 * </ul>
 * See individual guide pages for each domain.
 *
 * @since 1.0
 */
package models.ml.Preprocessing;