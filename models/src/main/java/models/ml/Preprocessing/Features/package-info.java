/**
 * Generic feature extraction interfaces and composable extractors.
 * <p>
 * <ul>
 *   <li>{@link FeatureExtractor} — Core interface for sparse/dense extraction</li>
 *   <li>{@link AbstractDenseFeatureExtractor} — Base class for dense extractors</li>
 *   <li>{@link CompositeFeatureExtractor} — Concatenate multiple extractors</li>
 *   <li>{@link ScalingFeatureExtractor} — Extractor + scaler composition</li>
 *   <li>{@link FeatureVectors} — Sparse/dense conversion utilities</li>
 * </ul>
 * Custom extractors should extend {@link AbstractDenseFeatureExtractor}.
 * See <a href="../../../../../../../docs/pipelines.html">Pipelines Guide</a>.
 *
 * @since 1.0
 */
package models.ml.Preprocessing.Features;