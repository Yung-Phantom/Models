/**
 * End-to-end pipelines combining feature extraction, label encoding, and models.
 * <p>
 * <ul>
 *   <li>{@link FeaturePipeline} — Generic pipeline for any {@link FeatureExtractor}</li>
 *   <li>{@link TextPipeline} — Text-specific pipeline with vectorizers</li>
 *   <li>{@link DataMode} — SPARSE or DENSE mode</li>
 * </ul>
 * Failure semantics: validation first; on training failure, pipeline left unfitted.
 * See <a href="../../../../../../../docs/pipelines.html">Pipelines Guide</a>.
 *
 * @since 1.0
 */
package models.ml.Pipeline;