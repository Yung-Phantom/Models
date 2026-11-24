/**
 * Dataset loading and splitting utilities.
 * <p>
 * Main entry points:
 * <ul>
 *   <li>{@link DatasetLoader} — Load CSV, TSV, TXT, JSON, SQLite</li>
 *   <li>{@link ImageDatasetLoader} — Load folder-per-class images</li>
 *   <li>{@link AudioDatasetLoader} — Load folder-per-class audio</li>
 *   <li>{@link TextDatasetLoader} — Load text classification data</li>
 * </ul>
 * Split strategies: {@link DatasetSplitter} (random, stratified, timeseries).
 * See <a href="../../../../../../../docs/data-loading.html">Data Loading Guide</a>.
 *
 * @since 1.0
 */
package models.ml.DatasetHandler;