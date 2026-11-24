/**
 * Core dataset types and split utilities.
 * <p>
 * <ul>
 *   <li>{@link Dataset} — In-memory table with header and label map</li>
 *   <li>{@link DatasetSplit} — Train/test pair from splitting</li>
 *   <li>{@link DatasetConfig} — Header/ID column configuration</li>
 *   <li>{@link LabeledDataset} — Generic labeled data with stratified split</li>
 *   <li>{@link ClassFolderScanner} — Scan folder-per-class directories</li>
 * </ul>
 * See <a href="../../../../../../../docs/data-loading.html">Data Loading Guide</a>.
 *
 * @since 1.0
 */
package models.ml.DatasetHandler.helpers;