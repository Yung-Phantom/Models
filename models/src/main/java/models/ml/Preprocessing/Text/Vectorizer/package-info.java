/**
 * Text vectorization and classification pipeline.
 * <p>
 * <ul>
 *   <li>{@link BagOfWordsVectorizer} — Token counts or binary presence (sparse)</li>
 *   <li>{@link TfidfVectorizer} — TF-IDF with optional sublinear TF and L2 normalization (sparse)</li>
 *   <li>{@link TextPipeline} — Vectorizer + LabelEncoder + Model</li>
 * </ul>
 * Both vectorizers support n-grams, stopwords, custom token patterns, JSON round-trip.
 * <p>
 * <b>Note:</b> {@link models.ml.Preprocessing.Text.WordToVec.Word2Vec} and
 * {@link models.ml.Preprocessing.Text.DocToVec.Doc2Vec} are simplified implementations
 * for educational purposes only.
 * <p>
 * See <a href="../../../../../../../docs/text.html">Text Processing Guide</a>.
 *
 * @since 1.0
 */
package models.ml.Preprocessing.Text.Vectorizer;