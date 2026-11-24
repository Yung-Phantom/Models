/**
 * Base {@link Model} class and evaluation metrics.
 * <p>
 * All models extend {@link Model} which provides:
 * <ul>
 *   <li>Dense and sparse {@code fit()} overloads</li>
 *   <li>{@code predict()}, {@code predictLabels()}, {@code predictProbabilities()}</li>
 *   <li>{@code evaluate()} with metric names</li>
 *   <li>{@code save()}/{@code load()} via Java serialization</li>
 *   <li>{@code refresh()} to rebuild after hyperparameter changes</li>
 * </ul>
 * Metrics: {@link Metrics} (accuracy, precision, recall, F1, MSE, RMSE, MAE, R²).
 * See <a href="../../../../../../../docs/models.html">Models Guide</a> and
 * <a href="../../../../../../../docs/evaluation.html">Evaluation Guide</a>.
 *
 * @since 1.0
 */
package models.ml.Model;