/**
 * Numeric feature scaling and imputation.
 * <p>
 * All implement {@link Transformer} with {@code fit()}, {@code transform()}, {@code fitTransform()}, {@code toJson()}/{@code fromJson()}.
 * <ul>
 *   <li>{@link ZScoreScaler} — Zero mean, unit variance (dense only)</li>
 *   <li>{@link MinMaxScaler} — Scale to [0, 1] (dense only)</li>
 *   <li>{@link MaxAbsScaler} — Scale by max absolute value (dense + sparse)</li>
 *   <li>{@link SimpleImputer} — Fill NaN with mean/median/mode/constant (dense only)</li>
 *   <li>{@link StandardScaler} — <b>Warning:</b> Only splits features from labels; does NOT scale.</li>
 * </ul>
 * See <a href="../../../../../../../docs/preprocessing.html">Preprocessing Guide</a>.
 *
 * @since 1.0
 */
package models.ml.Preprocessing.Scaling;