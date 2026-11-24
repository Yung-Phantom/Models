/**
 * Audio data representation and feature extraction.
 * <p>
 * <ul>
 *   <li>{@link AudioData} — Immutable samples in [-1,1]</li>
 *   <li>{@link AudioDecoder} — WAV PCM 8/16-bit mono/stereo → AudioData</li>
 *   <li>{@link AudioOps} — Resample (linear interp, no anti-alias), fix length</li>
 *   <li>{@link Fft} — Radix-2 magnitude spectrum</li>
 *   <li>{@link Windows} — Hann window generation/application</li>
 *   <li>{@link SpectralStatsExtractor} — 8 stats: mean/std RMS, ZCR, centroid, rolloff (dense)</li>
 *   <li>{@link MfccExtractor} — 2×numCoeffs MFCCs (coefficient-major mean+std) (dense)</li>
 *   <li>{@link FrameStats} — Population mean/std utility</li>
 * </ul>
 * <b>Resampling caveat:</b> Linear interpolation only, no anti-aliasing filter.
 * See <a href="../../../../../../../docs/audio.html">Audio Processing Guide</a>.
 *
 * @since 1.0
 */
package models.ml.Preprocessing.Audio;