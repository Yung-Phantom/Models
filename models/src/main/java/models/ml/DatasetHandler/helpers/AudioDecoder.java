package models.ml.DatasetHandler.helpers;

import models.ml.Preprocessing.Audio.AudioData;

import javax.sound.sampled.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Decodes WAV audio files into {@link AudioData} using {@link AudioSystem}.
 * <p>
 * Supports PCM 8-bit unsigned and 16-bit signed little-endian, mono or stereo.
 * Stereo channels are averaged to mono.
 * <p>
 * Not instantiable; all methods are static.
 */
public final class AudioDecoder {

    private AudioDecoder() {}

    /**
     * Reads a WAV file and returns it as {@link AudioData} with samples in [-1.0, 1.0].
     *
     * @param file path to WAV file; must not be null
     * @return AudioData with decoded samples
     * @throws IllegalArgumentException if file is null, not a regular file,
     *         not a WAV file, or uses an unsupported encoding
     */
    public static AudioData decode(Path file) {
        if (file == null) {
            throw new IllegalArgumentException("file path cannot be null");
        }
        if (!Files.isRegularFile(file)) {
            throw new IllegalArgumentException("file does not exist or is not a regular file: " + file);
        }
        File jfile = file.toFile();
        // Validate file format
        AudioFileFormat fileFormat;
        try {
            fileFormat = AudioSystem.getAudioFileFormat(jfile);
        } catch (UnsupportedAudioFileException e) {
            throw new IllegalArgumentException("file is not a valid WAV file: " + file, e);
        } catch (IOException e) {
            throw new IllegalArgumentException("failed to read audio file: " + file, e);
        }
        if (fileFormat.getType() != AudioFileFormat.Type.WAVE) {
            throw new IllegalArgumentException("file is not a WAV file: " + file);
        }

        AudioFormat format = fileFormat.getFormat();

        // Check supported encodings: only PCM_SIGNED 16-bit little-endian and PCM_UNSIGNED 8-bit
        if (format.getEncoding() == AudioFormat.Encoding.PCM_SIGNED) {
            if (format.getSampleSizeInBits() != 16 || format.isBigEndian()) {
                throw new IllegalArgumentException(
                        "unsupported encoding: only PCM_SIGNED 16-bit little-endian supported, got " + format);
            }
        } else if (format.getEncoding() == AudioFormat.Encoding.PCM_UNSIGNED) {
            if (format.getSampleSizeInBits() != 8) {
                throw new IllegalArgumentException(
                        "unsupported encoding: only PCM_UNSIGNED 8-bit supported, got " + format);
            }
        } else {
            throw new IllegalArgumentException(
                    "unsupported encoding " + format.getEncoding() + " in file: " + file);
        }
        if (format.getChannels() > 2) {
            throw new IllegalArgumentException(
                    "unsupported channel count " + format.getChannels() + " in file: " + file);
        }

        // Read all bytes using try-with-resources
        byte[] bytes;
        try (AudioInputStream ais = AudioSystem.getAudioInputStream(jfile)) {
            bytes = ais.readAllBytes();
        } catch (UnsupportedAudioFileException e) {
            throw new IllegalArgumentException("file is not a valid WAV file: " + file, e);
        } catch (IOException e) {
            throw new IllegalArgumentException("failed to read audio file: " + file, e);
        }

        int sampleSize = format.getSampleSizeInBits();
        int channels = format.getChannels();
        int frameSize = format.getFrameSize();
        int numFrames = bytes.length / frameSize;

        double[] samples = new double[numFrames];

        if (sampleSize == 16) {
            // PCM_SIGNED 16-bit little-endian: values in [-32768, 32767]
            for (int i = 0; i < numFrames; i++) {
                int offset = i * frameSize;
                // Read as unsigned 16-bit, then convert to signed
                int sample = ((bytes[offset + 1] & 0xFF) << 8) | (bytes[offset] & 0xFF);
                if (sample >= 32768) {
                    sample -= 65536; // Convert to signed 16-bit
                }
                if (channels == 2) {
                    int sampleR = ((bytes[offset + 3] & 0xFF) << 8) | (bytes[offset + 2] & 0xFF);
                    if (sampleR >= 32768) {
                        sampleR -= 65536;
                    }
                    samples[i] = ((sample + sampleR) / 2.0) / 32768.0;
                } else {
                    samples[i] = sample / 32768.0;
                }
            }
        } else if (sampleSize == 8) {
            // PCM_UNSIGNED 8-bit
            for (int i = 0; i < numFrames; i++) {
                int offset = i * channels;
                int sample = bytes[offset] & 0xFF;
                if (channels == 2) {
                    int sampleR = bytes[offset + 1] & 0xFF;
                    samples[i] = ((sample + sampleR) / 2.0 - 128.0) / 128.0;
                } else {
                    samples[i] = (sample - 128.0) / 128.0;
                }
            }
        } else {
            throw new IllegalArgumentException(
                    "unsupported sample size " + sampleSize + " in file: " + file);
        }

        return AudioData.of((int) format.getSampleRate(), samples);
    }
}