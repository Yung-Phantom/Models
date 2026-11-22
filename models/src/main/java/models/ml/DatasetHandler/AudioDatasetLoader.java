package models.ml.DatasetHandler;

import models.ml.DatasetHandler.helpers.ClassFolderScanner;
import models.ml.DatasetHandler.helpers.LabeledDataset;
import models.ml.DatasetHandler.helpers.AudioDecoder;
import models.ml.Preprocessing.Audio.AudioData;

import java.nio.file.*;
import java.util.*;

/**
 * Loads audio files from a folder-per-class directory structure.
 * <p>
 * Expected layout:
 * <pre>
 * root/
 *   classA/
 *     audio1.wav
 *     audio2.wav
 *   classB/
 *     audio3.wav
 * </pre>
 * <p>
 * Supported extension (case-insensitive): .wav
 * Hidden files (starting with '.') and hidden directories (starting with '.') and nested subdirectories are ignored.
 * Files with .wav extension that cannot be decoded throw an exception.
 * <p>
 * Deterministic ordering: classes sorted by name (String.compareTo), files sorted by name (String.compareTo) within a class.
 * Not thread-safe.
 */
public final class AudioDatasetLoader {

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of("wav");

    private AudioDatasetLoader() {}

    /**
     * Loads all audio files from a root directory with class subdirectories.
     *
     * @param rootDirectory root directory containing class subdirectories; must not be null
     * @return LabeledDataset of AudioData
     * @throws IllegalArgumentException if root is null, not a directory, no class subdirectories,
     *         or no audio files found
     */
    public static LabeledDataset<AudioData> load(Path rootDirectory) {
        ClassFolderScanner.ScanResult scan = ClassFolderScanner.scan(rootDirectory, SUPPORTED_EXTENSIONS);

        List<AudioData> items = new ArrayList<>(scan.files.size());
        for (Path audioFile : scan.files) {
            AudioData audioData = AudioDecoder.decode(audioFile);
            items.add(audioData);
        }

        return new LabeledDataset<>(items, scan.labels, scan.sources);
    }
}