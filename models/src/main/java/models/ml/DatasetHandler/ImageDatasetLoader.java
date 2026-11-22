package models.ml.DatasetHandler;

import models.ml.DatasetHandler.helpers.ClassFolderScanner;
import models.ml.DatasetHandler.helpers.ImageDataset;
import models.ml.DatasetHandler.helpers.ImageDecoder;
import models.ml.Preprocessing.Image.ImageData;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;

/**
 * Loads images from a folder-per-class directory structure.
 * <p>
 * Expected layout:
 * <pre>
 * root/
 *   classA/
 *     img1.jpg
 *     img2.png
 *   classB/
 *     img3.png
 * </pre>
 * <p>
 * Supported extensions (case-insensitive): .png, .jpg, .jpeg, .gif, .bmp
 * Hidden files (starting with '.') and hidden directories (starting with '.') and nested subdirectories are ignored.
 * Files with supported extensions that cannot be decoded throw an exception.
 * <p>
 * Deterministic ordering: classes sorted by name (String.compareTo), files sorted by name (String.compareTo) within a class.
 * Not thread-safe.
 */
public final class ImageDatasetLoader {

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of(
            "png", "jpg", "jpeg", "gif", "bmp"
    );

    private ImageDatasetLoader() {}

    /**
     * Loads all images from a root directory with class subdirectories.
     *
     * @param rootDirectory root directory containing class subdirectories; must not be null
     * @return ImageDataset with all loaded images
     * @throws IllegalArgumentException if root is null, not a directory, no class subdirectories,
     *         or no images found
     */
    public static ImageDataset load(Path rootDirectory) {
        ClassFolderScanner.ScanResult scan = ClassFolderScanner.scan(rootDirectory, SUPPORTED_EXTENSIONS);

        List<ImageData> images = new ArrayList<>(scan.files.size());
        for (Path imageFile : scan.files) {
            ImageData imageData = ImageDecoder.decode(imageFile);
            images.add(imageData);
        }

        return new ImageDataset(images, scan.labels, scan.sources);
    }
}