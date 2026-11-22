package models.ml.DatasetHandler;

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
        if (rootDirectory == null) {
            throw new IllegalArgumentException("rootDirectory cannot be null");
        }
        if (!Files.isDirectory(rootDirectory)) {
            throw new IllegalArgumentException("rootDirectory is not a directory: " + rootDirectory);
        }

        List<Path> classDirs = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(rootDirectory)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    String dirName = entry.getFileName().toString();
                    // Skip hidden directories
                    if (dirName.startsWith(".")) {
                        continue;
                    }
                    classDirs.add(entry);
                }
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to read root directory: " + rootDirectory, e);
        }

        if (classDirs.isEmpty()) {
            throw new IllegalArgumentException("no class subdirectories found in: " + rootDirectory);
        }

        // Sort class directories by name (String.compareTo) for deterministic ordering
        classDirs.sort(Comparator.comparing(p -> p.getFileName().toString()));

        List<ImageData> images = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        List<String> sources = new ArrayList<>();

        for (Path classDir : classDirs) {
            String className = classDir.getFileName().toString();
            List<Path> imageFiles = new ArrayList<>();

            try (DirectoryStream<Path> stream = Files.newDirectoryStream(classDir)) {
                for (Path entry : stream) {
                    if (!Files.isRegularFile(entry)) {
                        continue;
                    }
                    String fileName = entry.getFileName().toString();
                    // Skip hidden files
                    if (fileName.startsWith(".")) {
                        continue;
                    }
                    // Check extension
                    String ext = getExtension(fileName);
                    if (ext == null || !SUPPORTED_EXTENSIONS.contains(ext.toLowerCase())) {
                        continue;
                    }
                    imageFiles.add(entry);
                }
            } catch (IOException e) {
                throw new IllegalArgumentException("Failed to read class directory: " + classDir, e);
            }

            // Sort files by name (String.compareTo) for deterministic ordering
            imageFiles.sort(Comparator.comparing(p -> p.getFileName().toString()));

            if (imageFiles.isEmpty()) {
                continue; // Skip empty class directories
            }

            for (Path imageFile : imageFiles) {
                ImageData imageData = ImageDecoder.decode(imageFile);
                images.add(imageData);
                labels.add(className);
                sources.add(imageFile.toString());
            }
        }

        if (images.isEmpty()) {
            throw new IllegalArgumentException("no images found in any class subdirectory of: " + rootDirectory);
        }

        return new ImageDataset(images, labels, sources);
    }

    private static String getExtension(String fileName) {
        int dot = fileName.lastIndexOf('.');
        if (dot <= 0 || dot == fileName.length() - 1) {
            return null;
        }
        return fileName.substring(dot + 1);
    }
}