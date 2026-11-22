package models.ml.DatasetHandler.helpers;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.Locale;

/**
 * Utility for scanning folder-per-class directory structures.
 * <p>
 * Not instantiable; all methods are static.
 * Not thread-safe.
 */
public final class ClassFolderScanner {

    private ClassFolderScanner() {}

    /**
     * Scans a root directory for class subdirectories and files with matching extensions.
     *
     * @param rootDirectory root directory containing class subdirectories; must not be null
     * @param extensions    set of allowed extensions (lowercase, without dot); must not be null
     * @return a tuple of [classDirs, classNames, filePaths] where each is a list parallel by class
     * @throws IllegalArgumentException if root is null, not a directory, no class subdirectories,
     *         or no files found
     */
    public static ScanResult scan(Path rootDirectory, Set<String> extensions) {
        if (rootDirectory == null) {
            throw new IllegalArgumentException("rootDirectory cannot be null");
        }
        if (!Files.isDirectory(rootDirectory)) {
            throw new IllegalArgumentException("rootDirectory is not a directory: " + rootDirectory);
        }
        if (extensions == null) {
            throw new IllegalArgumentException("extensions cannot be null");
        }

        List<Path> classDirs = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(rootDirectory)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    String dirName = entry.getFileName().toString();
                    if (dirName.startsWith(".")) {
                        continue; // Skip hidden directories
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

        // Sort by name (String.compareTo)
        classDirs.sort(Comparator.comparing(p -> p.getFileName().toString()));

        List<Path> allFiles = new ArrayList<>();
        List<String> allLabels = new ArrayList<>();
        List<String> allSources = new ArrayList<>();

        for (Path classDir : classDirs) {
            String className = classDir.getFileName().toString();
            List<Path> imageFiles = new ArrayList<>();

            try (DirectoryStream<Path> stream = Files.newDirectoryStream(classDir)) {
                for (Path entry : stream) {
                    if (!Files.isRegularFile(entry)) {
                        continue;
                    }
                    String fileName = entry.getFileName().toString();
                    if (fileName.startsWith(".")) {
                        continue; // Skip hidden files
                    }
                    String ext = getExtension(fileName);
                    if (ext == null || !extensions.contains(ext.toLowerCase(Locale.ROOT))) {
                        continue;
                    }
                    imageFiles.add(entry);
                }
            } catch (IOException e) {
                throw new IllegalArgumentException("Failed to read class directory: " + classDir, e);
            }

            // Sort files by name (String.compareTo)
            imageFiles.sort(Comparator.comparing(p -> p.getFileName().toString()));

            if (imageFiles.isEmpty()) {
                continue; // Skip empty class directories
            }

            for (Path imageFile : imageFiles) {
                allFiles.add(imageFile);
                allLabels.add(className);
                allSources.add(imageFile.toString());
            }
        }

        if (allFiles.isEmpty()) {
            throw new IllegalArgumentException("no files found in any class subdirectory of: " + rootDirectory);
        }

        return new ScanResult(allFiles, allLabels, allSources);
    }

    private static String getExtension(String fileName) {
        int dot = fileName.lastIndexOf('.');
        if (dot <= 0 || dot == fileName.length() - 1) {
            return null;
        }
        return fileName.substring(dot + 1);
    }

    /**
     * Result of a class folder scan.
     */
    public static final class ScanResult {
        public final List<Path> files;
        public final List<String> labels;
        public final List<String> sources;

        public ScanResult(List<Path> files, List<String> labels, List<String> sources) {
            this.files = Collections.unmodifiableList(new ArrayList<>(files));
            this.labels = Collections.unmodifiableList(new ArrayList<>(labels));
            this.sources = Collections.unmodifiableList(new ArrayList<>(sources));
        }
    }
}