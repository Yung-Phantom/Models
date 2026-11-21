package models.ml.DatasetHandler.helpers;

import models.ml.Preprocessing.Image.ImageData;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Immutable container for a labeled image dataset.
 * <p>
 * Parallel lists of images, labels, and source file paths.
 * Not thread-safe; immutable after construction.
 */
public final class ImageDataset {

    private final List<ImageData> images;
    private final List<String> labels;
    private final List<String> sources;

    /**
     * Creates an ImageDataset with validation.
     *
     * @param images  list of images; must not be null or contain nulls
     * @param labels  list of labels; must not be null or contain nulls
     * @param sources list of source file paths; must not be null or contain nulls
     * @throws IllegalArgumentException if lists are null, have different sizes, or contain nulls
     */
    public ImageDataset(List<ImageData> images, List<String> labels, List<String> sources) {
        if (images == null) {
            throw new IllegalArgumentException("images list cannot be null");
        }
        if (labels == null) {
            throw new IllegalArgumentException("labels list cannot be null");
        }
        if (sources == null) {
            throw new IllegalArgumentException("sources list cannot be null");
        }
        if (images.size() != labels.size() || images.size() != sources.size()) {
            throw new IllegalArgumentException("images, labels, and sources must have the same size");
        }
        for (int i = 0; i < images.size(); i++) {
            if (images.get(i) == null) {
                throw new IllegalArgumentException("image at index " + i + " cannot be null");
            }
            if (labels.get(i) == null) {
                throw new IllegalArgumentException("label at index " + i + " cannot be null");
            }
            if (sources.get(i) == null) {
                throw new IllegalArgumentException("source at index " + i + " cannot be null");
            }
        }
        // Defensive copies into unmodifiable lists
        this.images = Collections.unmodifiableList(new ArrayList<>(images));
        this.labels = Collections.unmodifiableList(new ArrayList<>(labels));
        this.sources = Collections.unmodifiableList(new ArrayList<>(sources));
    }

    /**
     * Returns the number of samples.
     */
    public int size() {
        return images.size();
    }

    /**
     * Returns an unmodifiable view of the images list.
     */
    public List<ImageData> getImages() {
        return images;
    }

    /**
     * Returns an unmodifiable view of the labels list.
     */
    public List<String> getLabels() {
        return labels;
    }

    /**
     * Returns an unmodifiable view of the sources list.
     */
    public List<String> getSources() {
        return sources;
    }

    /**
     * Stratified split into train and test sets.
     *
     * @param testFraction fraction of data to use for test; must be in (0, 1)
     * @param seed         random seed for deterministic shuffling
     * @return array of length 2: [train, test]
     * @throws IllegalArgumentException if testFraction not in (0, 1)
     */
    public ImageDataset[] split(double testFraction, long seed) {
        if (testFraction <= 0.0 || testFraction >= 1.0) {
            throw new IllegalArgumentException("testFraction must be in (0, 1), got " + testFraction);
        }
        if (images.isEmpty()) {
            return new ImageDataset[] { this, new ImageDataset(Collections.emptyList(), Collections.emptyList(), Collections.emptyList()) };
        }

        // Group indices by label
        Map<String, List<Integer>> labelToIndices = new LinkedHashMap<>();
        for (int i = 0; i < labels.size(); i++) {
            labelToIndices.computeIfAbsent(labels.get(i), k -> new ArrayList<>()).add(i);
        }

        // Deterministic RNG per seed
        Random rng = new Random(seed);
        List<Integer> trainIndices = new ArrayList<>();
        List<Integer> testIndices = new ArrayList<>();

        for (List<Integer> indices : labelToIndices.values()) {
            if (indices.size() < 2) {
                // Class with < 2 samples goes entirely to train
                trainIndices.addAll(indices);
                continue;
            }
            // Shuffle a copy
            List<Integer> shuffled = new ArrayList<>(indices);
            Collections.shuffle(shuffled, rng);
            int testCount = (int) Math.round(indices.size() * testFraction);
            // Ensure at least 1 in each
            testCount = Math.max(1, Math.min(testCount, indices.size() - 1));
            testIndices.addAll(shuffled.subList(0, testCount));
            trainIndices.addAll(shuffled.subList(testCount, indices.size()));
        }

        // Sort for deterministic output
        trainIndices.sort(Integer::compareTo);
        testIndices.sort(Integer::compareTo);

        return new ImageDataset[] {
            extractSubset(trainIndices),
            extractSubset(testIndices)
        };
    }

    private ImageDataset extractSubset(List<Integer> indices) {
        List<ImageData> subImages = new ArrayList<>(indices.size());
        List<String> subLabels = new ArrayList<>(indices.size());
        List<String> subSources = new ArrayList<>(indices.size());
        for (int idx : indices) {
            subImages.add(images.get(idx));
            subLabels.add(labels.get(idx));
            subSources.add(sources.get(idx));
        }
        return new ImageDataset(subImages, subLabels, subSources);
    }
}