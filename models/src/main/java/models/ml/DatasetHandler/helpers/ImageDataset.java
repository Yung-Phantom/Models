package models.ml.DatasetHandler.helpers;

import models.ml.Preprocessing.Image.ImageData;

import java.util.*;

/**
 * Immutable container for a labeled image dataset.
 * <p>
 * Parallel lists of images, labels, and source file paths.
 * Delegates to {@link LabeledDataset} for core functionality.
 * Not thread-safe; immutable after construction.
 */
public final class ImageDataset {

    private final LabeledDataset<ImageData> delegate;

    /**
     * Creates an ImageDataset with validation.
     *
     * @param images  list of images; must not be null or contain nulls
     * @param labels  list of labels; must not be null or contain nulls
     * @param sources list of source file paths; must not be null or contain nulls
     * @throws IllegalArgumentException if lists are null, have different sizes, or contain nulls
     */
    public ImageDataset(List<ImageData> images, List<String> labels, List<String> sources) {
        this.delegate = new LabeledDataset<>(images, labels, sources,
                "images", "image");
    }

    /**
     * Returns the number of samples.
     */
    public int size() {
        return delegate.size();
    }

    /**
     * Returns an unmodifiable view of the images list.
     */
    public List<ImageData> getImages() {
        return delegate.getItems();
    }

    /**
     * Returns an unmodifiable view of the labels list.
     */
    public List<String> getLabels() {
        return delegate.getLabels();
    }

    /**
     * Returns an unmodifiable view of the sources list.
     */
    public List<String> getSources() {
        return delegate.getSources();
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
        LabeledDataset<ImageData>[] split = delegate.split(testFraction, seed);
        return new ImageDataset[] {
            new ImageDataset(split[0].getItems(), split[0].getLabels(), split[0].getSources()),
            new ImageDataset(split[1].getItems(), split[1].getLabels(), split[1].getSources())
        };
    }
}