package models.ml.DatasetHandler.helpers;

import java.util.*;

/**
 * Generic labeled dataset with stratified split capability.
 * <p>
 * Immutable after construction.
 * Not thread-safe.
 *
 * @param <T> the item type
 */
public final class LabeledDataset<T> {

    private final List<T> items;
    private final List<String> labels;
    private final List<String> sources;

    /**
     * Creates a LabeledDataset with validation using default item names ("items"/"item").
     *
     * @param items   list of items; must not be null or contain nulls
     * @param labels  list of labels; must not be null or contain nulls
     * @param sources list of source identifiers; must not be null or contain nulls
     * @throws IllegalArgumentException if lists are null, have different sizes, or contain nulls
     */
    public LabeledDataset(List<T> items, List<String> labels, List<String> sources) {
        this(items, labels, sources, "items", "item");
    }

    /**
     * Creates a LabeledDataset with validation using custom item names for error messages.
     * Label and source names are fixed ("labels"/"label", "sources"/"source").
     *
     * @param items         list of items; must not be null or contain nulls
     * @param labels        list of labels; must not be null or contain nulls
     * @param sources       list of source identifiers; must not be null or contain nulls
     * @param itemsPlural   plural name for items in error messages (e.g., "images")
     * @param itemSingular  singular name for item in error messages (e.g., "image")
     * @throws IllegalArgumentException if lists are null, have different sizes, or contain nulls
     */
    public LabeledDataset(List<T> items, List<String> labels, List<String> sources,
                          String itemsPlural, String itemSingular) {
        if (items == null) {
            throw new IllegalArgumentException(itemsPlural + " list cannot be null");
        }
        if (labels == null) {
            throw new IllegalArgumentException("labels list cannot be null");
        }
        if (sources == null) {
            throw new IllegalArgumentException("sources list cannot be null");
        }
        if (items.size() != labels.size() || items.size() != sources.size()) {
            throw new IllegalArgumentException(itemsPlural + ", labels, and sources must have the same size");
        }
        for (int i = 0; i < items.size(); i++) {
            if (items.get(i) == null) {
                throw new IllegalArgumentException(itemSingular + " at index " + i + " cannot be null");
            }
            if (labels.get(i) == null) {
                throw new IllegalArgumentException("label at index " + i + " cannot be null");
            }
            if (sources.get(i) == null) {
                throw new IllegalArgumentException("source at index " + i + " cannot be null");
            }
        }
        this.items = Collections.unmodifiableList(new ArrayList<>(items));
        this.labels = Collections.unmodifiableList(new ArrayList<>(labels));
        this.sources = Collections.unmodifiableList(new ArrayList<>(sources));
    }

    /**
     * Returns the number of samples.
     */
    public int size() {
        return items.size();
    }

    /**
     * Returns an unmodifiable view of the items.
     */
    public List<T> getItems() {
        return items;
    }

    /**
     * Returns an unmodifiable view of the labels.
     */
    public List<String> getLabels() {
        return labels;
    }

    /**
     * Returns an unmodifiable view of the sources.
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
    public LabeledDataset<T>[] split(double testFraction, long seed) {
        if (testFraction <= 0.0 || testFraction >= 1.0) {
            throw new IllegalArgumentException("testFraction must be in (0, 1), got " + testFraction);
        }
        if (items.isEmpty()) {
            @SuppressWarnings("unchecked")
            LabeledDataset<T> empty = new LabeledDataset<>(Collections.emptyList(), Collections.emptyList(), Collections.emptyList());
            return new LabeledDataset[] { this, empty };
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

        return new LabeledDataset[] {
            extractSubset(trainIndices),
            extractSubset(testIndices)
        };
    }

    private LabeledDataset<T> extractSubset(List<Integer> indices) {
        List<T> subItems = new ArrayList<>(indices.size());
        List<String> subLabels = new ArrayList<>(indices.size());
        List<String> subSources = new ArrayList<>(indices.size());
        for (int idx : indices) {
            subItems.add(items.get(idx));
            subLabels.add(labels.get(idx));
            subSources.add(sources.get(idx));
        }
        return new LabeledDataset<>(subItems, subLabels, subSources);
    }
}