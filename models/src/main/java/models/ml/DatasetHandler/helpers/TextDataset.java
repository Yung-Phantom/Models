package models.ml.DatasetHandler.helpers;

import java.util.List;
import java.util.Map;

public class TextDataset {
    public final List<String> texts;
    public final double[] labels;
    public final Map<String, Integer> labelToIndex;
    public final String[] header;

    public TextDataset(List<String> texts,
                       double[] labels,
                       Map<String, Integer> labelToIndex,
                       String[] header) {
        this.texts = texts;
        this.labels = labels;
        this.labelToIndex = labelToIndex;
        this.header = header;
    }

    public int size() {
        return texts.size();
    }
}