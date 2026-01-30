package models.ml.DatasetHandler.helpers;

import java.util.Map;

public class Dataset {

    public final double[][] data;
    public final String[] header;
    public final Map<String, Integer> labelToIndex;

    public Dataset(double[][] data,
            String[] header,
            Map<String, Integer> labelToIndex) {

        this.data = data;
        this.header = header;
        this.labelToIndex = labelToIndex;
    }

    public int rows() {
        return data.length;
    }

    public int cols() {
        int cols = 0;
        if (data.length > 0) {
            cols = data[0].length;
        }
        return cols;
    }
}
