package models.ml.Preprocessing.Scaling;

public class StandardScaler {
    public double[][] extractFeatures(double[][] dataset) {
        if (dataset == null) {
            throw new NullPointerException("dataset is null");
        }
        // Legacy behavior: throw NPE for null rows
        for (double[] row : dataset) {
            if (row == null) {
                throw new NullPointerException("row is null");
            }
        }
        return FeatureLabelSplitter.extractFeatures(dataset);
    }

    public double[] extractLabels(double[][] dataset) {
        if (dataset == null) {
            throw new NullPointerException("dataset is null");
        }
        // Legacy behavior: throw NPE for null rows
        for (double[] row : dataset) {
            if (row == null) {
                throw new NullPointerException("row is null");
            }
        }
        return FeatureLabelSplitter.extractLabels(dataset);
    }

    public double[] extractLabelsRegression(double[][] dataset) {
        if (dataset == null) {
            throw new NullPointerException("dataset is null");
        }
        // Legacy behavior: throw NPE for null rows
        for (double[] row : dataset) {
            if (row == null) {
                throw new NullPointerException("row is null");
            }
        }
        return FeatureLabelSplitter.extractLabelsRegression(dataset);
    }
}
