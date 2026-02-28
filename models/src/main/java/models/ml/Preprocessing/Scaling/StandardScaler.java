package models.ml.Preprocessing.Scaling;

public class StandardScaler {
    public double[][] extractFeatures(double[][] dataset) {
        if (dataset[0].length < 2)
            throw new IllegalArgumentException("Dataset must have at least one feature and one label column.");

        double[][] datasetIn = new double[dataset.length][dataset[0].length - 1];
        for (int i = 0; i < dataset.length; i++) {
            System.arraycopy(dataset[i], 0, datasetIn[i], 0, dataset[0].length - 1);
        }
        return datasetIn;
    }
    public double[] extractLabels(double[][] dataset) {
        if (dataset[0].length < 2)
            throw new IllegalArgumentException("Dataset must have at least one feature and one label column.");

        double[] labels = new double[dataset.length];
        for (int i = 0; i < dataset.length; i++) {
            labels[i] = dataset[i][dataset[0].length - 1];
            if (labels[i] != (int) labels[i])
                throw new IllegalArgumentException("Labels must be integers");
        }
        return labels;
    }
}
