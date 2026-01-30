package models.ml.DatasetHandler.helpers;

public class DatasetSplit {

    public final double[][] train;
    public final double[][] test;

    public DatasetSplit(double[][] train, double[][] test) {
        this.train = train;
        this.test = test;
    }
}
