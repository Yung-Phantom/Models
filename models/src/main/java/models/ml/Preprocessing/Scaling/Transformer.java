package models.ml.Preprocessing.Scaling;

public interface Transformer {
    void fit(double[][] X);
    double[][] transform(double[][] X);
    double[] transform(double[] row);
    double[][] fitTransform(double[][] X);
    String toJson();
}