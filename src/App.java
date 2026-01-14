import java.util.Map;

import KNN.KNN;

public class App {
    static double[][] dataset = {
            { 5.1, 3.5, 1.0 }, // A i=0, j = 0, 1
            { 4.9, 3.0, 1.0 }, // A i=1, j = 0, 1
            { 6.2, 3.4, 2.0 }, // B
            { 5.9, 3.0, 2.0 }, // B
            { 7.0, 3.2, 3.0 }, // C
            { 6.4, 3.2, 3.0 }, // C
            { 5.0, 3.6, 1.0 }, // A
            { 6.7, 3.1, 2.0 }, // B
            { 6.9, 3.1, 3.0 }, // C
            { 5.5, 2.3, 1.0 }, // A
            { 5.6, 2.7, 1.0 }, // A
            { 6.1, 2.8, 2.0 }, // B
            { 7.1, 2.9, 3.0 }, // C
            { 5.7, 2.9, 1.0 }, // A
            { 6.3, 3.0, 2.0 }, // B
            { 7.3, 3.1, 3.0 }, // C
            { 5.8, 3.1, 1.0 }, // A
            { 6.5, 3.2, 2.0 }, // B
            { 7.5, 3.3, 3.0 }, // C
            { 5.9, 3.3, 1.0 }, // A
            { 6.6, 3.4, 2.0 }, // B
            { 7.6, 3.4, 3.0 }, // C
            { 5.1, 3.7, 1.0 }, // A
            { 6.8, 3.5, 2.0 }, // B
            { 7.7, 3.5, 3.0 }, // C
            { 5.2, 3.8, 1.0 }, // A
            { 7.0, 3.6, 2.0 }, // B
            { 7.8, 3.6, 3.0 }, // C
            { 5.3, 3.9, 1.0 }, // A
            { 7.1, 3.7, 2.0 }, // B
            { 7.9, 3.7, 3.0 }, // C
            { 5.4, 3.10, 1.0 }, // A
            { 7.2, 3.8, 2.0 }, // B
            { 8.0, 3.8, 3.0 }, // C
            { 5.5, 3.11, 1.0 }, // A
            { 7.3, 3.9, 2.0 }, // B
            { 8.1, 3.9, 3.0 }, // C
            { 5.6, 3.12, 1.0 }, // A
            { 7.4, 4.0, 2.0 }, // B
            { 8.2, 4.0, 3.0 }, // C
            { 5.7, 3.13, 1.0 }, // A
            { 7.5, 4.1, 2.0 }, // B
            { 8.3, 4.1, 3.0 }, // C
            { 5.8, 3.14, 1.0 }, // A
            { 7.6, 4.2, 2.0 }, // B
            { 8.4, 4.2, 3.0 }, // C
            { 5.9, 3.15, 1.0 }, // A
            { 7.7, 4.3, 2.0 }, // B
            { 8.5, 4.3, 3.0 }, // C
            { 6.0, 3.16, 1.0 }, // A
            { 7.8, 4.4, 2.0 }, // B
            { 8.6, 4.4, 3.0 }, // C
            { 6.1, 3.17, 1.0 }, // A
            { 7.9, 4.5, 2.0 }, // B
            { 8.7, 4.5, 3.0 }, // C
            { 6.2, 3.18, 1.0 }, // A
            { 8.0, 4.6, 2.0 }, // B
            { 8.8, 4.6, 3.0 }, // C
            { 6.3, 3.19, 1.0 }, // A
            { 8.1, 4.7, 2.0 }, // B
            { 8.9, 4.7, 3.0 }, // C
            { 6.4, 3.20, 1.0 }, // A
            { 8.2, 4.8, 2.0 }, // B
            { 9.0, 4.8, 3.0 }, // C
            { 6.5, 3.21, 1.0 }, // A
            { 8.3, 4.9, 2.0 }, // B
            { 9.1, 4.9, 3.0 }, // C
            { 6.6, 3.22, 1.0 }, // A
            { 8.4, 5.0, 2.0 }, // B
            { 9.2, 5.0, 3.0 }, // C
            { 6.7, 3.23, 1.0 }, // A
            { 8.5, 5.1, 2.0 }, // B
            { 9.3, 5.1, 3.0 }, // C
    };
    static double[][] datasetNeg = {
            // Cluster A (label 1.0)
            { 4.9, 3.0, 1.0 }, { 5.0, 3.1, 1.0 }, { 5.1, 3.2, 1.0 }, { 5.2, 3.3, 1.0 }, { 5.3, 3.4, 1.0 },

            // Cluster B (label 2.0)
            { 6.5, 3.0, 2.0 }, { 6.6, 3.1, 2.0 }, { 6.7, 3.2, 2.0 }, { 6.8, 3.3, 2.0 }, { 6.9, 3.4, 2.0 },

            // Cluster C (label 3.0)
            { 7.5, 3.5, 3.0 }, { 7.6, 3.6, 3.0 }, { 7.7, 3.7, 3.0 }, { 7.8, 3.8, 3.0 }, { 7.9, 3.9, 3.0 },

            // Outlier (label 2.0)
            { 9.3, 5.1, 2.0 }
    };

    /**
     * The main method of the program.
     * 
     * @param args The command line arguments
     * @throws Exception If an exception occurs
     */
    public static void main(String[] args) throws Exception {
        // Create a KNN object with the dataset and the negated version of the dataset
        KNN knn = new KNN(dataset, datasetNeg, "Minkowski", 5);
    
        Map<Integer, double[]> neighbors = knn.getNeighbours(0);
        for (Map.Entry<Integer, double[]> entry : neighbors.entrySet()) {
            System.out.print(entry.getKey() + ": ");
            for (double d : entry.getValue()) {
                System.out.print(d + " ");
            }
            System.out.println();
        }
        // Get probability maps for all query points
        // List<Map<Integer, Double>> probsList = knn.predictAllProbability();
    
        // // Iterate over each query point's probability map
        // for (int i = 0; i < probsList.size(); i++) {
        //     Map<Integer, Double> probs = probsList.get(i);
        //     for (Map.Entry<Integer, Double> e : probs.entrySet()) {
        //         System.out.print(e.getKey() + ": " + e.getValue() + " ");
        //     }
        //     System.out.println();
        // }
    }
    
}
