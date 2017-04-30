package wandell.ml;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;



public class Main {

    public static void main(String[] args) {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);

        // Create traning data
        INDArray X = Nd4j.create(new float[][]{
                {3.0f, 5.0f},
                {5.0f, 1.0f},
                {10f, 2f},
                {1f, 24f}
        });

        INDArray max = X.max(0);
        for(int i = 0; i < X.shape()[0]; i++){
            X.getRow(i).divi(max);
        }

        // Create output target
        INDArray y = Nd4j.create(new float[][]{
                {75f}, {82f}, {93f}, {10f}
        });

        y.divi(100);

        NeuralNetwork n = new NeuralNetwork(2, 3, 1);
        n.setMaxCost(0.001f);
        n.setEpochReport(1000);
        n.setVerbose(true);
        n.setLearningRate(0.2f);
        n.trainSgd(X, y);

        // Guess 1 - should be close to 75
        INDArray guess1 = n.guess(X.getRow(0));

        // Guess 2 - should be close to 82
        INDArray guess2 = n.guess(X.getRow(1));

        // Guess 1 - should be close to 75
        INDArray guess3 = n.guess(X.getRow(2));

        // Guess 1 - should be close to 75
        INDArray guess4 = n.guess(X.getRow(3));

        System.out.println(guess1.toString());
        System.out.println(guess2.toString());
        System.out.println(guess3.toString());
        System.out.println(guess4.toString());
    }
}
