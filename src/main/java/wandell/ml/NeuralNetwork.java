package wandell.ml;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

public class NeuralNetwork {

    private INDArray W2;
    private INDArray W1;
    private int inSize;
    private int hiddenSize;
    private int outSize;
    private float learningRate = 0.00001f;
    private boolean verbose = false;
    private float tol = 0.001f;
    private int maxEpoc = 0;
    private boolean useMaxEpoc = false;
    private int epocReport = 10;
    private float maxCost = 0.001f;
    private boolean useMaxCost = false;
    private INDArray Z2;
    private INDArray A2;
    private INDArray Z3;
    private INDArray yHat;


    public NeuralNetwork(int inSize, int hiddenSize, int outSize) {
        this.inSize = inSize + 1;
        this.hiddenSize = hiddenSize;
        this.outSize = outSize;

        this.W1 = Nd4j.randn(this.inSize, this.hiddenSize);
        this.W2 = Nd4j.randn(this.hiddenSize, this.outSize);
    }

    private INDArray formatInput(INDArray Xin) {
        INDArray X = Nd4j.create(Xin.shape()[0], Xin.shape()[1] + 1);

        for(int i = 0; i < Xin.shape()[0]; i++){
            INDArray tmp = Xin.getRow(i);
            INDArray tmp1 = Nd4j.ones(1, tmp.shape()[1] + 1);
            for(int j = 0; j < tmp.shape()[1]; j++){
                tmp1.put(j, tmp.getColumn(j));
            }
            X.putRow(i, tmp1);
        }
        return X;
    }

    public void trainSgd(INDArray Xin, INDArray y) {
        INDArray X = this.formatInput(Xin);

        float alpha = this.learningRate;
        int epoch = 0;
        Random r = new Random();

        while(!this.useMaxEpoc || epoch < this.maxEpoc) {
            INDArray s = Nd4j.arange(X.shape()[0]);
            Nd4j.shuffle(s, 0);

            for(int i = 0; i < s.shape()[1]; i++){
                int learn = s.getInt(i);
                INDArray[] djdw1_djdw2 = this.costFunctionPrime(X.getRow(learn), y.getRow(learn));
                INDArray[] plainParams = this.getPlainParams();

                int[] oShape = djdw1_djdw2[0].shape();
                INDArray t1 = djdw1_djdw2[0].ravel();
                t1.muli(alpha);
                t1 = t1.reshape(oShape);
                plainParams[0].subi(t1);

                int[] oShape1 = djdw1_djdw2[1].shape();
                INDArray t2 = djdw1_djdw2[1].ravel();
                t2.muli(alpha);
                t2 = t2.reshape(oShape1);
                plainParams[1].subi(t2);

                this.setPlainParams(plainParams);
            }

            epoch += 1;
            INDArray cost = this.costFunction(X, y);
            float sum = cost.sumNumber().floatValue();

            if(this.verbose && epoch % this.epocReport == 0) {
                System.out.println("Epoch: " + Integer.toString(epoch) + " Cost: " + Float.toString(sum));
            }

            if(this.useMaxCost && sum < this.maxCost){
                if(this.verbose){
                    System.out.println("Cost level: " + Float.toString(this.maxCost) + " reached");
                    break;
                }
            }
        }

    }

    private void setPlainParams(INDArray[] plainParams) {
        this.W1 = plainParams[0];
        this.W2 = plainParams[1];
    }

    private INDArray[] getPlainParams() {
        return new INDArray[]{this.W1, this.W2};
    }

    public INDArray guess(INDArray Xin) {
        INDArray X = this.formatInput(Xin);
        return this.forward(X);
    }

    private INDArray forward(INDArray X) {
        this.Z2 = X.mmul(this.W1);
        this.A2 = this.mySigmoid(this.Z2.add(1));
        this.Z3 = this.A2.mmul(this.W2);
        return this.mySigmoid(this.Z3.add(1));
    }

    private INDArray costFunction(INDArray X, INDArray y) {
        this.yHat = this.forward(X);
        INDArray diff = y.sub(this.yHat);
        INDArray t = diff.transpose();
        INDArray squared = t.mmul(diff);
        return squared.div(X.shape()[1]);
    }

    private INDArray[] costFunctionPrime(INDArray X, INDArray y) {
        this.yHat = this.forward(X);
        INDArray ymyh = y.sub(this.yHat);
        INDArray rsub = ymyh.mul(-1);
        INDArray delta3 = this.sigmoidPrime(this.Z3);
        delta3.muli(rsub);
        INDArray djdw2 = this.A2.transpose().mmul(delta3);
        INDArray d3mw2 = delta3.mmul(this.W2.transpose());
        INDArray sig = this.sigmoidPrime(this.Z2);
        INDArray delta2 = d3mw2.transpose().mmul(sig);
        INDArray xtrans = X.transpose().reshape(delta2.rows(), 1);
        INDArray djdw1 = xtrans.mmul(delta2);
        return new INDArray[]{djdw1, djdw2};
    }

    private INDArray mySigmoid(INDArray z) {
        return sigmoid(z);
    }

    private INDArray sigmoidPrime(INDArray z) {
        INDArray sig = this.mySigmoid(z);
        INDArray oneMinus = sig.rsub(1.0f);
        INDArray mul = sig.mul(oneMinus);
        return mul;
    }

    public void setMaxCost(float maxCost) {
        this.maxCost = maxCost;
        this.useMaxCost = true;
    }

    public void setEpochReport(int epocReport) {
        this.epocReport = epocReport;
    }

    public void setMaxEpoc(int maxEpoc) {
        this.maxEpoc = maxEpoc;
        this.useMaxEpoc = true;
    }

    public void setTol(float tol) {
        this.tol = tol;
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("\nNeuralNetwork:");
        sb.append("\n    Input Size: " + Integer.toString(this.inSize));
        sb.append("\n    Hidden Size: " + Integer.toString(this.hiddenSize));
        sb.append("\n    Out Size: " + Integer.toString(this.outSize));
        sb.append("\n    Weights 1: \n" + this.W1.toString());
        sb.append("\n    Weights 2: \n" + this.W2.toString());

        return sb.toString();
    }
}
