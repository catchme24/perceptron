package function;

import org.apache.commons.math3.linear.RealMatrix;

public class Softmax implements ActivationFunc {

    @Override
    public RealMatrix calculate(RealMatrix x) {
        RealMatrix result = x.copy();
        double summ = 0;
        double[] vector = x.getColumn(0);
        for (int i = 0; i < vector.length; i++) {
            summ = summ + Math.exp(vector[i]);
        }
        for (int i = 0; i < result.getRowDimension(); i++) {
            result.setEntry(i, 0, Math.exp(result.getEntry(i, 0)) / summ);
        }
        return result;
    }

    @Override
    public RealMatrix calculateDerivation(RealMatrix x) {
        return ActivationFunc.super.calculateDerivation(x);
    }

    @Override
    public double calculate(double x) {
        return 0;
    }

    @Override
    public double calculateDerivation(double x) {
        return 0;
    }
}
