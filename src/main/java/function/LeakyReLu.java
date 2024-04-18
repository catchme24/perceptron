package function;

import org.apache.commons.math3.linear.RealMatrix;

public class LeakyReLu implements ActivationFunc {

    @Override
    public double calculate(double x) {
        return x > 0 ? x : 0.1 * x;
    }

    @Override
    public double calculateDerivation(double x) {
        return x > 0 ? 1 : 0.1;
    }
}
