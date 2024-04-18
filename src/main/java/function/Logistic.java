package function;

import org.apache.commons.math3.linear.RealMatrix;

public class Logistic implements ActivationFunc {

    @Override
    public double calculate(double x) {
        return 1.0/(1 + Math.exp(-x));
    }

    @Override
    public double calculateDerivation(double x) {
        return calculate(x) * (1 - calculate(x));
    }
}
