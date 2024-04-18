package function;

import org.apache.commons.math3.linear.RealMatrix;

public interface ActivationFunc {

    default RealMatrix calculate(RealMatrix x) {
        RealMatrix result = x.copy();
        for (int i = 0; i < result.getRowDimension(); i++) {
            result.setEntry(i, 0, calculate(x.getEntry(i, 0)));
        }
        return result;
    };

    default RealMatrix calculateDerivation(RealMatrix x) {
        RealMatrix result = x.copy();
        for (int i = 0; i < result.getRowDimension(); i++) {
            double calculated = calculateDerivation(x.getEntry(i, 0));
//            System.out.println(calculated);
            result.setEntry(i, 0, calculated);
        }
        return result;
    };

    double calculate(double x);
    double calculateDerivation(double x);
}
