package function;

public class ReLu implements ActivationFunc {
    @Override
    public double calculate(double x) {
        return x > 0 ? x : 0;
    }

    @Override
    public double calculateDerivation(double x) {
        return x > 0 ? 1 : 0;
    }
}
