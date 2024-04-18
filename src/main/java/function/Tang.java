package function;

public class Tang implements ActivationFunc {

    @Override
    public double calculate(double x) {
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
    }

    @Override
    public double calculateDerivation(double x) {
        return 1 - Math.pow(2, calculate(x));
    }
}
