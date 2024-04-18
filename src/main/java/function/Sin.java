package function;

public class Sin implements ActivationFunc {
    @Override
    public double calculate(double x) {
        return Math.sin(Math.PI * x / 180);
    }

    @Override
    public double calculateDerivation(double x) {
        return Math.cos(Math.PI * x / 180);
    }
}
