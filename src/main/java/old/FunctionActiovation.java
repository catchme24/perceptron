package old;

public class FunctionActiovation {
    private final String function;

    public FunctionActiovation(String function) {
        this.function = function.toLowerCase();
    }

    private double tangens(double x) { return ( (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)) ); }

    private double tangensDeriv(double x) { return 1 - Math.pow(2, tangens(x)); }
    private double logistic(double x) {

        return 1.0/(1 + Math.exp(-x));
    }

    private double logisticDeriv(double x) { return logistic(x) * (1 - logistic(x)); }

    private double sin(double x) {
        return Math.sin(Math.PI * x / 180);
    }

    private double sinDeriv(double x) { return Math.cos(Math.PI * x / 180); }

    private double ReLu(double x) {
        if (x > 0) {
            return x;
        } else {
            return 0;
        }
    }

    private double ReLuDeriv(double x) {
        if (x > 0) {
            return 1;
        } else {
            return 0;
        }
    }

    private double LeakyReLu(double x) {
        if (x > 0) {
            return x;
        } else {
            return 0.1 * x;
        }
    }

    private double LeakyReLuDeriv(double x) {
        if (x > 0) {
            return 1;
        } else {
            return 0.1;
        }
    }
    public double calculate(double x) {
        switch (function) {
            case "log":
                return logistic(x);
            case "sin":
                return sin(x);
            case "tan":
                return tangens(x);
            case "relu":
                return ReLu(x);
            case "leakyrelu":
                return LeakyReLu(x);
        }

        return 0;
    }

    public double calculateDeriv(double x) {
        switch (function) {
            case "log":
                return logisticDeriv(x);
            case "sin":
                return sinDeriv(x);
            case "tan":
                return tangensDeriv(x);
            case "relu":
                return ReLuDeriv(x);
            case "leakyrelu":
                return LeakyReLuDeriv(x);
        }

        return 0;
    }
}
