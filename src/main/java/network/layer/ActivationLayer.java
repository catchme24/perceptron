package network.layer;

import function.ActivationFunc;
import lombok.extern.slf4j.Slf4j;
import network.NetworkConfigException;
import org.apache.commons.math3.linear.RealMatrix;
import util.MatrixUtils;

@Slf4j
public class ActivationLayer implements Layer  {

    private RealMatrix preActivation;

    private RealMatrix postActivation;

    private Layer prev;

    private ActivationFunc activationFunc;

    private int neuronsCount;

    public ActivationLayer(ActivationFunc func) {
        if (func == null) {
            throw new NetworkConfigException("Activation function cannot be null!");
        }
        this.activationFunc = func;
    }

    public void setPrevious(Layer prev) {
        if (prev == null) {
            throw new NetworkConfigException("Prev layer for activation layer cannot be null!");
        }
        this.prev = prev;
        this.neuronsCount = prev.getSize();
        log.debug("ActivationLayer layer: {} prev size", prev.getSize());
        log.debug("ActivationLayer layer: {} size", neuronsCount);
    }

    @Override
    public RealMatrix propogateForward(RealMatrix inputVector) {
        log.debug("ActivationLayer: Start propogateForward with:");
        MatrixUtils.printMatrix(inputVector);
        if (inputVector.getColumnDimension() != 1) {
            throw new NetworkConfigException(   "Input vector has size: " + inputVector.getRowDimension() +
                    "x" + inputVector.getColumnDimension() +
                    ". Count of columns must be 1!"
            );
        }

        preActivation = inputVector.copy();
        //Высчитывает сигнал с оффсетом, если он установлен
        RealMatrix output = activationFunc.calculate(inputVector);
        postActivation = output.copy();
        log.debug("ActivationLayer: End propogateForward with:");
        MatrixUtils.printMatrix(output);
        return output;
    }

    @Override
    public void correctWeights(double learnRate) {

    }

    @Override
    public RealMatrix propogateBackward(RealMatrix errorVector) {
        log.debug("ActivationLayer: Start propogateBackward with error vector:");
        MatrixUtils.printMatrix(errorVector);

        log.debug("ActivationLayer: preActivation vector:");
        MatrixUtils.printMatrix(preActivation);

        RealMatrix derivation = activationFunc.calculateDerivation(preActivation);

        log.debug("ActivationLayer: derivations vector:");
        MatrixUtils.printMatrix(derivation);

        RealMatrix localGradients = derivation.copy();

        for (int i = 0; i < errorVector.getRowDimension(); i++) {
            localGradients.setEntry(i,
                                    0,
                                    errorVector.getEntry(i, 0) * derivation.getEntry(i, 0));
        }
        log.debug("ActivationLayer: End propogateBackward with local gradient:");
        MatrixUtils.printMatrix(localGradients);
        return localGradients;
    }

    @Override
    public int getSize() {
        return neuronsCount;
    }
}
