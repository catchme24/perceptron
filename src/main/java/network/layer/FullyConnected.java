package network.layer;

import lombok.extern.slf4j.Slf4j;
import network.NetworkConfigException;
import org.apache.commons.math3.linear.RealMatrix;
import util.MatrixUtils;

@Slf4j
public class FullyConnected implements Layer {
    private RealMatrix offset;
    private RealMatrix offsetGradient;
    private RealMatrix weights;
    private RealMatrix weightsGradient;

    private RealMatrix preActivation;

    private RealMatrix postActivation;

    private Layer prev;

    private int neuronsCount;

    private int inputsCount = 0;

    public FullyConnected(int neuronsCount) {
        this.neuronsCount = neuronsCount;
        this.offset = MatrixUtils.createInstance(neuronsCount, 1);
        MatrixUtils.fillRandom(offset);
    }

    public FullyConnected(int neuronsCount, int inputsCount) {
        this(neuronsCount);
        this.inputsCount = inputsCount;
        weights = MatrixUtils.createInstance(inputsCount, neuronsCount);
        MatrixUtils.fillRandom(weights);
    }

    public void setPrevious(Layer prev) {
        if (inputsCount <= 0) {
            if (prev == null) {
                throw new NetworkConfigException("Prev layer for fully connected cannot be null!");
            }
            this.prev = prev;
            weights = MatrixUtils.createInstance(prev.getSize(), neuronsCount);
            log.debug("FullyConnected layer: {} prev size", prev.getSize());
            log.debug("FullyConnected layer: {} size", neuronsCount);
        } else {
            log.debug("FullyConnected layer: {} prev size", inputsCount);
            log.debug("FullyConnected layer: {} size", neuronsCount);
        }
        MatrixUtils.fillRandom(weights);
    }

    @Override
    public RealMatrix propogateForward(RealMatrix inputVector) {
        log.debug("FullyConneted: Start propogateForward with:");
        MatrixUtils.printMatrix(inputVector);
        if (inputVector.getColumnDimension() != 1) {
            throw new NetworkConfigException(   "Input vector has size: " + inputVector.getRowDimension() +
                    "x" + inputVector.getColumnDimension() +
                    ". Count of columns must be 1!"
            );
        }

        if (weights == null) {
            throw new NetworkConfigException("First FullyConnected layer must have inputs count!");
        }

        log.debug("МАТРИЦА ВЕСОВ");
        MatrixUtils.printMatrix(weights);
        log.debug("ТРАНСПОНИРОВАННАЯ МАТРИЦА ВЕСОВ");
        MatrixUtils.printMatrix(weights.transpose());
        log.debug("МАТРИЦА СМЕЩЕНИЯ");
        MatrixUtils.printMatrix(offset);

        preActivation = inputVector.copy();
        //Высчитывает сигнал с оффсетом
        RealMatrix output = weights.transpose().multiply(inputVector).add(offset);
        postActivation = output.copy();
        log.debug("FullyConneted: End propogateForward with:");
        MatrixUtils.printMatrix(output);
        return output;
    }


    @Override
    public RealMatrix propogateBackward(RealMatrix localGradients) {
        log.debug("FullyConneted: Start propogateBackward with local gradient activation layer:");
        MatrixUtils.printMatrix(localGradients);

        log.debug("FullyConneted: preActivation:");
        MatrixUtils.printMatrix(preActivation.transpose());

        offsetGradient = localGradients;
        log.debug("FullyConneted: offsetGradient:");
        MatrixUtils.printMatrix(offsetGradient);

        weightsGradient = localGradients.multiply(preActivation.transpose());
        log.debug("FullyConneted: weightsGradient:");
        MatrixUtils.printMatrix(weightsGradient);

        RealMatrix errorVector = weights.multiply(localGradients);
        log.debug("FullyConneted: End propogateBackward with error vector:");
        MatrixUtils.printMatrix(errorVector);
        return errorVector;
    }

    @Override
    public void correctWeights(double learnRate) {
        weights = weights.subtract(weightsGradient.transpose().scalarMultiply(learnRate));
        offset = offset.subtract(offsetGradient.scalarMultiply(learnRate));
    }


    @Override
    public int getSize() {
        return neuronsCount;
    }
}
