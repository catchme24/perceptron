package network.layer;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import util.MatrixUtils;

@Slf4j
public class FullyConnectedTest {

    private double[] array;

    @BeforeEach
    public void setup() {
        //Зафиксировать сид, чтобы тесты выдавали однотипные данные!
        MatrixUtils.setSeed(10);
        array = new double[]{0.12315, 0.65323, 1.42235, -2.4324234};
    }

    @Test
    public void testIsolatedLayerWithoutInput() {
        int neurousCount = 4;
        FullyConnected layer = new FullyConnected(neurousCount);

        RealMatrix input = MatrixUtils.createRowFromArray(array).transpose();
        log.info("ВХОДНАЯ МАТРИЦА");
        MatrixUtils.printMatrixTest(input);

        RealMatrix output = layer.propogateForward(input);
        log.info("ВЫХОДНАЯ МАТРИЦА");
        MatrixUtils.printMatrixTest(output);
    }

    @Test
    public void testIsolatedLayerWithInput() {
        int neurousCount = 5;
        int inputsCount = 4;
        FullyConnected layer = new FullyConnected(neurousCount, inputsCount);

        RealMatrix input = MatrixUtils.createRowFromArray(array).transpose();
        log.info("ВХОДНАЯ МАТРИЦА");
        MatrixUtils.printMatrixTest(input);

        RealMatrix output = layer.propogateForward(input);
        log.info("ВЫХОДНАЯ МАТРИЦА");
        MatrixUtils.printMatrixTest(output);
    }
}
