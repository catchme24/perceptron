package network.layer;

import function.ActivationFunc;
import function.ReLu;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import util.MatrixUtils;

@Slf4j
class ActivationLayerTest {

    private double[] array;

    @BeforeEach
    public void setup() {
        //Зафиксировать сид, чтобы тесты выдавали однотипные данные!
        MatrixUtils.setSeed(10);
        array = new double[]{0.12315, 0.65323, 1.42235, -2.4324234};
    }

    @Test
    void testIsolatedLayerWithReluFunction() {
        ActivationFunc func = new ReLu();
        ActivationLayer layer = new ActivationLayer(func);

        RealMatrix input = MatrixUtils.createRowFromArray(array).transpose();
        log.info("ВХОДНАЯ МАТРИЦА");
        MatrixUtils.printMatrixTest(input);

        RealMatrix output = layer.propogateForward(input);
        log.info("ВЫХОДНАЯ МАТРИЦА");
        MatrixUtils.printMatrixTest(output);

        Assertions.assertEquals(input.getEntry(0, 0), output.getEntry(0, 0));
        Assertions.assertEquals(input.getEntry(1, 0), output.getEntry(1, 0));
        Assertions.assertEquals(input.getEntry(2, 0), output.getEntry(2, 0));
        Assertions.assertEquals(0, output.getEntry(3, 0));
    }

}