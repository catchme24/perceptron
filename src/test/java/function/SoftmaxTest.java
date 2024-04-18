package function;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.jupiter.api.Test;
import util.MatrixUtils;

@Slf4j
public class SoftmaxTest {

    @Test
    void test() {
        Softmax softmax = new Softmax();
        int error = 5;

        RealMatrix input = MatrixUtils.createVector(5);
        MatrixUtils.fillRandom(input);

        log.info("ВХОД");
        MatrixUtils.printMatrixTest(input);

        RealMatrix calculated = softmax.calculate(input);

        log.info("ВЫХОД");
        MatrixUtils.printMatrixTest(calculated);

        double result = 0;

        for (int i = 0; i < calculated.getRowDimension(); i++) {
            result += calculated.getEntry(i, 0);
        }

        log.info("СУММА ДОЛЖНА БЫТЬ 1: {}", result);
    }
}
