package function;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.jupiter.api.Test;
import util.MatrixUtils;

@Slf4j
public class ReLuTest {

    @Test
    void testOneValueDerivation() {
        ReLu relu = new ReLu();
        double value = -0.9627782487392131;

        double calculated = relu.calculate(value);
        log.info("result valuet: {}", calculated);
    }

    @Test
    void testVectorDerivation() {
        ReLu relu = new ReLu();
        double[] test = new double[]{-0.9432, 0.1231, 12.123123};

        RealMatrix input = MatrixUtils.createInstanceFromRow(test);
        input = input.transpose();

        log.info("ВХОД");
        MatrixUtils.printMatrixTest(input);

        RealMatrix calculated = relu.calculateDerivation(input);

        log.info("ВЫХОД");
        MatrixUtils.printMatrixTest(calculated);

    }
}
