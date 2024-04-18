package util;

import org.apache.commons.math3.linear.RealMatrix;
import org.junit.jupiter.api.Test;

public class MatrixUtilsTest {

    @Test
    public void testGetGroundTruth() {
        double classNumber = 2.0;
        int classesCount = 10;

        RealMatrix result = MatrixUtils.getGroundTruth(classNumber, classesCount);
        MatrixUtils.printMatrixTest(result);
    }
}
