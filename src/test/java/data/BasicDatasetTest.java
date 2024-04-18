package data;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import util.MatrixUtils;

@Slf4j
public class BasicDatasetTest {

    //15 строк * 0.63 = 9.45, округлит до 10 кол-во в learning data
    @Test
    void autoRoundToBigger() {
        int learnPercent = 63;
        BasicDataset dataset = new BasicDataset(learnPercent);
        int rowsCount = 15;
        int columnsCount = 5;

        RealMatrix matrix = MatrixUtils.createInstance(rowsCount, columnsCount);
        MatrixUtils.fillRandom(matrix);

        for (int i = 0; i < matrix.getRowDimension(); i++) {
            dataset.addRow(matrix.getRow(i));
        }

        MatrixUtils.printDatasetItemsTest(dataset.getLearnData());
        log.info("РАЗДЕЛЕНИЕ");
        MatrixUtils.printDatasetItemsTest(dataset.getTestData());
        Assertions.assertEquals(dataset.getTestData().size() + dataset.getLearnData().size(), rowsCount);
        Assertions.assertEquals(dataset.getLearnPercent(), learnPercent);
    }

    //15 строк * 0.67 = 10.05, округлит до 11 кол-во в learning data
    @Test
    void autoRoundToSmaller() {
        int learnProcent = 67;
        BasicDataset dataset = new BasicDataset(learnProcent);
        int rowsCount = 15;
        int columnsCount = 5;

        RealMatrix matrix = MatrixUtils.createInstance(rowsCount, columnsCount);
        MatrixUtils.fillRandom(matrix);

        for (int i = 0; i < matrix.getRowDimension(); i++) {
            dataset.addRow(matrix.getRow(i));
        }

        MatrixUtils.printDatasetItemsTest(dataset.getLearnData());
        log.info("РАЗДЕЛЕНИЕ");
        MatrixUtils.printDatasetItemsTest(dataset.getTestData());
        Assertions.assertEquals(dataset.getTestData().size() + dataset.getLearnData().size(), rowsCount);
        Assertions.assertEquals(dataset.getLearnPercent(), learnProcent);
    }
}
