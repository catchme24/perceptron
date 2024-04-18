package util;

import data.DatasetItem;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Slf4j
public class MatrixUtils {

    private static final Array2DRowRealMatrix worker = new Array2DRowRealMatrix();

    private static final Random random = new Random();

    private MatrixUtils() {};

    public static void setSeed(long seed) {
        random.setSeed(seed);
    }

    /*
    Заполняет матрицу рандомными значениями в диапазоне [0, 1]
     */
    public static RealMatrix fillRandom(RealMatrix matrix) {
        for(int i = 0; i < matrix.getRowDimension(); i++) {
            for(int j = 0; j < matrix.getColumnDimension(); j++) {
//                matrix.setEntry(i, j, Math.random() * 2 - 1);
                matrix.setEntry(i, j, random.nextDouble() * 2 - 1);
            }
        }
        return matrix;
    }

    public static RealMatrix createInstance(int row, int column) {
        return worker.createMatrix(row, column);
    }

    public static RealMatrix createRowFromArray(double[] row) {
        RealMatrix matrix = worker.createMatrix(1, row.length);
        matrix.setRow(0, row);
        return matrix;
    }

    public static RealMatrix createInstanceFromRow(double[] row) {
        RealMatrix matrix = worker.createMatrix(1, row.length);
        matrix.setRow(0, row);
        return matrix;
    }

    public static void printMatrix(RealMatrix matrix) {
        for (int g = 0; g < matrix.getRowDimension(); g++) {
            log.debug(Arrays.toString(matrix.getRow(g)));
        }
    }

    public static void printMatrixTest(RealMatrix matrix) {
        for (int g = 0; g < matrix.getRowDimension(); g++) {
            log.info(Arrays.toString(matrix.getRow(g)));
        }
    }

    public static void printDatasetItemsTest(List<DatasetItem> items) {
        for (DatasetItem item : items) {
            log.info(Arrays.toString(item.getAttributes()));
        }
    }

    public static RealMatrix createVector(int row) {
        return worker.createMatrix(row, 1);
    }

    public static RealMatrix getGroundTruth(double classNumber, int classesCount) {
        RealMatrix matrix = worker.createMatrix(classesCount, 1);
        matrix.setEntry(Double.valueOf(classNumber).intValue() - 1, 0, 1.0);
        return matrix;
    }
}
