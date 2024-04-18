package old.big;

import old.DatasetWorker;
import old.FunctionActiovation;
import old.big.NeuroNetwork;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class Test {

    public static void main(String[] args) throws IOException {


        String pathCar = "D:\\Обучение\\Магистратура\\projects\\Нейронные сети\\AI-lab1\\src\\main\\resources\\car.data";
        String pathIris = "D:\\Обучение\\Магистратура\\projects\\Нейронные сети\\AI-lab1\\src\\main\\resources\\iris.data";
        File file = new File(pathIris);
        Array2DRowRealMatrix matrixWorker = new Array2DRowRealMatrix();
        DatasetWorker worker = new DatasetWorker();

        int layers_count = 2;
        int epochStart = 500;
        int epochEnd = 3000;
        int inputs_count = 4;
        int neuronsCountStart = 3;
        int neuronsCountEnd = 10;
        int outputs_count = 3;
        double learnRateStart = 0.1;
        double learnRateEnd = 0.3;
        int percent = 80;

        int trueClasses = 0;
        int max = 0;

        for(int i = neuronsCountStart; i <= neuronsCountEnd; i++) {
            for(int j = epochStart; j <= epochEnd; j = j + 10) {
                for(double k = learnRateStart; k <= learnRateEnd; k = k + 0.05) {
                    NeuroNetwork network = new NeuroNetwork(layers_count, inputs_count, i, outputs_count, matrixWorker, worker);
                    FunctionActiovation functionActiovation = new FunctionActiovation("log");
                    network.setFunctionActiovation(functionActiovation);

                    network.loadDataset(file, percent);
                    network.initializationWeights();
                    network.learning(j, k);
                    trueClasses = network.testing();
                    if(trueClasses > max) {
                        max = trueClasses;
                    }
                    System.out.println("Узлов: " + i + ", эпох: " + j + ", коэф. обучения: " + k + ", правильных классов " + trueClasses);
                }
            }
        }
        System.out.println("Максимальное число правильных классов: " + max);
    }

    public static double logistic(double x) {
        return 1.0/(1 + Math.exp(-x));
    }

    public static double sin(double x) {
        return Math.sin(x);
    }

    public static double logistic_deriv(double x) {
        return logistic(x) * (1 - logistic(x));
    }

    public static RealMatrix fillRandom(RealMatrix matrix) {
        for(int i = 0; i < matrix.getRowDimension(); i++) {
            for(int j = 0; j < matrix.getColumnDimension(); j++) {
                matrix.setEntry(i, j, Math.random() * 2 - 1);
            }
        }
        return matrix;
    }

    public static void printMatrix(RealMatrix matrix) {
        for(int g = 0; g < matrix.getRowDimension(); g++) {
            System.out.println(Arrays.toString(matrix.getRow(g)));
        }
    }

    public static RealMatrix formOutputMatrix(RealMatrix matrix, Array2DRowRealMatrix matrixWorker, int outputs_count) {
        RealMatrix output_vector_2 = matrix.getSubMatrix(0, matrix.getRowDimension() - 1, matrix.getColumnDimension()-1,
                matrix.getColumnDimension()-1);
        output_vector_2 = output_vector_2.transpose();

        RealMatrix output_matrix_2 = matrixWorker.createMatrix(output_vector_2.getColumnDimension(), outputs_count);
        for(int i = 0; i < output_matrix_2.getRowDimension(); i++) {
            for(int j = 1; j <= 4; j++) {
                if(output_vector_2.getEntry(0, i) == j) {
                    output_matrix_2.setEntry(i, j - 1, 1);
                }
            }
        }
        return output_matrix_2;
    }

    public static RealMatrix formOutputMatrix(RealMatrix matrix, Array2DRowRealMatrix matrixWorker, int output_count, boolean Predication) {
        RealMatrix output_vector = matrix.getSubMatrix(0, matrix.getRowDimension() - 1, matrix.getColumnDimension()-1,
                matrix.getColumnDimension()-1);

        return output_vector;
    }
}
