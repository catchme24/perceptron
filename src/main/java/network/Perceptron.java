package network;

import data.Dataset;
import data.DatasetItem;
import lombok.extern.slf4j.Slf4j;
import network.layer.Layer;
import org.apache.commons.math3.linear.RealMatrix;
import util.MatrixUtils;

import java.util.Deque;
import java.util.Iterator;
import java.util.List;

@Slf4j
public class Perceptron implements Network {

    private Deque<Layer> layers;

    Perceptron(Deque<Layer> layers) {
        this.layers = layers;
        Layer prev = null;
        for (Layer layer: layers) {
            layer.setPrevious(prev);
            prev = layer;
        }
    }


    public void learn(int epoch, double learnRate, Dataset dataset) {
        List<DatasetItem> learn = dataset.getLearnData();
        double rmsePrev = 0;
        int selectionCount = learn.size();

        for (int epochIndex = 0; epochIndex < epoch; epochIndex++) {
            //Среднеквадратичная ошибка - целевая функция, считается в конце каждой эпохи и должна уменьшаться от эпохи к эпохе
            double rmse = 0;

            //Цикл выборок
            for (int selectionIndex = 0; selectionIndex < selectionCount; selectionIndex++) {
                //ПРЯМОЕ РАСПРОСТРОНЕНИЕ
                RealMatrix networkInput = MatrixUtils.createInstanceFromRow(learn.get(selectionIndex).getAttributes()).transpose();
                log.debug("NETWORK INPUT VECTOR VECTOR:");
                MatrixUtils.printMatrix(networkInput);

                RealMatrix networkOutput = stepForward(networkInput);
                log.debug("NETWORK OUTPUT VECTOR VECTOR:");
                MatrixUtils.printMatrix(networkOutput);

                //Формирование выходного вектора класса, например [0, 1, 0, 0, 0] - для 2-ого класса из 5-и
                RealMatrix groundTruth  = MatrixUtils.getGroundTruth(learn.get(selectionIndex).getClassNumber(), dataset.getClassesCount());
                log.debug("GROUND TRUTH VECTOR:");
                MatrixUtils.printMatrix(groundTruth);

                //Подсчет вектора ошибки
                RealMatrix errorVector = networkOutput.subtract(groundTruth);
                log.debug("ERROR VECTOR:");
                MatrixUtils.printMatrix(errorVector);

                //Ошибка итерации суммарная - сумма квадратов всех ошибок
                for (int k = 0; k < errorVector.getColumnDimension(); k++) {
                    rmse = rmse + errorVector.getEntry(0, k) * errorVector.getEntry(0, k);
                }
                stepBackward(errorVector);
                correctWeights(learnRate);
            }

            rmse = Math.sqrt(rmse / selectionCount);
//            if(rmse == rmsePrev) {
//                throw new GradientException();
//            }
            System.out.println("RMSE " + epochIndex + "-ой эпохи равна " + rmse);
            rmsePrev = rmse;
        }
    }



    private RealMatrix stepForward(RealMatrix input) {
        log.debug("Start STEP ---> with:");
        MatrixUtils.printMatrix(input);

        RealMatrix output = input.copy();

        for (Layer layer: layers) {
            output = layer.propogateForward(output);
        }
        log.debug("End STEP ---> with:");
        MatrixUtils.printMatrix(output);
        return output;
    }

    private RealMatrix stepBackward(RealMatrix input) {
        log.debug("Start STEP <--- with:");
        MatrixUtils.printMatrix(input);

        RealMatrix output = input;

        for (Iterator<Layer> it = layers.descendingIterator(); it.hasNext(); ) {
            Layer layer = it.next();
            output = layer.propogateBackward(output);
        }

        log.debug("End STEP <--- with:");
        MatrixUtils.printMatrix(output);
        return output;
    }

    private void correctWeights(double learnRate) {
        log.debug("Start CORRECT WEIGHTS");
        for (Layer layer: layers) {
            layer.correctWeights(learnRate);
        }
        log.debug("End CORRECT WEIGHTS");
    }

    @Override
    public void test(Dataset dataset) {
        List<DatasetItem> test = dataset.getTestData();
//        double selectionCount = 10;
        int selectionCount = test.size();
        int trueClasses = 0;

        //Среднеквадратичная ошибка - целевая функция, считается в конце каждой эпохи и должна уменьшаться от эпохи к эпохе
        double rmse = 0;
        //Цикл выборок
        for (int selectionIndex = 0; selectionIndex < selectionCount; selectionIndex++) {
            //Входной вектор
            RealMatrix networkInput = MatrixUtils.createInstanceFromRow(test.get(selectionIndex).getAttributes()).transpose();
            log.debug("NETWORK INPUT VECTOR VECTOR:");
            MatrixUtils.printMatrix(networkInput);

            //Прямое распространение
            RealMatrix networkOutput = stepForward(networkInput);
            log.debug("NETWORK OUTPUT VECTOR:");
            MatrixUtils.printMatrix(networkOutput);

            //Формирование выходного вектора класса, например [0, 1, 0, 0, 0] - для 2-ого класса из 5-и
            RealMatrix groundTruth  = MatrixUtils.getGroundTruth(test.get(selectionIndex).getClassNumber(), dataset.getClassesCount());
            log.debug("GROUND TRUTH VECTOR:");
            MatrixUtils.printMatrix(groundTruth);

            //Подсчет вектора ошибки
            RealMatrix errorVector = networkOutput.subtract(groundTruth);
            log.debug("ERROR VECTOR:");
            MatrixUtils.printMatrix(errorVector);

            //Ошибка итерации суммарная - сумма квадратов всех ошибок
            for (int k = 0; k < errorVector.getColumnDimension(); k++) {
                rmse = rmse + errorVector.getEntry(0, k) * errorVector.getEntry(0, k);
            }

            double max = networkOutput.getEntry(0, 0);
            int findedClassNumber = 1;
            int classNumber = Double.valueOf(test.get(selectionIndex).getClassNumber()).intValue();
            for (int i = 0; i < networkOutput.getRowDimension(); i++) {
                double currentValue = networkOutput.getEntry(i, 0);
                if (currentValue > max) {
                    max = currentValue;
                    findedClassNumber = i + 1;
                }
            }
            System.out.println("Итерация " + selectionIndex + "; номер найденного: " + findedClassNumber + "; номер прав: " + classNumber);
            if (findedClassNumber == classNumber) {
                trueClasses++;
            }
        }

        rmse = Math.sqrt(rmse / selectionCount);
        System.out.println("RMSE тестирования: " + rmse);
        System.out.println("Кол-во примеров в тестовой выборке: " + selectionCount);
        System.out.println("Правильно определенных классов: " + trueClasses);
    }

}
