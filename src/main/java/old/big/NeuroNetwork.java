package old.big;

import lombok.Data;
import old.DatasetWorker;
import old.FunctionActiovation;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static old.big.Test.*;


@Data
public class NeuroNetwork {
    private int layers_count;
    private int inputs_count;
    private int neurons_count;
    private int outputs_count;
    private Array2DRowRealMatrix matrixWorker;
    private DatasetWorker worker;
    private RealMatrix dataset_matrix;
    private FunctionActiovation functionActiovation;

    private RealMatrix weight_matrix_in;
    private RealMatrix weight_matrix_offset_in;

    private RealMatrix weight_matrix_out;
    private RealMatrix weight_matrix_offset_out;

    private RealMatrix[] weight_matrix;
    private RealMatrix[] weight_matrix_offset;

    private RealMatrix[] preActivation;
    private RealMatrix preActivation_out;

    private RealMatrix[] postActivation;
    private RealMatrix postActivation_out;


    public NeuroNetwork(int layers_count, int inputs_count, int neurons_count, int outputs_count,
                        Array2DRowRealMatrix matrixWorker, DatasetWorker worker) {
        this.layers_count = layers_count;
        this.inputs_count = inputs_count;
        this.neurons_count = neurons_count;
        this.outputs_count = outputs_count;
        this.matrixWorker = matrixWorker;
        this.worker = worker;
        this.weight_matrix = new RealMatrix[layers_count - 1];
        this.weight_matrix_offset = new RealMatrix[layers_count - 1];
        this.preActivation = new RealMatrix[layers_count];
        this.postActivation = new RealMatrix[layers_count];
    }

    public void loadDataset(File file, int percent) throws IOException {
        worker.prepareDataset(file, percent);
    }

    public void initializationWeights() {
        //Матрица весов с рандомными значениями ВХОДНОГО СЛОЯ ПРИ СКРЫТОМ СЛОЕ = 1
        weight_matrix_in = matrixWorker.createMatrix(inputs_count, neurons_count);
        weight_matrix_offset_in = matrixWorker.createMatrix(1, neurons_count);
        fillRandom(weight_matrix_in);
        fillRandom(weight_matrix_offset_in);

        //Матрица весов с рандомными значениями ДЛЯ ПРОМЕЖУТОЧНЫХ СЛОЕВ
        for (int i = 0; i < layers_count - 1; i++) {
            weight_matrix[i] = matrixWorker.createMatrix(neurons_count, neurons_count);
            weight_matrix_offset[i] = matrixWorker.createMatrix(1, neurons_count);
            fillRandom(weight_matrix_offset[i]);
            fillRandom(weight_matrix[i]);
        }

        //Матрица весов с рандомными значениями ВЫХОДНОГО СЛОЯ ПРИ СКРЫТОМ СЛОЕ = 1
        weight_matrix_out = matrixWorker.createMatrix(neurons_count, outputs_count);
        weight_matrix_offset_out = matrixWorker.createMatrix(1, outputs_count);
        fillRandom(weight_matrix_out);
        fillRandom(weight_matrix_offset_out);

        //Матрицы пре-активации и пост-активации для каждого словая
        for (int i = 0; i < layers_count; i++) {
            preActivation[i] = matrixWorker.createMatrix(1, neurons_count);
            postActivation[i] = matrixWorker.createMatrix(1, neurons_count);
        }
        preActivation_out = matrixWorker.createMatrix(1, outputs_count);
        postActivation_out = matrixWorker.createMatrix(1, outputs_count);
    }

    public void learning(int epoch, double learn_rate) throws IOException {
        //Обучение
        RealMatrix learn_matrix = worker.getLearnMatrix();

//        for (int i = 0; i < learn_matrix.getRowDimension(); i++) {
//            System.out.println(Arrays.toString(learn_matrix.getRow(i)) + "  " + Arrays.toString(learnShuffled.getRow(i)));
//        }
//        System.out.println("Матрица обуч");

        //Матрица входов
//        for (int i = 0; i < learn_matrix.getRowDimension(); i++) {
//            System.out.println(Arrays.toString(learnShuffled.getRow(i)) + "  " + Arrays.toString(input_matrix.getRow(i)));
//        }

        //Матрица выходов

//        for (int i = 0; i < learn_matrix.getRowDimension(); i++) {
//            System.out.println(Arrays.toString(learnShuffled.getRow(i)) + "  " + Arrays.toString(output_matrix.getRow(i)));
//        }

        int training_count = learn_matrix.getRowDimension();
        //int training_count = 1;

//        System.out.println("МАТРИЦА 1-ЫХ ВЕСОВ");
//        printMatrix(weight_matrix_in);
//        System.out.println("МАТРИЦА 2-ЫХ ВЕСОВ");
//        printMatrix(weight_matrix[0]);
//        System.out.println("МАТРИЦА 3-ИХ ВЕСОВ");
//        printMatrix(weight_matrix_out);

        //Цикл эпох
        for (int i = 0; i < epoch; i++) {
            double error = 0;

            RealMatrix learnShuffled = shuffle(learn_matrix);
            RealMatrix input_matrix = learnShuffled.getSubMatrix(0, learn_matrix.getRowDimension() - 1, 0, learn_matrix.getColumnDimension() - 2);
            RealMatrix output_matrix = formOutputMatrix(learnShuffled, matrixWorker, outputs_count);

            //Цикл выборок
            for (int j = 0; j < training_count; j++) {

                //ПРЯМОЕ РАСПРОСТРОНЕНИЕ
//                System.out.println("Эпоха: " + epoch + ", тренировочная итерация: " + j);
                //Входной вектор признаков
                RealMatrix learn_vector = input_matrix.getSubMatrix(j, j, 0, input_matrix.getColumnDimension() - 1);
                //Выходной вектор классов
                RealMatrix output_vector = output_matrix.getSubMatrix(j, j, 0, output_matrix.getColumnDimension() - 1);
                //printMatrix(output_vector);

//                System.out.println("0-ой слой матр");
//                printMatrix(weight_matrix_in);
//                System.out.println("1-ой слой матр");
//                printMatrix(weight_matrix[0]);
//                System.out.println("Вых матр");
//                printMatrix(weight_matrix_out);

                //Подсчет сигнала ПРЕ-АКТИВАЦИИ и ПОСТ-АКТИВАЦИИ для 0-ого скрытого слоя
                //System.out.println(0 + "-ый слой");
                preActivation[0] = (learn_vector).multiply(weight_matrix_in);
//                System.out.println("Сигнал преактивации без оффсета:");
//                printMatrix(preActivation[0]);
                preActivation[0] = preActivation[0].add(weight_matrix_offset_in);
//                System.out.println("Сигнал преактивации:");
//                printMatrix(preActivation[0]);
                //Цикл обрабокти сигнала каждого нейрона l-ого скрытого слоя
                for (int k = 0; k < neurons_count; k++) {
                    postActivation[0].setEntry(0, k, functionActiovation.calculate(preActivation[0].getEntry(0, k)));
                }
//                System.out.println("Сигнал постактивации:");
//                printMatrix(postActivation[0]);

                //Подсчет сигнала ПРЕ-АКТИВАЦИИ и ПОСТ-АКТИВАЦИИ для скрытых слоев
                for (int l = 1; l < layers_count; l++) {
//                    System.out.println(l + "-ый слой");
                    preActivation[l] = postActivation[l - 1].multiply(weight_matrix[l - 1]);
                    preActivation[l] = preActivation[l].add(weight_matrix_offset[l - 1]);
//                    System.out.println("Сигнал преактивации:");
//                    printMatrix(preActivation[l]);

                    //Цикл обрабокти сигнала каждого нейрона l-ого скрытого слоя
                    for (int k = 0; k < neurons_count; k++) {
                        postActivation[l].setEntry(0, k, functionActiovation.calculate(preActivation[l].getEntry(0, k)));
                    }
//                    System.out.println("Сигнал постактивации:");
//                    printMatrix(postActivation[l]);
                }

                //Подсчет сигнала ПРЕ-АКТИВАЦИИ для выходого слоя
//                System.out.println("Выходной слой");
                preActivation_out = postActivation[layers_count - 1].multiply(weight_matrix_out);
                preActivation_out = preActivation_out.add(weight_matrix_offset_out);
//                System.out.println("Сигнал преактивации:");
//                printMatrix(preActivation_out);
                //Цикл обрабокти сигнала каждого нейрона выходного слоя
                for (int k = 0; k < outputs_count; k++) {
                    postActivation_out.setEntry(0, k, functionActiovation.calculate(preActivation_out.getEntry(0, k)));
                }
//                System.out.println("Сигнал постактивации:");
//                printMatrix(postActivation_out);

                //Подсчет вектора ошибки (Yi - Di)
                RealMatrix FE = postActivation_out.subtract(output_vector);
//                System.out.println("Выходной вектор:");
//                printMatrix(output_vector);
//                System.out.println("Вектор ошибки:");
//                printMatrix(FE);
                //Ошибка итерации суммарная
                for (int k = 0; k < FE.getColumnDimension(); k++) {
                    error = error + FE.getEntry(0, k) * FE.getEntry(0, k);
                }
//                System.out.println("Ошибка итерации: " + error);

                //СЕТЬ ОБРАТНОГО РАСПРОСТРОНЕНИЯ

                //Пре-активация сигнала ошибки выходного слоя
                RealMatrix preActivation_out_err = matrixWorker.createMatrix(1,outputs_count);
                preActivation_out_err.setRow(0, FE.getRow(0));
//                System.out.println("Преактивация вых ош");
//                printMatrix(preActivation_out_err);
                //Пост-активация сигнала ошибки выходного слоя
                RealMatrix postActivation_out_err = matrixWorker.createMatrix(1, outputs_count);
                for(int k = 0; k < postActivation_out_err.getColumnDimension(); k++) {
                    postActivation_out_err.setEntry(0, k, functionActiovation.calculateDeriv(preActivation_out.getEntry(0, k)) * preActivation_out_err.getEntry(0, k));
//                    System.out.println("Произв " + functionActiovation.calculateDeriv(preActivation_out.getEntry(0, k)));
//                    System.out.println("Пре актив вых" + preActivation_out_err.getEntry(0, k));
                }
//                System.out.println("Постактивая вых ош");
//                printMatrix(postActivation_out_err);
                //Пре-активация сигнала ошибки 1-ого слоя
                RealMatrix preActivation_1_err = matrixWorker.createMatrix(1, neurons_count);
                preActivation_1_err = postActivation_out_err.multiply(weight_matrix_out.transpose());
//                System.out.println("Матрица вых коэф");
//                printMatrix(weight_matrix_out);
//                System.out.println("Преактивация 1 слоя ош");
//                printMatrix(preActivation_1_err);
                //Пост-активация сигнала ошибки 1-ого слоя
                RealMatrix postActivation_1_err = matrixWorker.createMatrix(1,neurons_count);
                for(int k = 0; k < postActivation_1_err.getColumnDimension(); k++) {
                    postActivation_1_err.setEntry(0, k, functionActiovation.calculateDeriv(preActivation[1].getEntry(0, k))
                            * preActivation_1_err.getEntry(0, k));
                }
//                System.out.println("Постактивая 1 слоя ош");
//                printMatrix(postActivation_1_err);
                //Пре-активация сигнала ошибки 0-ого слоя
                RealMatrix preActivation_0_err = matrixWorker.createMatrix(1,neurons_count);
                preActivation_0_err = postActivation_1_err.multiply(weight_matrix[0].transpose());
//                System.out.println("Преактивация 0 слоя ош");
//                printMatrix(preActivation_0_err);
                //Пост-активация сигнала ошибки 0-ого слоя
                RealMatrix postActivation_0_err = matrixWorker.createMatrix(1,neurons_count);
                for(int k = 0; k < postActivation_0_err.getColumnDimension(); k++) {
                    postActivation_0_err.setEntry(0, k, functionActiovation.calculateDeriv(preActivation[0].getEntry(0, k))
                            * preActivation_0_err.getEntry(0, k));
                }
//                System.out.println("Постактивая 0 слоя ош");
//                printMatrix(postActivation_0_err);

                //Вычисление градиентов по каждому слою

                //Градиент выходного слоя
                RealMatrix grad_out = (postActivation_out_err.transpose()).multiply(postActivation[1]);
                grad_out = grad_out.transpose();
                RealMatrix grad_out_offset = postActivation_out_err.copy();
                //Градиент 1-ого слоя
                RealMatrix grad_1 = (postActivation_1_err.transpose()).multiply(postActivation[0]);
                grad_1 = grad_1.transpose();
                RealMatrix grad_1_offset = postActivation_1_err.copy();
                //Градиент 0-ого слоя
                RealMatrix grad_0 = (postActivation_0_err.transpose()).multiply(learn_vector);
                grad_0 = grad_0.transpose();
                RealMatrix grad_0_offset = postActivation_0_err.copy();
                //Вывод градиентов
//                System.out.println("Град вых слоя");
//                printMatrix(grad_out);
//                System.out.println("Град 1 слоя");
//                printMatrix(grad_1);
//                System.out.println("Град 0 слоя");
//                printMatrix(grad_0);

                //Корректировка весов нейронов
//                System.out.println("Градиент вх");
//                printMatrix(grad_0);
//                System.out.println("МАТРИЦА 1-ЫХ ВЕСОВ ДО КОР");
//                printMatrix(weight_matrix_in);
                weight_matrix_in = weight_matrix_in.subtract(grad_0.scalarMultiply(learn_rate));
//                System.out.println("МАТРИЦА 1-ЫХ ВЕСОВ ПОСЛЕ КОР");
//                printMatrix(weight_matrix_in);
                weight_matrix[0] = weight_matrix[0].subtract(grad_1.scalarMultiply(learn_rate));
//                System.out.println("Градиент вых");
//                printMatrix(grad_out);
//                System.out.println("Матрица весов вых ");
//                printMatrix(weight_matrix_out);
                weight_matrix_out = weight_matrix_out.subtract(grad_out.scalarMultiply(learn_rate));
//                System.out.println("Матрица весов вых корр");
//                printMatrix(weight_matrix_out);
                //Корректировка весов нейронов-смещения
                weight_matrix_offset_out = weight_matrix_offset_out.subtract(grad_out_offset.scalarMultiply(learn_rate));
                weight_matrix_offset[0] = weight_matrix_offset[0].subtract(grad_1_offset.scalarMultiply(learn_rate));
                weight_matrix_offset_in = weight_matrix_offset_in.subtract(grad_0_offset.scalarMultiply(learn_rate));

//                System.out.println("МАТРИЦА 1-ЫХ ВЕСОВ");
//                printMatrix(weight_matrix_in);

            }
//            System.out.println("Ошибка " + i + "-ой эпохи равна " + error/2);
        }

//        System.out.println("МАТРИЦА 1-ЫХ ВЕСОВ");
//        printMatrix(weight_matrix_in);
//        System.out.println("МАТРИЦА 2-ЫХ ВЕСОВ");
//        printMatrix(weight_matrix[0]);
//        System.out.println("МАТРИЦА 3-ИХ ВЕСОВ");
//        printMatrix(weight_matrix_out);
        }


        public int testing () {
            //Тестирование
            //
            RealMatrix test_matrix = worker.getTestMatrix();
            int example_count = test_matrix.getRowDimension();
            //int example_count = 1;

            RealMatrix input_matrix_2 = test_matrix.getSubMatrix(0, test_matrix.getRowDimension() - 1, 0, test_matrix.getColumnDimension() - 2);
            RealMatrix output_matrix_2 = formOutputMatrix(test_matrix, matrixWorker, outputs_count);

//            for (int i = 0; i < test_matrix.getRowDimension(); i++) {
//                System.out.println(Arrays.toString(test_matrix.getRow(i)) + "  " + Arrays.toString(input_matrix_2.getRow(i)));
//            }
//
//            for (int i = 0; i < test_matrix.getRowDimension(); i++) {
//                System.out.println(Arrays.toString(test_matrix.getRow(i)) + "  " + Arrays.toString(output_matrix_2.getRow(i)));
//            }

            int true_classes = 0;

            int classes = output_matrix_2.getRowDimension();

            for (int i = 0; i < example_count; i++) {
                //ПРЯМОЕ РАСПРОСТРОНЕНИЕ
                RealMatrix test_vector = input_matrix_2.getSubMatrix(i, i, 0, input_matrix_2.getColumnDimension() - 1);
                //System.out.println("1ST LAYER!");
                //printMatrix(test_vector);

                //Подсчет сигнала ПРЕ-АКТИВАЦИИ, т.е. подсчет сумм, которые входят в нейроны
                preActivation[0] = test_vector.multiply(weight_matrix_in);
                preActivation[0] = preActivation[0].add(weight_matrix_offset_in);
//                System.out.println("СИГНАЛ ПРЕАКТИВАЦИИ C ОФФЕСТОМ:");
//                printMatrix(preActivation[0]);

                //Цикл обрабокти сигнала каждого нейрона 1-ого слоя
                for (int k = 0; k < neurons_count; k++) {
                    postActivation[0].setEntry(0, k, functionActiovation.calculate(preActivation[0].getEntry(0, k)));
                }
//                System.out.println("СИГНАЛ ПОСТАКТИВАЦИИ:");
//                printMatrix(postActivation[0]);

                //System.out.println("2ND LAYER!");

                //Подсчет сигнала ПРЕ-АКТИВАЦИИ, т.е. подсчет сумм, которые входят в нейроны
                preActivation[1] = postActivation[1].multiply(weight_matrix[0]);
                preActivation[1] = preActivation[1].add(weight_matrix_offset[0]);
//                System.out.println("СИГНАЛ ПРЕАКТИВАЦИИ C ОФФЕСТОМ:");
//                printMatrix(preActivation[1]);

                //Цикл обрабокти сигнала каждого нейрона 2-ого слоя
                for (int k = 0; k < neurons_count; k++) {
                    postActivation[1].setEntry(0, k, functionActiovation.calculate(preActivation[1].getEntry(0, k)));
                }
//                System.out.println("СИГНАЛ ПОСТАКТИВАЦИИ:");
//                printMatrix(postActivation[1]);

                //System.out.println("OUTPUT LAYER!");

                //Подсчет сигнала ПРЕ-АКТИВАЦИИ, т.е. подсчет сумм, которые входят в нейроны
                preActivation_out = postActivation[1].multiply(weight_matrix_out);
                preActivation_out = preActivation_out.add(weight_matrix_offset_out);
//                System.out.println("СИГНАЛ ПРЕАКТИВАЦИИ C ОФФЕСТОМ:");
//                printMatrix(preActivation_out);


                //Цикл обрабокти сигнала каждого нейрона 3-ого (ВЫХОДНОГО) слоя
                for (int k = 0; k < outputs_count; k++) {
                    postActivation_out.setEntry(0, k, functionActiovation.calculate(preActivation_out.getEntry(0, k)));
                }
//                System.out.println("СИГНАЛ ПОСТАКТИВАЦИИ:");
//                printMatrix(postActivation_out);



                double[] array = postActivation_out.getRow(0);
                double max = array[0];
                int index = 0;
                for (int h = 0; h < array.length; h++) {
                    if (array[h] > max) {
                        max = array[h];
                        index = h;
                    }
                }

                double[] array_2 = output_matrix_2.getRow(i);
                double max_2 = array_2[0];
                int index_2 = 0;
                for (int h = 0; h < array_2.length; h++) {
                    if (array_2[h] > max_2) {
                        max_2 = array_2[h];
                        index_2 = h;
                    }
                }

//                System.out.println("Индекс вектора пост-активации " + index + " индекс тестовго вектора " + index_2);
//                printMatrix(test_vector);
//                printMatrix(postActivation_out);
//                System.out.println(Arrays.toString(output_matrix_2.getRow(i)));

                if (index == index_2) {
                    true_classes++;
//                    switch (index) {
//                        case 0:
//                            first_class++;
//                            break;
//                        case 1:
//                            second_class++;
//                            break;
//                        case 2:
//                            third_class++;
//                            break;
//                        case 3:
//                            four_class++;
//                            break;
//                    }
                }
            }

//            System.out.println("Total classes: " + classes + "\n" + "True classes: " + true_classes);
//            System.out.println("From them: " + first_class + " " + second_class + " " + third_class + " " + four_class);
            return true_classes;
        }

        public RealMatrix shuffle(RealMatrix matrix) {

            List<Integer> list = new ArrayList();
            for(int i = 0; i < matrix.getRowDimension(); i++) {
                list.add(i);
            }

            Collections.shuffle(list);

            RealMatrix newMatrix = matrixWorker.createMatrix(matrix.getRowDimension(), matrix.getColumnDimension());

            for(int i = 0; i < matrix.getRowDimension(); i++) {
                newMatrix.setRow(i, matrix.getRow(list.get(i)));
            }

            return newMatrix;
        }

}

