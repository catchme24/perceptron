package old.small;

import old.DatasetWorker;
import old.FunctionActiovation;
import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.util.*;

import static old.big.Test.*;
@Data
public class SmallNeuroNetwork {
    private int inputs_count;
    private int neurons_count;
    private int outputs_count;
    private Array2DRowRealMatrix matrixWorker;
    private DatasetWorker worker;
    private RealMatrix dataset_matrix;

    private FunctionActiovation functionActiovationHidden;

    private FunctionActiovation functionActiovationOut;

    private RealMatrix weight_matrix_in;
    private RealMatrix weight_matrix_offset_in;
    private RealMatrix weight_matrix_out;
    private RealMatrix weight_matrix_offset_out;

    private RealMatrix preActivation_in;
    private RealMatrix preActivation_out;

    private RealMatrix postActivation_in;
    private RealMatrix postActivation_out;

    private RealMatrix grad_out;
    private RealMatrix grad_in;
    private RealMatrix grad_offset_out;
    private RealMatrix grad_offset_in;

    private double learn_rate;

    private RealMatrix weight_matrix_in_post;
    private RealMatrix weight_matrix_out_post;

    private RealMatrix weight_matrix_offset_in_post;
    private RealMatrix weight_matrix_offset_out_post;
    private double moment;
    private String algorithm = "basic";

    private boolean classification;


    public SmallNeuroNetwork(int inputs_count, int neurons_count, int outputs_count,
                        Array2DRowRealMatrix matrixWorker, DatasetWorker worker, boolean classification) {
        this.inputs_count = inputs_count;
        this.neurons_count = neurons_count;
        this.outputs_count = outputs_count;
        this.matrixWorker = matrixWorker;
        this.worker = worker;
        this.classification = classification;
        this.functionActiovationHidden = new FunctionActiovation("log");
        this.functionActiovationOut =  new FunctionActiovation("log");
    }

    public void loadDataset(File file, int learnProcent) throws IOException {
        worker.prepareDataset(file, learnProcent);
    }

    public void loadDataset(FileReader fileReader, int records, int testExampleCount, int samplingStep, int offset) throws IOException, ParseException {
        worker.prepareDataset(fileReader, records, testExampleCount, samplingStep, offset);
    }

    public void loadDataset(int records, int testExampleCount, int samplingStep, double offset) throws IOException {
        worker.prepareDataset(records, testExampleCount, samplingStep, offset);
    }

    public void initializationWeights() {
        //Инициализация градиентов
        grad_out = matrixWorker.createMatrix(outputs_count, neurons_count);
        grad_in = matrixWorker.createMatrix(neurons_count, inputs_count);
        grad_offset_out = matrixWorker.createMatrix(outputs_count, 1);
        grad_offset_in = matrixWorker.createMatrix(neurons_count, 1);

        //Матрица весов с рандомными значениями ВХОДНОГО СЛОЯ
        weight_matrix_in = matrixWorker.createMatrix(inputs_count, neurons_count);
        weight_matrix_offset_in = matrixWorker.createMatrix(1, neurons_count);
        fillRandom(weight_matrix_in);
//        printMatrix(weight_matrix_in);
        fillRandom(weight_matrix_offset_in);
//        printMatrix(weight_matrix_offset_in);

        //Матрица весов с рандомными значениями ВЫХОДНОГО СЛОЯ
        weight_matrix_out = matrixWorker.createMatrix(neurons_count, outputs_count);
        weight_matrix_offset_out = matrixWorker.createMatrix(1, outputs_count);
        fillRandom(weight_matrix_out);
        fillRandom(weight_matrix_offset_out);

        //Матрицы пре-активации и пост-активации для каждого словая
        preActivation_in = matrixWorker.createMatrix(1, neurons_count);
        postActivation_in = matrixWorker.createMatrix(1, neurons_count);
        preActivation_out = matrixWorker.createMatrix(1, outputs_count);
        postActivation_out = matrixWorker.createMatrix(1, outputs_count);
    }
    public void learning(int epoch, double learn_rate, double moment) throws IOException {
        this.moment = moment;
        learning(epoch, learn_rate);
    }

    public void learning(int epoch, double learn_rate) throws IOException {
        this.learn_rate = learn_rate;

        //Обучение
        RealMatrix learn_matrix = worker.getLearnMatrix();
//        System.out.println("learn");
//        printMatrix(learn_matrix);
        int training_count = learn_matrix.getRowDimension();
        double prev_error = 0;

        //Цикл эпох
        for (int i = 0; i < epoch; i++) {
            //Среднеквадратичная ошибка - целевая функция, считается в конце каждой эпохи и должна уменьшаться от эпохи к эпохе
            double error = 0;
            double sko = 0;
            double average = 0;

            //Перемешивание случайным образом входной матрицы для каждой эпохи
            //RealMatrix learnShuffled = shuffle(learn_matrix);

            //Формирование матрицы входов и матрицы выходов
            //RealMatrix input_matrix = learnShuffled.getSubMatrix(0, learn_matrix.getRowDimension() - 1, 0, learn_matrix.getColumnDimension() - 2);

            RealMatrix input_matrix = learn_matrix.getSubMatrix(0, learn_matrix.getRowDimension() - 1, 0, learn_matrix.getColumnDimension() - 2);
            RealMatrix output_matrix;
            if (classification == true) {
                output_matrix = formOutputMatrix(learn_matrix, matrixWorker, outputs_count);
            } else {
                output_matrix = formOutputMatrix(learn_matrix, matrixWorker, outputs_count, true);
            }

            //Цикл выборок
            for (int j = 0; j < training_count; j++) {
                //ПРЯМОЕ РАСПРОСТРОНЕНИЕ
                //Входной вектор признаков
                RealMatrix learn_vector = input_matrix.getSubMatrix(j, j, 0, input_matrix.getColumnDimension() - 1);
                //Выходной вектор классов
                RealMatrix output_vector = output_matrix.getSubMatrix(j, j, 0, output_matrix.getColumnDimension() - 1);

                //Подсчет сигнала ПРЕ-АКТИВАЦИИ и ПОСТ-АКТИВАЦИИ для 0-ого скрытого слоя
                preActivation_in = (learn_vector).multiply(weight_matrix_in);
                preActivation_in = preActivation_in.add(weight_matrix_offset_in);

                //Цикл обрабокти сигнала каждого нейрона скрытого слоя
                for (int k = 0; k < neurons_count; k++) {
                    postActivation_in.setEntry(0, k, functionActiovationHidden.calculate(preActivation_in.getEntry(0, k)));
                }

                //Подсчет сигнала ПРЕ-АКТИВАЦИИ для выходого слоя
                preActivation_out = postActivation_in.multiply(weight_matrix_out);
                preActivation_out = preActivation_out.add(weight_matrix_offset_out);

                //Цикл обрабокти сигнала каждого нейрона выходного слоя
                for (int k = 0; k < outputs_count; k++) {
                    postActivation_out.setEntry(0, k, functionActiovationOut.calculate(preActivation_out.getEntry(0, k)));
                }

                //Подсчет вектора ошибки
                RealMatrix FE = postActivation_out.subtract(output_vector);
                //Ошибка итерации суммарная - сумма квадратов всех ошибок
                for (int k = 0; k < FE.getColumnDimension(); k++) {
                    error = error + FE.getEntry(0, k) * FE.getEntry(0, k);
                }

                //Обратное распространение
                //Градиент выходного слоя
                for(int k = 0; k < outputs_count; k++) {
                    for(int l = 0; l < neurons_count; l++) {
                        grad_out.setEntry(k,l, FE.getEntry(0, k) *
                                functionActiovationOut.calculateDeriv(preActivation_out.getEntry(0, k)) *
                                postActivation_in.getEntry(0, l));
                    }
                }

                //Градиент входного слоя
                for(int k = 0; k < neurons_count; k++) {
                    for(int l = 0; l < inputs_count; l++) {
                        double summError = 0;
                        for(int m = 0; m < outputs_count; m++) {
                            summError += FE.getEntry(0, m) *
                            functionActiovationOut.calculateDeriv(preActivation_out.getEntry(0, m)) *
                            weight_matrix_out.getEntry(k, m);
                        }
                        grad_in.setEntry(k, l, summError *
                                functionActiovationHidden.calculateDeriv(preActivation_in.getEntry(0, k)) *
                                learn_vector.getEntry(0, l));
                    }
                }

                //Градиент смещения выходного слоя
                for(int k = 0; k < outputs_count; k++) {
                    grad_offset_out.setEntry(k,0, FE.getEntry(0, k) *
                            functionActiovationOut.calculateDeriv(preActivation_out.getEntry(0, k)));
                }

                //Градиент смещения входного слоя
                for(int k = 0; k < neurons_count; k++) {
                    double summError = 0;
                    for(int m = 0; m < outputs_count; m++) {
                        summError += FE.getEntry(0, m) *
                                functionActiovationOut.calculateDeriv(preActivation_out.getEntry(0, m)) *
                                weight_matrix_out.getEntry(k, m);
                    }
                    grad_offset_in.setEntry(k, 0, summError *
                            functionActiovationHidden.calculateDeriv(preActivation_in.getEntry(0, k)));
                }

                //Корректировка весов нейронов
                switch (algorithm) {
                    case "withmoment":
                        correctWeightWithMoment(j);
                        break;
                    default:
                        correctWeight();
                }
            }

            error = Math.sqrt(error / training_count);
            if(error == prev_error) {
                throw new GradientException();
            }
//            if(i == epoch - 1) {
//            System.out.println("MSE " + i + "-ой эпохи равна " + error / 2);
//            System.out.println("Предыдущая RMSE равна " + prev_error);
            System.out.println("RMSE " + i + "-ой эпохи равна " + error);
            prev_error = error;
//            }
        }

    }

    public int testing () {
        //Тестирование
        RealMatrix test_matrix = worker.getTestMatrix();
//        RealMatrix test_matrix = worker.getLearnMatrix();
        System.out.println(test_matrix.getRowDimension());

        int example_count = test_matrix.getRowDimension();
        double sko = 0;
        double average = 0;
        double error = 0;
        ArrayList<Double> devi = new ArrayList<>();
        ArrayList<Double> outputValueCalculated = new ArrayList();
        ArrayList<Double> outputValueDataset = new ArrayList();

        //Матрицы входов и выходов для тестирования
        RealMatrix input_matrix_2 = test_matrix.getSubMatrix(0, test_matrix.getRowDimension() - 1, 0, test_matrix.getColumnDimension() - 2);
        RealMatrix output_matrix_2;
        if (classification == true) {
            output_matrix_2 = formOutputMatrix(test_matrix, matrixWorker, outputs_count);
        } else {
            output_matrix_2 = formOutputMatrix(test_matrix, matrixWorker, outputs_count, true);
        }

        int true_classes = 0;
        int classes = output_matrix_2.getRowDimension();
        int[] eachTrueClassCount = new int[worker.getClassesCount()];
        Arrays.fill(eachTrueClassCount, 0);
        ArrayList<String> classesNames = worker.getClassesNames();

        for (int i = 0; i < example_count; i++) {
            //ПРЯМОЕ РАСПРОСТРОНЕНИЕ
            //Входной тестовый вектор
            RealMatrix test_vector = input_matrix_2.getSubMatrix(i, i, 0, input_matrix_2.getColumnDimension() - 1);

            //Подсчет сигнала ПРЕ-АКТИВАЦИИ, т.е. подсчет сумм, которые входят в нейроны
            preActivation_in = test_vector.multiply(weight_matrix_in);
            preActivation_in = preActivation_in.add(weight_matrix_offset_in);

            //Цикл обрабокти сигнала каждого нейрона скрытого
            for (int k = 0; k < neurons_count; k++) {
                postActivation_in.setEntry(0, k, functionActiovationHidden.calculate(preActivation_in.getEntry(0, k)));
            }

            //Подсчет сигнала ПРЕ-АКТИВАЦИИ, т.е. подсчет сумм, которые входят в нейроны
            preActivation_out = postActivation_in.multiply(weight_matrix_out);
            preActivation_out = preActivation_out.add(weight_matrix_offset_out);

            //Цикл обрабокти сигнала каждого нейрона выходного
            for (int k = 0; k < outputs_count; k++) {
                postActivation_out.setEntry(0, k, functionActiovationOut.calculate(preActivation_out.getEntry(0, k)));
            }

            //Подсчет вектора ошибки
            RealMatrix FE = postActivation_out.subtract(output_matrix_2.getRowMatrix(i));
            for (int k = 0; k < FE.getColumnDimension(); k++) {
                error = error + FE.getEntry(0, k) * FE.getEntry(0, k);
            }

            if(classification == false) {
                //Определение правильного предсказания
//                System.out.println(i + "-ая итерация");
//                System.out.println("Входной вектор для теста");
//                printMatrix(test_vector);
//                System.out.println("Преактивация");
//                printMatrix(preActivation_out);
//                System.out.println("Постактивация");
//                printMatrix(postActivation_out);
//                System.out.println("Должно быть");
//                System.out.println(Arrays.toString(output_matrix_2.getRow(i)));
                double mathExpectation = worker.getMathExpectation();
                double dispersion = worker.getDispersion();

                double obtainedValue = ((postActivation_out.getEntry(0,0) - 2) * Math.sqrt(dispersion)) + mathExpectation;
                double testValue = ((output_matrix_2.getEntry(i,0) -2) * Math.sqrt(dispersion)) + mathExpectation;
                double deviation = (testValue - obtainedValue)/testValue * 100;
//                double obtainedValue = postActivation_out.getEntry(0, 0);
//                double testValue = output_matrix_2.getEntry(i,0);
//                double deviation = (output_matrix_2.getEntry(i,0) - postActivation_out.getEntry(0, 0))
//                        /output_matrix_2.getEntry(i,0) * 100;

                outputValueCalculated.add(obtainedValue);
                outputValueDataset.add(testValue);
                devi.add(deviation);
//                System.out.println("Полученное и тестовое значения");
//                System.out.println(obtainedValue);
//                System.out.println(testValue);
            } else {
                //Определение количества правильных классов
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
                if (index == index_2) {
                    true_classes++;
                    eachTrueClassCount[index]++;
                }
            }

        }

        //График
        XYSeries series2 = new XYSeries("Ожидаемые значения");
        XYSeries series1 = new XYSeries("Спрогнозированные значения");
//        System.out.println(outputValueCalculated.toString());
        for(int i = 0; i < outputValueCalculated.size(); i++) {
            series1.add(i, (Double) outputValueCalculated.get(i));
        }
        for(int i = 0; i < outputValueDataset.size(); i++) {
            series2.add(i, (Double) outputValueDataset.get(i));
        }
        XYSeriesCollection xyDataset = new XYSeriesCollection(series2);
        xyDataset.addSeries(series1);

        String text = "Катировки акций Sberbank 2016-2017 (Многослойный персептрон)";

        JFreeChart chart = ChartFactory.createXYLineChart("Прогнозирование цен акций (обучающая выборка)", "x", "y",
                    xyDataset,
                    PlotOrientation.VERTICAL,
                    true, true, true);

        XYPlot xyPlot = chart.getXYPlot();
        ValueAxis domainAxis = xyPlot.getDomainAxis();
        ValueAxis rangeAxis = xyPlot.getRangeAxis();
        domainAxis.setRange(0.0, 190);
        rangeAxis.setRange(80, 170);

        JFrame frame = new JFrame("MinimalStaticChart");
        // Помещаем график на фрейм
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.getContentPane().add(new ChartPanel(chart));
        frame.setSize(400,300);
        frame.setVisible(true);

//        //Подсчет СКО
//        for(int j = 0; j < outputValueCalculated.size(); j++) {
//            average = average + outputValueCalculated.get(j);
//        }
//        average = average/outputValueCalculated.size();
//        //System.out.println("среднее " + average);
//        for(int j = 0; j < outputValueCalculated.size(); j++) {
//            double current = outputValueCalculated.get(j);
//            double difference = current - average;
//            sko = sko + Math.pow(difference, 2);
//        }
//        sko = sko/outputValueCalculated.size();
//        sko = Math.sqrt(sko);

        if(classification == false) {
            System.out.print("Отклонения: ");
            for (int i = 0; i < devi.size(); i++) {
                //String digit = devi.get(i);
                System.out.print(String.format("%.2f", devi.get(i)) + "% ");
            }
            System.out.println();
            System.out.println("RMSE тестирования " + Math.sqrt(error / example_count));
            //System.out.println("СКО тестирования: " + sko);
        } else {
            System.out.println("Total classes: " + classes + "\n" + "True classes: " + true_classes);
            System.out.println("From them: ");
            System.out.println(example_count);
            for (int i = 0; i < eachTrueClassCount.length; i++) {
                System.out.print(classesNames.get(i) + " class: " + eachTrueClassCount[i] + "; ");
            }
            System.out.println();
        }
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

    public void correctWeight() {
        weight_matrix_in = weight_matrix_in.subtract(grad_in.transpose().scalarMultiply(learn_rate));
        weight_matrix_out = weight_matrix_out.subtract(grad_out.transpose().scalarMultiply(learn_rate));
        weight_matrix_offset_out = weight_matrix_offset_out.subtract(grad_offset_out.transpose().scalarMultiply(learn_rate));
        weight_matrix_offset_in = weight_matrix_offset_in.subtract(grad_offset_in.transpose().scalarMultiply(learn_rate));
    }

    public void correctWeightWithMoment(int j) {

        weight_matrix_in_post = weight_matrix_in;
        weight_matrix_out_post = weight_matrix_out;
        weight_matrix_offset_out_post = weight_matrix_offset_out;
        weight_matrix_offset_in_post = weight_matrix_offset_in;

        if(j >= 1) {
            weight_matrix_in =
                    weight_matrix_in.subtract(grad_in.transpose().scalarMultiply(learn_rate).
                            add(weight_matrix_in.subtract(weight_matrix_in_post).scalarMultiply(moment)));
            weight_matrix_out =
                    weight_matrix_out.subtract(grad_out.transpose().scalarMultiply(learn_rate).
                            add(weight_matrix_out.subtract(weight_matrix_out_post).scalarMultiply(moment)));
            weight_matrix_offset_in =
                    weight_matrix_offset_in.subtract(grad_offset_in.transpose().scalarMultiply(learn_rate).
                            add(weight_matrix_offset_in.subtract(weight_matrix_offset_in_post).scalarMultiply(moment)));
            weight_matrix_offset_out_post =
                    weight_matrix_offset_out_post.subtract(grad_offset_out.transpose().scalarMultiply(learn_rate).
                            add(weight_matrix_offset_out_post.subtract(weight_matrix_offset_out_post).scalarMultiply(moment)));
        }
    }

    public void showWeightsMatrix() {
        System.out.println("МАТРИЦА ВХОДНЫХ ВЕСОВ");
        printMatrix(weight_matrix_in);
        System.out.println("МАТРИЦА ВЫХОДНЫХ ВЕСОВ");
        printMatrix(weight_matrix_out);
    }
}