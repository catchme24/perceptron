package old.small;

import old.DatasetWorker;
import old.FunctionActiovation;
import old.small.SmallNeuroNetwork;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;

public class SmallNeuroNetworkTest {

    @Test
    void testClassification() throws IOException {
        String pathCar = "src/main/resources/car.data";
        String pathIris = "src/main/resources/iris.data";
        String pathWine = "src/main/resources/wine.data";
        File file = new File(pathIris);
        Array2DRowRealMatrix matrixWorker = new Array2DRowRealMatrix();
        int percentDatasetForLearning = 80;

        DatasetWorker worker = new DatasetWorker();

        int epoch = 700;
        int inputsCount = 4;
        int neuronsCount = 8;
        int outputsCount = 3;
        double learnRate = 0.35;
        double moment = 0.1;

        String algCorr_1 = "basic";
        String algCorr_2 = "withMoment";

        int trueClasses;
        int max = 0;

        SmallNeuroNetwork network = new SmallNeuroNetwork(inputsCount, neuronsCount, outputsCount, matrixWorker, worker, true);
        //network.setAlgorithm(algCorr_2);
        FunctionActiovation functionActiovation = new FunctionActiovation("log");
        network.setFunctionActiovationHidden(functionActiovation);
        network.setFunctionActiovationOut(functionActiovation);

        network.loadDataset(file, percentDatasetForLearning);
        network.initializationWeights();
        network.learning(epoch, learnRate, moment);
        trueClasses = network.testing();
        if (trueClasses > max) {
            max = trueClasses;
        }
        System.out.println("Узлов: " + neuronsCount + ", эпох: " + epoch + ", коэф. обучения: " + learnRate + ", правильных классов " + trueClasses);
    }

    @Test
    void testPredication() throws IOException, ParseException {
        DatasetWorker worker = new DatasetWorker();
        String fileName = "src/main/resources/SBER.csv";
        //FileReader fileReader = new FileReader(fileName);
        Array2DRowRealMatrix matrixWorker = new Array2DRowRealMatrix();
        FunctionActiovation functionActiovationHidden = new FunctionActiovation("sin");
        FunctionActiovation functionActiovationOut = new FunctionActiovation("relu");

        //Характеристики сети
        int epoch = 400;
        int inputsCount = 5;
        int neuronsCount = 4;
        int outputsCount = 1;
        double learnRate = 0.01;
        double moment = 0.1;
        String algCorr_1 = "basic";
        String algCorr_2 = "withMoment";

        //Характеристики выборки
        int testExampleCount = 10;
        int samplingStep = 5;
        int records = 200;
        int offset = 0;



        SmallNeuroNetwork network = new SmallNeuroNetwork(inputsCount, neuronsCount, outputsCount, matrixWorker, worker, false);
        network.setAlgorithm(algCorr_2);
        network.setFunctionActiovationHidden(functionActiovationHidden);
        network.setFunctionActiovationOut(functionActiovationOut);
        FileReader fileReader = new FileReader(fileName);

        worker.setNormalize(true);
        network.loadDataset(fileReader, records, testExampleCount, samplingStep, offset);
        network.initializationWeights();
        network.learning(epoch, learnRate);
        network.testing();

        System.out.println();
        System.out.println("Узлов: " + neuronsCount + ", эпох: " + epoch + ", коэф. обучения: " + learnRate);
    }

    @Test
    void findBestParamsClassification() throws IOException {
        String pathCar = "src/main/resources/car.data";
        String pathIris = "src/main/resources/iris.data";
        String pathWine = "src/main/resources/wine.data";
        File file = new File(pathIris);
        Array2DRowRealMatrix matrixWorker = new Array2DRowRealMatrix();
        int percentStart = 80;
        int percentEnd = 80;


        for(int h = percentStart; h <= percentEnd; h = h + 5) {

            DatasetWorker worker = new DatasetWorker();

            int epochStart = 700;
            int epochEnd = 700;
            int inputs_count = 4;
            int neuronsStart = 8;
            int neuronsEnd = 8;
            int outputs_count = 3;
            double learnRateStart = 0.35;
            double lernRateEnd = 0.35;
            double moment = 0.1;
            String algCorr_1 = "basic";
            String algCorr_2 = "withMoment";

            int trueClasses;
            int max = 0;

            for (int i = neuronsStart; i <= neuronsEnd; i++) {
                for (int j = epochStart; j <= epochEnd; j = j + 100) {
                    for (double k = learnRateStart; k <= lernRateEnd; k = k + 0.05) {

                        SmallNeuroNetwork network = new SmallNeuroNetwork(inputs_count, i, outputs_count, matrixWorker, worker, true);
                        //network.setAlgorithm(algCorr_2);
                        FunctionActiovation functionActiovation = new FunctionActiovation("log");
                        network.setFunctionActiovationHidden(functionActiovation);
                        network.setFunctionActiovationOut(functionActiovation);

                        network.loadDataset(file, h);
                        network.initializationWeights();
                        network.learning(j, k, moment);
                        trueClasses = network.testing();
                        if (trueClasses > max) {
                            max = trueClasses;
                        }
                        System.out.println("Узлов: " + i + ", эпох: " + j + ", коэф. обучения: " + k + ", правильных классов " + trueClasses);
                    }
                }
            }
            System.out.println("Максимальное число правильных классов: " + max);
        }
    }

    @Test
    void findBestParamsPredication() throws IOException, ParseException {
        DatasetWorker worker = new DatasetWorker();
        String fileName = "src/main/resources/SBER.csv";
        //FileReader fileReader = new FileReader(fileName);
        Array2DRowRealMatrix matrixWorker = new Array2DRowRealMatrix();
        FunctionActiovation functionActiovationHidden = new FunctionActiovation("sin");
        FunctionActiovation functionActiovationOut = new FunctionActiovation("relu");

        //Характеристики сети
        int epochStart = 400;
        int epochEnd = 400;
        int inputs_count = 5;
        int neuronsStart = 4;
        int neuronsEnd = 4;
        int outputs_count = 1;
        double learnRateStart = 0.01;
        double lernRateEnd = 0.01;
        double moment = 0.1;
        String algCorr_1 = "basic";
        String algCorr_2 = "withMoment";

        //Характеристики выборки
        int testExampleCount = 10;
        int samplingStep = 5;
        int records = 200;
        int offset = 0;

        for (int i = neuronsStart; i <= neuronsEnd; i++) {
            for (int j = epochStart; j <= epochEnd; j = j + 100) {
                for (double k = learnRateStart; k <= lernRateEnd; k = k + 0.05) {

                    SmallNeuroNetwork network = new SmallNeuroNetwork(inputs_count, i, outputs_count, matrixWorker, worker, false);
                    network.setAlgorithm(algCorr_2);
                    network.setFunctionActiovationHidden(functionActiovationHidden);
                    network.setFunctionActiovationOut(functionActiovationOut);
                    FileReader fileReader = new FileReader(fileName);

                    worker.setNormalize(true);
//                    network.loadDataset(records, testExampleCount, samplingStep, offset);
                    network.loadDataset(fileReader, records, testExampleCount, samplingStep, offset);
                    network.initializationWeights();
                    network.learning(j, k);
                    network.testing();

                    System.out.println();
                    //System.out.println("Узлов: " + i + ", эпох: " + j + ", коэф. обучения: " + k);
                }
            }
        }
    }
}
