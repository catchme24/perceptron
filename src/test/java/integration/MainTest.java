package integration;

import data.*;
import data.converter.DatasetRowConverter;
import data.converter.IrisDatasetConverter;
import function.*;
import network.Network;
import network.NetworkBuilder;
import network.layer.ActivationLayer;
import network.layer.FullyConnected;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

public class MainTest {


    @BeforeEach


    @Test
    void FULLY_RELU_FULLY_RELU() throws IOException {
        DatasetRowConverter converter = new IrisDatasetConverter(4, ",");
        DatasetHelper datasetHelper = new DatasetHelperImpl();

        String pathIris = "src/main/resources/iris.data";
        File file = new File(pathIris);

        Dataset dataset = datasetHelper.prepareDataset(file, 70, converter);
        dataset.shuffleDataset();

        //GOOOOOOOOOD
        Network network = NetworkBuilder.builder()
                .append(new FullyConnected(4, 4))
                .append(new ActivationLayer(new ReLu()))
                .append(new FullyConnected(3))
                .append(new ActivationLayer(new ReLu()))
                .build();

        double learnRate = 0.01;
        int epoch = 70;

        network.learn(epoch, learnRate, dataset);
        network.test(dataset);
    }

    @Test
    void FULLY_LEAKY_FULLY_SIN_FULLY_LEAKY_FULLY_RELU_FULLY_LEAKY() throws IOException {
        DatasetRowConverter converter = new IrisDatasetConverter(4, ",");
        DatasetHelper datasetHelper = new DatasetHelperImpl();

        String pathIris = "src/main/resources/iris.data";
        File file = new File(pathIris);

        Dataset dataset = datasetHelper.prepareDataset(file, 75, converter);
        dataset.shuffleDataset();

        Network network = NetworkBuilder.builder()
                .append(new FullyConnected(3, 4))
                .append(new ActivationLayer(new LeakyReLu()))
                .append(new FullyConnected(6))
                .append(new ActivationLayer(new Sin()))
                .append(new FullyConnected(8))
                .append(new ActivationLayer(new LeakyReLu()))
                .append(new FullyConnected(10))
                .append(new ActivationLayer(new ReLu()))
                .append(new FullyConnected(3))
                .append(new ActivationLayer(new LeakyReLu()))
                .build();

        double learnRate = 0.01;
        int epoch = 120;

        network.learn(epoch, learnRate, dataset);
        network.test(dataset);
    }

    @Test
    void FULLY_RELU_FULLY_SOFTMAX() throws IOException {
        DatasetRowConverter converter = new IrisDatasetConverter(4, ",");
        DatasetHelper datasetHelper = new DatasetHelperImpl();

        String pathIris = "src/main/resources/iris.data";
        File file = new File(pathIris);

        Dataset dataset = datasetHelper.prepareDataset(file, 65, converter);

        Network network = NetworkBuilder.builder()
                .append(new FullyConnected(4, 4))
                .append(new ActivationLayer(new ReLu()))
                .append(new FullyConnected(3))
                .append(new ActivationLayer(new Softmax()))
                .build();

        double learnRate = 0.35;
        int epoch = 5;

        network.learn(epoch, learnRate, dataset);
//        network.test(dataset.getTestData());
    }
}
