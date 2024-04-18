package network;

import data.Dataset;

public interface Network {

    void learn(int epoch, double learnRate, Dataset dataset);

    void test(Dataset dataset);


}
