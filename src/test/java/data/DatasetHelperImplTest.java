package data;

import data.converter.DatasetRowConverter;
import data.converter.IrisDatasetConverter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

public class DatasetHelperImplTest {

    @Test
    void calculateOutput() throws IOException {
        DatasetRowConverter converter = new IrisDatasetConverter(4, ",");
        int learnPercent  = 63;
        int rowsCount = 150;

        String pathIris = "src/main/resources/iris.data";
        File file = new File(pathIris);

        DatasetHelperImpl datasetHelper = new DatasetHelperImpl();

        Dataset dataset = datasetHelper.prepareDataset(file, learnPercent, converter);

        Assertions.assertEquals(dataset.getTestData().size() + dataset.getLearnData().size(), rowsCount);
        Assertions.assertEquals(dataset.getLearnPercent(), learnPercent);
    }
}
