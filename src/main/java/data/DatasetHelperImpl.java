package data;

import data.converter.DatasetRowConverter;

import java.io.*;

public class DatasetHelperImpl implements DatasetHelper {

    public DatasetHelperImpl() {}

    @Override
    public Dataset prepareDataset(File file, int learnPercent, DatasetRowConverter converter) throws IOException {
        Reader reader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(reader);
        BasicDataset dataset = new BasicDataset(learnPercent);

        while (bufferedReader.ready()) {
            String line = bufferedReader.readLine();
            double[] values = converter.convert(line);
            dataset.addRow(values);
        }
        return dataset;
    }
}
