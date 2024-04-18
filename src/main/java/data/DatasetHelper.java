package data;

import data.converter.DatasetRowConverter;

import java.io.File;
import java.io.IOException;

public interface DatasetHelper {

    Dataset prepareDataset(File file, int learnPercent, DatasetRowConverter converter) throws IOException;
}
