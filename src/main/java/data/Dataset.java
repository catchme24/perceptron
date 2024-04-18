package data;

import java.util.List;

public interface Dataset {

    List<DatasetItem> getLearnData();

    List<DatasetItem>  getTestData();

    void shuffleDataset();

    int getLearnPercent();

    int getClassesCount();
}
