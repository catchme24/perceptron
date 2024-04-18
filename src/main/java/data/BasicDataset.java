package data;

import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

@Slf4j
public class BasicDataset implements Dataset {

    private final List<DatasetItem> items;
    private final HashSet<Double> groundTruthCount;
    private final int learnPercent;

    public BasicDataset(int learnPercent) {
        this.learnPercent = learnPercent;
        this.items = new ArrayList<>();
        this.groundTruthCount = new HashSet<>();
    }

    void addRow(double[] row) {
        log.debug("Adding row: {}", row);
        groundTruthCount.add(getLastElement(row));
        items.add(new DatasetItemImpl(row));
        log.debug("Row added");
    }

    private int getIndexOfTestData() {
        return (items.size() * learnPercent / 100);
    }

    private double getLastElement(double[] row) {
        return row[row.length - 1];
    }

    @Override
    public List<DatasetItem> getLearnData() {
        List<DatasetItem> learn = new ArrayList<>();
        for (int i = 0; i < getIndexOfTestData(); i++) {
            learn.add(items.get(i));
        }
        return learn;
    }

    @Override
    public List<DatasetItem> getTestData() {
        List<DatasetItem> test = new ArrayList<>();
        for (int i = getIndexOfTestData(); i < items.size(); i++) {
            test.add(items.get(i));
        }
        return test;
    }

    public void shuffleDataset() {
        Collections.shuffle(items);
    }

    @Override
    public int getLearnPercent() {
        return learnPercent;
    }

    @Override
    public int getClassesCount() {
        return groundTruthCount.size();
    }

}