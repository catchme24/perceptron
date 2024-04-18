package data;

public class DatasetItemImpl implements DatasetItem  {

    private double[] row;
    public DatasetItemImpl(double[] row) {
        this.row = row;
    }

    @Override
    public double[] getAttributes() {
        double[] attributes = new double[row.length - 1];
        System.arraycopy(row, 0, attributes, 0, row.length - 1);
        return attributes;
    }

    @Override
    public double getClassNumber() {
        return row[row.length - 1];
    }


    @Override
    public String toString() {
        return row.toString();
    }
}
