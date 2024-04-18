package data.converter;

import data.DatasetConvertingException;

public abstract class AbstractDatasetRowConverter implements DatasetRowConverter {

    private final int attributesCount;
    private final String delimeter;

    public AbstractDatasetRowConverter(int attributesCount, String delimeter) {
        if (delimeter == null) {
            throw new DatasetConvertingException("Delimeter cannot be null!");
        }
        this.attributesCount = attributesCount;
        this.delimeter = delimeter;
    }
    @Override
    public double[] convert(String datasetRow) {
        String[] values = datasetRow.split(delimeter);
        if (values.length < 2) {
            throw new DatasetConvertingException(   "Count of values in dataset row:" + values.length +
                                                    "\n" +
                                                    "Dataset row must have at least 2 values: attribute and class");
        }
        double[] result = new double[attributesCount + 1];

        for (int i = 0; i < result.length - 1; i++) {
            result[i] = convertAttribute(values[i]);
        }
        result[attributesCount] = convertClassName(values[values.length - 1]);
        return result;
    }

}
