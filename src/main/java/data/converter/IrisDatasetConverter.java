package data.converter;

public class IrisDatasetConverter extends AbstractDatasetRowConverter {

    public IrisDatasetConverter(int attributesCount, String delimeter) {
        super(attributesCount, delimeter);
    }

    @Override
    public double convertClassName(String className) {
        switch (className) {
            case "Iris-setosa":
                return 1.0;
            case "Iris-versicolor":
                return 2.0;
            case "Iris-virginica":
                return 3.0;
            default:
                return 1.0;
        }
    }

    @Override
    public double convertAttribute(String attribute) {
        return Double.parseDouble(attribute);
    }


}
