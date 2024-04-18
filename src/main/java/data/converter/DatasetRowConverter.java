package data.converter;

/*
    Конвертирует строку датасета в массив.
    Минимальная длинна возвращаемого массива - 2.
    Последнее значение всегда "класс" или лучше сказать "правильное" значение.
 */
public interface DatasetRowConverter {

    double[] convert(String datasetRow);

    double convertClassName(String className);

    double convertAttribute(String attribute);
}
