package old;

import au.com.bytecode.opencsv.CSVReader;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.*;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.*;

public class DatasetWorker {
    private File file;
    private final Array2DRowRealMatrix matrixWorker;
    private int learnPercent;
    private int classesCount;
    private RealMatrix dataset_matrix;
    private RealMatrix learnMatrix;
    private RealMatrix testMatrix;
    private ArrayList<String> classesNames;
    private ArrayList generatedList;

    private double offset = 0;

    private double mathExpectation = 0;
    private double dispersion = 0;

    private boolean normalize = true;

    public DatasetWorker() {
        this.matrixWorker = new Array2DRowRealMatrix();
    }

    public int getLearnPercent() {
        return learnPercent;
    }
    public void setLearnPercent(int learnPercent) {
        this.learnPercent = learnPercent;
    }

    public ArrayList<String> getClassesNames() {
        return classesNames;
    }
    public void setClassesNames(ArrayList<String> classesNames) {
        this.classesNames = classesNames;
    }


    public File getFile() {
        return file;
    }
    public void setFile(File file) {
        this.file = file;
    }

    public int getClassesCount() {
        return classesCount;
    }
    public void setClassesCount(int classesCount) {
        this.classesCount = classesCount;
    }

    public double getMathExpectation() {return mathExpectation;}
    public void setMathExpectation(double mathExpectation) {this.mathExpectation = mathExpectation;}

    public double getDispersion() {return dispersion;}
    public void setDispersion(double dispersion) {this.dispersion = dispersion;}

    public boolean isNormalize() {return normalize;}
    public void setNormalize(boolean normalize) {this.normalize = normalize;}

    //Методы воркера
    private ArrayList<String> datasetToList() throws IOException {
        ArrayList<String> list = new ArrayList<>();

        Reader reader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(reader);

        while (bufferedReader.ready()) {
            list.add(bufferedReader.readLine());
        }
        return list;
    }

    private ArrayList<double[]> prepareList(ArrayList<String> list) {
        ArrayList<double[]> newList = new ArrayList<>();
        classesNames = new ArrayList<>();

            for (int i = 0; i < list.size(); i++) {
                String[] line = list.get(i).split(",");
                newList.add(new double[line.length]);
                if(!classesNames.contains(line[line.length - 1])) classesNames.add(line[line.length - 1]);
                for (int j = 0; j < newList.get(i).length; j++) {
                    if(j == newList.get(i).length - 1) {
                        if(classesNames.contains(line[j])) {
                            newList.get(i)[j] = classesNames.indexOf(line[j]) + 1;
                        }
                    } else {
                        newList.get(i)[j] = Double.parseDouble(line[j]);
                    }
                }
            }
        classesCount = classesNames.size();
        return newList;
    }
    private ArrayList<double[]> prepareListCar(ArrayList<String> list) {
        ArrayList<double[]> newList = new ArrayList<>();
        classesNames = new ArrayList<>();

        for (int i = 0; i < list.size(); i++) {
            String[] line = list.get(i).split(",");
            newList.add(new double[line.length]);

            if(!classesNames.contains(line[line.length - 1])) classesNames.add(line[line.length - 1]);

            switch (line[0]) {
                case "low":
                    newList.get(i)[0] = 1.0;
                    break;
                case "med":
                    newList.get(i)[0] = 2.0;
                    break;
                case "high":
                    newList.get(i)[0] = 3.0;
                    break;
                case "vhigh":
                    newList.get(i)[0] = 4.0;
                    break;
            }

            switch (line[1]) {
                case "low":
                    newList.get(i)[1] = 1.0;
                    break;
                case "med":
                    newList.get(i)[1] = 2.0;
                    break;
                case "high":
                    newList.get(i)[1] = 3.0;
                    break;
                case "vhigh":
                    newList.get(i)[1] = 4.0;
                    break;
            }

            switch (line[2]) {
                case "2":
                    newList.get(i)[2] = 1.0;
                    break;
                case "3":
                    newList.get(i)[2] = 2.0;
                    break;
                case "4":
                    newList.get(i)[2] = 3.0;
                    break;
                case "5more":
                    newList.get(i)[2] = 4.0;
                    break;
            }

            switch (line[3]) {
                case "2":
                    newList.get(i)[3] = 1.0;
                    break;
                case "4":
                    newList.get(i)[3] = 2.0;
                    break;
                case "more":
                    newList.get(i)[3] = 3.0;
                    break;
            }

            switch (line[4]) {
                case "small":
                    newList.get(i)[4] = 1.0;
                    break;
                case "med":
                    newList.get(i)[4] = 2.0;
                    break;
                case "big":
                    newList.get(i)[4] = 3.0;
                    break;
            }

            switch (line[5]) {
                case "low":
                    newList.get(i)[5] = 1.0;
                    break;
                case "med":
                    newList.get(i)[5] = 2.0;
                    break;
                case "high":
                    newList.get(i)[5] = 3.0;
                    break;
            }

            switch (line[6]) {
                case "unacc":
                    newList.get(i)[6] = 1.0;
                    break;
                case "acc":
                    newList.get(i)[6] = 2.0;
                    break;
                case "good":
                    newList.get(i)[6] = 3.0;
                    break;
                case "vgood":
                    newList.get(i)[6] = 4.0;
                    break;
            }


        }

        double[][] theArray = new double[newList.size()][newList.get(0).length];

        for(int i = 0; i < theArray.length; i++) {
            theArray[i] = newList.get(i);
        }

        Arrays.sort(theArray, (b, a) -> (int) (b[6] - a[6]));


        for(int i = 0; i < theArray.length; i++) {
            newList.set(i, theArray[i]);
        }

        ArrayList<double[]> lastList = new ArrayList<>();
        int index = 0;

        for(int i = 1; i <= classesNames.size(); i++) {
            int count = 0;
            for (int j = 0; j < newList.size(); j++) {
                if(newList.get(j)[6] == i) {
                    if(count < 65) {
                        lastList.add(newList.get(j));
                        index++;
                        count++;
                    } else {
                        break;
                    }
                }
            }
        }

        classesCount = classesNames.size();
        return lastList;
    }

    private ArrayList<String> prepareListWine(ArrayList<String> list) {
        ArrayList<String> newList = new ArrayList<>();
        ArrayList<String> lastList = new ArrayList<>();

        for(int k = 1; k <= 3; k++) {
            int count = 0;
            for (int i = 0; i < list.size(); i++) {
                String[] line = list.get(i).split(",");

                if(count < 48 && line[0].equals(Integer.toString(k))) {
                    String first = line[0];
                    String last = line[line.length - 1];

                    line[0] = last;
                    line[line.length - 1] = first;

                    StringBuilder sb = new StringBuilder();
                    for (int j = 0; j < line.length; j++) {
                        sb.append(line[j]);
                        if (j != line.length - 1) {
                            sb.append(",");
                        }
                    }
                    newList.add(sb.toString());
                    count++;
                }
            }
        }

        return newList;
    }
    private void prepareLearnAndTestMatrix(RealMatrix dataset_matrix) {
        int learnCount = (dataset_matrix.getRowDimension() * learnPercent) / 100;
        int learnClassLength = learnCount / classesCount;
        int testCount = (dataset_matrix.getRowDimension() * (100 - learnPercent)) / 100;
        int testClassLength = testCount/ classesCount;

        learnMatrix = matrixWorker.createMatrix(learnCount, dataset_matrix.getColumnDimension());
        testMatrix = matrixWorker.createMatrix(testCount, dataset_matrix.getColumnDimension());

        int indexLearn = 0;
        int indexTest = 0;
        int index = 0;

        for (int i = 0; i < classesCount; i ++) {
            int learnRecorded = 0;
            int testRecorded = 0;

            for (int j = index; j < dataset_matrix.getRowDimension(); j++) {

                if (learnRecorded == learnClassLength && testRecorded == testClassLength) {
                    break;
                } else {
                    if (learnRecorded != learnClassLength) {
                        learnMatrix.setRow(indexLearn, dataset_matrix.getRow(j));
                        learnRecorded++;
                        indexLearn++;
                    } else {
                        testMatrix.setRow(indexTest, dataset_matrix.getRow(j));
                        testRecorded++;
                        indexTest++;
                    }
                }
                index++;
            }
        }

    }
    public RealMatrix getLearnMatrix() {

        return learnMatrix;
    }
    public RealMatrix getTestMatrix() {

        return testMatrix;
    }

    public void showClassesCount(RealMatrix matrix) {
        ArrayList<Integer> classes = new ArrayList<>();
        ArrayList<Integer> indexes = new ArrayList<>();

        for (int i = 0; i < matrix.getRowDimension(); i++) {
            int numberOfClass = (int) matrix.getEntry(i, matrix.getColumnDimension() - 1);
            try {
                classes.get(numberOfClass - 1);
            } catch (Exception e) {
                for(int j = 0; j < numberOfClass - classes.size(); j++) {classes.add(0);}
            } finally {
                classes.set(numberOfClass - 1, classes.get(numberOfClass - 1) + 1);
            }
        }

        for (int i = 0; i < classes.size(); i++) {
            System.out.print("Примеров " + (i + 1) + "-ого класса: " + classes.get(i) + " ");
        }
        System.out.println();
    }
    private RealMatrix toMatrixDataset(ArrayList<double[]> dataset) {

        RealMatrix table = matrixWorker.createMatrix(dataset.size(), dataset.get(0).length);

        for(int i = 0; i < dataset.size(); i++) {
            table.setRow(i, dataset.get(i));
        }

        return table;
    }

    public ArrayList generateSin(int record, int testExampleCount, int samplingStep, double offset) {
        this.offset = offset;
        return generateSin(record, testExampleCount, samplingStep);
    }

    public ArrayList generateSin(int record, int testExampleCount, int samplingStep) {
        double step = 2 * Math.PI / record;
        generatedList = new ArrayList();
        double x = - Math.PI - this.offset;
        //Заполнение листа значениями синуса
//        for(int i = 0; i < record; i++, x+= step) {
//            generatedList.add(Math.sin(x));
//        }
        for(int i = 0; i < record; i++) {
            generatedList.add(Math.sin(i));
        }

        //Создание обучающей и тестовой мартиц
        learnMatrix = matrixWorker.createMatrix(record - testExampleCount - samplingStep, samplingStep + 1);
        testMatrix = matrixWorker.createMatrix(testExampleCount, samplingStep + 1);

        //Заполнение матриц значениями из генерированного листа
        int i = 0;
        int indexTestMatrix;

        for(i = 0; i < generatedList.size() - testExampleCount - samplingStep; i++) {
            for(int j = 0; j < samplingStep + 1; j++) {
                learnMatrix.setEntry(i, j, (double) generatedList.get(i + j));
            }
        }

        int k = 0;

        for(i = generatedList.size() - testExampleCount - samplingStep, indexTestMatrix = 0; i < generatedList.size() - samplingStep; i++, indexTestMatrix++) {
            for(int j = 0; j < samplingStep + 1; j++) {
                testMatrix.setEntry(indexTestMatrix, j, (double) generatedList.get(i + j));
            }
        }

        System.out.println(generatedList.toString());
        return generatedList;
    }

    public void prepareDataset(File file, int learnPercent) throws IOException {
        this.file = file;
        this.learnPercent = learnPercent;
        ArrayList<String> rawDataset = datasetToList();
        ArrayList<double[]> dataset = prepareList(rawDataset);
        //ArrayList<double[]> dataset = prepareListCar(rawDataset);
        dataset_matrix = toMatrixDataset(dataset);
        prepareLearnAndTestMatrix(dataset_matrix);
    }

    public void prepareDataset(int record, int testExampleCount, int samplingStep) {
        generateSin(record, testExampleCount, samplingStep);
    }

    public void prepareDataset(int record, int testExampleCount, int samplingStep, double offset) {
        generateSin(record, testExampleCount, samplingStep, offset);
    }

    public void readFile(FileReader fileReader) throws IOException {
        CSVReader reader = new CSVReader(fileReader, ',', '"', 1);
        String[] nextLine;
        //Создание листа
        this.generatedList = new ArrayList();
        NumberFormat format = NumberFormat.getInstance(Locale.getDefault());
        //Заполнение листа
        int count = 0;
        while ((nextLine = reader.readNext()) != null) {
            generatedList.add(nextLine[1]);
            count++;
        }
    }

    public void offsetData(int offset, int record) throws ParseException {
        //Переворот листа
        NumberFormat format = NumberFormat.getInstance(Locale.getDefault());
        Collections.reverse(this.generatedList);
        //Добавление смещения
        ArrayList timeList = new ArrayList();
        for(int j = 0; j < this.generatedList.size(); j++) {
            timeList.add(this.generatedList.get(j));
        }

        this.generatedList.removeAll(this.generatedList);

        for(int j = 0; j < record; j++) {
            this.generatedList.add(format.parse((String) timeList.get(j + offset)).doubleValue());
        }
    }

    public void prepareDataset(FileReader fileReader, int record, int testExampleCount, int samplingStep, int offset) throws IOException, ParseException {
        //Чтение строк из файла
        readFile(fileReader);
        //Взятие нужного кол-ва строк + добавление смещения
        offsetData(offset, record);
        System.out.println("Значения      : " + this.generatedList.toString());
        //Нормализация данных с помощью мат ожидания и дисперсии
        if(normalize) {
            normalize();
        }
        //Создание обучающей и тестовой мартиц
        learnMatrix = matrixWorker.createMatrix(record - testExampleCount - samplingStep, samplingStep + 1);
        testMatrix = matrixWorker.createMatrix(testExampleCount, samplingStep + 1);
        //Заполнение матриц значениями из генерированного листа
        fillTestAndLearnMatrix(record, testExampleCount, samplingStep);

//        XYSeries series1 = new XYSeries("200 рабочих дней");
//        for(int i = 0; i < generatedList.size(); i++) {
//            series1.add(i, (double) generatedList.get(i));
//        }
//        XYSeriesCollection xyDataset = new XYSeriesCollection(series1);
//
//        JFreeChart chart = ChartFactory.createXYLineChart("Цена акции", "x", "y",
//                xyDataset,
//                PlotOrientation.VERTICAL,
//                true, true, true);
//
//        XYPlot xyPlot = chart.getXYPlot();
//        ValueAxis domainAxis = xyPlot.getDomainAxis();
//        ValueAxis rangeAxis = xyPlot.getRangeAxis();
//        domainAxis.setRange(0.0, 200);
//        //domainAxis.setTickUnit(new NumberTickUnit(0.1));
//        rangeAxis.setRange(80, 170);
//        //rangeAxis.setTickUnit(new NumberTickUnit(0.05));
//
//        JFrame frame = new JFrame("MinimalStaticChart");
//        // Помещаем график на фрейм
//        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
//        frame.getContentPane().add(new ChartPanel(chart));
//        frame.setSize(400,300);
//        frame.setVisible(true);

    }

    private void fillTestAndLearnMatrix(int record, int testExampleCount, int samplingStep) throws ParseException {
        int i = 0;
        int indexList = 0;
        int indexTestMatrix;

        for(i = 0; i < generatedList.size() - testExampleCount - samplingStep; i++) {
            for(int j = 0; j < samplingStep + 1; j++) {
                learnMatrix.setEntry(i, j, (Double) generatedList.get(i + j));
            }
        }

        int k = 0;

        for(i = generatedList.size() - testExampleCount - samplingStep, indexTestMatrix = 0; i < generatedList.size() - samplingStep; i++, indexTestMatrix++) {
            for(int j = 0; j < samplingStep + 1; j++) {
                testMatrix.setEntry(indexTestMatrix, j, (Double) generatedList.get(i + j));
            }
        }
    }

    public void normalize() throws ParseException {
        //Подсчет мат. ожидания и дисперсии
        //
        NumberFormat format = NumberFormat.getInstance(Locale.getDefault());
        double mathExpectation = 0;
        double dispersion = 0;
        //подсчет общей суммы для мат ожидание
        double total = 0;
        for(int j = 0; j < generatedList.size(); j++) {
            double current = (double) generatedList.get(j);
            total += current;
        }
        //Само мат ожидание
        mathExpectation = total/generatedList.size();
//        System.out.println("Мат.ожид.     : " + mathExpectation);
        this.mathExpectation = mathExpectation;
        //подсчет суммы разности в квадрате для дисперсии
        double totalBrace = 0;

        ArrayList differenceList = new ArrayList();
        ArrayList powList = new ArrayList();

        for(int j = 0; j < generatedList.size(); j++) {
            double current = (double) generatedList.get(j);
            double difference = current - mathExpectation;
            differenceList.add(difference);
            totalBrace = totalBrace + Math.pow(difference, 2);
            powList.add(Math.pow(difference, 2));
        }
//        System.out.println("Отклонения    : " + differenceList.toString());
//        System.out.println("Квадраты откл.: " + powList.toString());
        //Сама дисперсия
        dispersion = totalBrace/(generatedList.size() - 1);
//        System.out.println("Дисперсия     : " + dispersion);
//        System.out.println("Корень дисперс: " + Math.sqrt(dispersion));
        this.dispersion = dispersion;
        //Заполнение листа нормализованными данными
        for(int j = 0; j < generatedList.size(); j++) {
            double current = (double) generatedList.get(j);
            double normalizedCurrent = (current - mathExpectation)/Math.sqrt(dispersion);
//            double normalizedCurrent = current / 1000;
            generatedList.set(j, normalizedCurrent + 2);
        }
//        System.out.println("Итого нормализ: " + generatedList.toString());
    }
}
