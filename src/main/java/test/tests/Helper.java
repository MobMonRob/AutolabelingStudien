package test.tests;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

/*
Diese Klasse enhält Helper-Methoden, die für die Tests verwendet werden können.

Hinweis: Helper-Klassen sind Indikator für schlechtes Design. Alle folgenden Methoden könnten auch in eigenen Klassen
implementiert werden.
*/
public class Helper {

    /*Speichern eines trainierten Models mit der entsprechenden Methode von DL4J. Mit den Parametern kann der Speicherort
    * und der Dateiname spezifiziert werden. Jeder Dateiname wird automatisch mit einem Index versehen, sodass das
    * Training mehrfach augeführt werden kann, ohne einen neuen Dateinamen anzugeben */
    public static void saveModel(Model model, String saveDirectoryPath, String modelName) throws IOException {
        File modelSaveFile = new File(saveDirectoryPath + "\\" + modelName + "-v1.zip");
        while (modelSaveFile.exists()) {
            modelSaveFile = Helper.getNextPossibleFile(modelSaveFile);
        }
        ModelSerializer.writeModel(model, modelSaveFile, true);
    }

    //Ausgeben aller Elemente eines INDArray
    public static void printINDArray(INDArray indArray) {
        double[] array = indArray.toDoubleVector();
        for (int i = 0; i < array.length; i++) {
            System.out.print(i + ": " + array[i] + " | ");
        }
        System.out.println();
    }

    //Methode, die den nächsten freien Dateinamen auf Basis eines Indexes zurückgibt.
    public static File getNextPossibleFile(File modelSaveFile) {
        String currentPath = modelSaveFile.getPath();
        String fileName = modelSaveFile.getName();

        int indexOfVersion = currentPath.indexOf("-v");
        String modelName = fileName.substring(0,fileName.indexOf("-v"));
        int integer = Integer.valueOf(currentPath.substring(indexOfVersion + 2, currentPath.lastIndexOf(".")));
        return new File(modelSaveFile.getParent() + "\\" + modelName + "-v" + ++integer + getFileExtension(modelSaveFile));
    }

    //Rückgabe des File-Extension einer Datei
    public static String getFileExtension(File file) {
        String name = file.getName();
        int lastIndexOf = name.lastIndexOf(".");
        if (lastIndexOf == -1) {
            return "";
        }
        return name.substring(lastIndexOf);
    }

    //Methode zum Ausgeben einer HashMap
    public static <T, V> void logHashMap(HashMap<T, V> hashMap) {
        for (T t : hashMap.keySet()) {
            System.out.println("Key: " + t + ", Value: " + hashMap.get(t));
        }
    }

    //Ausgeben von Evaluationsdetails eines Testdatensatzes
    public static void logSingleEvaluationDetails(MultiLayerNetwork network, DataSetIterator testIterator) {
        System.out.println("start evaluation");
        testIterator.reset();

        org.nd4j.linalg.dataset.DataSet testData = testIterator.next();
        INDArray features = testData.getFeatures();
        INDArray prediction = network.output(features);
        System.out.println("single eval:");
        System.out.println("Num Labels: " + network.numLabels());
        Evaluation evaluation = new Evaluation(network.numLabels());
        evaluation.eval(testData.getLabels().getRow(0), prediction.getRow(0));
        System.out.println(evaluation.stats(false, true));
        System.out.println("Datensatz 1 --> Features: ");
        Helper.printINDArray(features.getRow(0));
        System.out.println("Datensatz 1 --> Prediction: ");
        Helper.printINDArray(prediction.getRow(0));
        System.out.println("geschätzter Wert: ");
        System.out.println(prediction.getRow(0).maxNumber());
    }
}
