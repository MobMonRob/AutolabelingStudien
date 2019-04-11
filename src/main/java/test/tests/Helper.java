package test.tests;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class Helper {

    public static void saveModel(Model model, String saveDirectoryPath, String modelName) throws IOException {
        File modelSaveFile = new File(saveDirectoryPath + "\\" + modelName + "-v1.zip");
        while (modelSaveFile.exists()) {
            modelSaveFile = Helper.getNextPossibleFile(modelSaveFile);
        }
        ModelSerializer.writeModel(model, modelSaveFile, true);
    }

    public static Model loadModel(File modelFile) throws IOException {
        return ModelSerializer.restoreMultiLayerNetwork(modelFile);
    }

    public static void printINDArray(INDArray indArray) {
        double[] array = indArray.toDoubleVector();
        for (int i = 0; i < array.length; i++) {
            System.out.print(i + ": " + array[i] + " | ");
        }
        System.out.println();
    }

    public static File getNextPossibleFile(File modelSaveFile) {
        String currentPath = modelSaveFile.getPath();
        int indexOfVersion = currentPath.indexOf("-v");
        int integer = Integer.valueOf(currentPath.substring(indexOfVersion + 2, currentPath.lastIndexOf(".")));
        return new File(modelSaveFile.getParent() + ++integer + getFileExtension(modelSaveFile));
    }

    static String getFileExtension(File file) {
        String name = file.getName();
        int lastIndexOf = name.lastIndexOf(".");
        if (lastIndexOf == -1) {
            return "";
        }
        return name.substring(lastIndexOf);
    }

    public static <T, V> void logHashMap(HashMap<T, V> hashMap) {
        for (T t : hashMap.keySet()) {
            System.out.println("Key: " + t + ", Value: " + hashMap.get(t));
        }
    }

    public static ArrayList<MultiLayerConfiguration> getConfigsFromJSON(ArrayList<String> jsonStrings) {
        ArrayList<MultiLayerConfiguration> configurations = new ArrayList<>();
        for (String s : jsonStrings) {
            configurations.add(MultiLayerConfiguration.fromJson(s));
        }
        return configurations;
    }

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
