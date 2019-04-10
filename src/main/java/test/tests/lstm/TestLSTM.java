package test.tests.lstm;

import datavec.SequentialMarkerwiseTrialRecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import preprocess_data.SequentialDataPreprocessor;
import preprocess_data.builders.TrialDataManagerBuilder;
import preprocess_data.builders.TrialDataTransformationBuilder;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.labeling.FrameLabelingStrategy;
import preprocess_data.labeling.NoLabeling;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.TreeSet;

public class TestLSTM {

    public static void main(String[] args) throws Exception {
        File trainDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\trainData\\train");
        File testDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\testData\\test");
        String[] markerLabels = {"C7", "CLAV", "LASI", "LELB", "LELBW", "LHUM4", "LHUMA", "LHUMP", "LHUMS", "LRAD", "LSCAP1", "LSCAP2", "LSCAP3", "LSCAP4", "LULN", "RASI", "RELB", "RELBW", "RHUM4", "RHUMA", "RHUMP", "RHUMS", "RRAD", "RSCAP1", "RSCAP2", "RSCAP3", "RSCAP4", "RULN", "SACR", "STRN", "T10", "THRX1", "THRX2", "THRX3", "THRX4"};
        TreeSet<String> selectedLabels = new TreeSet<>(Arrays.asList(markerLabels));

        FrameLabelingStrategy frameLabelingStrategy = new NoLabeling();
        TrialDataManagerBuilder dataManager = TrialDataManagerBuilder
                .addTransformation(TrialDataTransformationBuilder
                        .addLabelingStrategy(frameLabelingStrategy)
                        .build())
                .withNormalization(new CentroidNormalization(-1,1));

        SequentialDataPreprocessor dataPreprocessor = new SequentialDataPreprocessor();
        String saveDirectoryTrain = "C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\save\\lstm\\train";
        String saveDirectoryTest = "C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\save\\lstm\\test";
        SequenceRecordReader recordReaderTrain;
        SequenceRecordReader recordReaderTest;
        if (SequentialDataPreprocessor.directoryHasData(saveDirectoryTest)
                && SequentialDataPreprocessor.directoryHasData(saveDirectoryTrain)) {
            recordReaderTrain = dataPreprocessor.getReader(saveDirectoryTrain);
            recordReaderTest = dataPreprocessor.getReader(saveDirectoryTest);
        } else {
            SequenceRecordReader train = new SequentialMarkerwiseTrialRecordReader(dataManager.build(), selectedLabels);
            SequenceRecordReader test = new SequentialMarkerwiseTrialRecordReader(dataManager.build(), selectedLabels);
            train.initialize(new FileSplit(trainDirectory));
            test.initialize(new FileSplit(testDirectory));

            recordReaderTrain = dataPreprocessor.saveDataToFile(train, saveDirectoryTrain);
            recordReaderTest = dataPreprocessor.saveDataToFile(test, saveDirectoryTest);
        }
        SequenceRecordReaderDataSetIterator trainIterator = new SequenceRecordReaderDataSetIterator(recordReaderTrain,
                1, 2,3,true); //numPossLabels has to be greater than 1 to allow multiple labels
        SequenceRecordReaderDataSetIterator testIterator = new SequenceRecordReaderDataSetIterator(recordReaderTest,
                1, 2,3,true);

        final MultiLayerConfiguration config = LSTMConfigs.simpleLSTM();
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(config);
        TrainingListener[] listeners = {
                new PerformanceListener(1,true),
        };
        /*DataSet next = trainIterator.next();
        List<INDArray> indArrays = multiLayerNetwork.feedForwardToLayer(1, next.getFeatures());

        System.out.println("Inputs");
        System.out.println(indArrays.get(0));
        System.out.println("LSTM");
        System.out.println(indArrays.get(1));*/
        multiLayerNetwork.setListeners(listeners);

        for (int i = 0; i < 5; i++) {
            multiLayerNetwork.fit(trainIterator);
            RegressionEvaluation regressionEvaluation = multiLayerNetwork.evaluateRegression(testIterator);
            System.out.println(regressionEvaluation.stats());
        }

        testIterator.reset();
        DataSet next = testIterator.next();
        System.out.println(next);
        List<INDArray> indArrays = multiLayerNetwork.feedForward(next.getFeatures());
        System.out.println("LSTM");
        System.out.println(indArrays.get(1));
        System.out.println("OutputLayer");
        System.out.println(indArrays.get(2));
    }
}
