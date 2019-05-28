package test.tests.one_marker_labeling.convolution;

import datavec.RandomizedTrialRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.evaluation.classification.Evaluation;
import preprocess_data.builders.TrialDataManagerBuilder;
import preprocess_data.builders.TrialDataTransformationBuilder;
import preprocess_data.data_manipulaton.FrameShuffleManipulator;
import preprocess_data.data_manipulaton.RandomFrameRotationManipulator;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.labeling.OneTargetLabeling;
import preprocess_data.preprocessors.FrameDataPreprocessor;

import java.io.File;
import java.util.Arrays;
import java.util.TreeSet;

public class TestCNNConfigs {
    public static void main(String[] args) throws Exception {

        //Trainings- und Testdaten --> Pfade anpassen!
        File trainDirectory = new File("C:\\Users\\...");
        File testDirectory = new File("C:\\Users\\...");
        String[] markerLabels = {"C7", "CLAV", "LASI", "LELB", "LELBW", "LHUM4", "LHUMA", "LHUMP", "LHUMS", "LRAD", "LSCAP1", "LSCAP2", "LSCAP3", "LSCAP4", "LULN", "RASI", "RELB", "RELBW", "RHUM4", "RHUMA", "RHUMP", "RHUMS", "RRAD", "RSCAP1", "RSCAP2", "RSCAP3", "RSCAP4", "RULN", "SACR", "STRN", "T10", "THRX1", "THRX2", "THRX3", "THRX4"};
        TreeSet<String> selectedLabels = new TreeSet<>(Arrays.asList(markerLabels));
        int shuffles = selectedLabels.size();


        //Konfiguration der Daten
        TrialDataManagerBuilder dataManagerBuilder = TrialDataManagerBuilder.addTransformation(TrialDataTransformationBuilder
                .addLabelingStrategy(new OneTargetLabeling("LELB", selectedLabels.size()))
                .withManipulation(new FrameShuffleManipulator(shuffles))
                .withManipulation(new RandomFrameRotationManipulator(5))
                .build())
                .withNormalization(new CentroidNormalization(-1, 1))
                .filterMarkers(selectedLabels);


        //Persistieren der Daten nach der Vorverarbeitung
        FrameDataPreprocessor preprocessor = new FrameDataPreprocessor(500);
        //Pfad zum Save-Directory --> anpassen!
        String saveTrain = "C:\\Users\\...";
        String saveTest = "C:\\Users\\...";

        int batchSize = 10;
        int recordReaderStorage = 50000;
        RecordReaderDataSetIterator trainIterator;
        RecordReaderDataSetIterator testIterator;

        //Falls bereits Daten im Save-Direcory sind wird der Vorverarbeitungsprozess nicht noch einmal ausgeführt
        if (FrameDataPreprocessor.directoryHasData(saveTrain) && FrameDataPreprocessor.directoryHasData(saveTest)) {
            trainIterator = new RecordReaderDataSetIterator(preprocessor.getReader(saveTrain), batchSize, 105, 35);
            testIterator = new RecordReaderDataSetIterator(preprocessor.getReader(saveTest), batchSize, 105, 35);
        } else {
            //Persistieren der Daten
            RandomizedTrialRecordReader train = new RandomizedTrialRecordReader(dataManagerBuilder.build(), recordReaderStorage);
            RandomizedTrialRecordReader test = new RandomizedTrialRecordReader(dataManagerBuilder.build(), recordReaderStorage);
            train.initialize(new FileSplit(trainDirectory));
            test.initialize(new FileSplit(testDirectory));
            trainIterator = new RecordReaderDataSetIterator(preprocessor.saveDataToFile(train, saveTrain), batchSize, 105, 35);
            testIterator = new RecordReaderDataSetIterator(preprocessor.saveDataToFile(test, saveTest), batchSize, 105, 35);
        }

        //Listener für das Training
        TrainingListener[] listeners = {
                new EvaluativeListener(testIterator, 1, InvocationType.EPOCH_END),
                new PerformanceListener(10000,true)
        };

        //Initialisieren der CNNs (ComputationGraphConfigs)
        ComputationGraphConfiguration graph = ConvolutionConfigs.twoReshapes(selectedLabels, batchSize, 40, 20);
        ComputationGraph computationGraph = new ComputationGraph(graph);
        computationGraph.init();
        computationGraph.addListeners(listeners);

        //Training und Evaluation
        computationGraph.fit(trainIterator, 20);

        Evaluation eval = computationGraph.evaluate(testIterator);
        System.out.println(eval.stats(true, true));
    }
}

