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
import preprocess_data.builders.TrialDataManagerBuilder;
import preprocess_data.builders.TrialDataTransformationBuilder;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.labeling.FrameLabelingStrategy;
import preprocess_data.labeling.NoLabeling;
import preprocess_data.preprocessors.SequentialDataPreprocessor;

import java.io.File;
import java.util.Arrays;
import java.util.TreeSet;

//Klasse zum Testen von LSTMs, Empfehlung: GPU verwenden! --> siehe pom.xml
public class TestLSTM {

    public static void main(String[] args) throws Exception {

        //Pfad zu den Trainings- und Testdaten. Pfade entsprechend anpassen!
        File trainDirectory = new File("C:\\Users\\...");
        File testDirectory = new File("C:\\Users\\...");

        //Auswahl der Marker, die eingelesen werden sollen.
        //Hier: alle Marker --> Array wird nur zur Vereinfachung möglicher Anpassungen/Tests erstellt...)
        String[] markerLabels = {"C7", "CLAV", "LASI", "LELB", "LELBW", "LHUM4", "LHUMA", "LHUMP", "LHUMS", "LRAD", "LSCAP1", "LSCAP2", "LSCAP3", "LSCAP4", "LULN", "RASI", "RELB", "RELBW", "RHUM4", "RHUMA", "RHUMP", "RHUMS", "RRAD", "RSCAP1", "RSCAP2", "RSCAP3", "RSCAP4", "RULN", "SACR", "STRN", "T10", "THRX1", "THRX2", "THRX3", "THRX4"};
        TreeSet<String> selectedLabels = new TreeSet<>(Arrays.asList(markerLabels));

        //Konfiguration der Daten. Anmerkung: Stand jetzt ist noch keine Vermehrung mit sequentiellen Daten möglich.
        //Bei der Verwendung von Manipulation-Strategien dürfen die Daten daher nicht vermehrt werden.
        FrameLabelingStrategy frameLabelingStrategy = new NoLabeling();
        TrialDataManagerBuilder dataManager = TrialDataManagerBuilder
                .addTransformation(TrialDataTransformationBuilder
                        .addLabelingStrategy(frameLabelingStrategy)
                        .build())
                .withNormalization(new CentroidNormalization(-1, 1));


        //*************************************************************************************************************
        //Zur Optimierung der Performance werden die Daten persistiert. Die Vorverabeitung wird dadurch
        //nur einmal durchgeführt
        SequentialDataPreprocessor dataPreprocessor = new SequentialDataPreprocessor();

        //Pfad zu den Speicherorten für die vorverarbeiteten Daten --> anpassen!
        String saveDirectoryTrain = "C:\\Users\\...";
        String saveDirectoryTest = "C:\\Users\\...";

        SequenceRecordReader recordReaderTrain;
        SequenceRecordReader recordReaderTest;
        //Überprüfen ob bereits Daten in diesem Directory gespeichert wurden. Warum? --> Dadurch kann der Code dieser
        //Klasse beliebig oft ausgeführt werden, ohne das der Vorverarbeitunsprozess wiederholt wird.
        if (SequentialDataPreprocessor.directoryHasData(saveDirectoryTest)
                && SequentialDataPreprocessor.directoryHasData(saveDirectoryTrain)) {
            //Safe-Directory hat Daten --> einfaches Inititalisieren von SequenceRecordReadern
            recordReaderTrain = dataPreprocessor.getReader(saveDirectoryTrain);
            recordReaderTest = dataPreprocessor.getReader(saveDirectoryTest);
        } else {
            //Speichern der Daten im angegebenen Speicherort

            //Initialisierung der RecordReader mit der Konfiguration der Daten (TrialDataManager)
            SequenceRecordReader train = new SequentialMarkerwiseTrialRecordReader(dataManager.build(), selectedLabels);
            SequenceRecordReader test = new SequentialMarkerwiseTrialRecordReader(dataManager.build(), selectedLabels);
            train.initialize(new FileSplit(trainDirectory));
            test.initialize(new FileSplit(testDirectory));

            //Verwendung des DataPreprocessors zum Speichern der Daten
            recordReaderTrain = dataPreprocessor.saveDataToFile(train, saveDirectoryTrain);
            recordReaderTest = dataPreprocessor.saveDataToFile(test, saveDirectoryTest);
        }

        //Initialisierung eines DataSetIterators zum Trainieren/Testen
        //Vorgabe von DL4J --> numPossLabels muss größer als 1 sein, um mehrere Label zu haben (bei Regression)
        //labelIndex = 3 --> Label beginnen ab dem Index 3.
        SequenceRecordReaderDataSetIterator trainIterator = new SequenceRecordReaderDataSetIterator(recordReaderTrain,
                10, 2, 3, true);
        SequenceRecordReaderDataSetIterator testIterator = new SequenceRecordReaderDataSetIterator(recordReaderTest,
                10, 2, 3, true);

        //Initialisierung des LSTM
        final MultiLayerConfiguration config = LSTMConfigs.simpleLSTM(); //Model-Konfiguration
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(config);
        //Listener für das Training
        TrainingListener[] listeners = {
                new PerformanceListener(1000, true),
        };
        multiLayerNetwork.setListeners(listeners);

        //Training und Evaluation für jede Epoche
        int epochs = 10;
        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch: " + (i + 1));
            multiLayerNetwork.fit(trainIterator);
            RegressionEvaluation regressionEvaluation = multiLayerNetwork.evaluateRegression(testIterator);
            System.out.println(regressionEvaluation.stats());
        }

        //Speichern des trainierten Modells:
        /*String saveDirectory = "C:\\Users\\...";
        Helper.saveModel(multiLayerNetwork, saveDirectory, "lstm");*/
    }
}
