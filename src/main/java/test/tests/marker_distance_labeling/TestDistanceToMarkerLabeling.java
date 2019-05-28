package test.tests.marker_distance_labeling;

import datavec.JsonTrialRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import preprocess_data.TrialDataTransformation;
import preprocess_data.builders.TrialDataManagerBuilder;
import preprocess_data.data_manipulaton.FrameReorderingManipulator;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.data_normalization.TrialNormalizationStrategy;
import preprocess_data.labeling.DistanceToMarkerLabeling;
import test.tests.Helper;

import java.io.File;

//Training mit den Distanzen jedes Markers zu den anderen Markern. Überprüfen ob eine bestimmte Reihenfolge vorliegt.
public class TestDistanceToMarkerLabeling {

    public static void main(String[] args) throws Exception {

        String[] allowedFileFormat = {"json"};
        //Input Data --> Pfade anpassen!
        File trainDirectory = new File("C:\\Users\\Nico Rinck\\...");
        File testDirectory = new File("C:\\Users\\Nico Rinck\\...");
        FileSplit fileSplitTrain = new FileSplit(trainDirectory, allowedFileFormat);
        FileSplit fileSplitTest = new FileSplit(testDirectory, allowedFileFormat);

        //Konfiguration der Daten*************************************************************************************
        //richtige (default) Reihenfolge:
        String[] orderedLabels = {"C7", "CLAV", "LASI", "LELB", "LELBW", "LHUM4", "LHUMA", "LHUMP", "LHUMS", "LRAD", "LSCAP1", "LSCAP2", "LSCAP3", "LSCAP4", "LULN", "RASI", "RELB", "RELBW", "RHUM4", "RHUMA", "RHUMP", "RHUMS", "RRAD", "RSCAP1", "RSCAP2", "RSCAP3", "RSCAP4", "RULN", "SACR", "STRN", "T10", "THRX1", "THRX2", "THRX3", "THRX4"};
        //Initialisierung der Labeling-Strategie mit der korrekten Reihenfolge
        DistanceToMarkerLabeling frameLabelingStrategy = new DistanceToMarkerLabeling(orderedLabels);
        TrialNormalizationStrategy normalizationStrategy = new CentroidNormalization();
        //Umsortierung und Vermehrung der Daten. Die Marker werden 5 mal in einer zuälligen falschen Reihenfolge und
        //2-mal in richtiger Reihenfolge zurückgegeben.
        FrameReorderingManipulator frameReorderingManipulator = new FrameReorderingManipulator(5, 2);
        TrialDataTransformation transformation = new TrialDataTransformation(frameLabelingStrategy, frameReorderingManipulator);
        TrialDataManagerBuilder trialDataManager = new TrialDataManagerBuilder(transformation).withNormalization(normalizationStrategy);

        //Initialisierung von RecordReader und DataSetIterator
        JsonTrialRecordReader trainDataReader = new JsonTrialRecordReader(trialDataManager.build());
        trainDataReader.initialize(fileSplitTrain);
        JsonTrialRecordReader testDataReader = new JsonTrialRecordReader(trialDataManager.build());
        testDataReader.initialize(fileSplitTest);

        DataSetIterator trainData = new RecordReaderDataSetIterator(trainDataReader, 20);
        DataSetIterator testData = new RecordReaderDataSetIterator(testDataReader, 20);

        //Netzwerk-Konfiguration und Initialisierung
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .seed(234)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.01))
                .list()
                .layer(new DenseLayer.Builder().activation(Activation.SIGMOID).weightInit(WeightInit.SIGMOID_UNIFORM).nIn(595).nOut(119).build())
                .layer(new DenseLayer.Builder().activation(Activation.SIGMOID).weightInit(WeightInit.SIGMOID_UNIFORM).nIn(119).nOut(119).build())
                .layer(new DenseLayer.Builder().activation(Activation.SIGMOID).weightInit(WeightInit.SIGMOID_UNIFORM).nIn(119).nOut(17).build())
                .layer(new DenseLayer.Builder().activation(Activation.SIGMOID).weightInit(WeightInit.SIGMOID_UNIFORM).nIn(17).nOut(17).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).activation(Activation.SIGMOID).weightInit(WeightInit.SIGMOID_UNIFORM).nIn(17).nOut(2).build())
                .build();

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(multiLayerConfiguration);
        multiLayerNetwork.init();

        //Anmerkung: Stand jetzt werden die Daten noch nicht skaliert! Folgend: entsprechender Code zur Skalierung
        /*
        int rangeMin = -1;
        int rangeMax = 1;
        NormalizerMinMaxScaler normalizerMinMaxScaler = new NormalizerMinMaxScaler(rangeMin, rangeMax);
        normalizerMinMaxScaler.fit(trainData);
        trainData.setPreProcessor(normalizerMinMaxScaler);
        NormalizerMinMaxScaler normalizerMinMaxScaler1 = new NormalizerMinMaxScaler(rangeMin, rangeMax);
        normalizerMinMaxScaler1.fit(testData);
        testData.setPreProcessor(normalizerMinMaxScaler1);
        */

        //Training
        int epochs = 1;
        multiLayerNetwork.fit(trainData, epochs);
        frameLabelingStrategy.logCount();
        frameLabelingStrategy.resetCount();

        //Evaluierung und Ausgabe der Ergebnisse
        Evaluation evaluate = multiLayerNetwork.evaluate(testData);
        String stats = evaluate.stats(false, true);
        System.out.println(stats);
        frameLabelingStrategy.logCount();

        //Evaluation und Ausgabe eines einzelenen Trainingsdatensatzes --> Überprüfung von Ein- und Ausgabe des Netzes
        testData.reset();
        DataSet test = testData.next();
        INDArray features = test.getFeatures();
        INDArray prediction = multiLayerNetwork.output(features);
        System.out.println("single eval:");
        for (int i = 1; i <= 10; i++) {
            Evaluation evaluation = new Evaluation(2);
            evaluation.eval(test.getLabels().getRow(i), prediction.getRow(i));
            System.out.println(evaluation.stats(false, false));
            System.out.println("Datensatz " + i + " --> Features: ");
            Helper.printINDArray(features.getRow(i));
            System.out.println("Datensatz " + i + " --> Prediction: ");
            Helper.printINDArray(prediction.getRow(i));
            System.out.println("geschätzter Wert: ");
            System.out.println(prediction.getRow(i).maxNumber());
        }

        //save models
        /*String saveDirectory = "C:\\Users\\...";
        Helper.saveModel(multiLayerNetwork, saveDirectory, "distanceToMarkerLabeling");*/
    }
}
