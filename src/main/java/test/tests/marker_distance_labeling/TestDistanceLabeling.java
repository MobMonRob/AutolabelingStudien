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
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import preprocess_data.TrialDataTransformation;
import preprocess_data.builders.TrialDataManagerBuilder;
import preprocess_data.data_manipulaton.FrameManipulationStrategy;
import preprocess_data.data_manipulaton.FrameShuffleManipulator;
import preprocess_data.data_model.Coordinate3D;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.data_normalization.TrialNormalizationStrategy;
import preprocess_data.labeling.FrameLabelingStrategy;
import preprocess_data.labeling.OneTargetDistanceLabeling;
import test.tests.Helper;

import java.io.File;

/*
Testen des Trainings der Distanzen zum Nullpunkt (entspricht Schwerpunkt, da normalisiert).
*/
public class TestDistanceLabeling {
    public static void main(String[] args) throws Exception {

        String[] allowedFileFormat = {"json"}; //Falls sich andere Dateien im Directory befinden

        //Input Data --> Pfade anpassen!
        File trainDirectory = new File("C:\\Users\\...");
        File testDirectory = new File("C:\\Users\\...");
        FileSplit fileSplitTrain = new FileSplit(trainDirectory, allowedFileFormat);
        FileSplit fileSplitTest = new FileSplit(testDirectory, allowedFileFormat);

        //Konfiguration der Datensätze:
        //Labeling --> OneTargetDistanceLabeling:
        //  - Label: Position eines Markers (hier: LELB)
        //  - Features: Distanz zum Mittelpunkt (Coordinate3D)
        FrameLabelingStrategy frameLabelingStrategy = new OneTargetDistanceLabeling(new Coordinate3D(0, 0, 0), "LELB", 35);
        //3-fache Vermehrung der Daten (Mischen der Markerreihenfolge)
        FrameManipulationStrategy manipulationStrategy = new FrameShuffleManipulator(3);
        //Normalisierung: Abziehen des Schwerpunkts und Skalierung in den Wertebreich [-1,1]
        TrialNormalizationStrategy normalizationStrategy = new CentroidNormalization(-1, 1);
        TrialDataTransformation transformation = new TrialDataTransformation(frameLabelingStrategy, manipulationStrategy);
        TrialDataManagerBuilder trialDataManager = new TrialDataManagerBuilder(transformation).withNormalization(normalizationStrategy);

        //Initialisieren von RecordReader und DataSetIterator
        JsonTrialRecordReader trainDataReader = new JsonTrialRecordReader(trialDataManager.build());
        JsonTrialRecordReader testDataReader = new JsonTrialRecordReader(trialDataManager.build());
        trainDataReader.initialize(fileSplitTrain);
        testDataReader.initialize(fileSplitTest);

        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator(trainDataReader, 20);
        RecordReaderDataSetIterator testIterator = new RecordReaderDataSetIterator(testDataReader, 20);

        //NN-Config ****************************************************************************************************
        final int numInputs = 35;
        final int outputNum = 35;
        final long seed = 1014L;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.TANH)
                .weightInit(WeightInit.NORMAL)
                .updater(new Sgd(0.4))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(35).build())
                .layer(1, new DenseLayer.Builder().nIn(35).nOut(35).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).nIn(35).nOut(outputNum).build())
                .build();
        //**************************************************************************************************************

        //Normalization mit DL4J --> nicht verwendet! (nur als Beispiel)
        /*
        int rangeMin = -1;
        int rangeMax = 1;
        NormalizerMinMaxScaler normalizerMinMaxScaler = new NormalizerMinMaxScaler(rangeMin, rangeMax);
        normalizerMinMaxScaler.fit(trainIterator);
        trainIterator.setPreProcessor(normalizerMinMaxScaler);
        NormalizerMinMaxScaler normalizerMinMaxScaler1 = new NormalizerMinMaxScaler(rangeMin, rangeMax);
        normalizerMinMaxScaler1.fit(testIterator);
        testIterator.setPreProcessor(normalizerMinMaxScaler1);
        */

        //Init MultiLayerNetwork
        MultiLayerNetwork nn = new MultiLayerNetwork(conf);
        nn.init();
        EvaluativeListener evaluativeListener = new EvaluativeListener(testIterator, 1, InvocationType.EPOCH_END);
        nn.setListeners(new ScoreIterationListener(10000), evaluativeListener);

        //Training
        int epochs = 5;
        nn.fit(trainIterator, 5);

        //Ausgabe der Trainingsdaten über die Epochen
        IEvaluation[] evaluations = evaluativeListener.getEvaluations();
        for (IEvaluation singleEvaluation : evaluations) {
            String s = singleEvaluation.stats();
            String[] split = s.split("\n");
            int i = 0;
            for (String s1 : split) {
                if (s1.contains("Accuracy")) {
                    System.out.println("Präzision über die Epochen:");
                    System.out.println("Epoche " + i++ + ": " + s1);
                }
            }
        }

        //finale Evaluation mit Confusion-Matrix
        System.out.println("start evaluation");
        testIterator.reset();
        Evaluation eval = nn.evaluate(testIterator);
        System.out.println(eval.stats(false, true));

        //Evaluation und Ausgabe eines einzelnen Datensatzes --> Überprüfung der Eingabe und Ausgabewerte
        testIterator.reset();
        DataSet testData = testIterator.next();
        INDArray features = testData.getFeatures();
        INDArray prediction = nn.output(features);
        System.out.println("single eval:");
        Evaluation evaluation = new Evaluation(outputNum);
        eval.eval(testData.getLabels().getRow(0), prediction.getRow(0));
        System.out.println(evaluation.stats(false, true));
        System.out.println("Datensatz 1 --> Features: ");
        Helper.printINDArray(features.getRow(0));
        System.out.println("Datensatz 1 --> Prediction: ");
        Helper.printINDArray(prediction.getRow(0));
        System.out.println("geschätzter Wert: ");
        System.out.println(prediction.getRow(0).maxNumber());
        System.out.println(nn.getLayerWiseConfigurations());
    }
}
