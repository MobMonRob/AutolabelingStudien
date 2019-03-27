package deep_learning.tests;

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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import preprocess_data.TrialDataManager;
import preprocess_data.TrialDataManagerBuilder;
import preprocess_data.TrialDataTransformation;
import preprocess_data.data_manipulaton.FrameReorderingManipulator;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.data_normalization.TrialNormalizationStrategy;
import preprocess_data.labeling.DistanceToMarkerLabeling;

import java.io.File;

public class TestDistanceToMarkerLabeling {

    public static void main(String[] args) throws Exception {
        String[] allowedFileFormat = {"json"};
        //Input Data
        File trainDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\trainData\\trainDistance");
        File testDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\testData\\testDistance");
        FileSplit fileSplitTrain = new FileSplit(trainDirectory, allowedFileFormat);
        FileSplit fileSplitTest = new FileSplit(testDirectory, allowedFileFormat);

        //Strategies/Assets
        //falsche Reihenfolge:
        String[] orderedLabels = {"RULN", "LSCAP4", "RHUMS", "RRAD", "T10", "THRX4", "RSCAP3", "LELB", "LSCAP3", "C7", "RHUMP", "THRX3", "LHUMA", "RSCAP4", "RHUM4", "CLAV", "STRN", "RELB", "RHUMA", "LULN", "LSCAP2", "LHUM4", "SACR", "THRX2", "RSCAP1", "LASI", "LRAD", "LSCAP1", "LHUMP", "RELBW", "RASI", "THRX1", "LELBW", "RSCAP2", "LHUMS"};
        //richtige (default) Reihenfolge;
        String[] orderedLabels2 = {"C7", "CLAV", "LASI", "LELB", "LELBW", "LHUM4", "LHUMA", "LHUMP", "LHUMS", "LRAD", "LSCAP1", "LSCAP2", "LSCAP3", "LSCAP4", "LULN", "RASI", "RELB", "RELBW", "RHUM4", "RHUMA", "RHUMP", "RHUMS", "RRAD", "RSCAP1", "RSCAP2", "RSCAP3", "RSCAP4", "RULN", "SACR", "STRN", "T10", "THRX1", "THRX2", "THRX3", "THRX4"};
        DistanceToMarkerLabeling frameLabelingStrategy = new DistanceToMarkerLabeling(orderedLabels2);
        TrialNormalizationStrategy normalizationStrategy = new CentroidNormalization();
        FrameReorderingManipulator frameReorderingManipulator = new FrameReorderingManipulator(5, 2);
        TrialDataTransformation transformation = new TrialDataTransformation(frameLabelingStrategy, frameReorderingManipulator);
        TrialDataManager trialDataManager = new TrialDataManagerBuilder(transformation).withNormalization(normalizationStrategy).build();

        //DataSet Iterators
        JsonTrialRecordReader trainDataReader = new JsonTrialRecordReader(trialDataManager);
        trainDataReader.initialize(fileSplitTrain);
        JsonTrialRecordReader testDataReader = new JsonTrialRecordReader(trialDataManager);
        testDataReader.initialize(fileSplitTest);

        DataSetIterator trainData = new RecordReaderDataSetIterator(trainDataReader, 20);
        DataSetIterator testData = new RecordReaderDataSetIterator(testDataReader, 20);


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

        multiLayerNetwork.fit(trainData, 1);
        frameLabelingStrategy.logCount();
        frameLabelingStrategy.resetCount();

        Evaluation evaluate = multiLayerNetwork.evaluate(testData);
        String stats = evaluate.stats(false, true);
        System.out.println(stats);
        frameLabelingStrategy.logCount();

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

/*
        for (Layer layer : multiLayerNetwork.getLayers()) {
            System.out.println(layer.toString());
        }
        System.out.println(multiLayerNetwork.getLayerWiseConfigurations().toString());
*/

        //save models
        File modelSaveFile = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\models\\testModel-v1.zip");
        while(modelSaveFile.exists()) {
            modelSaveFile = Helper.getNextPossibleFile(modelSaveFile);
        }
        ModelSerializer.writeModel(multiLayerNetwork,modelSaveFile,true);
    }
}
