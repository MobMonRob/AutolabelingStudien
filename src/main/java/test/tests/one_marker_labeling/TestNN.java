package test.tests.one_marker_labeling;

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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import preprocess_data.TrialDataManager;
import preprocess_data.TrialDataTransformation;
import preprocess_data.data_manipulaton.FrameManipulationStrategy;
import preprocess_data.data_manipulaton.FrameShuffleManipulator;
import preprocess_data.labeling.FrameLabelingStrategy;
import preprocess_data.labeling.OneTargetLabeling;

import java.io.File;

public class TestNN {

    public static void main(String[] args) throws Exception {

        String[] allowedFileFormat = {"json"};
        //Input Data
        File trainDirectory = new File("C:\\Users\\nico.rinck\\Desktop\\Daten_Studienarbeit\\ABD_train");
        FileSplit fileSplitTrain = new FileSplit(trainDirectory, allowedFileFormat);

        //Test Data
        File file = new File("C:\\Users\\nico.rinck\\Desktop\\Daten_Studienarbeit\\ABD_test");
        FileSplit fileSplitTest = new FileSplit(file, allowedFileFormat);

        //Strategies/Assets
        FrameLabelingStrategy frameLabelingStrategy = new OneTargetLabeling("RASI", 35);
        FrameManipulationStrategy manipulationStrategy = new FrameShuffleManipulator(10);
        TrialDataTransformation transformation = new TrialDataTransformation(frameLabelingStrategy,manipulationStrategy);
        TrialDataManager trialDataManagerTrain = new TrialDataManager(transformation);
        TrialDataManager trialDataManagerTest = new TrialDataManager(transformation);

        //DataSet Iterators
        JsonTrialRecordReader trainDataReader = new JsonTrialRecordReader(trialDataManagerTrain);
        trainDataReader.initialize(fileSplitTrain);
        JsonTrialRecordReader testDataReader = new JsonTrialRecordReader(trialDataManagerTest);
        testDataReader.initialize(fileSplitTest);

        //more is not needed. datavec assumes the last index of the data-array (record) as label.
        //To get output-labels of the NN the method RecordReader.getLabels() has to be implemented!
        RecordReaderDataSetIterator trainDataIterator = new RecordReaderDataSetIterator(trainDataReader,16);
        RecordReaderDataSetIterator testDataIterator = new RecordReaderDataSetIterator(testDataReader,16);

        //NN Config
        final int numInputs = 105;
        final int outputNum = 35;
        final long seed = 1014L;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.SIGMOID)
                .weightInit(WeightInit.SIGMOID_UNIFORM)
                .updater(new Sgd(0.1))
                .list()
                .layer(0,new DenseLayer.Builder().nIn(numInputs).nOut(53).build())
                .layer(1,new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS).nIn(53).nOut(outputNum).build())
                .build();

        //logs
        System.out.println(conf.getBackpropType());
        System.out.println(conf.getEpochCount());

        MultiLayerNetwork nn = new MultiLayerNetwork(conf);
        nn.init();
        nn.setListeners(new EvaluativeListener(testDataIterator,1, InvocationType.EPOCH_END), new ScoreIterationListener(1000));

        //Training
        nn.fit(trainDataIterator,10);
    }
}