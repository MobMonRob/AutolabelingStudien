package test.tests.one_marker_labeling;

import datavec.JsonTrialRecordReader;
import datavec.RandomizedTrialRecordReader;
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
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import preprocess_data.TrialDataManager;
import preprocess_data.TrialDataTransformation;
import preprocess_data.builders.TrialDataManagerBuilder;
import preprocess_data.data_manipulaton.FrameManipulationStrategy;
import preprocess_data.data_manipulaton.FrameShuffleManipulator;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.labeling.FrameLabelingStrategy;
import preprocess_data.labeling.OneTargetLabeling;

import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;

public class TestNN {

    public static void main(String[] args) throws Exception {

        String[] allowedFileFormat = {"json"};
        //Input Data
        File trainDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\trainData\\trainDistanceSimple");
        FileSplit fileSplitTrain = new FileSplit(trainDirectory, allowedFileFormat);

        //Test Data
        File file = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\testData\\testDistanceSimple");
        FileSplit fileSplitTest = new FileSplit(file, allowedFileFormat);

        String[] markerLabels = {"C7", "CLAV", "LASI", "LELB", "LELBW", "LHUM4", "LHUMA", "LHUMP", "LHUMS", "LRAD", "LSCAP1", "LSCAP2", "LSCAP3", "LSCAP4", "LULN", "RASI", "RELB", "RELBW", "RHUM4", "RHUMA", "RHUMP", "RHUMS", "RRAD", "RSCAP1", "RSCAP2", "RSCAP3", "RSCAP4", "RULN", "SACR", "STRN", "T10", "THRX1", "THRX2", "THRX3", "THRX4"};
        TreeSet<String> selectedLabels = new TreeSet<>();
        selectedLabels.add(markerLabels[0]);
        selectedLabels.add(markerLabels[1]);
        selectedLabels.add(markerLabels[2]);
        selectedLabels.add(markerLabels[3]);

        for (int i = 4; i < markerLabels.length; i++) {
            selectedLabels.add(markerLabels[i]);
            System.out.println("current amount of markers: " + selectedLabels.size());

            FrameLabelingStrategy frameLabelingStrategy = new OneTargetLabeling("LELB", selectedLabels.size());
            FrameManipulationStrategy manipulationStrategy = new FrameShuffleManipulator(20);
            TrialDataTransformation transformation = new TrialDataTransformation(frameLabelingStrategy,manipulationStrategy);

            TrialDataManager trialDataManager = TrialDataManagerBuilder
                    .addTransformation(transformation)
                    .withNormalization(new CentroidNormalization(-1,1))
                    .filterMarkers(selectedLabels).build();
            JsonTrialRecordReader recordReaderTrain = new RandomizedTrialRecordReader(trialDataManager, 5000);
            JsonTrialRecordReader recordReaderTest = new RandomizedTrialRecordReader(trialDataManager, 5000);
            recordReaderTrain.initialize(fileSplitTrain);
            recordReaderTest.initialize(fileSplitTest);
            RecordReaderDataSetIterator trainDataIterator = new RecordReaderDataSetIterator(recordReaderTrain,20);
            RecordReaderDataSetIterator testDataIterator = new RecordReaderDataSetIterator(recordReaderTest,20);

            MultiLayerNetwork network = new MultiLayerNetwork(getNNConfig(selectedLabels));
            network.fit(trainDataIterator,1);

            Evaluation evaluation = network.evaluate(testDataIterator);
            System.out.println(evaluation.stats(false,true));
        }

    }

    private static MultiLayerConfiguration getNNConfig(Set<String> markers) {
        final int numInputs = markers.size() * 3;
        final int hiddenInputs = markers.size() * 2;
        final int outputNum = markers.size();
        final long seed = 1014L;

        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.1))
                .list()
                .layer(0,new DenseLayer.Builder().nIn(numInputs).nOut(hiddenInputs).activation(Activation.TANH).weightInit(WeightInit.NORMAL).build())
                .layer(1,new DenseLayer.Builder().nIn(hiddenInputs).nOut(hiddenInputs).activation(Activation.RELU).weightInit(WeightInit.RELU_UNIFORM).build())
                .layer(2,new DenseLayer.Builder().nIn(hiddenInputs).nOut(hiddenInputs).activation(Activation.RELU).weightInit(WeightInit.RELU_UNIFORM).build())
                .layer(3,new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).nIn(hiddenInputs).nOut(outputNum).build())
                .build();
    }
}