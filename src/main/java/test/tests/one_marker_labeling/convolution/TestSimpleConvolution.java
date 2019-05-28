package test.tests.one_marker_labeling.convolution;

import datavec.RandomizedTrialRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
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
import preprocess_data.builders.TrialDataManagerBuilder;
import preprocess_data.builders.TrialDataTransformationBuilder;
import preprocess_data.data_manipulaton.FrameShuffleManipulator;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.labeling.OneTargetLabeling;
import test.tests.Helper;

import java.io.File;
import java.io.IOException;
import java.util.Set;
import java.util.TreeSet;

//einfacher Test mit simpler CNN-Config und wenigen Markern
public class TestSimpleConvolution {

    public static void main(String[] args) throws IOException, InterruptedException {

        //Auswahl der Trainings- und Testdaten sowie der beachteten Markern --> Pfade anpassen!
        File trainDirectory = new File("C:\\Users\\...");
        File testDirectory = new File("C:\\Users\\...");
        String[] markerLabels = {"C7", "CLAV", "LASI", "LELB", "LELBW", "LHUM4", "LHUMA", "LHUMP", "LHUMS", "LRAD", "LSCAP1", "LSCAP2", "LSCAP3", "LSCAP4", "LULN", "RASI", "RELB", "RELBW", "RHUM4", "RHUMA", "RHUMP", "RHUMS", "RRAD", "RSCAP1", "RSCAP2", "RSCAP3", "RSCAP4", "RULN", "SACR", "STRN", "T10", "THRX1", "THRX2", "THRX3", "THRX4"};
        TreeSet<String> selectedLabels = new TreeSet<>();
        selectedLabels.add(markerLabels[1]);
        selectedLabels.add(markerLabels[2]);
        selectedLabels.add(markerLabels[3]);
        selectedLabels.add(markerLabels[4]);
        selectedLabels.add(markerLabels[5]);

        //Konfiguration der Daten --> Spezialisierung auf LELB
        TrialDataManagerBuilder trialDataManager = TrialDataManagerBuilder.addTransformation(TrialDataTransformationBuilder
                .addLabelingStrategy(new OneTargetLabeling("LELB", selectedLabels.size()))
                .withManipulation(new FrameShuffleManipulator(10)).build())
                .withNormalization(new CentroidNormalization(0, 1))
                .filterMarkers(selectedLabels);

        //Initialisierung der RecordReader und DataSetIterator
        RandomizedTrialRecordReader train = new RandomizedTrialRecordReader(trialDataManager.build(), 50000);
        RandomizedTrialRecordReader test = new RandomizedTrialRecordReader(trialDataManager.build(), 50000);
        train.initialize(new FileSplit(trainDirectory));
        test.initialize(new FileSplit(testDirectory));

        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator(train, 20);
        RecordReaderDataSetIterator testIterator = new RecordReaderDataSetIterator(test, 20);

        //Initialisierung des Netzwerks
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(nnConfig(selectedLabels));
        multiLayerNetwork.init();
        EvaluativeListener evaluativeListener = new EvaluativeListener(testIterator, 1, InvocationType.EPOCH_END);
        multiLayerNetwork.setListeners(new ScoreIterationListener(10000), evaluativeListener);

        //Training und Evaluation
        multiLayerNetwork.fit(trainIterator, 5);
        Evaluation evaluate = multiLayerNetwork.evaluate(testIterator);
        System.out.println(evaluate.stats(false, true));
        Helper.logSingleEvaluationDetails(multiLayerNetwork, testIterator);
    }

    //Konfiguration des CNN
    private static MultiLayerConfiguration nnConfig(Set<String> selectedLabels) {
        int inputSize = selectedLabels.size() * 3;
        int cnnOutputChannels = 20;
        int cnnOutputSize = cnnOutputChannels * 5;

        return new NeuralNetConfiguration.Builder()
                .seed(523)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.01))
                .list()
                //nIn --> Channels
                //nOut --> wie oft wird der Filter angewendet
                .layer(new ConvolutionLayer.Builder().nIn(1).nOut(cnnOutputChannels).kernelSize(3, 1).stride(3, 1).build())
                .layer(new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.NORMAL)
                        .nIn(cnnOutputSize).nOut(cnnOutputSize).build())
                .layer(new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.NORMAL)
                        .nIn(cnnOutputSize).nOut(cnnOutputSize).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(cnnOutputSize).nOut(5).build())
                .setInputType(InputType.convolutionalFlat(inputSize, 1, 1))
                .build();

    }
}
