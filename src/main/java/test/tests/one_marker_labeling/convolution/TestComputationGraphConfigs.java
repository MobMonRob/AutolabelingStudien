package test.tests.one_marker_labeling.convolution;

import datavec.RandomizedTrialRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import preprocess_data.TrialDataManager;
import preprocess_data.builders.TrialDataManagerBuilder;
import preprocess_data.builders.TrialDataTransformationBuilder;
import preprocess_data.data_manipulaton.FrameShuffleManipulator;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.labeling.OneTargetLabeling;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.TreeSet;

public class TestComputationGraphConfigs {
    public static void main(String[] args) throws IOException, InterruptedException {

        File trainDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\trainData\\train");
        File testDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\testData\\test");
        String[] markerLabels = {"C7", "CLAV", "LASI", "LELB", "LELBW", "LHUM4", "LHUMA", "LHUMP", "LHUMS", "LRAD", "LSCAP1", "LSCAP2", "LSCAP3", "LSCAP4", "LULN", "RASI", "RELB", "RELBW", "RHUM4", "RHUMA", "RHUMP", "RHUMS", "RRAD", "RSCAP1", "RSCAP2", "RSCAP3", "RSCAP4", "RULN", "SACR", "STRN", "T10", "THRX1", "THRX2", "THRX3", "THRX4"};
        TreeSet<String> selectedLabels = new TreeSet<>(Arrays.asList(markerLabels));
        int batchSize = 20;
        int shuffles = selectedLabels.size() + 5;
        int recordReaderStorage = shuffles * 25000;

        TrialDataManager trialDataManager = TrialDataManagerBuilder.addTransformation(TrialDataTransformationBuilder
                .addLabelingStrategy(new OneTargetLabeling("LELB", selectedLabels.size()))
                .withManipulation(new FrameShuffleManipulator(shuffles))
                .build())
                .withNormalization(new CentroidNormalization(-1, 1))
                .filterMarkers(selectedLabels)
                .build();


        RandomizedTrialRecordReader train = new RandomizedTrialRecordReader(trialDataManager, recordReaderStorage);
        RandomizedTrialRecordReader test = new RandomizedTrialRecordReader(trialDataManager, recordReaderStorage);
        train.initialize(new FileSplit(trainDirectory));
        test.initialize(new FileSplit(testDirectory));

        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator(train, batchSize);
        RecordReaderDataSetIterator testIterator = new RecordReaderDataSetIterator(test, batchSize);

        //best: multipleReshapes(,20,40,20) in 10 Epochen
        ComputationGraph graph = new ComputationGraph(ConvolutionConfigs.multipleReshapes(selectedLabels, batchSize, 20, 10));
        System.out.println();
        graph.init();
        EvaluativeListener evaluativeListener = new EvaluativeListener(testIterator, 1, InvocationType.EPOCH_END);
        graph.setListeners(new ScoreIterationListener(10000), evaluativeListener);

        graph.fit(trainIterator, 5);
        Evaluation evaluate = graph.evaluate(testIterator);
        System.out.println(evaluate.stats(false, true));
    }

    /* feedforward and log path to test layer outputs.
        To test with same in each run use JsonTrialRecordReader and FrameShuffleManipulator with seed.
        DataSet next = trainIterator.next();
        INDArray features = next.getFeatures();
        Map<String, INDArray> indArrays = graph.feedForward(features, false);
        for (String s : indArrays.keySet()) {
            System.out.println(s);
        }
        System.out.println("Batch 0: _________________________________");
        System.out.println("    Inputs:");
        System.out.println(indArrays.get("1").getRow(0));
        System.out.println("    CNN 1:");
        System.out.println(indArrays.get("CNN1").getRow(0));
        System.out.println("    mergeChannels:");
        System.out.println(indArrays.get("mergeChannels").getRow(0));
        System.out.println("    CNN2:");
        System.out.println(indArrays.get("CNN2").getRow(0));
        System.out.println("    Layer 2:");
        System.out.println(indArrays.get("Layer 2").getRow(0));*/

    //Current Best (5 Markers) --> 99%
    //batchsize = 20
    //RandomizedRR(50000)
    //FrameShuffles(10)
    //CentroidNorm(-1,1)
    //channels = 10
   /* return new NeuralNetConfiguration.Builder()
            .seed(523)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.01))
            .graphBuilder()
                .addInputs("1")
                .layer("CNN1", new ConvolutionLayer.Builder().nIn(1).nOut(cnn1Channels).kernelSize(3, 1).stride(3, 1).build(), "1")
            .addVertex("mergeChannels",
                               new ReshapeVertex(batchSize, 1, cnn1Channels, cnnOutputSize), "CNN1")
            .layer("CNN2", new ConvolutionLayer.Builder().nIn(1).nOut(1).kernelSize(cnn1Channels, 1).stride(1, 1).build(), "mergeChannels")
            .layer("Layer 2", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                        .nIn(cnnOutputSize).nOut(cnnOutputSize).build(), "CNN2")
            .layer("Layer 3", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                        .nIn(cnnOutputSize).nOut(cnnOutputSize).build(), "Layer 2")
            .layer("Layer 4", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(cnnOutputSize).nOut(5).build(), "Layer 3")
            .setOutputs("Layer 4")
                .setInputTypes(InputType.convolutionalFlat(inputSize, 1, 1))
            .build();*/
}