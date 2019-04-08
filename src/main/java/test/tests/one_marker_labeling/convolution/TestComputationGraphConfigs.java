package test.tests.one_marker_labeling.convolution;

import datavec.RandomizedTrialRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import preprocess_data.FrameDataPreprocessor;
import preprocess_data.builders.TrialDataManagerBuilder;
import preprocess_data.builders.TrialDataTransformationBuilder;
import preprocess_data.data_manipulaton.FrameShuffleManipulator;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.labeling.OneTargetLabeling;
import test.execution.DL4JNetworkTrainer;

import java.io.File;
import java.util.Arrays;
import java.util.TreeSet;

public class TestComputationGraphConfigs {
    public static void main(String[] args) throws Exception {

        File trainDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\trainData\\train"); //\01_SS_O1_S1_Abd.json
        File testDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\testData\\test");
        String[] markerLabels = {"C7", "CLAV", "LASI", "LELB", "LELBW", "LHUM4", "LHUMA", "LHUMP", "LHUMS", "LRAD", "LSCAP1", "LSCAP2", "LSCAP3", "LSCAP4", "LULN", "RASI", "RELB", "RELBW", "RHUM4", "RHUMA", "RHUMP", "RHUMS", "RRAD", "RSCAP1", "RSCAP2", "RSCAP3", "RSCAP4", "RULN", "SACR", "STRN", "T10", "THRX1", "THRX2", "THRX3", "THRX4"};
        TreeSet<String> selectedLabels = new TreeSet<>(Arrays.asList(markerLabels));
        int shuffles = selectedLabels.size();
        int batchSize = 20; //total amount of data should be multiple of batchSize (prevent error in last batch)
        int recordReaderStorage = shuffles * 2000;

        TrialDataManagerBuilder dataManagerBuilder = TrialDataManagerBuilder.addTransformation(TrialDataTransformationBuilder
                .addLabelingStrategy(new OneTargetLabeling("LELB", selectedLabels.size()))
                .withManipulation(new FrameShuffleManipulator(shuffles))
                .build())
                .withNormalization(new CentroidNormalization(-1, 1))
                .filterMarkers(selectedLabels);


        FrameDataPreprocessor preprocessor = new FrameDataPreprocessor(recordReaderStorage * 2);
        String saveTrain = "C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\save\\train";
        String saveTest = "C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\save\\test";

        RecordReaderDataSetIterator trainIterator;
        RecordReaderDataSetIterator testIterator;
        if (FrameDataPreprocessor.directoryHasData(saveTrain) && FrameDataPreprocessor.directoryHasData(saveTest)) {
            trainIterator = new RecordReaderDataSetIterator(preprocessor.getReader(saveTrain), batchSize, 105, 35);
            testIterator = new RecordReaderDataSetIterator(preprocessor.getReader(saveTest), batchSize, 105, 35);
        } else {
            RandomizedTrialRecordReader train = new RandomizedTrialRecordReader(dataManagerBuilder.build(), recordReaderStorage);
            RandomizedTrialRecordReader test = new RandomizedTrialRecordReader(dataManagerBuilder.build(), recordReaderStorage);
            train.initialize(new FileSplit(trainDirectory));
            test.initialize(new FileSplit(testDirectory));
            trainIterator = new RecordReaderDataSetIterator(preprocessor.saveDataToFile(train, saveTrain), batchSize, 105, 35);
            testIterator = new RecordReaderDataSetIterator(preprocessor.saveDataToFile(train, saveTest), batchSize, 105, 35);
        }

        //best: multipleReshapes(,20,20,10) in 5 Epochen
        //beobachtung: hoher Wert bei cnn2Channels --> am anfang lernt es sehr langsam, dann am besten (bei 10)
        DL4JNetworkTrainer networkTrainer = new DL4JNetworkTrainer(trainIterator);
        TrainingListener[] listeners = {new EvaluativeListener(testIterator, 1, InvocationType.EPOCH_END),
                new ScoreIterationListener(10000)};
        networkTrainer.addListeners(listeners);

        //ComputationGraphConfiguration graph = ConvolutionConfigs.treeReshapesOneDeepLayer(selectedLabels, batchSize, 20, 10, 1); --> 92%
        //ComputationGraphConfiguration graph = ConvolutionConfigs.treeReshapesOneDeepLayer(selectedLabels, batchSize, 40, 20, 10); --> 95,4% (20 Epochen)
        //ComputationGraphConfiguration graph = ConvolutionConfigs.treeReshapesMultipleDeepLayers(selectedLabels, batchSize, 40, 20, 10) --> 95,2 (10 Epochen)
        //ComputationGraphConfiguration graph = ConvolutionConfigs.treeReshapesMultipleDeepLayers(selectedLabels, batchSize, 50, 10, 5);
        /*ComputationGraphConfiguration graph = ConvolutionConfigs.treeReshapesMultipleDeepLayers(selectedLabels, batchSize, 20, 10, 5);*/ /*(20 Epochen) --> 96%*/
        /*ComputationGraphConfiguration graph = ConvolutionConfigs.treeReshapesMultipleDeepLayers(selectedLabels, batchSize, 20, 20, 5); 10 Epochen --> 89%*/
        ComputationGraphConfiguration graph = ConvolutionConfigs.twoReshapes(selectedLabels, batchSize, 40, 20);
        ComputationGraph computationGraph = new ComputationGraph(graph);
        computationGraph.init();
        computationGraph.addListeners(listeners);
        computationGraph.fit(trainIterator, 20);

        Evaluation eval = computationGraph.evaluate(testIterator);
        System.out.println(eval.stats(true, true));
    }

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