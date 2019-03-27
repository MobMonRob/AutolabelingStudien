package deep_learning.tests;

import datavec.JsonTrialRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import preprocess_data.TrialDataManager;
import preprocess_data.TrialDataManagerBuilder;
import preprocess_data.TrialDataTransformation;
import preprocess_data.data_manipulaton.FrameManipulationStrategy;
import preprocess_data.data_manipulaton.FrameShuffleManipulator;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.data_normalization.TrialNormalizationStrategy;
import preprocess_data.labeling.FrameLabelingStrategy;
import preprocess_data.labeling.OneTargetLabeling;

import java.io.File;

public class TestNormalization {
    //alle json-dateien besitzen die richtigen marker, allerdings ist in einigen Dateien der Wert für den Marker thrx NAN
    //dadurch können die bekannten fehler entstehen

    public static void main(String[] args) throws Exception {
        String[] allowedFileFormat = {"json"};
        //Input Data
        File trainDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\train");
        File testDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\test");
        FileSplit fileSplitTrain = new FileSplit(trainDirectory, allowedFileFormat);
        FileSplit fileSplitTest = new FileSplit(testDirectory,allowedFileFormat);

        //Strategies/Assets
        FrameLabelingStrategy frameLabelingStrategy = new OneTargetLabeling("LELB", 35);
        FrameManipulationStrategy manipulationStrategy = new FrameShuffleManipulator(10);
        TrialNormalizationStrategy normalizationStrategy = new CentroidNormalization(0,1);
        TrialDataTransformation transformation = new TrialDataTransformation(frameLabelingStrategy, manipulationStrategy);
        TrialDataManager trialDataManager = new TrialDataManagerBuilder(transformation)
                .withNormalization(normalizationStrategy).build();

        //DataSet Iterators
        JsonTrialRecordReader trainDataReader = new JsonTrialRecordReader(trialDataManager);
        JsonTrialRecordReader testDataReader = new JsonTrialRecordReader(trialDataManager);
        trainDataReader.initialize(fileSplitTrain);
        testDataReader.initialize(fileSplitTest);
        for (Writable writable : trainDataReader.next()) {
            System.out.println(writable);
        }

        //NN Config
        final int numInputs = 105;
        final int outputNum = 35;
        final long seed = 1014L;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                /*.activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)*/
                .updater(new Sgd(0.1))
                .list()
                .layer(0, new DenseLayer.Builder().
                        nIn(numInputs).nOut(53).
                        weightInit(WeightInit.SIGMOID_UNIFORM).
                        activation(Activation.SIGMOID).build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(53).nOut(35)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.LEAKYRELU).build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(35).nOut(35)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.LEAKYRELU).build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(35).nOut(35)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.LEAKYRELU).build())
                .layer(4, new DenseLayer.Builder()
                        .nIn(35).nOut(35)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.LEAKYRELU).build())
                .layer(5, new DenseLayer.Builder()
                        .nIn(35).nOut(35)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.LEAKYRELU).build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(35).nOut(outputNum)
                        .weightInit(WeightInit.NORMAL)
                        .activation(Activation.SOFTMAX).build())
                .build();
                //hidden variieren (RELU)-->mehr layer (RELU)
                //hidden softmax
                //nochmal variieren --> Negloglike
                //bias?
                //l1,l2 optimierung

                //Tanh, Relu ,Softmax mit MXCENT --> mehr Deep Layer besseres Ergebnis
                //Softmax --> Hidden --> nein
                //Relu in den hidden Layer sorgt für stärkeres Lernen einzelner Neuronen im Output
                        //wenn Lernrate niedriger ist dieser Effekt nicht so start
                        //Sigmoid verteilt besser, Ergebnisse verbessern sich aber nicht
                //negloglike kommt mit normal weights besser zurecht als mcxent

        //dataset iterator
        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator(trainDataReader, 20);
        RecordReaderDataSetIterator testIterator = new RecordReaderDataSetIterator(testDataReader,20); //ändert das was?

        //Normalization
        int rangeMin = -1;
        int rangeMax = 1;
        NormalizerMinMaxScaler normalizerMinMaxScaler = new NormalizerMinMaxScaler(rangeMin,rangeMax);
        normalizerMinMaxScaler.fit(trainIterator);
        trainIterator.setPreProcessor(normalizerMinMaxScaler);
        NormalizerMinMaxScaler normalizerMinMaxScaler1 = new NormalizerMinMaxScaler(rangeMin,rangeMax);
        normalizerMinMaxScaler1.fit(testIterator);
        testIterator.setPreProcessor(normalizerMinMaxScaler1);

        //smallDataSetNegRange nn
        MultiLayerNetwork nn = new MultiLayerNetwork(conf);
        nn.init();
        EvaluativeListener evaluativeListener = new EvaluativeListener(testIterator,1,InvocationType.EPOCH_END);
        nn.setListeners(new ScoreIterationListener(10000), evaluativeListener);

        //Training
        nn.fit(trainIterator, 2);

        System.out.println("start evaluation");
        testIterator.reset();
        Evaluation eval = nn.evaluate(testIterator);
        System.out.println(eval.stats(false, true));

        //single eval
        testIterator.reset();
        for (int i = 0; i < 100; i++) {
            testIterator.next();
        }
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
        System.out.println("layerwise config" + nn.getLayerWiseConfigurations().toJson());
    }
}
