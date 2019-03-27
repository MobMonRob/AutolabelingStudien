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
import org.nd4j.evaluation.IEvaluation;
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
import preprocess_data.data_model.Coordinate3D;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.data_normalization.TrialNormalizationStrategy;
import preprocess_data.labeling.FrameLabelingStrategy;
import preprocess_data.labeling.OneTargetDistanceLabeling;

import java.io.File;

public class TestDistanceLabeling {
    public static void main(String[] args) throws Exception {
        String[] allowedFileFormat = {"json"};
        //Input Data
        File trainDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\trainData\\train");
        File testDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\testData\\test");
        FileSplit fileSplitTrain = new FileSplit(trainDirectory, allowedFileFormat);
        FileSplit fileSplitTest = new FileSplit(testDirectory,allowedFileFormat);

        //Strategies/Assets
        FrameLabelingStrategy frameLabelingStrategy = new OneTargetDistanceLabeling(new Coordinate3D(0,0,0), "LELB", 35);
        FrameManipulationStrategy manipulationStrategy = new FrameShuffleManipulator(3);
        TrialNormalizationStrategy normalizationStrategy = new CentroidNormalization(-100,100);
        TrialDataTransformation transformation = new TrialDataTransformation(frameLabelingStrategy, manipulationStrategy);
        TrialDataManager trialDataManager = new TrialDataManagerBuilder(transformation).withNormalization(normalizationStrategy).build();

        //DataSet Iterators
        JsonTrialRecordReader trainDataReader = new JsonTrialRecordReader(trialDataManager);
        JsonTrialRecordReader testDataReader = new JsonTrialRecordReader(trialDataManager);
        trainDataReader.initialize(fileSplitTrain);
        testDataReader.initialize(fileSplitTest);
        for (Writable writable : trainDataReader.next()) {
            System.out.println(writable);
        }

        //NN Config
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

        //dataset iterator
        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator(trainDataReader, 20);
        RecordReaderDataSetIterator testIterator = new RecordReaderDataSetIterator(testDataReader,20);
        DataSet next = trainIterator.next();
        Helper.printINDArray(next.getFeatures().getRow(0));
        Helper.printINDArray(next.getLabels().getRow(0));

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
        nn.fit(trainIterator, 1);

        //epochs
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

        System.out.println("start evaluation");
        testIterator.reset();
        Evaluation eval = nn.evaluate(testIterator);
        System.out.println(eval.stats(false, true));

        //single eval
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
