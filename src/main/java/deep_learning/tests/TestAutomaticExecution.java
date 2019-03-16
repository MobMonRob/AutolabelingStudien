package deep_learning.tests;

import deep_learning.execution.AutomaticConfigExecutor;
import deep_learning.execution.config_generation.ConfigVariator;
import deep_learning.execution.config_generation.LayerConfigVariator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import preprocess_data.TrialDataManager;
import preprocess_data.TrialDataTransformation;
import preprocess_data.data_manipulaton.FrameManipulationStrategy;
import preprocess_data.data_manipulaton.FrameShuffleManipulator;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.data_normalization.TrialNormalizationStrategy;
import preprocess_data.labeling.FrameLabelingStrategy;
import preprocess_data.labeling.OneTargetLabeling;

import java.io.File;

public class TestAutomaticExecution {

    public static void main(String[] args) throws Exception {
        File trainDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\trainData\\trainDistanceSimple");
        File testDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\testData\\testDistanceSimple");
        File logFile = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\logs\\logFile-v1.txt");

        //Strategies/Assets
        FrameLabelingStrategy frameLabelingStrategy = new OneTargetLabeling("LELB", 35);
        FrameManipulationStrategy manipulationStrategy = new FrameShuffleManipulator(5);
        TrialNormalizationStrategy normalizationStrategy = new CentroidNormalization(0, 1);
        TrialDataTransformation transformation = new TrialDataTransformation(frameLabelingStrategy, manipulationStrategy);
        TrialDataManager trialDataManager = new TrialDataManager(transformation, normalizationStrategy);

        LayerConfigVariator layerConfigVariator = new LayerConfigVariator(3, 5, 7);
        layerConfigVariator.addInputLayers(LayerConfigs.INPUT_LAYERS);
        layerConfigVariator.addHiddenLayers(LayerConfigs.getHiddenLayersFromIndexes(0, 1, 2, 3, 4, 5));
        layerConfigVariator.addOutputLayers(LayerConfigs.getOutputLayersFromIndexes(0, 2, 4, 5, 6));

        ConfigVariator configVariator = new ConfigVariator(24, layerConfigVariator);
        IUpdater[] updaters = {new Sgd(0.1), new Sgd(0.01), new Sgd(0.001)};
        configVariator.addUpdater(updaters);

        AutomaticConfigExecutor configExecutor = new AutomaticConfigExecutor(trainDirectory, testDirectory, logFile,
                trialDataManager, 20);
        configExecutor.executeConfigs(configVariator.getConfigs(), 1, 3);
    }
}
