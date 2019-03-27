package deep_learning.tests;

import deep_learning.execution.AutomaticConfigExecutor;
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

public class ExecutorConfigs {

    static AutomaticConfigExecutor smallDataSetNegRange(int shuffles) throws Exception {
        File trainDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\trainData\\trainDistanceSimple");
        File testDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\testData\\testDistanceSimple");
        File logFile = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\logs\\logFile(-1,1)-v7.txt");

        //Strategies/Assets
        FrameLabelingStrategy frameLabelingStrategy = new OneTargetLabeling("LELB", 35);
        FrameManipulationStrategy manipulationStrategy = new FrameShuffleManipulator(shuffles);
        TrialNormalizationStrategy normalizationStrategy = new CentroidNormalization(-1, 1);
        TrialDataTransformation transformation = new TrialDataTransformation(frameLabelingStrategy, manipulationStrategy);
        TrialDataManager trialDataManager = new TrialDataManagerBuilder(transformation).withNormalization(normalizationStrategy).build();
        return new AutomaticConfigExecutor(trainDirectory, testDirectory, logFile,
                trialDataManager, 20);
    }

    static AutomaticConfigExecutor largeDataSetNegRange() throws Exception {
        File trainDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\trainData\\train");
        File testDirectory = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\testData\\test");
        File logFile = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\logs\\logFile(-1,1)-v5.txt");

        //Strategies/Assets
        FrameLabelingStrategy frameLabelingStrategy = new OneTargetLabeling("LELB", 35);
        FrameManipulationStrategy manipulationStrategy = new FrameShuffleManipulator(20);
        TrialNormalizationStrategy normalizationStrategy = new CentroidNormalization(-1, 1);
        TrialDataTransformation transformation = new TrialDataTransformation(frameLabelingStrategy, manipulationStrategy);
        TrialDataManager trialDataManager = new TrialDataManagerBuilder(transformation).withNormalization(normalizationStrategy).build();
        return new AutomaticConfigExecutor(trainDirectory, testDirectory, logFile,
                trialDataManager, 20);
    }
}
