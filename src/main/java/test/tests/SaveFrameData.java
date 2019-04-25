package test.tests;

import datavec.JsonTrialRecordReader;
import org.datavec.api.split.FileSplit;
import preprocess_data.FrameDataPreprocessor;
import preprocess_data.builders.TrialDataManagerBuilder;
import preprocess_data.builders.TrialDataTransformationBuilder;
import preprocess_data.data_manipulaton.FrameRotationManipulator;
import preprocess_data.data_normalization.CentroidNormalization;
import preprocess_data.labeling.FrameLabelingStrategy;
import preprocess_data.labeling.NoLabeling;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class SaveFrameData {

    public static void main(String[] args) throws Exception {
        File rootData = new File("C:\\Users\\Nico Rinck\\Desktop\\oneProband.json");
        /*File rootData = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\trainData\\train\\02_JI_O1_S1_Abd.json");*/
        String[] orderedLabels2 = {"RRAD", "RULN", "RASI", "LASI", "LRAD", "LULN", "T10",
                "RELB", "LELB", "RHUM4", "LHUM4", "RSCAP2", "LSCAP2", "C7", "THRX1", "THRX3", "SACR", "STRN"
        };
        Set<String> stringSets = new HashSet<>(Arrays.asList(orderedLabels2));

        FrameLabelingStrategy frameLabelingStrategy = new NoLabeling();
        double[] rotationAngles = {22.5, 45.0, 90.0, 135};
        TrialDataManagerBuilder dataManager = TrialDataManagerBuilder
                .addTransformation(TrialDataTransformationBuilder
                        .addLabelingStrategy(frameLabelingStrategy)
                        .withManipulation(new FrameRotationManipulator(rotationAngles))
                        .build())
                .withNormalization(new CentroidNormalization())
                .filterMarkers(stringSets);

        FrameDataPreprocessor dataPreprocessor = new FrameDataPreprocessor(5000);
        String saveDirectory = "C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\save\\visualize";
        JsonTrialRecordReader train = new JsonTrialRecordReader(dataManager.build());
        train.initialize(new FileSplit(rootData));

        dataPreprocessor.saveDataToFile(train, saveDirectory);
    }
}
