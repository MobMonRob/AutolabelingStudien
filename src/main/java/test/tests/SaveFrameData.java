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
import java.util.*;

public class SaveFrameData {

    public static void main(String[] args) throws Exception {
        File rootData = new File("C:\\Users\\Nico Rinck\\Desktop\\oneProband.json");
        /*File rootData = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\trainData\\train\\02_JI_O1_S1_Abd.json");*/
        String[] orderedLabels2 = {
                "RASI","SACR", "LASI",
                       "STRN",
                       "THRX3",
                       "THRX1",
                        "C7", //höchster Punkt (Mittel der Armlinien)
                "RSCAP2",
                "RHUM4",
                "RELB",
                "RULN",
                              "LSCAP2",
                              "LHUM4",
                              "LELB",
                              "LULN",

        };
        Set<String> stringSets = new LinkedHashSet<>(Arrays.asList(orderedLabels2));
        stringSets.forEach(System.out::println);

        FrameLabelingStrategy frameLabelingStrategy = new NoLabeling();
        double[] rotationAngles = {22.5};
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
