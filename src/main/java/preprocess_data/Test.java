package preprocess_data;

import datavec.JsonTrialRecordReader;
import org.datavec.api.split.FileSplit;
import preprocess_data.data_manipulator.FrameShuffleManipulator;
import preprocess_data.labeling.OneTargetLabelingStrategy;

import java.io.File;
import java.io.IOException;

public class Test {

    private static final String testString = "01_SS_O1_S1_Abd-TEST";

    public static void main(String[] args) {
        JsonTrialRecordReader jsonTrialRecordReader = new JsonTrialRecordReader(
                new TrialDataManager(new OneTargetLabelingStrategy("RASI"),new FrameShuffleManipulator(3)));
        File file = new File("C:\\Users\\Nico Rinck\\IdeaProjects\\autolabeling\\src\\main\\resources\\01_SS_O1_S1_Abd-TEST.json");
        FileSplit fileSplit = new FileSplit(file);
        try {
            jsonTrialRecordReader.initialize(fileSplit);
            while (jsonTrialRecordReader.hasNext()) {
                System.out.println("Dataset: " + jsonTrialRecordReader.next());
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }

}
