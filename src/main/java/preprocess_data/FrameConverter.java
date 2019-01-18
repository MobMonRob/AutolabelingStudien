package preprocess_data;

import org.datavec.api.writable.Writable;
import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;
import preprocess_data.labeling.FrameLabelingStrategy;

import java.util.ArrayList;

public class FrameConverter {

    private final FrameLabelingStrategy frameLabelingStrategy;

    public FrameConverter(FrameLabelingStrategy frameLabelingStrategy) {
        this.frameLabelingStrategy = frameLabelingStrategy;
    }

    public ArrayList<ArrayList<Writable>> convertFrameToListOfWritables(Frame frame) {
        ArrayList<ArrayList<Writable>> resultList = new ArrayList<ArrayList<Writable>>();
        resultList.add(frameLabelingStrategy.getLabeledWritableList(frame));
        return resultList;
    }

    public ArrayList<ArrayList<Writable>> convertFramesToListOfWritables(ArrayList<Frame> frames) {
        ArrayList<ArrayList<Writable>> resultList = new ArrayList<ArrayList<Writable>>();
        for (Frame frame : frames) {
            resultList.add(frameLabelingStrategy.getLabeledWritableList(frame));
        }
        return resultList;
    }
}
