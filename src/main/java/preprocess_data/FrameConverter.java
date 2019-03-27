package preprocess_data;

import org.datavec.api.writable.Writable;
import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;
import preprocess_data.labeling.FrameLabelingStrategy;

import java.util.ArrayList;

//applies labeling strategy
public class FrameConverter {

    private final FrameLabelingStrategy frameLabelingStrategy;

    FrameConverter(FrameLabelingStrategy frameLabelingStrategy) {
        this.frameLabelingStrategy = frameLabelingStrategy;
    }

    ArrayList<ArrayList<Writable>> convertFrameToListOfWritables(Frame frame) {
        ArrayList<ArrayList<Writable>> resultList = new ArrayList<ArrayList<Writable>>();
        resultList.add(frameLabelingStrategy.getLabeledWritableList(frame));
        return resultList;
    }

    ArrayList<ArrayList<Writable>> convertFramesToListOfWritables(ArrayList<Frame> frames) {
        ArrayList<ArrayList<Writable>> resultList = new ArrayList<ArrayList<Writable>>();
        for (Frame frame : frames) {
            resultList.add(frameLabelingStrategy.getLabeledWritableList(frame));
        }
        return resultList;
    }

    public FrameLabelingStrategy getFrameLabelingStrategy() {
        return frameLabelingStrategy;
    }
}
