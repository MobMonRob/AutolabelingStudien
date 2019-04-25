package preprocess_data;

import org.datavec.api.writable.Writable;
import preprocess_data.data_manipulaton.FrameManipulationStrategy;
import preprocess_data.data_manipulaton.FrameShuffleManipulator;
import preprocess_data.data_manipulaton.RandomFrameRotationManipulator;
import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;
import preprocess_data.labeling.FrameLabelingStrategy;
import preprocess_data.labeling.NoLabeling;

import java.util.ArrayList;

public class TrialDataTransformation {

    private final ArrayList<FrameManipulationStrategy> manipulators;
    private final FrameConverter converter;

    //defines how a frame of marker-data is converted to a list of writables (datavec-format)
    public TrialDataTransformation(FrameLabelingStrategy frameLabelingStrategy,
                                   ArrayList<FrameManipulationStrategy> manipulators) {
        this.converter = new FrameConverter(frameLabelingStrategy);
        this.manipulators = manipulators;
    }

    public TrialDataTransformation(FrameLabelingStrategy labelingStrategy, FrameManipulationStrategy manipulator) {
        this.converter = new FrameConverter(labelingStrategy);
        this.manipulators = new ArrayList<>();
        this.manipulators.add(manipulator);
    }

    ArrayList<ArrayList<Writable>> transformFrameData(final Frame frame) {
        if (manipulators != null && manipulators.size() > 0) {
            return converter.convertFramesToListOfWritables(this.manipulateFrame(frame));
        }
        return converter.convertFrameToListOfWritables(frame);
    }

    public FrameConverter getConverter() {
        return converter;
    }

    private ArrayList<Frame> manipulateFrame(final Frame frame) {
        ArrayList<Frame> resultList = new ArrayList<>();
        resultList.add(frame);
        for (FrameManipulationStrategy manipulationStrategy : manipulators) {
            resultList = manipulateFrames(resultList, manipulationStrategy);
        }
        return resultList;
    }

    private ArrayList<Frame> manipulateFrames(ArrayList<Frame> frames, FrameManipulationStrategy manipulationStrategy) {
        ArrayList<Frame> resultList = new ArrayList<>();
        for (Frame frame : frames) {
            resultList.addAll(manipulationStrategy.manipulateFrame(frame));
        }
        return resultList;
    }

    String getInfoString() {
        return "Manipulation: " + manipulators.toString() +
                "\nLabeling: " + converter.getFrameLabelingStrategy().toString();
    }
}