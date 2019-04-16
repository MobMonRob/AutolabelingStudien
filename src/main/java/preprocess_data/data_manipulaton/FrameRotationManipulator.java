package preprocess_data.data_manipulaton;

import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;

import java.util.ArrayList;

//centroid normalization has to be applied first
public class FrameRotationManipulator implements FrameManipulationStrategy {

    private final double[] angles; //length > 1 --> returns multiple frames
    private final FrameRotator frameRotator = new FrameRotator();

    public FrameRotationManipulator(double[] angles) {
        this.angles = angles;
    }

    @Override
    public ArrayList<Frame> manipulateFrame(Frame frame) {
        final ArrayList<Frame> resultList = new ArrayList<>();
        for (double angle : angles) {
            resultList.add(frameRotator.rotateFrame(frame,angle));
        }
        return resultList;
    }
}
