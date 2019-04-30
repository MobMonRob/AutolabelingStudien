package preprocess_data.labeling;

import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;

import java.util.ArrayList;
import java.util.List;

//no labeling --> regression problems
public class NoLabeling implements FrameLabelingStrategy {
    @Override
    public ArrayList<Writable> getLabeledWritableList(Frame frame) {
        ArrayList<Writable> resultList = new ArrayList<>();
        for (Marker marker : frame.getMarkers()) {
            resultList.add(new DoubleWritable(marker.getX()));
            resultList.add(new DoubleWritable(marker.getY()));
            resultList.add(new DoubleWritable(marker.getZ()));
        }
        return resultList;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }
}
