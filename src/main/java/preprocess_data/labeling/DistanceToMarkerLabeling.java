package preprocess_data.labeling;

import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//To use this labeling strategy effectively, the frames have to be shuffled. (-> FrameShuffleManipulator)
//If not frame will be labeled with only "1" or only "0". ("1" -> labels == frame.markers, "0" -> labels != frame.markers)
public class DistanceToMarkerLabeling implements FrameLabelingStrategy {
    //feste Reihenfolge --> ArrayList mit allen Labels müssen reingegeben werden.
    //Dann nach Reihenfolge der d(1,2) d(1,3) .....

    private final String[] labels;

    public DistanceToMarkerLabeling(String[] labels) {
        this.labels = labels;
    }

    @Override
    public ArrayList<Writable> getLabeledWritableList(Frame frame) {
        final ArrayList<Writable> resultList = new ArrayList<>();
        final ArrayList<Marker> markers = frame.getMarkers();
        boolean hasSameContent = true; //test if order in labels is the same as in frame
        for (int i = 0; i < labels.length; i++) {
            hasSameContent &= labels[i].equalsIgnoreCase(markers.get(i).getLabel());
            for (int j = i + 1; j < labels.length; j++) {
                resultList.add(new DoubleWritable(DistanceCalculator.getDistanceToMarker(markers.get(i),markers.get(j))));
            }
        }
        if (hasSameContent) {
            resultList.add(new Text("1"));
            return resultList;
        }
        resultList.add(new Text("0"));
        return resultList;
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("0", "1");
    }
}
