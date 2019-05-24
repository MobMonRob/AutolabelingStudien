package preprocess_data.labeling;

import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/*
Features: Distanz jedes Markers zu jedem anderen
Label: 1 --> Reihenfolge entspricht der vorgegebenen (labels), 0 --> Reihenfolge ist vertauscht
*/
public class DistanceToMarkerLabeling implements FrameLabelingStrategy {

    private final String[] labels;
    private int count1 = 0;
    private int count0 = 0;
    private final ArrayList<Writable> resultList = new ArrayList<>();

    //To use this labeling strategy effectively, the frames have to be reordered.
    //--> Always use in Combination with FrameReorderingManipulator
    //If not frame will be labeled with only "1" or only "0". ("1" -> labels == frame.markers, "0" -> labels != frame.markers)
    public DistanceToMarkerLabeling(String[] labels) {
        this.labels = labels;
    }

    @Override
    public ArrayList<Writable> getLabeledWritableList(Frame frame) {
        resultList.clear();
        final ArrayList<Marker> markers = frame.getMarkers();
        boolean hasSameContent = true; //test if order in labels is the same as in frame
        for (int i = 0; i < labels.length; i++) {
            hasSameContent &= labels[i].equalsIgnoreCase(markers.get(i).getLabel());
            for (int j = i + 1; j < labels.length; j++) {
                resultList.add(new DoubleWritable(DistanceCalculator.getDistanceToMarker(markers.get(i),markers.get(j))));
            }
        }
        if (hasSameContent) {
            count1++;
            resultList.add(new Text("1"));
            return new ArrayList<>(resultList);
        }
        count0++;
        resultList.add(new Text("0"));
        return new ArrayList<>(resultList);
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("0", "1");
    }

    @Override
    public String toString() {
        return "DistanceToMarkerLabeling(labels: " + Arrays.asList(labels).toString() + ")";
    }

    public void logCount() {
        System.out.println("1: " + this.count1 + ", 0: " + this.count0);
    }

    public void resetCount() {
        count0 = 0;
        count1 = 0;
    }
}
