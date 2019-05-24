package preprocess_data.labeling;

import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;

import java.util.ArrayList;
import java.util.List;

/*
Features: x,y,z Koordinatenwerte
Labels: Position eines bestimmten Markers

targetLabel: Marker auf den das neuronale Netz trainiert wird (Label = Position des Markers in den Eingabedaten)
*/
public class OneTargetLabeling implements FrameLabelingStrategy {

    private final String targetLabel;
    private final int amountOfLabels;

    //amountOfLabels: Wichtig für getLabels()
    public OneTargetLabeling(String targetLabel, int amountOfLabels) {
        this.targetLabel = targetLabel;
        this.amountOfLabels = amountOfLabels;
    }

    public OneTargetLabeling(String targetLabel) {
        this.targetLabel = targetLabel;
        this.amountOfLabels = -1;
    }

    public ArrayList<Writable> getLabeledWritableList(Frame frame) {
        int indexOfTarget = -1;
        final ArrayList<Writable> resultList = new ArrayList<Writable>();
        for (int i = 0; i < frame.getMarkers().size(); i++) {
            final Marker currentMarker = frame.getMarkers().get(i);
            resultList.add(new DoubleWritable(currentMarker.getX()));
            resultList.add(new DoubleWritable(currentMarker.getY()));
            resultList.add(new DoubleWritable(currentMarker.getZ()));
            if (currentMarker.getLabel().equalsIgnoreCase(targetLabel)) {
                indexOfTarget = i;
            }
        }
        resultList.add(new Text(indexOfTarget + ""));
        return resultList;
    }

    public List<String> getLabels() {
        if (amountOfLabels > 0) {
            final ArrayList<String> resultList = new ArrayList<String>();
            for (int i = 0; i < amountOfLabels; i++) {
                resultList.add(i + "");
            }
            return resultList;
        }
        return null;
    }

    @Override
    public String toString() {
        return "OneTargetDistanceLabeling(targetLabel: " + targetLabel + ", amountOfLabels: " + amountOfLabels + ")";
    }
}
