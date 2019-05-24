package preprocess_data.labeling;

import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import preprocess_data.data_model.Coordinate3D;
import preprocess_data.data_model.Frame;

import java.util.ArrayList;
import java.util.List;

/*
Features: Distanzen jedes Markers zu einem TargetPoint werden als Features verwendet. Der TargetPoint kann beliebig
gewählt werden.
Labels: Position eines bestimmten Markers

targetPoint: Punkt zu dem die Distanz berechnet wird.
targetLabel: Marker auf den das neuronale Netz trainiert wird (Label = Postion des Markers in den Eingabedaten)
*/
public class OneTargetDistanceLabeling implements FrameLabelingStrategy {

    private final Coordinate3D targetPoint;
    private final String targetLabel;
    private final int amountOfLabels;

    public OneTargetDistanceLabeling(final Coordinate3D targetPoint, String targetLabel) {
        this.targetPoint = targetPoint;
        this.targetLabel = targetLabel;
        this.amountOfLabels = -1;
    }
    public OneTargetDistanceLabeling(final Coordinate3D targetPoint, String targetLabel, int amountOfLabels) {
        this.targetPoint = targetPoint;
        this.targetLabel = targetLabel;
        this.amountOfLabels = amountOfLabels;
    }

    public ArrayList<Writable> getLabeledWritableList(Frame frame) {
        final ArrayList<Writable> resultList = new ArrayList<Writable>();
        int indexOfTarget = -1;
        for (int i = 0; i < frame.getMarkers().size(); i++) {
            final double distance = DistanceCalculator.getDistanceToCoordinate(frame.getMarkers().get(i),targetPoint);
            resultList.add(new DoubleWritable(distance));
            if (frame.getMarkers().get(i).getLabel().equalsIgnoreCase(targetLabel)) {
                indexOfTarget = i;
            }
        }
        resultList.add(new Text(indexOfTarget + ""));
        return resultList;
    }

    public List<String> getLabels() {
        if (amountOfLabels > 0) {
            final ArrayList<String> resultList = new ArrayList<>();
            for (int i = 0; i < amountOfLabels; i++) {
                resultList.add(i + "");
            }
            return resultList;
        }
        return null;
    }

    @Override
    public String toString() {
        return "OneTargetDistanceLabeling(targetPoint: " + targetPoint.toString() + ", targetlabel: " + targetLabel +
                ", amountOfLabels: " +  amountOfLabels + ")";
    }
}
