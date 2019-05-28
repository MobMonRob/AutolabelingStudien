package preprocess_data.data_manipulaton;

import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;

import java.util.ArrayList;
import java.util.Collections;

/*
Hierbei handelt es sich um eine Strategie zum Umsortieren der Markerreihenfolge. (Wird zum Trainieren einer bestimmten
Markerreihenfolge verwendet)
Eingabe: Frame
Ausgabe: mehrere Frames. Eine bestimmte Anzahl wird dabei umsortiert
*/
public class FrameReorderingManipulator implements FrameManipulationStrategy {

    private final ArrayList<String> correctOrder; //in case order in json-files is different
    private final int reorderedFrames;
    private final int originalFrames;

    public FrameReorderingManipulator(int reorderedFrames, int originalFrames, ArrayList<String> correctOrder) {
        this.correctOrder = correctOrder;
        this.reorderedFrames = reorderedFrames;
        this.originalFrames = originalFrames;
    }

    public FrameReorderingManipulator(int reorderedFrames, int originalFrames) {
        this(reorderedFrames,originalFrames,null);
    }

    @Override
    public ArrayList<Frame> manipulateFrame(Frame frame) {
        Frame currentFrame = getOrderedFrame(frame);
        final ArrayList<Frame> resultList = new ArrayList<>();
        for (int i = 0; i < reorderedFrames; i++) {
            final ArrayList<Marker> markers = new ArrayList<>(currentFrame.getMarkers());
            Collections.shuffle(markers);
            resultList.add(new Frame(markers));
        }
        for (int i = 0; i < originalFrames; i++) {
            resultList.add(currentFrame);
        }
        Collections.shuffle(resultList);
        return resultList;
    }

    private Frame getOrderedFrame(Frame frame) {
        if (correctOrder != null) {
            ArrayList<Marker> correctMarkers = new ArrayList<>();
            for (String s : correctOrder) {
                for (Marker marker : frame.getMarkers()) {
                    if (marker.getLabel().equalsIgnoreCase(s)) {
                        correctMarkers.add(marker);
                    }
                }
            }
            return new Frame(correctMarkers);
        }
        return frame;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        correctOrder.forEach(s -> stringBuilder.append(s).append(", "));
        return "FrameReorderingManipulator(reorderedFrames: " + reorderedFrames + ", originalFrames: " + originalFrames
                + ", originalOrder: "  + stringBuilder.toString() + ")";
    }
}
