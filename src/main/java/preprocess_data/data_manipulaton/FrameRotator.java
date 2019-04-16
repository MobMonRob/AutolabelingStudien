package preprocess_data.data_manipulaton;

import org.yaml.snakeyaml.error.Mark;
import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;

import java.util.ArrayList;

//this implementation always takes the z-axis as rotation axis
public class FrameRotator {

    Frame rotateFrame(final Frame frame, double angle) {
        ArrayList<Marker> newMarkers = new ArrayList<>();
        for (Marker marker : frame.getMarkers()) {
            newMarkers.add(rotateMarker(marker, angle));
        }
        return new Frame(newMarkers);
    }

    private Marker rotateMarker(final Marker marker, double angle) {
        double newX = marker.getX() * Math.cos(angle) + marker.getY() * Math.sin(angle);
        double newY = -marker.getY() * Math.sin(angle) + marker.getX() * Math.cos(angle);

        return new Marker(marker.getLabel(), newX, newY, marker.getZ());
    }
}
