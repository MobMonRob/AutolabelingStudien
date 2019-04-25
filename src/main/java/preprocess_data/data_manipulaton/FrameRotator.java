package preprocess_data.data_manipulaton;

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
        double newX = marker.getX() * Math.cos(Math.toRadians(angle)) - marker.getY() * Math.sin(Math.toRadians(angle));
        double newY = marker.getX() * Math.sin(Math.toRadians(angle)) + marker.getY() * Math.cos(Math.toRadians(angle));

        return new Marker(marker.getLabel(), newX, newY, marker.getZ());
    }
}
