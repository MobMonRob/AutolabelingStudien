package preprocess_data.data_normalization;

import preprocess_data.data_model.Coordinate3D;
import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;

import java.util.ArrayList;

//subtract centroid from any marker in trial. If set, all values will be scaled between minValue and maxValue.
//1. centroid subtraction 2. scale new (subtracted) values (executed in this order)
public class CentroidNormalization implements TrialNormalizationStrategy {

    private final Coordinate3D addedValues = new Coordinate3D(0, 0, 0);
    private final MarkerValueScaler markerValueScaler;
    private int counter = 0;
    private Coordinate3D centroid;

    public CentroidNormalization(int minValue, int maxValue) {
        this.markerValueScaler = new MarkerValueScaler(minValue, maxValue);
    }

    public CentroidNormalization() {
        this.markerValueScaler = null;
    }

    public Frame normalizeFrame(final Frame frame) {
        calculateCentroid();
        final ArrayList<Marker> newMarkers = new ArrayList<Marker>();
        for (final Marker marker : frame.getMarkers()) {
            newMarkers.add(normalizeMarker(marker));
        }
        return new Frame(newMarkers);
    }

    private void calculateCentroid() {
        if (centroid == null) {
            centroid = new Coordinate3D(addedValues.getX() / counter,
                    addedValues.getY() / counter,
                    addedValues.getZ() / counter);
        }
    }

    private Marker normalizeMarker(Marker marker) {
        final Marker normalizedMarker = new Marker(marker.getLabel(), marker.getX() - centroid.getX(),
                marker.getY() - centroid.getY(),
                marker.getZ() - centroid.getZ());
        if (markerValueScaler == null) {
            return normalizedMarker;
        }
        return markerValueScaler.scaleMarker(normalizedMarker, centroid);
    }

    public void collectMarkerData(Marker marker) {
        if (markerValueScaler != null) {
            markerValueScaler.collectData(marker);
        }
        addedValues.add(marker.getX(), marker.getY(), marker.getZ());
        counter++;
    }

    public TrialNormalizationStrategy getNewInstance() {
        if (markerValueScaler == null) {
            return new CentroidNormalization();
        }
        return new CentroidNormalization(markerValueScaler.getMinValue(),markerValueScaler.getMaxValue());
    }

    @Override
    public String toString() {
        if (this.markerValueScaler == null) {
            return "CentroidNormalization()";
        }
        return "CentroidNormalization(minValue: " + markerValueScaler.getMinValue() + ", maxValue: "
                + markerValueScaler.getMaxValue() + ")";
    }
}
