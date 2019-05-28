package preprocess_data.data_normalization;

import preprocess_data.data_model.Coordinate3D;
import preprocess_data.data_model.Marker;

//Diese Klasse implementiert das Skalieren der Markerkoordinaten
class MarkerValueScaler {

    private final int minValue;
    private final int maxValue;
    private double[] minValues;
    private double[] maxValues;
    private boolean centroidSubtracted = false;

    MarkerValueScaler(int minValue, int maxValue) {
        this.minValue = minValue;
        this.maxValue = maxValue;
    }

    Marker scaleMarker(Marker marker, Coordinate3D centroid) {
        if (!centroidSubtracted) {
            subtractCentroidFromValues(minValues, centroid);
            subtractCentroidFromValues(maxValues, centroid);
            centroidSubtracted = true;
        }
        return new Marker(marker.getLabel(), getNewValue(marker.getX(), 0),
                getNewValue(marker.getY(), 1),
                getNewValue(marker.getZ(), 2));
    }

    private double getNewValue(double markerValue, int i) {
        return (markerValue - minValues[i]) / (maxValues[i] - minValues[i]) * (maxValue - minValue) + minValue;
    }

    private void subtractCentroidFromValues(double[] valueArray, Coordinate3D centroid) {
        valueArray[0] -= centroid.getX();
        valueArray[1] -= centroid.getY();
        valueArray[2] -= centroid.getZ();
    }

    void collectData(Marker marker) {
        if (!initValues(marker)) {
            checkNewExtremePoints(marker);
        }
    }

    private boolean initValues(Marker marker) {
        if (minValues == null) {
            this.minValues = new double[]{marker.getX(), marker.getY(), marker.getZ()};
            this.maxValues = new double[]{marker.getX(), marker.getY(), marker.getZ()};
            return true;
        }
        return false;
    }

    private void checkNewExtremePoints(Marker marker) {
        assignValueWhenTrue(minValues, 0, marker.getX(), marker.getX() < minValues[0]);
        assignValueWhenTrue(minValues, 1, marker.getY(), marker.getY() < minValues[1]);
        assignValueWhenTrue(minValues, 2, marker.getZ(), marker.getZ() < minValues[2]);
        assignValueWhenTrue(maxValues, 0, marker.getX(), marker.getX() > maxValues[0]);
        assignValueWhenTrue(maxValues, 1, marker.getY(), marker.getY() > maxValues[1]);
        assignValueWhenTrue(maxValues, 2, marker.getZ(), marker.getZ() > maxValues[2]);
    }

    private void assignValueWhenTrue(double[] values, int rowNumber, double compareValue, boolean condition) {
        if (condition) {
            values[rowNumber] = compareValue;
        }
    }

    int getMinValue() {
        return minValue;
    }

    int getMaxValue() {
        return maxValue;
    }
}
