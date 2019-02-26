package preprocess_data.data_generation;

import javafx.geometry.Point3D;
import preprocess_data.data_model.Marker;

class MarkerMovementGenerator {

    private final Marker baseMarker;
    private final MarkerMovementFunction movementFunction;
    private final DirectionFunction xyDirection;

    MarkerMovementGenerator(Marker baseMarker, MarkerMovementFunction movementFunction, DirectionFunction xyDirection) {
        this.baseMarker = baseMarker;
        this.movementFunction = movementFunction;
        this.xyDirection = xyDirection;
    }

    Marker getNextMarker(int stepCounter) {
        return getMarkerFromPoint(movementFunction.getNextMarker(xyDirection.getNextPosition(stepCounter)), baseMarker);
    }

    private Marker getMarkerFromPoint(Point3D markerPosition, Marker baseMarker) {
        return new Marker(baseMarker.getLabel(),
                baseMarker.getX() + markerPosition.getX(),
                baseMarker.getY() + markerPosition.getY(),
                baseMarker.getZ() + markerPosition.getZ());
    }
}
