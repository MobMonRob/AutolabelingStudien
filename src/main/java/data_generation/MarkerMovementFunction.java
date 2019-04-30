package data_generation;

import preprocess_data.data_model.Coordinate3D;

//TODO: refactor --> merge Direction function and this class --> takes only stepwidth from outside --> transform with params for each dimension
class MarkerMovementFunction {

    //3D-function: f(x,y) = coefficientX*x^exponentX + coefficientY*y^exponentY
    private final double coefficientX;
    private final double exponentX;
    private final double coefficientY;
    private final double exponentY;

    MarkerMovementFunction(double coefficientX, double exponentX, double coefficientY, double exponentY) {
        this.coefficientX = coefficientX;
        this.exponentX = exponentX;
        this.coefficientY = coefficientY;
        this.exponentY = exponentY;
    }

    MarkerMovementFunction(double coefficientX, double coefficientY) {
        this(coefficientX, 1, coefficientY, 1);
    }

    Coordinate3D getNextMarker(Coordinate3D newXYPosition) {
        return new Coordinate3D(newXYPosition.getX(), newXYPosition.getY(), getNextZ(newXYPosition));
    }

    private double getNextZ(Coordinate3D xy) {
        return coefficientX * Math.pow(xy.getX(), exponentX) + coefficientY * Math.pow(xy.getY(), exponentY);
    }
}
