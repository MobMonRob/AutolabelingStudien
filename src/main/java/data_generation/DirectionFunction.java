package data_generation;

import preprocess_data.data_model.Coordinate3D;

//function which decides in which direction the marker can move --> x,y
//f(x) = coefficient * x ^ exponent
//stepWidth to get new X
class DirectionFunction {

    private final double stepWidth;
    private final double coefficient;
    private final double exponent;

    DirectionFunction(double stepWidth, double coefficient, double exponent) {
        this.stepWidth = stepWidth;
        this.coefficient = coefficient;
        this.exponent = exponent;
    }

    DirectionFunction(double stepWidth, double coefficient) {
        this(stepWidth, coefficient, 1);
    }

    Coordinate3D getNextPosition(int stepCounter) {
        double newX = stepWidth * stepCounter;
        return new Coordinate3D(newX, calcY(newX),0);
    }

    private double calcY(double newX) {
        return coefficient * Math.pow(newX, exponent);
    }
}