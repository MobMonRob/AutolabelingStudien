package preprocess_data.data_manipulaton;

import preprocess_data.TrialDataTransformation;
import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;
import preprocess_data.labeling.NoLabeling;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Random;

//random angles for any frame
public class RandomFrameRotationManipulator implements FrameManipulationStrategy {

    private final int amountOfRotations;
    private final double[] range; //user-defined range for angles

    public RandomFrameRotationManipulator(int amountOfRotations) {
        this.amountOfRotations = amountOfRotations;
        this.range = new double[]{0, 360};
    }

    public RandomFrameRotationManipulator(int amountOfRotations, double rangeMin, double rangeMax) {
        this.amountOfRotations = amountOfRotations;
        this.range = new double[]{rangeMin, rangeMax};
    }

    @Override
    public ArrayList<Frame> manipulateFrame(Frame frame) {
        FrameManipulationStrategy manipulator = new FrameRotationManipulator(getRandomAngles());
        return manipulator.manipulateFrame(frame);
    }

    private double[] getRandomAngles() {
        double[] angles = new double[amountOfRotations];
        Random random = new Random();
        for (int i = 0; i < amountOfRotations; i++) {
            angles[i] = getRandomAngle(random);
        }
        return angles;
    }

    private double getRandomAngle(Random random) {
        return random.nextDouble() * (range[1] - range[0]) + range[0]; //random between 0 inclusive and 360 exclusive
    }
}
