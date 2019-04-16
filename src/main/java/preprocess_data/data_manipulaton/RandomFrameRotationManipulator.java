package preprocess_data.data_manipulaton;

import preprocess_data.data_model.Frame;

import java.util.ArrayList;
import java.util.Random;

//random angles for any frame
public class RandomFrameRotationManipulator implements FrameManipulationStrategy {

    private final int amountOfRotations;

    public RandomFrameRotationManipulator(int amountOfRotations) {
        this.amountOfRotations = amountOfRotations;
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
        return random.nextDouble() * 360; //random between 0 inclusive and 360 exclusive
    }
}
