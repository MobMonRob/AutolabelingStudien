package preprocess_data.data_manipulaton;

import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class FrameShuffleManipulator implements FrameManipulationStrategy {

    private final int amountOfShuffles;
    private Random random;
    private final boolean hasSeed;

    public FrameShuffleManipulator(int amountOfShuffles, long seed) {
        this.amountOfShuffles = amountOfShuffles;
        this.random = new Random(seed);
        this.hasSeed = true;
    }

    public FrameShuffleManipulator(final int amountOfShuffles) {
        this.amountOfShuffles = amountOfShuffles;
        this.hasSeed = false;
    }

    public ArrayList<Frame> manipulateFrame(Frame frame) {
        ArrayList<Frame> resultFrames = new ArrayList<Frame>();
        for (int i = amountOfShuffles; i > 0; i--) {
            resultFrames.add(getShuffledFrame(frame));
        }
        return resultFrames;
    }

    @Override
    public String toString() {
        return "FrameShuffleManipulator(amountOfShuffles: " + amountOfShuffles + ")";
    }

    private Frame getShuffledFrame(Frame frame) {
        ArrayList<Marker> newList = new ArrayList<Marker>(frame.getMarkers());
        if (hasSeed) {
            Collections.shuffle(newList, random);
            return new Frame(newList);
        }
        Collections.shuffle(newList);
        return new Frame(newList);
    }
}
