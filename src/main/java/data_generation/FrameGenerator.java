package data_generation;

import com.google.gson.stream.JsonWriter;

import java.io.IOException;
import java.util.ArrayList;

//forward frame includes the first Frame
class FrameGenerator {

    private final ArrayList<MarkerGenerator> markerGenerators;
    private final MarkerToJsonConverter markerToJsonConverter = new MarkerToJsonConverter();
    private final int amountOfFrames;
    private final int backwardFrames;
    private final int forwardFrames;

    FrameGenerator(ArrayList<MarkerGenerator> markerGenerators, int amountOfFrames) {
        this.markerGenerators = markerGenerators;
        this.amountOfFrames = amountOfFrames;
        this.forwardFrames = -1;
        this.backwardFrames = -1;
    }

    FrameGenerator(ArrayList<MarkerGenerator> markerGenerators, int forwardFrames, int backwardFrames) {
        this.markerGenerators = markerGenerators;
        this.amountOfFrames = forwardFrames + backwardFrames;
        this.forwardFrames = forwardFrames;
        this.backwardFrames = backwardFrames;
    }

    void generateFrames(JsonWriter jsonWriter, int repeatMovement) throws IOException {
        jsonWriter.beginArray();
        for (int i = 0; i < repeatMovement; i++) {
            generateFrameObjects(jsonWriter);
        }
        jsonWriter.endArray();
    }

    private void generateFrameObjects(JsonWriter jsonWriter) throws IOException {
        for (int i = 0; i < amountOfFrames; i++) {
            jsonWriter.beginObject();
            if (forwardFrames != -1 && backwardFrames != -1 && i >= forwardFrames) {
                addFrameObject(jsonWriter, countBackwards(i));
            } else {
                addFrameObject(jsonWriter, i);
            }
            jsonWriter.endObject();
        }
    }

    private void addFrameObject(JsonWriter jsonWriter, int counter) throws IOException {
        for (MarkerGenerator markerGenerator : markerGenerators) {
            markerToJsonConverter.addToJsonObject(jsonWriter, markerGenerator.getNextMarker(counter));
        }
    }

    private int countBackwards(int counter) {
        return forwardFrames - 2 - (counter - forwardFrames);
    }

}
