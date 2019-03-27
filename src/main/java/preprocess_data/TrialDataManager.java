package preprocess_data;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import org.datavec.api.writable.Writable;
import preprocess_data.data_model.Frame;
import preprocess_data.data_normalization.TrialNormalizationStrategy;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Set;

//manages json-parsing, normalization and transformation of trial-data
public class TrialDataManager {

    private Iterator<Frame> frameIterator;
    private final TrialDataTransformation dataTransformer;
    private final JsonToTrialParser jsonToTrialParser = new JsonToTrialParser();
    private TrialNormalizationStrategy normalizationStrategy;
    private final Set<String> acceptedMarkers;

    public TrialDataManager(TrialDataTransformation dataTransformer, TrialNormalizationStrategy normalizationStrategy,
                            Set<String> acceptedMarkers) {
        this.dataTransformer = dataTransformer;
        this.normalizationStrategy = normalizationStrategy;
        this.acceptedMarkers = acceptedMarkers;
        jsonToTrialParser.addFilter(acceptedMarkers);
    }

    public TrialDataManager(TrialDataTransformation dataTransformer) {
        this(dataTransformer,null,null);
    }

    public void setTrialContent(JsonArray trialData) {
        if (normalizationStrategy != null) {
            normalizationStrategy = normalizationStrategy.getNewInstance();
        }
        getFramesFromJson(trialData);
    }

    public ArrayList<ArrayList<Writable>> getNextTrialContent() {
        final Frame currentFrame = frameIterator.next();
        return new ArrayList<>(transformFrameToWritable(currentFrame));
    }

    public boolean hasNext() {
        return frameIterator.hasNext();
    }

    private ArrayList<ArrayList<Writable>> transformFrameToWritable(Frame frame) {
        if (normalizationStrategy != null) {
            return dataTransformer.transformFrameData(normalizationStrategy.normalizeFrame(frame));
        }
        return dataTransformer.transformFrameData(frame);
    }

    private void getFramesFromJson(JsonArray trialData) {
        final ArrayList<Frame> currentFrames = new ArrayList<>();
        for (JsonElement trialDatum : trialData) {
            Frame frame = jsonToTrialParser.getFrameFromJson(trialDatum.getAsJsonObject(), normalizationStrategy);
            currentFrames.add(frame);
        }
        this.frameIterator = currentFrames.iterator();
    }

    public TrialDataTransformation getDataTransformer() {
        return dataTransformer;
    }

    public String getInfoString() {
        return dataTransformer.getInfoString() + "\n" + "Normalisierung: " + normalizationStrategy.toString();
    }
}
