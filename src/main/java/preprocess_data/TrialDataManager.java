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

/*Eine bessere Design-Entscheidung wäre an dieser Stelle eine Schnittstelle,
 sodass das JSON-Format flexibel festgelegt werden kann */
public class TrialDataManager {

    private Iterator<Frame> frameIterator;
    private final TrialDataTransformation dataTransformer;
    private final JsonToTrialParser jsonToTrialParser = new JsonToTrialParser();
    private TrialNormalizationStrategy normalizationStrategy;
    private int currentAmountOfFrames;

    public TrialDataManager(TrialDataTransformation dataTransformer, TrialNormalizationStrategy normalizationStrategy,
                            Set<String> acceptedMarkers) {
        this.dataTransformer = dataTransformer;
        this.normalizationStrategy = normalizationStrategy;
        jsonToTrialParser.setFilter(acceptedMarkers);
    }

    public TrialDataManager(TrialDataTransformation dataTransformer) {
        this(dataTransformer, null, null);
    }

    public void setTrialContent(JsonArray trialData) {
        if (normalizationStrategy != null) {
            normalizationStrategy = normalizationStrategy.getNewInstance();
        }
        this.currentAmountOfFrames = trialData.size();
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

    /*Die JSON-Daten des Trails werden eingelesen. Alle Frames eines Trials werden in Frame-Objekte umgewandelt und in einer
     * Array-Liste gespeichert. Aus der Liste wird ein Iterator erzeugt*/
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

    public int getAmountOfFrames() {
        return this.currentAmountOfFrames;
    }

    public void setNewFilter(Set<String> filterMarkers) {
        jsonToTrialParser.setFilter(filterMarkers);
    }

    public String getInfoString() {
        return dataTransformer.getInfoString() + "\n" + "Normalisierung: " + normalizationStrategy.toString();
    }
}
