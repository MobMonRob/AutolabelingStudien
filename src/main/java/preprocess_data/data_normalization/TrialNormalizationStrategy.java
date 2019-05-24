package preprocess_data.data_normalization;

import preprocess_data.data_model.Frame;
import preprocess_data.data_model.Marker;

//Normalisierung eines einzelnen Frames
public interface TrialNormalizationStrategy {

    Frame normalizeFrame(Frame frame);

    //Method to collect data for the normalization process from outside (during Json-parsing)
    void collectMarkerData(Marker marker);

    //returns new instance of the strategy to reset values and calculations
    TrialNormalizationStrategy getNewInstance();
}
