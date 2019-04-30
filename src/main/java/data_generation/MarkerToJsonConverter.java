package data_generation;

import com.google.gson.stream.JsonWriter;
import preprocess_data.data_model.Marker;

import java.io.IOException;

class MarkerToJsonConverter {

    void addToJsonObject(JsonWriter jsonWriter, Marker marker) throws IOException {
        jsonWriter.name(marker.getLabel() + "_x").value(marker.getX());
        jsonWriter.name(marker.getLabel() + "_y").value(marker.getY());
        jsonWriter.name(marker.getLabel() + "_z").value(marker.getZ());
    }
}
