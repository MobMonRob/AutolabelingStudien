package datavec;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.datavec.api.split.FileSplit;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class TrialFileIterator implements Iterator<JsonArray> {

    private final FileSplit files;
    private final JsonParser jsonParser = new JsonParser();
    private int currentIndex = 0;
    private static final String TRIAL_PROPERTY = "trial";
    private static final String FRAMES_PROPERTY = "frames";

    //this iterator only accepts json files in a well defined structure
    public TrialFileIterator(FileSplit files) {
        this.files = files;
    }

    public boolean hasNext() {
        return currentIndex < files.locations().length;
    }

    public JsonArray next(){
        if (hasNext()) {
            try {
                System.out.println("current file: " + files.locations()[currentIndex]);
                InputStream inputStream = files.openInputStreamFor(files.locations()[currentIndex++].toString());
                JsonArray result = getFramesArray(inputStream);
                if (result != null) { //If Json is invalid, no result is returned
                    return result;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        throw new NoSuchElementException();
    }

    private JsonArray getFramesArray(InputStream inputStream) {
        JsonObject rootElement = jsonParser.parse(new InputStreamReader(inputStream)).getAsJsonObject();
        return rootElement.getAsJsonObject(TRIAL_PROPERTY).getAsJsonArray(FRAMES_PROPERTY);
    }

    public int getCurrentFileIndex() {
        return currentIndex;
    }

    public void remove() {
        throw new UnsupportedOperationException();
    }
}
