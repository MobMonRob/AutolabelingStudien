package data_generation;


import com.google.gson.stream.JsonWriter;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class TrialDataGenerator {

    private final FrameGenerator frameGenerator;
    private final String generationPath;

    public TrialDataGenerator(FrameGenerator frameGenerator, String generationPath) {
        this.frameGenerator = frameGenerator;
        this.generationPath = generationPath;
    }


    void generateTrial(String filename, int repeatMovement) {
        try {
            JsonWriter jsonWriter = new JsonWriter(new FileWriter(createFile(filename)));
            jsonWriter.beginObject().name("trial").beginObject().name("frames");
            frameGenerator.generateFrames(jsonWriter, repeatMovement);
            jsonWriter.endObject().endObject(); //close remaining Objects
            jsonWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private File createFile(String filename) {
        File file = new File(generationPath + "\\" + filename);

        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return file;
    }
}
