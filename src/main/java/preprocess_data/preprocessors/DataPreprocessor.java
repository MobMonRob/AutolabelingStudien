package preprocess_data.preprocessors;

import org.datavec.api.records.reader.RecordReader;

import java.io.File;
import java.io.IOException;

public abstract class DataPreprocessor<T extends RecordReader> {

    private static final String FILE_NAME = "save%";

    public abstract T getReader(String directoryPath) throws IOException, InterruptedException;

    public abstract void saveData(T reader, String directoryPath) throws Exception;

    public final T saveDataToFile(T reader, String directoryPath) throws Exception {
        SaveDirectoryManager.initDirectory(directoryPath, FILE_NAME);
        saveData(reader, directoryPath);
        reader.reset();
        return getReader(directoryPath);
    }

    public static boolean directoryHasData(String directoryPath) {
        File directory = new File(directoryPath);
        if (!directory.isDirectory()) {
            throw new IllegalArgumentException(directoryPath + " is not a directory path");
        }
        return SaveDirectoryManager.getSaveFiles(directory, FILE_NAME).length > 0;
    }

}
