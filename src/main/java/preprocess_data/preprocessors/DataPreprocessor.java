package preprocess_data.preprocessors;

import org.datavec.api.records.reader.RecordReader;

import java.io.File;
import java.io.IOException;

//Durchführung eines kompletten Vorverarbeitungsprozesses und anschließende Persitierung der Daten
public abstract class DataPreprocessor<T extends RecordReader> {

    private static final String FILE_NAME = "save%";

    /*Speichern der Daten in einem RecordReader im angegebenen Pfad*/
    public abstract void saveData(T reader, String directoryPath) throws Exception;

    /*
    Speichern der Daten in einem angegebenen Pfad und Rückgabe eines RecordReaders,
    mit dem direkt über die persistierten Daten iteriert werden kann.
     */
    public final T saveDataToFile(T reader, String directoryPath) throws Exception {
        SaveDirectoryManager.initDirectory(directoryPath, FILE_NAME);
        saveData(reader, directoryPath);
        reader.reset();
        return getReader(directoryPath);
    }

    public abstract T getReader(String directoryPath) throws IOException, InterruptedException;

    //Testen ob bereits Daten persisitert wurden
    public static boolean directoryHasData(String directoryPath) {
        File directory = new File(directoryPath);
        if (!directory.isDirectory()) {
            throw new IllegalArgumentException(directoryPath + " is not a directory path");
        }
        return SaveDirectoryManager.getSaveFiles(directory, FILE_NAME).length > 0;
    }

}
