package preprocess_data.preprocessors;

import java.io.File;

public class SaveDirectoryManager {

    static void initDirectory(String directoryString, String saveFileString) {
        File directory = new File(directoryString);
        if (!directory.mkdirs()) {
            System.out.println("Directory already exists");
        }
        clearSaveFiles(directory, saveFileString);
    }

    static void clearSaveFiles(File directory, String filter) {
        System.out.println("clear directory files...");
        File[] filteredFiles = getSaveFiles(directory,filter);
        for (File filteredFile : filteredFiles) {
            if (filteredFile != null && !filteredFile.delete()) {
                System.out.println("Delete " + filteredFile.getName() + " not successful!");
            }
        }
        System.out.println("finish clean-up");
    }

    static File[] getSaveFiles(File directory, String filter) {
        return directory.listFiles(((dir, name) -> name.contains(filter)));
    }
}
