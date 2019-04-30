package test.execution.result_logging;

import java.io.*;
import java.util.ArrayList;

public class FileManager {

    private final File file;
    private boolean valid = true;

    public FileManager(File file) {
        this.file = createFileIfNotExist(file);
    }

    public ArrayList<String> getFileContent() {
        ArrayList<String> resultList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            while (br.ready()) {
                String s = br.readLine();
                resultList.add(s);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return resultList;
    }
    private File createFileIfNotExist(File file) {
        if (!file.exists()) {
            try {
                this.valid = file.createNewFile();
            } catch (IOException e) {
                this.valid = false;
            }
        }
        return file;
    }
    boolean hasValidState() {
        return valid;
    }

    public void writeInFile(String content) {
        if (hasValidState()) {
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(file, true))) {
                bw.write(content);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
