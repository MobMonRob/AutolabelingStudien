package preprocess_data;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;

import java.io.File;
import java.io.IOException;

public class DataPreprocessor {

    private final int splitIndex;
    private static final String FILE_NAME = "save%";

    public DataPreprocessor(int splitIndex) {
        this.splitIndex = splitIndex;
    }

    public DataPreprocessor() {
        splitIndex = -1;
    }

    public CSVRecordReader saveDataToFile(RecordReader recordReader, String directoryPath) throws Exception {
        initDirectory(directoryPath);
        int amountOfSplit = 0;
        CSVRecordWriter recordWriter = getNextWriter(amountOfSplit, directoryPath);
        for (int i = 0; recordReader.hasNext(); i++) {
            if (splitIndex != -1 && i > 0 && i % splitIndex == 0) {
                amountOfSplit++;
                recordWriter.close();
                recordWriter = getNextWriter(amountOfSplit, directoryPath);
            }
            recordWriter.write(recordReader.next());
        }
        recordReader.reset();
        recordWriter.close();
        return getCSVReader(directoryPath);
    }

    private CSVRecordWriter getNextWriter(int amountOfSplit, String directoryPath) throws Exception {
        CSVRecordWriter recordWriter = new CSVRecordWriter();
        FileSplit fileSplit = new FileSplit(getNextFile(amountOfSplit, directoryPath));
        recordWriter.initialize(fileSplit, new NumberOfRecordsPartitioner());
        return recordWriter;
    }

    private CSVRecordReader getCSVReader(String directoryPath) throws IOException, InterruptedException {
        CSVRecordReader csvRecordReader = new CSVRecordReader();
        csvRecordReader.initialize(new FileSplit(new File(directoryPath)));
        return csvRecordReader;
    }

    private File getNextFile(int amountOfSplit, String directoryPath) {
        return new File(directoryPath + "\\" + FILE_NAME + amountOfSplit + ".txt");
    }

    private void initDirectory(String directoryString) {
        File directory = new File(directoryString);
        if (!directory.mkdirs()) {
            System.out.println("Directory already exists");
        }
        clearSaveFiles(directory);
    }

    private void clearSaveFiles(File directory) {
        System.out.println("clear directory files...");
        File[] filteredFiles = directory.listFiles(((dir, name) -> name.contains(FILE_NAME)));
        for (File filteredFile : filteredFiles) {
            if (filteredFile != null && !filteredFile.delete()) {
                System.out.println("Delete " + filteredFile.getName() + " not successful!");
            }
        }
        System.out.println("finish clean-up");
    }

}
