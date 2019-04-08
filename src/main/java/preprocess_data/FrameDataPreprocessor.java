package preprocess_data;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;

import java.io.File;
import java.io.IOException;

public class FrameDataPreprocessor extends DataPreprocessor<RecordReader> {

    private final int splitIndex;
    private static final String FILE_NAME = "save%";

    public FrameDataPreprocessor(int splitIndex) {
        this.splitIndex = splitIndex;
    }

    public FrameDataPreprocessor() {
        splitIndex = -1;
    }

    @Override
    public RecordReader getReader(String directoryPath) throws IOException, InterruptedException {
        CSVRecordReader csvRecordReader = new CSVRecordReader();
        csvRecordReader.initialize(new FileSplit(new File(directoryPath)));
        return csvRecordReader;
    }

    @Override
    public void saveData(RecordReader reader, String directoryPath) throws Exception {
        int amountOfSplit = 0;
        CSVRecordWriter recordWriter = getNextWriter(amountOfSplit, directoryPath);
        for (int i = 0; reader.hasNext(); i++) {
            if (splitIndex != -1 && i > 0 && i % splitIndex == 0) {
                amountOfSplit++;
                recordWriter.close();
                recordWriter = getNextWriter(amountOfSplit, directoryPath);
            }
            recordWriter.write(reader.next());
        }
        recordWriter.close();
    }

    static CSVRecordWriter getNextWriter(int amountOfSplit, String directoryPath) throws Exception {
        CSVRecordWriter recordWriter = new CSVRecordWriter();
        FileSplit fileSplit = new FileSplit(getNextFile(amountOfSplit, directoryPath));
        recordWriter.initialize(fileSplit, new NumberOfRecordsPartitioner());
        return recordWriter;
    }

    static File getNextFile(int amountOfSplit, String directoryPath) {
        return new File(directoryPath + "\\" + FILE_NAME + amountOfSplit + ".txt");
    }
}
