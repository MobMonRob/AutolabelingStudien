package preprocess_data.preprocessors;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class SequentialDataPreprocessor extends DataPreprocessor<SequenceRecordReader> {

    @Override
    public SequenceRecordReader getReader(String directoryPath) throws IOException, InterruptedException {
        CSVSequenceRecordReader sequenceRecordReader = new CSVSequenceRecordReader();
        sequenceRecordReader.initialize(new FileSplit(new File(directoryPath)));
        return sequenceRecordReader;
    }

    @Override
    public void saveData(SequenceRecordReader reader, String directoryPath) throws Exception {
        for (int i = 0; reader.hasNext(); i++) {
            CSVRecordWriter nextWriter = FrameDataPreprocessor.getNextWriter(i, directoryPath);
            writeRecords(reader.sequenceRecord(), nextWriter);
            nextWriter.close();
        }
    }

    private void writeRecords(List<List<Writable>> records, CSVRecordWriter writer) throws IOException {
        for (List<Writable> record : records) {
            writer.write(record);
        }
    }
}
