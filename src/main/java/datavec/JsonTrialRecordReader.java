package datavec;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import preprocess_data.TrialDataManager;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/*
RecordReader, der die Daten eines Trials aus einem fest definierten JSON-Format einliest. Dieses Format wird im
JsonToTrialParser definiert, der Teil des TrialDataManagers ist.

Die Konfiguration der Vorverarbeitung wird durch den bei der Initialisierung übergebenen TrialDataManager möglich.

Die Ausgabe des RecordReaders sind die Daten eines einzelnen Frames, die in Writables konvertiert wurden.
*/
public class JsonTrialRecordReader extends BaseRecordReader {

    TrialDataManager trialDataManager; //Regelt den kompletten Vorverarbeitungsprozess
    TrialFileIterator fileIterator; //Iteriert über die Dateien des FileSplits
    FileSplit fileSplit;
    Iterator<ArrayList<Writable>> fileContentIterator;

    public JsonTrialRecordReader(TrialDataManager trialDataManager) {
        this.trialDataManager = trialDataManager;
    }

    //only accept File inputSplit
    public void initialize(InputSplit inputSplit) throws IOException, InterruptedException, IllegalArgumentException {
        if (!(inputSplit instanceof FileSplit)) {
            throw new IllegalArgumentException("JsonTrialRecordReader is for file input only");
        }
        this.fileSplit = (FileSplit) inputSplit;
        initIterators(fileSplit);
    }

    void initIterators(final FileSplit fileSplit) {
        fileIterator = new TrialFileIterator(fileSplit);
        trialDataManager.setTrialContent(fileIterator.next());
        fileContentIterator = trialDataManager.getNextTrialContent().iterator();
    }

    public void initialize(Configuration configuration, InputSplit inputSplit) throws IOException, InterruptedException, IllegalArgumentException {
        initialize(inputSplit);
    }

    public List<Writable> next() {
        if (fileContentIterator.hasNext()) {
            return fileContentIterator.next();
        } else if (trialDataManager.hasNext()) {
            fileContentIterator = trialDataManager.getNextTrialContent().iterator();
            return fileContentIterator.next();
        } else if (fileIterator.hasNext()) {
            trialDataManager.setTrialContent(fileIterator.next());
            fileContentIterator = trialDataManager.getNextTrialContent().iterator();
            return fileContentIterator.next();
        } else {
            throw new NoSuchElementException();
        }
    }

    public boolean hasNext() {
        return !(!fileIterator.hasNext() && !trialDataManager.hasNext() && !fileContentIterator.hasNext());
    }

    public List<String> getLabels() {
        return trialDataManager.getDataTransformer().getConverter().getFrameLabelingStrategy().getLabels();
    }

    public void reset() {
        initIterators(fileSplit);
    }

    public boolean resetSupported() {
        return true;
    }

    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        return null;
    }

    public Record nextRecord() {
        List<Writable> next = next();
        //metadata --> fileIndex/location (get from TrialFileIterator). Closer look: https://github.com/deeplearning4j/DataVec/blob/master/datavec-api/src/main/java/org/datavec/api/records/reader/impl/csv/CSVRecordReader.java
        return new org.datavec.api.records.impl.Record(next, null); //quick fix
    }

    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return null;
    }

    public List<Record> loadFromMetaData(List<RecordMetaData> list) throws IOException {
        return null;
    }

    public void close() throws IOException {

    }

    public void setConf(Configuration configuration) {

    }

    public Configuration getConf() {
        return null;
    }
}
