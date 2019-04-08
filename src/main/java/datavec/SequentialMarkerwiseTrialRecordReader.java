package datavec;

import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import preprocess_data.TrialDataManager;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.*;

public class SequentialMarkerwiseTrialRecordReader extends JsonTrialRecordReader implements SequenceRecordReader {

    private final boolean hasSequenceLength;
    private int sequenceLength = -1;
    private Set<String> markerStrings;
    private Iterator<String> markerStringsIterator;
    private int currentTrialAmountOfFrames;
    private int currentFrameIndex;
    private List<Writable> removedElement;

    public SequentialMarkerwiseTrialRecordReader(TrialDataManager dataManager, Set<String> markerStrings, int sequenceLength) {
        super(dataManager);
        this.markerStrings = markerStrings;
        this.sequenceLength = sequenceLength + 1; //next element is needed to label.
        this.hasSequenceLength = true;
    }

    //last frame is final label --> sequenceLength = amount of frames in trial -1
    public SequentialMarkerwiseTrialRecordReader(TrialDataManager dataManager, Set<String> markerStrings) {
        super(dataManager);
        this.markerStrings = markerStrings;
        this.hasSequenceLength = false;
    }

    @Override
    public void initialize(InputSplit inputSplit) throws IOException, InterruptedException, IllegalArgumentException {
        initMarkerStringIterator(); //set first Marker as Filter
        this.fileSplit = (FileSplit) inputSplit;
        this.initIterators(fileSplit);
        initNewTrial();
    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        List<List<Writable>> resultSequence = getNextUnlabeledSequence();
        labelSequence(resultSequence);
        return resultSequence;
    }

    private void labelSequence(List<List<Writable>> nextUnlabeledSequence) {
        final int length = nextUnlabeledSequence.size();
        for (int i = 0; i < length; i++) {
            if (i < length - 1) {
                nextUnlabeledSequence.get(i).addAll(nextUnlabeledSequence.get(i + 1));
            }
        }
        //last element is final label
        this.removedElement = nextUnlabeledSequence.remove(nextUnlabeledSequence.size() - 1);
    }

    private List<List<Writable>> getNextUnlabeledSequence() {
        int framesLeftInTrial = framesLeftInTrial();
        if (framesLeftInTrial >= sequenceLength) {
            this.currentFrameIndex += sequenceLength - 1;
            return getNextTrialSequence(sequenceLength);
        } else if (framesLeftInTrial > 0) {
            return getNextTrialSequence(framesLeftInTrial);
        } else if (markerStringsIterator.hasNext()) {
            setMarkerFilter(markerStringsIterator.next());
            return sequenceRecord();
        } else if (fileIterator.hasNext()) {
            trialDataManager.setTrialContent(fileIterator.next());
            initNewTrial();
            initMarkerStringIterator();
            return sequenceRecord();
        } else
            throw new NoSuchElementException();
    }

    private void initMarkerStringIterator() {
        this.markerStringsIterator = markerStrings.iterator();
        setMarkerFilter(markerStringsIterator.next()); //set first Marker as Filter
    }

    //returns true with empty records
    public boolean hasNext() {
        return !(!sequenceLeftInTrial() && !markerStringsIterator.hasNext() && !fileIterator.hasNext());
    }

    private boolean sequenceLeftInTrial() {
        return currentFrameIndex + sequenceLength < currentTrialAmountOfFrames;
    }

    private List<List<Writable>> getNextTrialSequence(int sequenceLength) {
        ArrayList<List<Writable>> resultList = new ArrayList<>();
        int currentSequenceLength = sequenceLength;
        if (hasSequenceLength && removedElement != null) {
            resultList.add(removedElement);
            currentSequenceLength--;
        }
        for (int length = currentSequenceLength; length > 0; length--) {
            if (super.hasNext()) {
                resultList.add(super.next());
            }
        }
        return resultList;
    }

    private void setMarkerFilter(String markerString) {
        Set<String> newSet = new HashSet<>();
        newSet.add(markerString);
        this.trialDataManager.setNewFilter(newSet);
    }

    private void initNewTrial() {
        this.currentFrameIndex = 0;
        this.currentTrialAmountOfFrames = trialDataManager.getAmountOfFrames();
        if (!hasSequenceLength) {
            sequenceLength = currentTrialAmountOfFrames;
        }
    }

    private int framesLeftInTrial() {
        if (!hasSequenceLength) {
            return sequenceLength;
        }
        return currentTrialAmountOfFrames - (currentFrameIndex + sequenceLength);
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        return null;
    }

    @Override
    public SequenceRecord nextSequence() {
        return null;
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return null;
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> list) throws IOException {
        return null;
    }
}
