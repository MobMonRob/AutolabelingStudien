package datavec;

import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import preprocess_data.TrialDataManager;

import java.io.IOException;
import java.util.*;

/*
Diese Klasse ist eine Art Wrapper für den JsonTrialRecordReader. Ein festgelegte Anzahl an Frames wird in einem
Storage gehalten und geshufflet. Dadurch wird die Reihenfolge der Frames verändert

Bei der Initialisierung kann die Größe des Speichers festgelegt werden. (storageSize)
Achtung: Bei zu großem Storage kann die Performance der Vorverarbeitung durch den Garbage Collector beeinflusst werden.
*/
public class RandomizedTrialRecordReader extends JsonTrialRecordReader {

    private final int storageSize;
    private Iterator<List<Writable>> storageIterator;

    public RandomizedTrialRecordReader(TrialDataManager trialDataManager, int storageSize) {
        super(trialDataManager);
        this.storageSize = storageSize;
    }

    @Override
    public void initialize(InputSplit inputSplit) throws IOException, InterruptedException, IllegalArgumentException {
        super.initialize(inputSplit);
        this.storageIterator = getShuffledFrames().iterator();
    }

    public List<Writable> next() {
        if (storageIterator.hasNext()) {
            return storageIterator.next();
        } else if (super.hasNext()) {
            storageIterator = getShuffledFrames().iterator();
            return storageIterator.next();
        } else {
            throw new NoSuchElementException();
        }
    }

    public boolean hasNext() {
        return !(!storageIterator.hasNext() && !super.hasNext());
    }

    private ArrayList<List<Writable>> getStoredFrames() {
        ArrayList<List<Writable>> storedFrames = new ArrayList<>();
        int counter = 0;
        while (super.hasNext() && counter <= storageSize) {
            counter++;
            storedFrames.add(super.next());
        }
        return storedFrames;
    }

    private ArrayList<List<Writable>> getShuffledFrames() {
        ArrayList<List<Writable>> storedFrames = getStoredFrames();
        Collections.shuffle(storedFrames);
        return storedFrames;
    }
}
