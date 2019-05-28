package test.tests.lstm;

import datavec.SequentialMarkerwiseTrialRecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.FileSplit;
import preprocess_data.TrialDataManager;
import preprocess_data.builders.TrialDataManagerBuilder;
import preprocess_data.builders.TrialDataTransformationBuilder;
import preprocess_data.labeling.FrameLabelingStrategy;
import preprocess_data.labeling.NoLabeling;
import preprocess_data.preprocessors.SequentialDataPreprocessor;

import java.io.File;
import java.util.Arrays;
import java.util.TreeSet;

//einfache Testklasse für den SequentialRecordReader zur Überprüfung der einzelnen Datensätze, bzw. zum Debugging
public class TestSequentialRecordReader {

    public static void main(String[] args) throws Exception {

        //Pfad zu den Testdaten im JSON-Format
        File trainDirectory = new File("C:\\Users\\...");

        //Liste mit mit allen möglichen Markern
        String[] markerLabels = {"C7", "CLAV", "LASI", "LELB", "LELBW", "LHUM4", "LHUMA", "LHUMP", "LHUMS", "LRAD", "LSCAP1", "LSCAP2", "LSCAP3", "LSCAP4", "LULN", "RASI", "RELB", "RELBW", "RHUM4", "RHUMA", "RHUMP", "RHUMS", "RRAD", "RSCAP1", "RSCAP2", "RSCAP3", "RSCAP4", "RULN", "SACR", "STRN", "T10", "THRX1", "THRX2", "THRX3", "THRX4"};
        //Filter für die Marker, der im RecordReader gesetzt werden kann. Zum Testen von kleineren Datenmengen,
        //die Liste verkleinern
        TreeSet<String> selectedLabels = new TreeSet<>(Arrays.asList(markerLabels));

        //Konfiguration der Datensätze mit dem TrialDataManager. Alternativ können Strategien zur Normalisierung und
        //Manipulation gesetzt werden.
        FrameLabelingStrategy frameLabelingStrategy = new NoLabeling(); //NoLabeling, da Regression
        TrialDataManager dataManager = TrialDataManagerBuilder
                .addTransformation(TrialDataTransformationBuilder.addLabelingStrategy(frameLabelingStrategy).build())
                .build();


        //Initialisierung des RecordReaders
        int sequenceLength = 20;
        //Übergeben werden: TrialDataManager; Set mit allen Markern, die Eingelesen werden sollen; Länge der Sequenz
        SequenceRecordReader recordReader = new SequentialMarkerwiseTrialRecordReader(dataManager, selectedLabels, sequenceLength);
        recordReader.initialize(new FileSplit(trainDirectory));

        //Speichern der Daten im ausgegebenen Directory. Hierfür wird der SequentialDataPreprocessor verwendet.
        SequentialDataPreprocessor dataPreprocessor = new SequentialDataPreprocessor();
        //Pfad muss angepasst werden!
        String saveDirectory = "C:\\Users\\...";
        //Speichern der Daten im Directory. Zurückgegeben wird ein CSVSequenceRecordReader, der die gespeicherten Daten
        //ausgibt.
        SequenceRecordReader sequenceRecordReader = dataPreprocessor.saveDataToFile(recordReader, saveDirectory);

        //Iterieren über die gespeicherten Daten und Ausgabe
        while (sequenceRecordReader.hasNext()) {
            System.out.println(sequenceRecordReader.sequenceRecord());
        }
    }
}
