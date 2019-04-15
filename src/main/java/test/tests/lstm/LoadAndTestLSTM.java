package test.tests.lstm;

import org.datavec.api.records.reader.impl.csv.CSVNLinesSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import play.api.libs.iteratee.Input;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class LoadAndTestLSTM {

    public static void main(String[] args) throws IOException, InterruptedException {

        String saveDirectoryTest = "C:\\Users\\Nico\\Documents\\Studienarbeit\\Daten_Studienarbeit\\save\\lstm\\test";
        CSVNLinesSequenceRecordReader sequenceReader = new CSVNLinesSequenceRecordReader(1);
        sequenceReader.initialize(new FileSplit(new File(saveDirectoryTest)));

        SequenceRecordReaderDataSetIterator iterator = new SequenceRecordReaderDataSetIterator(
                sequenceReader, 1, 2, 3, true);

        DataSet next = iterator.next();
        System.out.println(next.getFeatures().getRow(0));


        String savePath1 = "C:\\Users\\Nico\\Documents\\Studienarbeit\\Daten_Studienarbeit\\save\\lstm\\models\\lstm-v6.zip";
        String savePath2 = "C:\\Users\\Nico\\Documents\\Studienarbeit\\Daten_Studienarbeit\\save\\lstm\\models\\lstm-v3.zip";
        MultiLayerNetwork multiLayerNetwork1 = ModelSerializer.restoreMultiLayerNetwork(savePath1);
        MultiLayerNetwork multiLayerNetwork2 = ModelSerializer.restoreMultiLayerNetwork(savePath2);
        List<INDArray> results1 = multiLayerNetwork1.feedForward(next.getFeatures());
        List<INDArray> results2 = multiLayerNetwork2.feedForward(next.getFeatures());


        System.out.println("LSTM2:");
        logINDArrayList(results1);
        System.out.println("Label");
        System.out.println(next.getLabels());

    }

    private static void logINDArrayList(List<INDArray> indArrays) {
        System.out.println("Input:");
        System.out.println(indArrays.get(0));
        System.out.println("LSTM:");
        System.out.println(indArrays.get(1));
        System.out.println("Output");
        System.out.println(indArrays.get(3));
    }
}
