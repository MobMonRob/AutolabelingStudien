package preprocess_data.labeling;

import org.datavec.api.writable.Writable;
import preprocess_data.data_model.Frame;

import java.util.ArrayList;
import java.util.List;

/*
Eine Labeling-Strategie definiert, welche Features und Label in den Trainingsdaten verwendet werden sollen.
Am Ende werden die Features und Label in entsprechende Writables der Datavec API umgewandelt.
*/
public interface FrameLabelingStrategy {

    ArrayList<Writable> getLabeledWritableList(Frame frame);

    //Used in RecordReader to get amount of Labels
    List<String> getLabels();
}
