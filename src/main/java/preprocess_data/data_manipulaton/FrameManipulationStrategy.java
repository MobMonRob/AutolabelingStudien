package preprocess_data.data_manipulaton;

import preprocess_data.data_model.Frame;

import java.util.ArrayList;

/*
Schnittstelle für alle Strategien zur Manipulation der Daten.
Die Art der Manipulation ist dabei völlig beliebig.

Mithilfe der Strategien wird unter anderem Data Augmentation umgesetzt.
*/

public interface FrameManipulationStrategy {

    //takes frame and returns one or more manipulated instances of it
    ArrayList<Frame> manipulateFrame(Frame frame);
}
