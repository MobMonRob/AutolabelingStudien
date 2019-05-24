package data_generation;

import preprocess_data.data_model.Marker;

import java.util.ArrayList;

/* Beispielklasse zum Generierung eines Trials */
public class TestGeneration {

    public static void main(String[] args) {

        /*
        erster Schritt ist das Erstellen mehrerer Marker. Jeder Marker verfügt über ein Label und über eine
        Ausgangsposition (x,y,z)
        */
        Marker marker1 = new Marker("1", 5, 10, 5);
        Marker marker2 = new Marker("2", 2, 2, 2);
        Marker marker3 = new Marker("3", 1, 1, 1);
        Marker marker4 = new Marker("4", 4, 0, 0);
        Marker marker5 = new Marker("5", 1, 1, -10);
        Marker marker6 = new Marker("6", 10, 4, 1);
        Marker marker7 = new Marker("7", 0, 0, 0);
        Marker marker8 = new Marker("8", 9, -9, -10);
        Marker marker9 = new Marker("9", 1, 1, 4);
        Marker marker10 = new Marker("10", 3, 8, -5);

        /*
        Erstellen der Marker-Generatoren. Für jeden Marker wird eine movement-function und eine direction-function
        definiert.
        */
        MarkerGenerator generator1 = new MarkerGenerator(marker1,
                new MarkerMovementFunction(1, 1),
                new DirectionFunction(-0.1, 1));
        MarkerGenerator generator2 = new MarkerGenerator(marker2,
                new MarkerMovementFunction(1, 1),
                new DirectionFunction(0, 1));
        MarkerGenerator generator3 = new MarkerGenerator(marker3,
                new MarkerMovementFunction(1, 1),
                new DirectionFunction(-0.1, -1));
        MarkerGenerator generator4 = new MarkerGenerator(marker4,
                new MarkerMovementFunction(1, 1),
                new DirectionFunction(0.1, 0));
        MarkerGenerator generator5 = new MarkerGenerator(marker5,
                new MarkerMovementFunction(1, 1),
                new DirectionFunction(0.1, -1));
        MarkerGenerator generator6 = new MarkerGenerator(marker6,
                new MarkerMovementFunction(1, 1),
                new DirectionFunction(0.1, 1));
        MarkerGenerator generator7 = new MarkerGenerator(marker7,
                new MarkerMovementFunction(1, 1),
                new DirectionFunction(-0.1, 1));
        MarkerGenerator generator8 = new MarkerGenerator(marker8,
                new MarkerMovementFunction(1, 1),
                new DirectionFunction(0.1, -1));
        MarkerGenerator generator9 = new MarkerGenerator(marker9,
                new MarkerMovementFunction(1, 1),
                new DirectionFunction(-0.1, -2));
        MarkerGenerator generator10 = new MarkerGenerator(marker10,
                new MarkerMovementFunction(1, 1),
                new DirectionFunction(0.1, 1));

        ArrayList<MarkerGenerator> markerGenerators = new ArrayList<MarkerGenerator>();
        markerGenerators.add(generator1);
        markerGenerators.add(generator2);
        markerGenerators.add(generator3);
        markerGenerators.add(generator4);
        markerGenerators.add(generator5);
        markerGenerators.add(generator6);
        markerGenerators.add(generator7);
        markerGenerators.add(generator8);
        markerGenerators.add(generator9);
        markerGenerators.add(generator10);

        /*
        Erstellen eines FrameGenerators mit einer Liste von Marker-Generatoren. Bei der Inititalisierung wird die Anzahl
        Frames hin und zurück definiert.
        */
        FrameGenerator frameGenerator = new FrameGenerator(markerGenerators, 1250, 1250);

        //Dieser Pfad muss entsprechend angepasst werden!
        String generationPath = "C:\\Users\\<...>";

        /*
        Erstellen des Trial-Generators. Mit dem generationPath wird der Ausgabepfad angegeben.
        */
        TrialDataGenerator trialDataGenerator = new TrialDataGenerator(frameGenerator, generationPath);
        for (int i = 0; i < 20; i++) {
            trialDataGenerator.generateTrial("test" + i + ".json", 1);
        }


    }
}
