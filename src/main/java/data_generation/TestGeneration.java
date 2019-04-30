package data_generation;

import preprocess_data.data_model.Marker;

import java.util.ArrayList;

public class TestGeneration {

    public static void main(String[] args) {

        Marker marker1 = new Marker("1", 5,10,5);
        Marker marker2 = new Marker("2", 2,2,2);
        Marker marker3 = new Marker("3", 1,1,1);
        Marker marker4 = new Marker("4", 4,0,0);
        Marker marker5 = new Marker("5", 1,1,-10);
        Marker marker6 = new Marker("6", 10,4,1);
        Marker marker7 = new Marker("7", 0,0,0);
        Marker marker8 = new Marker("8", 9,-9,-10);
        Marker marker9 = new Marker("9", 1,1,4);
        Marker marker10 = new Marker("10", 3,8,-5);

        MarkerGenerator generator1 = new MarkerGenerator(marker1,
                new MarkerMovementFunction(1,1),
                new DirectionFunction(-0.1,1));
        MarkerGenerator generator2 = new MarkerGenerator(marker2,
                new MarkerMovementFunction(1,1),
                new DirectionFunction(0,1));
        MarkerGenerator generator3 = new MarkerGenerator(marker3,
                new MarkerMovementFunction(1,1),
                new DirectionFunction(-0.1,-1));
        MarkerGenerator generator4 = new MarkerGenerator(marker4,
                new MarkerMovementFunction(1,1),
                new DirectionFunction(0.1,0));
        MarkerGenerator generator5 = new MarkerGenerator(marker5,
                new MarkerMovementFunction(1,1),
                new DirectionFunction(0.1,-1));
        MarkerGenerator generator6 = new MarkerGenerator(marker6,
                new MarkerMovementFunction(1,1),
                new DirectionFunction(0.1,1));
        MarkerGenerator generator7 = new MarkerGenerator(marker7,
                new MarkerMovementFunction(1,1),
                new DirectionFunction(-0.1,1));
        MarkerGenerator generator8 = new MarkerGenerator(marker8,
                new MarkerMovementFunction(1,1),
                new DirectionFunction(0.1,-1));
        MarkerGenerator generator9 = new MarkerGenerator(marker9,
                new MarkerMovementFunction(1,1),
                new DirectionFunction(-0.1,-2));
        MarkerGenerator generator10 = new MarkerGenerator(marker10,
                new MarkerMovementFunction(1,1),
                new DirectionFunction(0.1,1));

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

        FrameGenerator frameGenerator = new FrameGenerator(markerGenerators,1250,1250);

        String generationPath = "C:\\Users\\Nico Rinck\\Desktop\\Daten_Studienarbeit\\mock";
        TrialDataGenerator trialDataGenerator = new TrialDataGenerator(frameGenerator,generationPath);
        for (int i = 0; i < 20; i++) {
            trialDataGenerator.generateTrial("test" + i + ".json", 1);
        }


    }
}
