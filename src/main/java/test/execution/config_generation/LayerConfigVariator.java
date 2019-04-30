package test.execution.config_generation;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;

import java.util.ArrayList;

public class LayerConfigVariator {

    private final int[] amountOfLayers;
    private Layer[] inputLayers;
    private OutputLayer[] outputLayers;
    private Layer[] hiddenLayers;

    public LayerConfigVariator(int... amountOfLayers) {
        this.amountOfLayers = amountOfLayers;
    }

    public void addInputLayers(final Layer... layers) {
        inputLayers = layers;
    }

    public void addOutputLayers(final OutputLayer... outputLayers) {
        this.outputLayers = outputLayers;
    }

    public void addHiddenLayers(final Layer... layers) {
        this.hiddenLayers = layers;
    }

    public ArrayList<ArrayList<Layer>> getLayerVariants() {
        ArrayList<ArrayList<Layer>> finalResultList = new ArrayList<>();
        for (int amount: amountOfLayers) {
            ArrayList<ArrayList<Layer>> resultList = new ArrayList<>();
            for (Layer inputLayer : inputLayers) {
                ArrayList<Layer> layers = new ArrayList<>();
                layers.add(inputLayer);
                resultList.add(layers);
            }
            ArrayList<ArrayList<Layer>> newResultList = combineLayers(resultList,hiddenLayers,amount-2);
            finalResultList.addAll(combineLayers(newResultList,outputLayers,1));
        }
        return finalResultList;
    }

    private ArrayList<ArrayList<Layer>> combineLayers(ArrayList<ArrayList<Layer>> resultList, Layer[] layers, int iterations) {
        ArrayList<ArrayList<Layer>> newResultList = new ArrayList<>();
        for (Layer layer : layers) {
            for (ArrayList<Layer> currentLayers : resultList) {
                ArrayList<Layer> layersCopy = new ArrayList<>(currentLayers);
                for (int i = 0; i < iterations; i++) {
                    layersCopy.add(layer);
                }
                newResultList.add(layersCopy);
            }
        }
        return newResultList;
    }
}
