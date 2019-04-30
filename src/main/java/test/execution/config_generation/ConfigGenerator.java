package test.execution.config_generation;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.ArrayList;

class ConfigGenerator {

    private ArrayList<NeuralNetConfiguration.Builder> configBuilders = new ArrayList<>();

    ConfigGenerator(final Integer seed) {
        configBuilders.add(new NeuralNetConfiguration.Builder().seed(seed));
    }

    <T> void addValueToBuilder(T[] values, ConfigurationBuilderMethod<T> method) {
        ArrayList<NeuralNetConfiguration.Builder> newBuilders = new ArrayList<>();
        for (T value : values) {
            for (NeuralNetConfiguration.Builder configBuilder : configBuilders) {
                NeuralNetConfiguration.Builder builder = configBuilder.clone();
                newBuilders.add(method.addToBuilder(builder, value));
            }
        }
        configBuilders = newBuilders;
    }

    ArrayList<MultiLayerConfiguration> getConfigs(ArrayList<ArrayList<Layer>> layerConfigs) {
        final ArrayList<MultiLayerConfiguration> resultList = new ArrayList<>();
        for (NeuralNetConfiguration.Builder configBuilder : configBuilders) {
            for (ArrayList<Layer> layers : layerConfigs) {
                resultList.add(getConfigWithLayers(configBuilder, layers));
            }
        }
        return resultList;
    }

    private MultiLayerConfiguration getConfigWithLayers(NeuralNetConfiguration.Builder builder, ArrayList<Layer> layers) {
        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        for (Layer layer : layers) {
            Layer layerCopy = layer.clone();
            listBuilder.layer(layerCopy);
        }
        return listBuilder.build();
    }

    public ArrayList<NeuralNetConfiguration.Builder> getConfigBuilders() {
        return configBuilders;
    }
}