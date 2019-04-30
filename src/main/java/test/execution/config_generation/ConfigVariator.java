package test.execution.config_generation;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.ArrayList;

public class ConfigVariator {

    private final ConfigGenerator configGenerator;
    private final LayerConfigVariator layerConfigVariator;

    public ConfigVariator(Integer seed, LayerConfigVariator layerConfigVariator) {
        this.configGenerator = new ConfigGenerator(seed);
        this.layerConfigVariator = layerConfigVariator;
    }

    public void addUpdater(IUpdater... updaters) {
        configGenerator.addValueToBuilder(updaters, NeuralNetConfiguration.Builder::updater);
    }

    void addOptimizationAlgorithm(OptimizationAlgorithm... optimizationAlgorithms) {
        configGenerator.addValueToBuilder(optimizationAlgorithms, NeuralNetConfiguration.Builder::optimizationAlgo);
    }

    public ArrayList<MultiLayerConfiguration> getConfigs() {
        return configGenerator.getConfigs(layerConfigVariator.getLayerVariants());
    }
}
