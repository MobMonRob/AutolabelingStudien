package test.tests;

import test.execution.AutomaticConfigExecutor;
import test.execution.config_generation.ConfigVariator;
import test.execution.config_generation.LayerConfigVariator;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;

public class VariateAndTestConfigs {

    public static void main(String[] args) throws Exception {

        LayerConfigVariator layerConfigVariator = new LayerConfigVariator(5, 7);
        layerConfigVariator.addInputLayers(LayerConfigs.getInputLayersFromIndexes(2));
        layerConfigVariator.addHiddenLayers(LayerConfigs.getHiddenLayersFromIndexes(0, 1, 2, 7));
        layerConfigVariator.addOutputLayers(LayerConfigs.getOutputLayersFromIndexes(4, 5));

        ConfigVariator configVariator = new ConfigVariator(24, layerConfigVariator);
        IUpdater[] updaters = {new Sgd(0.01), new Sgd(0.001)};
        configVariator.addUpdater(updaters);

        AutomaticConfigExecutor automaticConfigExecutor = ExecutorConfigs.smallDataSetNegRange(20);
        automaticConfigExecutor.executeConfigs(configVariator.getConfigs(), 20, 2);
    }
}
