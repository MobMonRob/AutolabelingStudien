package test.execution;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class DL4JNetworkTrainer {

    private final DataSetIterator trainData;
    private TrainingListener[] listeners;

    public DL4JNetworkTrainer(DataSetIterator trainData) {
        this.trainData = trainData;
    }

    public void addListeners(TrainingListener[] listeners) {
        this.listeners = listeners;
    }

    public MultiLayerNetwork trainNetwork(final MultiLayerConfiguration config, int epochs) {
        final MultiLayerNetwork model = new MultiLayerNetwork(config.clone());
        initModel(model);
        model.fit(trainData, epochs);
        return model;
    }

    public ComputationGraph trainComputationGraph(final ComputationGraphConfiguration configuration, int epochs) {
        final ComputationGraph computationGraph = new ComputationGraph(configuration.clone());
        initModel(computationGraph);
        computationGraph.fit(trainData, epochs);
        return computationGraph;
    }

    private void initModel(final Model model) {
        model.init();
        if (listeners != null) {
            model.addListeners(listeners);
        }
    }
}
