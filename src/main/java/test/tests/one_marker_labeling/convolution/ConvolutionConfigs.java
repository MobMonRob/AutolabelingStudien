package test.tests.one_marker_labeling.convolution;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.graph.ReshapeVertex;
import org.deeplearning4j.nn.conf.graph.SubsetVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Set;

public class ConvolutionConfigs {

    //important! batchSize cannot vary during processing (--> last element)
    private static int inputSize;   //equals height of convolution input --> spatial: [inputSize,1]
    private static int outputSize;  //CNN kernel lowers amount of data (3 marker-coordinates to one value)
    //cnnChannels --> amount of kernel-operations on input data in cnn1
    //kernelSize of CNN1 = [3,1] --> 3 for x,y,z of each marker, 1 because width of data = 1
    //stride = [3,1] --> step over 3 elements to filter each marker individually

    static ComputationGraphConfiguration twoReshapes(Set<String> selectedLabels, int batchSize,
                                                     int cnn1Channels, int cnn2Channels) {
        initInputSize(selectedLabels);
        int[] newShape = {batchSize, 1, 1, cnn2Channels * outputSize}; //final shape of outputs

        return new NeuralNetConfiguration.Builder()
                .seed(523)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.01))
                .graphBuilder()
                .addInputs("1")
                .layer("CNN1", new ConvolutionLayer.Builder().nIn(1).nOut(cnn1Channels)
                        .kernelSize(3, 1).stride(3, 1).build(), "1")
                .addVertex("Reshape1",
                        new ReshapeVertex(batchSize, 1, cnn1Channels, outputSize), "CNN1")
                .layer("CNN2", new ConvolutionLayer.Builder().nIn(1).nOut(cnn2Channels)
                        .kernelSize(cnn1Channels, 1).stride(1, 1).build(), "Reshape1")
                //vertex to convert the inputs in order of markers
                .addVertex("Reshape2",
                        new ReshapeVertex('f', newShape, null), "CNN2")
                .layer("DL2", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                        .nIn(outputSize * cnn2Channels).nOut(outputSize * cnn2Channels).build(), "Reshape2")
                .layer("DL3", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                        .nIn(outputSize * cnn2Channels).nOut(outputSize).build(), "DL2")
                .layer("DL4", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(outputSize).nOut(outputSize).build(), "DL3")
                .setOutputs("DL4")
                .setInputTypes(InputType.convolutionalFlat(inputSize, 1, 1))
                .build();
    }

    static ComputationGraphConfiguration treeReshapesOneDeepLayer(Set<String> selectedLabels, int batchSize,
                                                                  int cnn1Channels, int cnn2Channels, int cnn3Channels) {
        initInputSize(selectedLabels);
        int[] newShape = {batchSize, 1, 1, cnn3Channels * outputSize}; //final shape of outputs

        return new NeuralNetConfiguration.Builder()
                .seed(523)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.01))
                .graphBuilder()
                .addInputs("1")
                .layer("CNN1", new ConvolutionLayer.Builder().nIn(1).nOut(cnn1Channels)
                        .kernelSize(3, 1).stride(3, 1).build(), "1")
                .addVertex("Reshape1",
                        new ReshapeVertex(batchSize, 1, cnn1Channels, outputSize), "CNN1")
                .layer("CNN2", new ConvolutionLayer.Builder().nIn(1).nOut(cnn2Channels)
                        .kernelSize(cnn1Channels, 1).stride(1, 1).build(), "Reshape1")
                .addVertex("Reshape2",
                        new ReshapeVertex(batchSize, 1, cnn2Channels, outputSize), "CNN2")
                .layer("CNN3", new ConvolutionLayer.Builder().nIn(1).nOut(cnn3Channels)
                        .kernelSize(cnn2Channels, 1).stride(1, 1).build(), "Reshape2")
                //vertex to convert the inputs in order of markers
                .addVertex("Reshape3",
                        new ReshapeVertex('f', newShape, null), "CNN3")
                .layer("DL2", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                        .nIn(outputSize * cnn3Channels).nOut(outputSize * cnn3Channels).build(), "Reshape3")
                .layer("DL3", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(outputSize * cnn3Channels).nOut(outputSize).build(), "DL2")
                .setOutputs("DL3")
                .setInputTypes(InputType.convolutionalFlat(inputSize, 1, 1))
                .build();
    }

    static ComputationGraphConfiguration treeReshapesMultipleDeepLayers(Set<String> selectedLabels, int batchSize,
                                                                        int cnn1Channels, int cnn2Channels, int cnn3Channels) {
        initInputSize(selectedLabels);
        int[] newShape = {batchSize, 1, 1, cnn3Channels * outputSize}; //final shape of outputs

        return new NeuralNetConfiguration.Builder()
                .seed(523)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.01))
                .graphBuilder()
                .addInputs("1")
                .layer("CNN1", new ConvolutionLayer.Builder().nIn(1).nOut(cnn1Channels)
                        .kernelSize(3, 1).stride(3, 1).build(), "1")
                .addVertex("Reshape1",
                        new ReshapeVertex(batchSize, 1, cnn1Channels, outputSize), "CNN1")
                .layer("CNN2", new ConvolutionLayer.Builder().nIn(1).nOut(cnn2Channels)
                        .kernelSize(cnn1Channels, 1).stride(1, 1).build(), "Reshape1")
                .addVertex("Reshape2",
                        new ReshapeVertex(batchSize, 1, cnn2Channels, outputSize), "CNN2")
                .layer("CNN3", new ConvolutionLayer.Builder().nIn(1).nOut(cnn3Channels)
                        .kernelSize(cnn2Channels, 1).stride(1, 1).build(), "Reshape2")
                //vertex to convert the inputs in order of markers
                .addVertex("Reshape3",
                        new ReshapeVertex('f', newShape, null), "CNN3")
                .layer("DL2", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                        .nIn(outputSize * cnn3Channels).nOut(outputSize * cnn3Channels).build(), "Reshape3")
                .layer("DL3", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                        .nIn(outputSize * cnn3Channels).nOut(outputSize * cnn3Channels).build(), "DL2")
                .layer("DL4", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(outputSize * cnn3Channels).nOut(outputSize).build(), "DL3")
                .setOutputs("DL4")
                .setInputTypes(InputType.convolutionalFlat(inputSize, 1, 1))
                .build();
    }

    static ComputationGraphConfiguration treeReshapesMultipleSplittedDeepLayers(Set<String> selectedLabels, int batchSize, int cnn1Channels, int cnn2Channels, int cnn3Channels) {
        initInputSize(selectedLabels);
        int[] newShape = {batchSize, cnn3Channels * outputSize, 1, 1}; //final shape of outputs --> channels for SubsetVertex

        ComputationGraphConfiguration.GraphBuilder build = new NeuralNetConfiguration.Builder()
                .seed(523)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.01))
                .graphBuilder()
                .addInputs("1")
                .layer("CNN1", new ConvolutionLayer.Builder().nIn(1).nOut(cnn1Channels)
                        .kernelSize(3, 1).stride(3, 1).build(), "1")
                .addVertex("Reshape1",
                        new ReshapeVertex(batchSize, 1, cnn1Channels, outputSize), "CNN1")
                .layer("CNN2", new ConvolutionLayer.Builder().nIn(1).nOut(cnn2Channels)
                        .kernelSize(cnn1Channels, 1).stride(1, 1).build(), "Reshape1")
                .addVertex("Reshape2",
                        new ReshapeVertex(batchSize, 1, cnn2Channels, outputSize), "CNN2")
                .layer("CNN3", new ConvolutionLayer.Builder().nIn(1).nOut(cnn3Channels)
                        .kernelSize(cnn2Channels, 1).stride(1, 1).build(), "Reshape2")
                //vertex to convert the inputs in order of markers
                .addVertex("Reshape3",
                        new ReshapeVertex('f', newShape, null), "CNN3")
                .addVertex("Preprocessor",
                        new PreprocessorVertex(new CnnToFeedForwardPreProcessor(1, 1, cnn3Channels * outputSize)),
                        "Reshape3");
        for (int i = 0; i < outputSize; i++) { //add 35 vertexes to split input data for any marker
            build.addVertex("Split" + i, new SubsetVertex(i * (cnn3Channels), ((i + 1) * cnn3Channels) - 1), "Preprocessor");
        }
        int deepLayerOut = 1;
        for (int i = 0; i < outputSize; i++) { //add layers for any marker individually
            build
                    /*.inputPreProcessor("DL" + i + ",1", new CnnToFeedForwardPreProcessor(1,1,10))*/
                    .layer("DL" + i + ",1", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                            .nIn(cnn3Channels).nOut(cnn3Channels).build(), "Split" + i)
                    .layer("DL" + i + ",2", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                            .nIn(cnn3Channels).nOut(deepLayerOut).build(), "DL" + i + ",1");
        }
        String[] allInputs = new String[deepLayerOut * outputSize];
        for (int i = 0; i < allInputs.length; i++) {
            allInputs[i] = "DL" + i + ",2";
        }
        build.addVertex("Merge", new MergeVertex(), allInputs)
                .layer("Out", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(outputSize).nOut(outputSize)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).build(), "Merge")
                .setOutputs("Out")
                .setInputTypes(InputType.convolutionalFlat(inputSize, 1, 1))
                .build();
        return build.build();
    }

    static ComputationGraphConfiguration singleReshape(Set<String> selectedLabels, int batchSize, int cnnChannels) {
        initInputSize(selectedLabels);
        int[] newShape = {batchSize, 1, 1, cnnChannels * outputSize}; //final shape of outputs
        return new NeuralNetConfiguration.Builder()
                .seed(523)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.01))
                .graphBuilder()
                .addInputs("1")
                .layer("CNN1", new ConvolutionLayer.Builder().nIn(1).nOut(cnnChannels)
                        .kernelSize(3, 1).stride(3, 1).build(), "1")
                //vertex to convert the inputs in order of markers
                .addVertex("Reshape",
                        new ReshapeVertex('f', newShape, null), "CNN1")
                .layer("DL2", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                        .nIn(outputSize * cnnChannels).nOut(outputSize * cnnChannels).build(), "Reshape")
                .layer("DL3", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                        .nIn(outputSize * cnnChannels).nOut(outputSize).build(), "DL2")
                .layer("DL4", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER)
                        .nIn(outputSize).nOut(outputSize).build(), "DL3")
                .setOutputs("DL4")
                .setInputTypes(InputType.convolutionalFlat(inputSize, 1, 1))
                .build();
    }



    private static void initInputSize(Set<String> selectedLabels) {
        inputSize = selectedLabels.size() * 3;
        outputSize = selectedLabels.size();
    }
}


/*static ComputationGraphConfiguration twoReshapesSplittedDeepLayers(Set<String> selectedLabels, int batchSize,
                                                                       int cnn1Channels, int cnn2Channels) {
        initInputSize(selectedLabels);
        int[] newShape = {batchSize, 1, 1, cnn2Channels * outputSize}; //final shape of outputs

        ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .seed(523)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.01))
                .graphBuilder()
                .addInputs("1")
                .layer("CNN1", new ConvolutionLayer.Builder().nIn(1).nOut(cnn1Channels)
                        .kernelSize(3, 1).stride(3, 1).build(), "1")
                .addVertex("Reshape1",
                        new ReshapeVertex(batchSize, 1, cnn1Channels, outputSize), "CNN1")
                .layer("CNN2", new ConvolutionLayer.Builder().nIn(1).nOut(cnn2Channels)
                        .kernelSize(cnn1Channels, 1).stride(1, 1).build(), "Reshape1")
                //vertex to convert the inputs in order of markers
                .addVertex("Reshape2",
                        new ReshapeVertex('f', newShape, null), "CNN2")
                .addVertex("Preprocessor",
                        new PreprocessorVertex(new CnnToFeedForwardPreProcessor(1, cnn2Channels * outputSize, 1)),
                        "Reshape2");
        int deepLayerOut = 1;
        for (int i = 0; i < outputSize; i++) { //add 35 vertexes to split input data for any marker
            graphBuilder.addVertex("Split" + i, new SubsetVertex(i * (cnn2Channels), ((i + 1) * cnn2Channels) - 1), "Preprocessor")
                    .layer("DL" + i + ",1", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                            .nIn(cnn2Channels).nOut(cnn2Channels).build(), "Split" + i)
                    .layer("DL" + i + ",2", new DenseLayer.Builder().activation(Activation.TANH).weightInit(WeightInit.XAVIER)
                            .nIn(cnn2Channels).nOut(deepLayerOut).build(), "DL" + i + ",1");
        }
        String[] allInputs = new String[deepLayerOut * outputSize];
        for (int i = 0; i < allInputs.length; i++) {
            allInputs[i] = "DL" + i + ",2";
        }
        graphBuilder.addVertex("Merge", new MergeVertex(), allInputs)
                .layer("Out", new LossLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).weightInit(WeightInit.XAVIER).build(), "Merge")
                .setOutputs("Out")
                .setInputTypes(InputType.convolutionalFlat(inputSize, 1, 1))
                .build();
        return graphBuilder.build();
    }*/
