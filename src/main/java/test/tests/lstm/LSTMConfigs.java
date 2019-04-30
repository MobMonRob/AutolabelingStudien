package test.tests.lstm;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LSTMConfigs {

    private final static int LSTM_INPUT_SIZE = 3; //marker x,y,z --> 3 inputs in each time-step

    public static MultiLayerConfiguration simpleLSTM() {
        return new NeuralNetConfiguration.Builder()
                .seed(140)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.0001))
                .list()
                .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(LSTM_INPUT_SIZE).nOut(LSTM_INPUT_SIZE)
                        .build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER).nIn(3).nOut(3).build())
                .build();
    }

    public static MultiLayerConfiguration simpleLSTMTruncated() {
        return new NeuralNetConfiguration.Builder()
                .seed(140)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.0015))
                .list()
                .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(LSTM_INPUT_SIZE).nOut(15)
                        .build())
                .layer(1, new LSTM.Builder().activation(Activation.TANH).nIn(15).nOut(15)
                        .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER).nIn(15).nOut(3).build())
                .build();
    }
}
