package test.execution;

import datavec.JsonTrialRecordReader;
import preprocess_data.builders.TrialDataManagerBuilder;
import test.execution.result_logging.ResultLogger;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import preprocess_data.TrialDataManager;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class AutomaticConfigExecutor {

    private static final String[] ALLOWED_FILE_FORMAT = {"json"};
    private final RecordReaderDataSetIterator trainIterator;
    private final RecordReaderDataSetIterator testIterator;
    private final ResultLogger resultLogger;
    private final ArrayList<TrainingListener> listeners = new ArrayList<>();
    private final DL4JNetworkTrainer networkExecutor;

    public AutomaticConfigExecutor(File train, File test, File logFile, TrialDataManagerBuilder dataManager, int batchSize,
                                   NormalizerMinMaxScaler normalizer) throws IOException, InterruptedException, InstantiationException, IllegalAccessException {
        this(train, test, logFile, dataManager, batchSize);
        addPreProcessors(normalizer, trainIterator);
        addPreProcessors(normalizer, testIterator);
        this.resultLogger.logNormalizer(normalizer);
    }

    public AutomaticConfigExecutor(File train, File test, File logFile, TrialDataManagerBuilder dataManager, int batchSize) throws IOException, InterruptedException {
        this.resultLogger = new ResultLogger(logFile);
        TrialDataManager trainDataManager = dataManager.build();
        this.trainIterator = initIterator(train, trainDataManager, batchSize);
        this.testIterator = initIterator(test, dataManager.build(), batchSize);
        this.resultLogger.logDataInfo(trainDataManager, batchSize);
        this.networkExecutor = new DL4JNetworkTrainer(trainIterator);
    }

    private void addPreProcessors(DataNormalization normalizer, RecordReaderDataSetIterator iterator) throws IllegalAccessException, InstantiationException {
        DataNormalization normalizerCopy = normalizer.getClass().newInstance();
        normalizer.fit(iterator);
        iterator.setPreProcessor(normalizerCopy);
    }

    private RecordReaderDataSetIterator initIterator(File file, TrialDataManager trialDataManager, int batchSize) throws IOException, InterruptedException {
        JsonTrialRecordReader recordReader = new JsonTrialRecordReader(trialDataManager);
        FileSplit fileSplit = new FileSplit(file, ALLOWED_FILE_FORMAT);
        this.resultLogger.logFiles(fileSplit, file);
        recordReader.initialize(fileSplit);
        return new RecordReaderDataSetIterator(recordReader, batchSize);
    }

    public ArrayList<TrainingListener> getListeners() {
        return this.listeners;
    }

    public void executeConfigs(ArrayList<MultiLayerConfiguration> configs, int epochsPerExecution, int repeats) {
        for (int i = 0; i < configs.size(); i++) {
            System.out.println("Konfiguration " + (i + 1) + "/" + configs.size());
            executeConfig(configs.get(i), repeats, epochsPerExecution);
        }
    }

    public void executeConfig(MultiLayerConfiguration config, int repeats, int epoch) {
        for (int i = 0; i < repeats; i++) {
            MultiLayerNetwork multiLayerNetwork = networkExecutor.trainNetwork(config, epoch);
            Evaluation evaluation = multiLayerNetwork.evaluate(testIterator);
            resultLogger.log(multiLayerNetwork, evaluation);
            testIterator.reset();
            trainIterator.reset();
        }
    }
}
