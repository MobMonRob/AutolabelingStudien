package test.execution.result_logging;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import preprocess_data.TrialDataManager;

import java.io.File;
import java.net.URI;

public class ResultLogger {

    private final FileManager logFileManager;
    private final FileManager configFileManager;
    private final ModelToStringConverter modelConverter = new ModelToStringConverter();
    private int modelCounter = 0;
    private String currentModelLine = "";

    public ResultLogger(File file) {
        this.logFileManager = new FileManager(file);
        this.configFileManager = getConfigFileManager(file);
    }

    static FileManager getConfigFileManager(File file) {
        return new FileManager(new File(file.getParent() + "\\"
                + file.getName().substring(0, file.getName().lastIndexOf(".")) + "_config.txt"));
    }

    public void log(MultiLayerNetwork model, Evaluation evaluation) {
        String modelString = modelConverter.modelToString(model);
        System.out.println(model.getLayer(model.getLayers().length - 1));
        System.out.println("Model: " + modelString);
        System.out.println(evaluation.stats(false, true));
        if (!modelString.equalsIgnoreCase(currentModelLine)) {
            modelCounter++;
            logFileManager.writeInFile("\n" + LinePrefixes.MODEL_LINE_PREFIX.getLinePrefix()
                    + modelCounter + "--> " + modelString);
            configFileManager.writeInFile(model.getLayerWiseConfigurations().toJson() + "\n"
                    + LinePrefixes.CONFIG_SEPARATOR + "\n");
            currentModelLine = modelString;
            System.out.println(currentModelLine);
            System.out.println(getEvaluationLogString(evaluation));
        }
        logFileManager.writeInFile(getEvaluationLogString(evaluation));
    }

    private String getEvaluationLogString(Evaluation evaluation) {
        return "\n" + LinePrefixes.RESULT_LINE_PREFIX.getLinePrefix() + evaluation.accuracy();
    }

    public void logDataInfo(TrialDataManager dataManager, int batchSize) {
        logFileManager.writeInFile("allgemeine Konfigurationen");
        logFileManager.writeInFile(dataManager.getInfoString());
        logFileManager.writeInFile("\nBatchsize: " + batchSize + "\n");
        logFileManager.writeInFile("----------------------------- \n");
    }

    public void logFiles(FileSplit fileSplit, File file) {
        final StringBuilder builder = new StringBuilder();
        for (URI uri : fileSplit.locations()) {
            builder.append(uri).append("\n");
        }
        String testOrTrain = file.getPath().contains("train") ? "train" : "test";
        logFileManager.writeInFile(testOrTrain + "-Files:\n" + builder.toString());
    }

    public void logNormalizer(NormalizerMinMaxScaler normalizer) {
        logFileManager.writeInFile("\nDL4JNormalization: NormalizerMinMaxScaler(minRange: "
                + normalizer.getTargetMin() + ", maxRange: " + normalizer.getTargetMax() + "\n");
    }
}
