package test.tests;

import test.execution.AutomaticConfigExecutor;
import test.execution.LogFileEvaluator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;

public class EvaluateExecutionResults {

    public static void main(String[] args) throws Exception {

        File file = new File("C:\\Users\\Nico Rinck\\Desktop\\Daten_Studienarbeit\\logs\\logfile(-1,1)-v4.txt");
        LogFileEvaluator logFileEvaluator = new LogFileEvaluator(file);

        HashMap<Integer, Double> bestResult = logFileEvaluator.getBestResult();
        Helper.logHashMap(bestResult);

        ArrayList<String> bestModelConfigs = logFileEvaluator.getBestModelConfigs(0.24);
        System.out.println(bestModelConfigs.size());

        ArrayList<MultiLayerConfiguration> configsFromJSON = Helper.getConfigsFromJSON(bestModelConfigs);
        int epochCount = configsFromJSON.get(0).getEpochCount();
        System.out.println(epochCount);
        AutomaticConfigExecutor automaticConfigExecutor = ExecutorConfigs.largeDataSetNegRange();
        automaticConfigExecutor.executeConfigs(configsFromJSON,15,1);
    }
}
