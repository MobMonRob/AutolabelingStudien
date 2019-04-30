package test.execution;

import test.execution.result_logging.ResultReader;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class LogFileEvaluator {

    private final ResultReader resultReader;

    public LogFileEvaluator(File logFile) {
        this.resultReader = new ResultReader(logFile);
    }

    public ArrayList<String> getBestModelConfigs(double minimalAccuracy) {
        ArrayList<String> resultList = new ArrayList<>();
        HashMap<Integer, ArrayList<Double>> results = resultReader.getModelIndexAndResults();
        ArrayList<String> modelJSONList = resultReader.getModelJSON();
        for (Integer integer : results.keySet()) {
            if (minimalAccuracy < Collections.max(results.get(integer))) {
                resultList.add(modelJSONList.get(integer));
            }
        }
        return resultList;
    }

    public HashMap<Integer, Double> getBestResult() {
        final HashMap<Integer, ArrayList<Double>> results = resultReader.getModelIndexAndResults();
        double currentBest = 0;
        int currentBestIndex = -1;
        for (Integer integer : results.keySet()) {
            Double max = Collections.max(results.get(integer));
            if (max > currentBest) {
                currentBest = max;
                currentBestIndex = integer;
            }
        }
        final HashMap<Integer,Double> result = new HashMap<>();
        result.put(currentBestIndex,currentBest);
        return result;
    }

    public HashMap<Integer, Double> getBestResults(double minimalAccuracy) {
        final HashMap<Integer,Double> resultList = new HashMap<>();
        final HashMap<Integer, ArrayList<Double>> results = resultReader.getModelIndexAndResults();
        for (Integer integer : results.keySet()) {
            Double max = Collections.max(results.get(integer));
            if (max > minimalAccuracy) {
                resultList.put(integer,max);
            }
        }
        return resultList;
    }
}
