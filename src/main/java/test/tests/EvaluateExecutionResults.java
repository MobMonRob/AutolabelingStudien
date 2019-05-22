package test.tests;

import test.execution.LogFileEvaluator;

import java.io.File;
import java.util.*;

public class EvaluateExecutionResults {

    public static void main(String[] args) throws Exception {

        File logFiles = new File("C:\\Users\\Nico Rinck\\Documents\\DHBW\\Studienarbeit\\Daten_Studienarbeit\\logs");
        File[] files = logFiles.listFiles();
        ArrayList<LogFileEvaluator> logFileEvaluators = new ArrayList<>();
        assert files != null;
        for (File file : files) {
            if (file.exists() && !file.getName().contains("config")) {
                System.out.println(file.getName());
                logFileEvaluators.add(new LogFileEvaluator(file));
            }
        }

        /*for (LogFileEvaluator logFileEvaluator : logFileEvaluators) {
            HashMap<Integer, Double> bestResults = logFileEvaluator.getBestResults(0.1);
            System.out.println("____________________________");
            for (Integer integer : bestResults.keySet()) {
                System.out.println(integer + ": " + bestResults.get(integer));
            }
        }*/

        HashMap<Integer, Double> bestResults = logFileEvaluators.get(0).getBestResults(0.112);
        Set<Integer> integers = bestResults.keySet();
        ArrayList<Integer> list = new ArrayList<>(integers);
        Collections.sort(list);
        for (Integer integer : list) {
            System.out.println(integer+1 + ": " + bestResults.get(integer)*2);
        }
    }
}
