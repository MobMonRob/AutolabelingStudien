package test.execution.result_logging;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;

public class ResultReader {

    private final FileManager logFileManager;
    private final FileManager configFileManager;

    public ResultReader(File logFile) {
        this.logFileManager = new FileManager(logFile);
        this.configFileManager = ResultLogger.getConfigFileManager(logFile);
    }

    public HashMap<Integer, ArrayList<Double>> getModelIndexAndResults() {
        HashMap<Integer, ArrayList<Double>> resultMap = new HashMap<>();
        ArrayList<String> lines = logFileManager.getFileContent();
        int modelIndex = 0;
        for (int i = 0; i < lines.size(); i++) {
            if (hasCertainPrefix(lines.get(i), LinePrefixes.MODEL_LINE_PREFIX.getLinePrefix())) {
                final ArrayList<String> modelResultsLines = getNextLinesFromPrefix(i, lines,
                        LinePrefixes.RESULT_LINE_PREFIX.getLinePrefix());
                final ArrayList<Double> modelResults = new ArrayList<>();
                modelResultsLines.forEach(d -> modelResults.add(Double.valueOf(d)));
                resultMap.put(modelIndex++, modelResults);
                i += modelResults.size();
            }
        }
        return resultMap;
    }

    private ArrayList<String> getNextLinesFromPrefix(int currentIndex, ArrayList<String> lines, String prefix) {
        final ArrayList<String> resultList = new ArrayList<>();
        boolean notNextModelLine = true;
        while (notNextModelLine && currentIndex < lines.size() - 1) {
            String line = lines.get(++currentIndex);
            if (line.contains(prefix)) {
                resultList.add(line.substring(prefix.length()));
            }
            if (hasCertainPrefix(line, LinePrefixes.MODEL_LINE_PREFIX.getLinePrefix())) {
                notNextModelLine = false;
            }
        }
        return resultList;
    }

    public String getModelString(int index) {
        ArrayList<String> fileContent = logFileManager.getFileContent();
        for (int i = 0; i < fileContent.size(); i++) {
            if (fileContent.get(i).contains(LinePrefixes.MODEL_LINE_PREFIX.getLinePrefix() + index)) {
                StringBuilder result = new StringBuilder(fileContent.get(i));
                ArrayList<String> nextLinesFromPrefix = getNextLinesFromPrefix(i, fileContent,
                        LinePrefixes.LAYER_LINE_PREFIX.getLinePrefix());
                for (String linesFromPrefix : nextLinesFromPrefix) {
                    result.append(linesFromPrefix);
                }
                return result.toString();
            }
        }
        return "";
    }

    public ArrayList<String> getModelJSON() {
        final ArrayList<String> fileContent = configFileManager.getFileContent();
        final ArrayList<String> resultList = new ArrayList<>();
        StringBuilder stringBuilder = new StringBuilder();
        for (String s: fileContent) {
            if (!s.contains(LinePrefixes.CONFIG_SEPARATOR.getLinePrefix())) {
                stringBuilder.append(s);
            } else {
                resultList.add(stringBuilder.toString());
                stringBuilder = new StringBuilder();
            }
        }
        return resultList;
    }

    private boolean hasCertainPrefix(String line, String prefix) {
        return line.length() >= prefix.length() && lineStartsWithPrefix(line, prefix);
    }

    private boolean lineStartsWithPrefix(String line, String prefix) {
        return line.substring(0, prefix.length()).equalsIgnoreCase(prefix);
    }
}
