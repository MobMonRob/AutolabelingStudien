package test.execution.result_logging;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

class ModelToStringConverter {

    String modelToString(MultiLayerNetwork model) {
        StringBuilder result = new StringBuilder(formatString("Evaluationen", model.getEpochCount()));
        result.append(formatString("Lernrate", model.getLearningRate(0)));
        result.append(formatString("Optimierung", model.getLayerWiseConfigurations().getConf(0).getOptimizationAlgo().toString()));
        result.append(formatString("Updater", getLayerProperty(model.getLayer(0).toString(), "iUpdater")));
        for (Layer layer : model.getLayers()) {
            String layerString = layer.toString();
            result.append("\n   Layer: [");
            result.append(formatString("Typ", getLayerProperty(layerString, "(layer")));
            result.append(formatString("Inputs", getLayerProperty(layerString, "nIn")));
            result.append(formatString("Neuronen", getLayerProperty(layerString, "nOut")));
            if (layerString.contains("OutputLayer")) {
                result.append(formatString("Kostenfunktion", getLayerProperty(layerString, "lossFn")));
            }
            result.append(formatString("Aktivierung", getLayerProperty(layerString, "activationFn")));
            result.append(formatString("Gewichte", getLayerProperty(layerString, "weightInit"), true));
            result.append("], ");
        }
        return result.toString();
    }

    private String formatString(String propertyLabel, Object propertyValue, boolean lastElement) {
        String result = propertyLabel + ": " + propertyValue;
        if (!lastElement) {
            result += ", ";
        }
        return result;
    }

    private String formatString(String propertyLabel, Object propertyValue) {
        return formatString(propertyLabel, propertyValue, false);
    }

    private String getLayerProperty(String layerString, String property) {
        int indexOfProperty = layerString.indexOf(property) + property.length() + 1;
        int indexOfNextCloseBracket = layerString.indexOf(")", indexOfProperty);
        int indexOfNextOpenBracket = layerString.indexOf("(", indexOfProperty);
        int indexOfNextComma = layerString.indexOf(",", indexOfProperty);

        if (indexOfNextOpenBracket != -1 && indexOfNextOpenBracket < indexOfNextComma) {
            return layerString.substring(indexOfProperty, indexOfNextOpenBracket).trim();
        }
        if (indexOfNextCloseBracket != -1 && indexOfNextCloseBracket < indexOfNextComma) {
            return layerString.substring(indexOfProperty, indexOfNextCloseBracket).trim();
        }
        return layerString.substring(indexOfProperty, indexOfNextComma).trim();
    }
}
