package preprocess_data.builders;

import preprocess_data.TrialDataManager;
import preprocess_data.TrialDataTransformation;
import preprocess_data.data_normalization.TrialNormalizationStrategy;

import java.util.Set;

public class TrialDataManagerBuilder implements Builder<TrialDataManager> {

    private TrialDataTransformation dataTransformation;
    private TrialNormalizationStrategy normalizationStrategy;
    private Set<String> acceptedMarkers;

    public TrialDataManagerBuilder() {}

    public TrialDataManagerBuilder(TrialDataTransformation dataTransformation) {
        this.dataTransformation = dataTransformation;
    }

    public static TrialDataManagerBuilder addTransformation(TrialDataTransformation transformation) {
        return new TrialDataManagerBuilder(transformation);
    }

    public TrialDataManagerBuilder withNormalization(TrialNormalizationStrategy normalizationStrategy) {
        this.normalizationStrategy = normalizationStrategy;
        return this;
    }

    public TrialDataManagerBuilder filterMarkers(Set<String> acceptedMarkers) {
        this.acceptedMarkers = acceptedMarkers;
        return this;
    }

    public TrialDataManager build() {
        return new TrialDataManager(dataTransformation, normalizationStrategy, acceptedMarkers);
    }
}
