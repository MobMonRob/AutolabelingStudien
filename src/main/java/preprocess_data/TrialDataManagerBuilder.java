package preprocess_data;

import preprocess_data.data_normalization.TrialNormalizationStrategy;

import java.util.Set;

public class TrialDataManagerBuilder {

    private final TrialDataTransformation dataTransformation;
    private TrialNormalizationStrategy normalizationStrategy;
    private Set<String> acceptedMarkers;


    public TrialDataManagerBuilder(TrialDataTransformation dataTransformation) {
        this.dataTransformation = dataTransformation;
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
