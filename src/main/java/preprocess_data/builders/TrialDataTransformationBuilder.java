package preprocess_data.builders;

import preprocess_data.TrialDataTransformation;
import preprocess_data.data_manipulaton.FrameManipulationStrategy;
import preprocess_data.labeling.FrameLabelingStrategy;

import java.util.ArrayList;

public class TrialDataTransformationBuilder implements Builder<TrialDataTransformation> {

    private FrameLabelingStrategy frameLabelingStrategy;
    private ArrayList<FrameManipulationStrategy> manipulationStrategy = new ArrayList<>();

    public TrialDataTransformationBuilder(FrameLabelingStrategy frameLabelingStrategy) {
        this.frameLabelingStrategy = frameLabelingStrategy;
    }

    public static TrialDataTransformationBuilder addLabelingStrategy(FrameLabelingStrategy frameLabelingStrategy) {
        return new TrialDataTransformationBuilder(frameLabelingStrategy);
    }

    public TrialDataTransformationBuilder withManipulation(FrameManipulationStrategy manipulationStrategy) {
        this.manipulationStrategy.add(manipulationStrategy);
        return this;
    }

    public TrialDataTransformation build() {
        return new TrialDataTransformation(frameLabelingStrategy, manipulationStrategy);
    }
}

