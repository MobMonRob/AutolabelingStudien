package preprocess_data.data_model;

public class Marker {

    private final String label;
    private final double x;
    private final double y;
    private final double z;

    public Marker(String label, double x, double y, double z) {
        this.label = label;
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double getZ() {
        return z;
    }

    public String getLabel() {
        return label;
    }

    @Override
    public String toString() {
        return "label: " + label + ", x: " + x + ", y: " + y + ", z: " + z;
    }
}
