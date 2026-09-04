package io.shashwat.ann.distance;

/** Which notion of "near" an index is built and searched with. Lower is always better. */
public enum Metric {

    /** Squared Euclidean distance. */
    L2,

    /** Negated inner product, so that the ordering convention stays "lower is better". */
    INNER_PRODUCT;

    /** Reference (scalar) computation, used by the oracle and by kernel tests. */
    public float scalar(float[] a, int aOff, float[] b, int bOff, int dim) {
        return switch (this) {
            case L2 -> ScalarDistance.l2Squared(a, aOff, b, bOff, dim);
            case INNER_PRODUCT -> ScalarDistance.negativeInnerProduct(a, aOff, b, bOff, dim);
        };
    }
}
