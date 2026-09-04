package io.shashwat.ann.distance;

/**
 * The distance entry point every index calls.
 *
 * <p>The scalar/SIMD choice is read once at class initialisation into a {@code static
 * final boolean}, so the JIT folds the branch away and the call site inlines straight
 * into whichever kernel was selected. Set {@code -Dann.simd=false} to run an index on the
 * scalar kernels — that is how the SIMD speedup is measured end-to-end rather than only
 * in the microbenchmark.
 */
public final class Distance {

    private static final boolean SIMD =
            Boolean.parseBoolean(System.getProperty("ann.simd", "true"));

    private Distance() {
    }

    public static boolean simdEnabled() {
        return SIMD;
    }

    public static String kernelName() {
        return SIMD ? "simd(" + SimdDistance.species() + ")" : "scalar";
    }

    public static float l2Squared(float[] a, int aOff, float[] b, int bOff, int dim) {
        return SIMD
                ? SimdDistance.l2SquaredUnrolled(a, aOff, b, bOff, dim)
                : ScalarDistance.l2Squared(a, aOff, b, bOff, dim);
    }

    public static float negativeInnerProduct(float[] a, int aOff, float[] b, int bOff, int dim) {
        return SIMD
                ? SimdDistance.negativeInnerProduct(a, aOff, b, bOff, dim)
                : ScalarDistance.negativeInnerProduct(a, aOff, b, bOff, dim);
    }

    /** Dispatch by metric. Used where the metric is a configuration value. */
    public static float compute(Metric metric, float[] a, int aOff, float[] b, int bOff, int dim) {
        return switch (metric) {
            case L2 -> l2Squared(a, aOff, b, bOff, dim);
            case INNER_PRODUCT -> negativeInnerProduct(a, aOff, b, bOff, dim);
        };
    }
}
