package io.shashwat.ann.distance;

/**
 * Plain-Java distance kernels over the flat vector layout.
 *
 * <p>These are the reference implementations: the SIMD kernels in {@link SimdDistance}
 * are tested against them, and the brute-force oracle uses them so that the thing every
 * index is validated against contains no clever code.
 */
public final class ScalarDistance {

    private ScalarDistance() {
    }

    /**
     * Squared Euclidean distance. The square root is never taken: it is monotonic, so it
     * changes no ordering, and skipping it removes a per-candidate sqrt from every scan.
     */
    public static float l2Squared(float[] a, int aOff, float[] b, int bOff, int dim) {
        float sum = 0f;
        for (int i = 0; i < dim; i++) {
            float d = a[aOff + i] - b[bOff + i];
            sum += d * d;
        }
        return sum;
    }

    public static float innerProduct(float[] a, int aOff, float[] b, int bOff, int dim) {
        float sum = 0f;
        for (int i = 0; i < dim; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    /**
     * Inner product negated so that lower is better, matching the convention every
     * candidate heap in this codebase assumes.
     */
    public static float negativeInnerProduct(float[] a, int aOff, float[] b, int bOff, int dim) {
        return -innerProduct(a, aOff, b, bOff, dim);
    }
}
