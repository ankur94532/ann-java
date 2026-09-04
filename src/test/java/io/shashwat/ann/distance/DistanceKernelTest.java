package io.shashwat.ann.distance;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The SIMD kernels must agree with the scalar reference. They will not agree bit-for-bit
 * — FMA and multiple accumulators reassociate the sum — so the check is a relative
 * tolerance, and the dimensions deliberately include values that are not multiples of the
 * vector lane count or of the unroll factor, which is where tail-handling bugs live.
 */
class DistanceKernelTest {

    private static final float REL_TOLERANCE = 1e-5f;

    /**
     * Lane count is 4 on the benchmark machine and the unroll factor is 4, so the
     * interesting dims are the ones around multiples of 16.
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 4, 5, 7, 8, 13, 15, 16, 17, 31, 32, 33, 63, 64, 65,
            127, 128, 129, 255, 384, 960, 961})
    void simdL2MatchesScalar(int dim) {
        Random rnd = new Random(dim * 31L + 7);
        for (int trial = 0; trial < 20; trial++) {
            float[] a = randomVector(rnd, dim, 5);
            float[] b = randomVector(rnd, dim, 5);
            float expected = ScalarDistance.l2Squared(a, 0, b, 0, dim);
            assertClose(expected, SimdDistance.l2Squared(a, 0, b, 0, dim), "l2 dim=" + dim);
            assertClose(expected, SimdDistance.l2SquaredUnrolled(a, 0, b, 0, dim),
                    "l2-unrolled dim=" + dim);
        }
    }

    /**
     * Inner products of sign-mixed vectors cancel: the terms are O(1) and the sum can land
     * near zero, so an error that is tiny next to the work done is enormous next to the
     * result. The tolerance is therefore scaled by the sum of the term magnitudes, which is
     * the quantity the rounding error is actually proportional to. Squared L2 has no such
     * problem — every term is non-negative, so nothing cancels — which is why the L2 test
     * above can use a plain relative tolerance.
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 3, 4, 13, 16, 17, 63, 128, 129, 960})
    void simdInnerProductMatchesScalar(int dim) {
        Random rnd = new Random(dim * 17L + 3);
        for (int trial = 0; trial < 20; trial++) {
            float[] a = randomVector(rnd, dim, 5);
            float[] b = randomVector(rnd, dim, 5);
            float scale = termMagnitude(a, b, dim);
            assertCloseScaled(ScalarDistance.innerProduct(a, 0, b, 0, dim),
                    SimdDistance.innerProduct(a, 0, b, 0, dim), scale, "ip dim=" + dim);
            assertCloseScaled(ScalarDistance.negativeInnerProduct(a, 0, b, 0, dim),
                    SimdDistance.negativeInnerProduct(a, 0, b, 0, dim), scale, "-ip dim=" + dim);
        }
    }

    /** Sum of |a_i * b_i|: the scale that inner-product rounding error is relative to. */
    private static float termMagnitude(float[] a, float[] b, int dim) {
        float sum = 0;
        for (int i = 0; i < dim; i++) {
            sum += Math.abs(a[i] * b[i]);
        }
        return sum;
    }

    /** Non-zero offsets are the normal case: vectors live inside one big flat array. */
    @ParameterizedTest
    @ValueSource(ints = {13, 16, 128, 960})
    void honoursOffsetsIntoAFlatArray(int dim) {
        Random rnd = new Random(dim);
        int n = 5;
        float[] flat = randomVector(rnd, n * dim, 10);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float expected = ScalarDistance.l2Squared(flat, i * dim, flat, j * dim, dim);
                assertClose(expected,
                        SimdDistance.l2SquaredUnrolled(flat, i * dim, flat, j * dim, dim),
                        "flat " + i + "," + j);
            }
        }
    }

    @Test
    void distanceToSelfIsZero() {
        Random rnd = new Random(99);
        float[] a = randomVector(rnd, 128, 100);
        assertEquals(0f, SimdDistance.l2SquaredUnrolled(a, 0, a, 0, 128));
        assertEquals(0f, ScalarDistance.l2Squared(a, 0, a, 0, 128));
    }

    @Test
    void facadeAgreesWithWhicheverKernelIsSelected() {
        Random rnd = new Random(5);
        float[] a = randomVector(rnd, 128, 10);
        float[] b = randomVector(rnd, 128, 10);
        float viaFacade = Distance.l2Squared(a, 0, b, 0, 128);
        assertClose(ScalarDistance.l2Squared(a, 0, b, 0, 128), viaFacade, "facade");
        assertEquals(Metric.L2.scalar(a, 0, b, 0, 128),
                ScalarDistance.l2Squared(a, 0, b, 0, 128));
    }

    /** SIFT vectors are small non-negative integers; make sure that regime is covered too. */
    @Test
    void matchesOnSiftLikeInputs() {
        Random rnd = new Random(1234);
        int dim = 128;
        float[] a = new float[dim];
        float[] b = new float[dim];
        for (int trial = 0; trial < 200; trial++) {
            for (int i = 0; i < dim; i++) {
                a[i] = rnd.nextInt(256);
                b[i] = rnd.nextInt(256);
            }
            assertClose(ScalarDistance.l2Squared(a, 0, b, 0, dim),
                    SimdDistance.l2SquaredUnrolled(a, 0, b, 0, dim), "sift-like");
        }
    }

    private static float[] randomVector(Random rnd, int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rnd.nextFloat() * 2 - 1) * scale;
        }
        return v;
    }

    private static void assertClose(float expected, float actual, String what) {
        assertCloseScaled(expected, actual, Math.abs(expected), what);
    }

    private static void assertCloseScaled(float expected, float actual, float scale, String what) {
        float tolerance = REL_TOLERANCE * Math.max(1f, scale);
        assertTrue(Math.abs(expected - actual) <= tolerance,
                () -> what + ": expected " + expected + " but got " + actual
                        + " (tolerance " + tolerance + ")");
    }
}
