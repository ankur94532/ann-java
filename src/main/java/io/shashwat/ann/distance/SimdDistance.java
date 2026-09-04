package io.shashwat.ann.distance;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Distance kernels written against the incubating Vector API.
 *
 * <p>Every method takes the same (array, offset, dim) shape as {@link ScalarDistance} and
 * is expected to agree with it to within float rounding — not bit-for-bit. The difference
 * is real and deliberate: these kernels use fused multiply-add and sum into several
 * partial accumulators, so the additions happen in a different order and with fewer
 * intermediate roundings than the scalar loop. The error is a few ULP on inputs of this
 * magnitude and cannot reorder two genuinely different neighbours.
 *
 * <p>On the benchmark machine {@code SPECIES_PREFERRED} is 128-bit (ARM NEON, 4 float
 * lanes), so 4x is the hard ceiling for these kernels. On an AVX-512 host the same code
 * runs 16 lanes wide with no change.
 */
public final class SimdDistance {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int LANES = SPECIES.length();

    /** How many vectors the multi-accumulator kernels process per iteration. */
    private static final int UNROLL = 4;

    private SimdDistance() {
    }

    public static VectorSpecies<Float> species() {
        return SPECIES;
    }

    /**
     * Squared Euclidean distance, one accumulator.
     *
     * <p>Each iteration's FMA depends on the previous iteration's accumulator, so this
     * loop runs at one vector per FMA-latency (3-4 cycles) however many FMA units the
     * core has. {@link #l2SquaredUnrolled} is the version that fixes that.
     */
    public static float l2Squared(float[] a, int aOff, float[] b, int bOff, int dim) {
        FloatVector acc = FloatVector.zero(SPECIES);
        int i = 0;
        int bound = SPECIES.loopBound(dim);
        for (; i < bound; i += LANES) {
            FloatVector diff = FloatVector.fromArray(SPECIES, a, aOff + i)
                    .sub(FloatVector.fromArray(SPECIES, b, bOff + i));
            acc = diff.fma(diff, acc);
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < dim; i++) {
            float d = a[aOff + i] - b[bOff + i];
            sum += d * d;
        }
        return sum;
    }

    /**
     * Squared Euclidean distance with four independent accumulators.
     *
     * <p>Four dependency chains instead of one, so the FMAs from different chains issue
     * back to back and the loop becomes throughput-bound rather than latency-bound. This
     * is the kernel the indexes use.
     */
    public static float l2SquaredUnrolled(float[] a, int aOff, float[] b, int bOff, int dim) {
        FloatVector acc0 = FloatVector.zero(SPECIES);
        FloatVector acc1 = FloatVector.zero(SPECIES);
        FloatVector acc2 = FloatVector.zero(SPECIES);
        FloatVector acc3 = FloatVector.zero(SPECIES);

        int step = LANES * UNROLL;
        int wideBound = dim - (dim % step);
        int i = 0;
        for (; i < wideBound; i += step) {
            FloatVector d0 = FloatVector.fromArray(SPECIES, a, aOff + i)
                    .sub(FloatVector.fromArray(SPECIES, b, bOff + i));
            FloatVector d1 = FloatVector.fromArray(SPECIES, a, aOff + i + LANES)
                    .sub(FloatVector.fromArray(SPECIES, b, bOff + i + LANES));
            FloatVector d2 = FloatVector.fromArray(SPECIES, a, aOff + i + 2 * LANES)
                    .sub(FloatVector.fromArray(SPECIES, b, bOff + i + 2 * LANES));
            FloatVector d3 = FloatVector.fromArray(SPECIES, a, aOff + i + 3 * LANES)
                    .sub(FloatVector.fromArray(SPECIES, b, bOff + i + 3 * LANES));
            acc0 = d0.fma(d0, acc0);
            acc1 = d1.fma(d1, acc1);
            acc2 = d2.fma(d2, acc2);
            acc3 = d3.fma(d3, acc3);
        }

        int bound = SPECIES.loopBound(dim);
        for (; i < bound; i += LANES) {
            FloatVector d = FloatVector.fromArray(SPECIES, a, aOff + i)
                    .sub(FloatVector.fromArray(SPECIES, b, bOff + i));
            acc0 = d.fma(d, acc0);
        }

        float sum = acc0.add(acc1).add(acc2.add(acc3)).reduceLanes(VectorOperators.ADD);
        for (; i < dim; i++) {
            float d = a[aOff + i] - b[bOff + i];
            sum += d * d;
        }
        return sum;
    }

    public static float innerProduct(float[] a, int aOff, float[] b, int bOff, int dim) {
        FloatVector acc0 = FloatVector.zero(SPECIES);
        FloatVector acc1 = FloatVector.zero(SPECIES);
        FloatVector acc2 = FloatVector.zero(SPECIES);
        FloatVector acc3 = FloatVector.zero(SPECIES);

        int step = LANES * UNROLL;
        int wideBound = dim - (dim % step);
        int i = 0;
        for (; i < wideBound; i += step) {
            acc0 = FloatVector.fromArray(SPECIES, a, aOff + i)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i), acc0);
            acc1 = FloatVector.fromArray(SPECIES, a, aOff + i + LANES)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i + LANES), acc1);
            acc2 = FloatVector.fromArray(SPECIES, a, aOff + i + 2 * LANES)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i + 2 * LANES), acc2);
            acc3 = FloatVector.fromArray(SPECIES, a, aOff + i + 3 * LANES)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i + 3 * LANES), acc3);
        }

        int bound = SPECIES.loopBound(dim);
        for (; i < bound; i += LANES) {
            acc0 = FloatVector.fromArray(SPECIES, a, aOff + i)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i), acc0);
        }

        float sum = acc0.add(acc1).add(acc2.add(acc3)).reduceLanes(VectorOperators.ADD);
        for (; i < dim; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    public static float negativeInnerProduct(float[] a, int aOff, float[] b, int bOff, int dim) {
        return -innerProduct(a, aOff, b, bOff, dim);
    }
}
