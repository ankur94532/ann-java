package io.shashwat.ann.index;

/**
 * Packs a (distance, id) pair into a single {@code long} so that heaps can be plain
 * {@code long[]} arrays instead of arrays of objects.
 *
 * <p>The float goes in the high 32 bits, transformed so that the natural signed ordering
 * of the packed longs is the ordering of the distances; the id goes in the low 32 bits,
 * which makes the ordering total and therefore deterministic across runs.
 */
public final class Packing {

    private Packing() {
    }

    /**
     * Maps a float onto an int whose signed order matches the float's order.
     *
     * <p>Non-negative floats already have that property, so they pass through untouched.
     * Negative floats have the wrong order among themselves (a bigger magnitude means a
     * bigger bit pattern) and the wrong position relative to positives, and flipping the
     * low 31 bits fixes both at once. NaN is not handled and must not reach here.
     */
    public static int sortableInt(float value) {
        int bits = Float.floatToIntBits(value);
        return bits ^ ((bits >> 31) & 0x7fffffff);
    }

    public static float fromSortableInt(int sortable) {
        int bits = sortable ^ ((sortable >> 31) & 0x7fffffff);
        return Float.intBitsToFloat(bits);
    }

    public static long pack(float distance, int id) {
        return ((long) sortableInt(distance) << 32) | (id & 0xffffffffL);
    }

    public static float distance(long packed) {
        return fromSortableInt((int) (packed >> 32));
    }

    public static int id(long packed) {
        return (int) packed;
    }
}
