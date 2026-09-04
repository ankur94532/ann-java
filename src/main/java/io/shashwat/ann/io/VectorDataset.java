package io.shashwat.ann.io;

/**
 * A set of {@code size} vectors of {@code dim} dimensions held in one flat array.
 *
 * <p>Vector {@code i} occupies {@code data[i*dim .. (i+1)*dim)}. The flat layout is
 * deliberate: it keeps a scan over consecutive vectors on consecutive cache lines and
 * lets the distance kernels read straight out of the array with no pointer chase, which
 * a {@code float[][]} cannot promise.
 */
public record VectorDataset(float[] data, int size, int dim) {

    public VectorDataset {
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be positive, got " + dim);
        }
        if (size < 0) {
            throw new IllegalArgumentException("size must be non-negative, got " + size);
        }
        long expected = (long) size * dim;
        if (data.length != expected) {
            throw new IllegalArgumentException(
                    "data length " + data.length + " != size*dim " + expected);
        }
    }

    /** Index into {@link #data} at which vector {@code i} starts. */
    public int offset(int i) {
        return i * dim;
    }

    /** Copy of vector {@code i}. Allocates; not for use on a hot path. */
    public float[] vector(int i) {
        float[] out = new float[dim];
        System.arraycopy(data, offset(i), out, 0, dim);
        return out;
    }

    /** Bytes occupied by the raw vectors, ignoring object headers. */
    public long rawBytes() {
        return (long) size * dim * Float.BYTES;
    }
}
