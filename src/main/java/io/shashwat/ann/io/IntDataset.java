package io.shashwat.ann.io;

/**
 * A set of {@code size} integer rows of width {@code dim} in one flat array, used for
 * {@code .ivecs} ground-truth files where row {@code i} lists the ids of the true
 * nearest neighbours of query {@code i}, nearest first.
 */
public record IntDataset(int[] data, int size, int dim) {

    public IntDataset {
        if (dim <= 0) {
            throw new IllegalArgumentException("dim must be positive, got " + dim);
        }
        long expected = (long) size * dim;
        if (data.length != expected) {
            throw new IllegalArgumentException(
                    "data length " + data.length + " != size*dim " + expected);
        }
    }

    public int offset(int i) {
        return i * dim;
    }

    /** The first {@code k} entries of row {@code i}. */
    public int[] row(int i, int k) {
        if (k > dim) {
            throw new IllegalArgumentException("k=" + k + " exceeds row width " + dim);
        }
        int[] out = new int[k];
        System.arraycopy(data, offset(i), out, 0, k);
        return out;
    }
}
