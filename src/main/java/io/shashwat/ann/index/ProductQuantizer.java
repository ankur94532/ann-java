package io.shashwat.ann.index;

import io.shashwat.ann.distance.Distance;

/**
 * Product quantization: a {@code dim}-dimensional vector is cut into {@code m} contiguous
 * subvectors, each subvector is replaced by the index of its nearest entry in a 256-entry
 * codebook trained for that subspace, and the vector becomes {@code m} bytes.
 *
 * <p>The compression is enormous — a 128-dim float vector is 512 bytes and its 16-byte
 * code is 32x smaller — and it comes from the codebooks being a *product*. Sixteen
 * independent 256-entry codebooks describe 256^16 ≈ 10^38 distinct points using only
 * 16 x 256 x 8 stored floats. A single codebook covering the same space would need 10^38
 * entries.
 *
 * <p>What is lost is exactness. A code names a cell, not a point, so a distance computed
 * from codes is the distance to the cell's representative. That error does not average out
 * — it puts a hard ceiling on recall that no amount of {@code nprobe} can lift, and
 * measuring where that ceiling sits for each {@code m} is the point of the Phase 5 sweep.
 */
public final class ProductQuantizer {

    /** 8 bits per subquantizer, so a code is exactly {@code m} bytes. */
    public static final int CENTROIDS_PER_SUBSPACE = 256;

    private final int dim;
    private final int m;
    private final int subDim;

    /** {@code m * 256 * subDim} floats: codebook j entry c starts at {@code (j*256+c)*subDim}. */
    private final float[] codebooks;

    private ProductQuantizer(int dim, int m, float[] codebooks) {
        this.dim = dim;
        this.m = m;
        this.subDim = dim / m;
        this.codebooks = codebooks;
    }

    /**
     * Trains one codebook per subspace on {@code n} training vectors.
     *
     * <p>The subspaces are trained independently, which is the assumption product
     * quantization rests on: that the components can be treated as if they were
     * uncorrelated across subspace boundaries. SIFT satisfies this well — its 128
     * dimensions are 16 spatial cells of 8 gradient orientations, so contiguous groups are
     * genuinely coherent. GIST satisfies it less well, and Phase 6 measures the difference.
     */
    public static ProductQuantizer train(float[] data, int n, int dim, int m,
                                         int iterations, long seed,
                                         SubspaceProgressListener progress) {
        if (dim % m != 0) {
            throw new IllegalArgumentException(
                    "m=" + m + " must divide dim=" + dim + " exactly");
        }
        int subDim = dim / m;
        int k = Math.min(CENTROIDS_PER_SUBSPACE, n);
        float[] codebooks = new float[m * CENTROIDS_PER_SUBSPACE * subDim];

        float[] subspace = new float[Math.multiplyExact(n, subDim)];
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                System.arraycopy(data, i * dim + j * subDim, subspace, i * subDim, subDim);
            }
            KMeans.Result result = KMeans.fit(subspace, n, subDim, k, iterations,
                    seed + 1000L * j, null);
            System.arraycopy(result.centroids(), 0, codebooks,
                    j * CENTROIDS_PER_SUBSPACE * subDim, k * subDim);
            // If there were fewer than 256 distinct training points, the unused entries stay
            // at the origin; they simply never win an encoding.
            if (progress != null) {
                progress.onSubspace(j + 1, m, result.inertia());
            }
        }
        return new ProductQuantizer(dim, m, codebooks);
    }

    public int m() {
        return m;
    }

    public int subDim() {
        return subDim;
    }

    public int dim() {
        return dim;
    }

    public float[] codebooks() {
        return codebooks;
    }

    /** Encodes one vector into {@code m} bytes. */
    public void encode(float[] vector, int offset, byte[] out, int outOffset) {
        for (int j = 0; j < m; j++) {
            int src = offset + j * subDim;
            int book = j * CENTROIDS_PER_SUBSPACE * subDim;
            int best = 0;
            float bestDistance = Float.MAX_VALUE;
            for (int c = 0; c < CENTROIDS_PER_SUBSPACE; c++) {
                float d = Distance.l2Squared(vector, src, codebooks, book + c * subDim, subDim);
                if (d < bestDistance) {
                    bestDistance = d;
                    best = c;
                }
            }
            out[outOffset + j] = (byte) best;
        }
    }

    /** Reconstructs the approximation a code stands for. Used by tests and diagnostics. */
    public void decode(byte[] codes, int codeOffset, float[] out, int outOffset) {
        for (int j = 0; j < m; j++) {
            int c = codes[codeOffset + j] & 0xff;
            System.arraycopy(codebooks, (j * CENTROIDS_PER_SUBSPACE + c) * subDim,
                    out, outOffset + j * subDim, subDim);
        }
    }

    /**
     * Fills {@code table} with the asymmetric distance lookup for one query.
     *
     * <p>{@code table[j*256 + c]} is the squared distance from the query's j-th subvector
     * to codeword c of subspace j. Because squared L2 is separable across the subspaces,
     * the distance from the query to any encoded vector is then the sum of {@code m} table
     * lookups — no arithmetic on the vector itself, and no need to store it.
     *
     * <p>"Asymmetric" is the important word: the query is <em>not</em> quantized. Both
     * sides could be encoded, which would make the lookup a pure table read, but the query
     * would then carry its own quantization error on top of the database vector's. Keeping
     * the query exact costs nothing at search time and roughly halves the error.
     */
    public void computeLookupTable(float[] query, int queryOffset, float[] table) {
        for (int j = 0; j < m; j++) {
            int src = queryOffset + j * subDim;
            int book = j * CENTROIDS_PER_SUBSPACE * subDim;
            int row = j * CENTROIDS_PER_SUBSPACE;
            for (int c = 0; c < CENTROIDS_PER_SUBSPACE; c++) {
                table[row + c] = Distance.l2Squared(query, src, codebooks,
                        book + c * subDim, subDim);
            }
        }
    }

    /** Sums the {@code m} table entries a code selects. */
    public float distanceFromTable(byte[] codes, int codeOffset, float[] table) {
        float sum = 0;
        for (int j = 0; j < m; j++) {
            sum += table[j * CENTROIDS_PER_SUBSPACE + (codes[codeOffset + j] & 0xff)];
        }
        return sum;
    }

    public int lookupTableSize() {
        return m * CENTROIDS_PER_SUBSPACE;
    }

    public long codebookBytes() {
        return (long) codebooks.length * Float.BYTES;
    }

    @FunctionalInterface
    public interface SubspaceProgressListener {
        void onSubspace(int done, int total, double inertia);
    }
}
