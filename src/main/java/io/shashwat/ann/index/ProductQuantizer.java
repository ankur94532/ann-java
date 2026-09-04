package io.shashwat.ann.index;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

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
 *
 * <h2>Codebook layout</h2>
 *
 * <p>The codebooks are stored <b>dimension-major within each subspace</b>: entry
 * {@code (subspace j, dimension d, codeword c)} lives at {@code (j*subDim + d)*256 + c}, so
 * the 256 codewords' values for one dimension are contiguous.
 *
 * <p>That is the whole performance story of this class. The obvious layout is
 * centroid-major, which makes a lookup table 256 calls to a distance kernel over a
 * subvector of {@code dim/m} elements — 4 or 8 at the sizes this project sweeps. At that
 * length a SIMD kernel is nearly all prologue: {@code docs/kernels.md} measures a fixed
 * ~4 ns for zeroing accumulators and the final horizontal reduction, against ~1 ns of
 * actual work. Since an IVF-PQ query rebuilds the table once per probed list, that
 * overhead is multiplied by {@code nprobe * m * 256}.
 *
 * <p>Transposed, the whole table is computed by vectorising across the <em>codeword</em>
 * axis instead of within the subvector: broadcast one query component, subtract 4 (or 16)
 * codeword components at once, accumulate in a register across the subspace's dimensions,
 * and store 4 finished distances. The vector width no longer has to divide {@code dim/m},
 * there is one reduction per 4 codewords rather than one per codeword, and the loop runs
 * at full width whatever {@code m} is.
 */
public final class ProductQuantizer {

    /** 8 bits per subquantizer, so a code is exactly {@code m} bytes. */
    public static final int CENTROIDS_PER_SUBSPACE = 256;

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int LANES = SPECIES.length();

    private final int dim;
    private final int m;
    private final int subDim;

    /**
     * {@code m * subDim * 256} floats, dimension-major within each subspace: the value of
     * dimension {@code d} of codeword {@code c} in subspace {@code j} is at
     * {@code (j*subDim + d)*256 + c}. See the class comment for why.
     */
    private final float[] codebooksT;

    private ProductQuantizer(int dim, int m, float[] codebooksT) {
        this.dim = dim;
        this.m = m;
        this.subDim = dim / m;
        this.codebooksT = codebooksT;
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
        return train(data, n, dim, m, iterations, seed, KMeans.Init.KMEANS_PLUS_PLUS,
                progress);
    }

    public static ProductQuantizer train(float[] data, int n, int dim, int m,
                                         int iterations, long seed, KMeans.Init init,
                                         SubspaceProgressListener progress) {
        if (dim % m != 0) {
            throw new IllegalArgumentException(
                    "m=" + m + " must divide dim=" + dim + " exactly");
        }
        int subDim = dim / m;
        int k = Math.min(CENTROIDS_PER_SUBSPACE, n);
        float[] codebooksT = new float[m * subDim * CENTROIDS_PER_SUBSPACE];

        float[] subspace = new float[Math.multiplyExact(n, subDim)];
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                System.arraycopy(data, i * dim + j * subDim, subspace, i * subDim, subDim);
            }
            KMeans.Result result = KMeans.fit(subspace, n, subDim, k, iterations,
                    seed + 1000L * j, init, null);
            // Transpose k-means' centroid-major output into the dimension-major layout.
            float[] centroids = result.centroids();
            for (int c = 0; c < k; c++) {
                for (int d = 0; d < subDim; d++) {
                    codebooksT[(j * subDim + d) * CENTROIDS_PER_SUBSPACE + c] =
                            centroids[c * subDim + d];
                }
            }
            // If there were fewer than 256 distinct training points, the unused entries stay
            // at the origin; they simply never win an encoding.
            if (progress != null) {
                progress.onSubspace(j + 1, m, result.inertia());
            }
        }
        return new ProductQuantizer(dim, m, codebooksT);
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

    /** The transposed codebooks. Exposed for tests and diagnostics only. */
    public float[] codebooks() {
        return codebooksT;
    }

    /**
     * Squared distances from one subvector to all 256 codewords of subspace {@code j},
     * written to {@code out[outOffset .. outOffset+256)}.
     *
     * <p>The vectorisation runs across codewords: one accumulator holds 4 (or 16) codewords'
     * partial sums while the loop walks the subspace's dimensions, so the register is live
     * across the whole inner loop and there is one store per lane-width of finished
     * distances. Nothing here depends on {@code subDim} being a multiple of the vector
     * width, which is the defect of the centroid-major formulation.
     */
    private void subspaceDistances(float[] vector, int offset, int j,
                                   float[] out, int outOffset) {
        int base = j * subDim * CENTROIDS_PER_SUBSPACE;
        int bound = SPECIES.loopBound(CENTROIDS_PER_SUBSPACE);
        int c = 0;
        for (; c < bound; c += LANES) {
            FloatVector acc = FloatVector.zero(SPECIES);
            for (int d = 0; d < subDim; d++) {
                FloatVector diff = FloatVector
                        .fromArray(SPECIES, codebooksT, base + d * CENTROIDS_PER_SUBSPACE + c)
                        .sub(vector[offset + d]);
                acc = diff.fma(diff, acc);
            }
            acc.intoArray(out, outOffset + c);
        }
        for (; c < CENTROIDS_PER_SUBSPACE; c++) {
            float sum = 0;
            for (int d = 0; d < subDim; d++) {
                float diff = codebooksT[base + d * CENTROIDS_PER_SUBSPACE + c]
                        - vector[offset + d];
                sum += diff * diff;
            }
            out[outOffset + c] = sum;
        }
    }

    /**
     * Encodes one vector into {@code m} bytes.
     *
     * @param scratch caller-supplied buffer of at least 256 floats, reused across calls so
     *                that encoding a million vectors allocates nothing
     */
    public void encode(float[] vector, int offset, byte[] out, int outOffset,
                       float[] scratch) {
        for (int j = 0; j < m; j++) {
            subspaceDistances(vector, offset + j * subDim, j, scratch, 0);
            int best = 0;
            float bestDistance = scratch[0];
            for (int c = 1; c < CENTROIDS_PER_SUBSPACE; c++) {
                if (scratch[c] < bestDistance) {
                    bestDistance = scratch[c];
                    best = c;
                }
            }
            out[outOffset + j] = (byte) best;
        }
    }

    /** Convenience for tests and cold paths; allocates. */
    public void encode(float[] vector, int offset, byte[] out, int outOffset) {
        encode(vector, offset, out, outOffset, new float[CENTROIDS_PER_SUBSPACE]);
    }

    /** Reconstructs the approximation a code stands for. Used by tests and diagnostics. */
    public void decode(byte[] codes, int codeOffset, float[] out, int outOffset) {
        for (int j = 0; j < m; j++) {
            int c = codes[codeOffset + j] & 0xff;
            for (int d = 0; d < subDim; d++) {
                out[outOffset + j * subDim + d] =
                        codebooksT[(j * subDim + d) * CENTROIDS_PER_SUBSPACE + c];
            }
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
            subspaceDistances(query, queryOffset + j * subDim, j,
                    table, j * CENTROIDS_PER_SUBSPACE);
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
        return (long) codebooksT.length * Float.BYTES;
    }

    @FunctionalInterface
    public interface SubspaceProgressListener {
        void onSubspace(int done, int total, double inertia);
    }
}
