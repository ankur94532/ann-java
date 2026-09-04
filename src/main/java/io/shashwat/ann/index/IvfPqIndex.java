package io.shashwat.ann.index;

import io.shashwat.ann.distance.Distance;
import io.shashwat.ann.distance.Metric;
import io.shashwat.ann.io.VectorDataset;

import java.util.Random;

/**
 * An inverted file with product-quantized residuals — the memory-efficient counterpart to
 * HNSW.
 *
 * <p>Build: cluster the base into {@code nlist} coarse centroids; assign every vector to
 * its nearest centroid and record it in that centroid's list; subtract the centroid from
 * the vector and product-quantize the <em>residual</em> into {@code m} bytes.
 *
 * <p>Search: find the {@code nprobe} nearest centroids to the query, and scan only those
 * lists, computing distances from the codes with a per-query lookup table.
 *
 * <p><b>Why the residual and not the vector.</b> Encoding {@code x} directly would ask one
 * set of 256-entry codebooks to cover the whole dataset. Encoding {@code x - c(x)} asks
 * them to cover only the spread <em>within</em> a cell, which is smaller by roughly the
 * factor the coarse quantizer already achieved. The same {@code m} bytes therefore buy a
 * far finer approximation. The cost is that the query's lookup table depends on which
 * centroid's list is being scanned — the query residual is {@code q - c}, different for
 * every probed list — so the table has to be rebuilt {@code nprobe} times per query rather
 * than once. That trade is visible in the latency curve at high {@code nprobe} and is
 * discussed in the analysis.
 *
 * <p>The index does not need the base vectors at search time and does not hold a reference
 * to them after construction. That is the whole point: SIFT1M is 512 MB of float32, and a
 * 16-byte code per vector is 16 MB.
 */
public final class IvfPqIndex implements VectorIndex {

    private final int dim;
    private final int size;
    private final IvfPqConfig config;

    /** {@code nlist * dim} coarse centroids. */
    private final float[] centroids;

    private final ProductQuantizer pq;

    /** Inverted lists, flattened: list {@code c} occupies {@code [offsets[c], offsets[c+1])}. */
    private final int[] listOffsets;
    private final int[] listIds;
    private final byte[] listCodes;

    /** Per-search scratch. Single-threaded search, per PROTOCOL.md §5. */
    private final float[] lookupTable;
    private final float[] queryResidual;
    private final BoundedMaxHeap coarseHeap;
    private final int[] probeIds;
    private final float[] probeDistances;

    /** Result heap, reused across queries so the timed region allocates nothing. */
    private BoundedMaxHeap results;

    private IvfPqIndex(int dim, int size, IvfPqConfig config, float[] centroids,
                       ProductQuantizer pq, int[] listOffsets, int[] listIds, byte[] listCodes) {
        this.dim = dim;
        this.size = size;
        this.config = config;
        this.centroids = centroids;
        this.pq = pq;
        this.listOffsets = listOffsets;
        this.listIds = listIds;
        this.listCodes = listCodes;
        this.lookupTable = new float[pq.lookupTableSize()];
        this.queryResidual = new float[dim];
        this.coarseHeap = new BoundedMaxHeap(config.nprobe());
        this.probeIds = new int[config.nprobe()];
        this.probeDistances = new float[config.nprobe()];
    }

    /** A view of the same index searched with a different {@code nprobe}. */
    public IvfPqIndex withNprobe(int nprobe) {
        return new IvfPqIndex(dim, size, config.withNprobe(nprobe), centroids, pq,
                listOffsets, listIds, listCodes);
    }

    // ---------------------------------------------------------------- build

    public static IvfPqIndex build(VectorDataset base, IvfPqConfig config,
                                   BuildProgressListener progress) {
        int n = base.size();
        int dim = base.dim();
        Random rng = new Random(config.seed());

        // --- coarse quantizer -------------------------------------------------------
        int coarseTrainingSize = config.coarseTrainingSize(n);
        report(progress, "coarse k-means on " + coarseTrainingSize + " of " + n
                + " vectors, nlist=" + config.nlist());
        float[] coarseTraining = sample(base, coarseTrainingSize, rng);
        KMeans.Result coarse = KMeans.fit(coarseTraining, coarseTrainingSize, dim,
                config.nlist(), config.trainIterations(), config.seed(),
                (iteration, max, inertia) -> report(progress,
                        String.format("  coarse iteration %d/%d, inertia %.4g",
                                iteration, max, inertia)));
        float[] centroids = coarse.centroids();

        // --- assign every base vector to a list -------------------------------------
        report(progress, "assigning " + n + " vectors to lists");
        int[] assignment = new int[n];
        int[] counts = new int[config.nlist()];
        for (int i = 0; i < n; i++) {
            int best = nearestCentroid(base.data(), base.offset(i), centroids,
                    config.nlist(), dim);
            assignment[i] = best;
            counts[best]++;
            if (progress != null && (i + 1) % 100_000 == 0) {
                report(progress, "  assigned " + (i + 1) + " / " + n);
            }
        }

        // --- product quantizer on residuals -----------------------------------------
        int pqTrainingSize = config.pqTrainingSize(n);
        report(progress, "training " + config.m() + " PQ codebooks on " + pqTrainingSize
                + " residuals");
        float[] pqTraining = residualSample(base, assignment, centroids, pqTrainingSize, rng);
        ProductQuantizer pq = ProductQuantizer.train(pqTraining, pqTrainingSize, dim,
                config.m(), config.trainIterations(), config.seed() + 7,
                (done, total, inertia) -> report(progress,
                        String.format("  subspace %d/%d, inertia %.4g", done, total, inertia)));

        // --- fill the inverted lists ------------------------------------------------
        report(progress, "encoding " + n + " residuals");
        int[] listOffsets = new int[config.nlist() + 1];
        for (int c = 0; c < config.nlist(); c++) {
            listOffsets[c + 1] = listOffsets[c] + counts[c];
        }
        int[] cursor = listOffsets.clone();
        int[] listIds = new int[n];
        byte[] listCodes = new byte[Math.multiplyExact(n, config.m())];
        float[] residual = new float[dim];

        for (int i = 0; i < n; i++) {
            int c = assignment[i];
            int slot = cursor[c]++;
            listIds[slot] = i;
            subtract(base.data(), base.offset(i), centroids, c * dim, residual, dim);
            pq.encode(residual, 0, listCodes, slot * config.m());
            if (progress != null && (i + 1) % 100_000 == 0) {
                report(progress, "  encoded " + (i + 1) + " / " + n);
            }
        }

        return new IvfPqIndex(dim, n, config, centroids, pq, listOffsets, listIds, listCodes);
    }

    private static void report(BuildProgressListener progress, String message) {
        if (progress != null) {
            progress.onStage(message);
        }
    }

    /** A uniform sample without replacement, copied into a fresh flat array. */
    private static float[] sample(VectorDataset base, int count, Random rng) {
        int n = base.size();
        int dim = base.dim();
        float[] out = new float[Math.multiplyExact(count, dim)];
        if (count == n) {
            System.arraycopy(base.data(), 0, out, 0, out.length);
            return out;
        }
        int[] picked = reservoir(n, count, rng);
        for (int i = 0; i < count; i++) {
            System.arraycopy(base.data(), base.offset(picked[i]), out, i * dim, dim);
        }
        return out;
    }

    private static float[] residualSample(VectorDataset base, int[] assignment,
                                          float[] centroids, int count, Random rng) {
        int n = base.size();
        int dim = base.dim();
        float[] out = new float[Math.multiplyExact(count, dim)];
        int[] picked = count == n ? null : reservoir(n, count, rng);
        for (int i = 0; i < count; i++) {
            int source = picked == null ? i : picked[i];
            subtract(base.data(), base.offset(source), centroids, assignment[source] * dim,
                    out, i * dim, dim);
        }
        return out;
    }

    private static int[] reservoir(int n, int count, Random rng) {
        int[] picked = new int[count];
        for (int i = 0; i < count; i++) {
            picked[i] = i;
        }
        for (int i = count; i < n; i++) {
            int j = rng.nextInt(i + 1);
            if (j < count) {
                picked[j] = i;
            }
        }
        return picked;
    }

    private static void subtract(float[] a, int aOff, float[] b, int bOff, float[] out, int dim) {
        subtract(a, aOff, b, bOff, out, 0, dim);
    }

    private static void subtract(float[] a, int aOff, float[] b, int bOff,
                                 float[] out, int outOff, int dim) {
        for (int i = 0; i < dim; i++) {
            out[outOff + i] = a[aOff + i] - b[bOff + i];
        }
    }

    private static int nearestCentroid(float[] data, int offset, float[] centroids,
                                       int nlist, int dim) {
        int best = 0;
        float bestDistance = Float.MAX_VALUE;
        for (int c = 0; c < nlist; c++) {
            float d = Distance.l2Squared(data, offset, centroids, c * dim, dim);
            if (d < bestDistance) {
                bestDistance = d;
                best = c;
            }
        }
        return best;
    }

    // ---------------------------------------------------------------- search

    @Override
    public int search(float[] query, int queryOffset, int k, int[] outIds, float[] outDistances) {
        if (results == null || results.capacity() != k) {
            results = new BoundedMaxHeap(k);
        } else {
            results.clear();
        }

        // 1. Coarse search: which lists are worth opening.
        coarseHeap.clear();
        for (int c = 0; c < config.nlist(); c++) {
            coarseHeap.offer(Distance.l2Squared(query, queryOffset, centroids, c * dim, dim), c);
        }
        int probes = coarseHeap.drainAscending(probeIds, probeDistances);

        // 2. Scan the chosen lists with a lookup table per list.
        int m = config.m();
        for (int p = 0; p < probes; p++) {
            int list = probeIds[p];
            int from = listOffsets[list];
            int to = listOffsets[list + 1];
            if (from == to) {
                continue;
            }
            subtract(query, queryOffset, centroids, list * dim, queryResidual, dim);
            pq.computeLookupTable(queryResidual, 0, lookupTable);
            for (int slot = from; slot < to; slot++) {
                results.offer(pq.distanceFromTable(listCodes, slot * m, lookupTable),
                        listIds[slot]);
            }
        }
        return results.drainAscending(outIds, outDistances);
    }

    // ---------------------------------------------------------------- reporting

    @Override
    public String name() {
        return "ivfpq(" + config.shortName() + ")";
    }

    @Override
    public int dim() {
        return dim;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public Metric metric() {
        return config.metric();
    }

    public IvfPqConfig config() {
        return config;
    }

    public ProductQuantizer productQuantizer() {
        return pq;
    }

    /** Number of vectors in each inverted list. */
    public int[] listSizes() {
        int[] sizes = new int[config.nlist()];
        for (int c = 0; c < config.nlist(); c++) {
            sizes[c] = listOffsets[c + 1] - listOffsets[c];
        }
        return sizes;
    }

    /**
     * Codes, ids, list offsets and both codebooks. Excludes the base vectors, which this
     * index genuinely does not keep — that exclusion is the comparison PROTOCOL.md §7 is
     * set up to make.
     */
    @Override
    public long estimatedBytes() {
        return (long) listCodes.length
                + (long) listIds.length * Integer.BYTES
                + (long) listOffsets.length * Integer.BYTES
                + (long) centroids.length * Float.BYTES
                + pq.codebookBytes();
    }

    /** Bytes the raw base vectors would occupy, for the compression ratio. */
    public long rawBytes() {
        return (long) size * dim * Float.BYTES;
    }

    @FunctionalInterface
    public interface BuildProgressListener {
        void onStage(String message);
    }
}
