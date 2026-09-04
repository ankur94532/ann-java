package io.shashwat.ann.index;

import io.shashwat.ann.distance.Distance;
import io.shashwat.ann.distance.Metric;
import io.shashwat.ann.io.VectorDataset;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Exact k-NN by scanning every vector. This is the oracle: every approximate index is
 * validated against it, so it is written for obviousness rather than speed.
 *
 * <p>It defaults to the scalar kernel for exactly that reason — the thing everything else is
 * checked against should contain no clever code. But a scalar scan costs 714 ns per distance
 * at 960 dimensions, which turns validating GIST1M into a thirteen-hour job, so the kernel
 * is selectable. The substitution is safe and not a matter of opinion: {@code
 * DistanceKernelTest} checks the two kernels against each other across dimensions that are
 * not multiples of the lane count or the unroll factor, and {@code OracleCommand}
 * re-verifies a subsample with the other kernel on every run.
 */
public final class BruteForceIndex implements VectorIndex {

    /** Which distance implementation the scan uses. */
    public enum Kernel {
        /** Plain loops. Obviously correct, and slow at high dimension. */
        SCALAR,
        /** Vector API. Agrees with scalar to float tolerance; see DistanceKernelTest. */
        SIMD
    }

    private final VectorDataset base;
    private final Metric metric;
    private final Kernel kernel;

    public BruteForceIndex(VectorDataset base, Metric metric) {
        this(base, metric, Kernel.SCALAR);
    }

    public BruteForceIndex(VectorDataset base, Metric metric, Kernel kernel) {
        this.base = base;
        this.metric = metric;
        this.kernel = kernel;
    }

    public Kernel kernel() {
        return kernel;
    }

    private float distance(float[] query, int queryOffset, int baseOffset) {
        return kernel == Kernel.SCALAR
                ? metric.scalar(query, queryOffset, base.data(), baseOffset, base.dim())
                : Distance.compute(metric, query, queryOffset, base.data(), baseOffset,
                        base.dim());
    }

    @Override
    public String name() {
        return "brute-force(" + metric + "," + kernel + ")";
    }

    @Override
    public int dim() {
        return base.dim();
    }

    @Override
    public int size() {
        return base.size();
    }

    @Override
    public Metric metric() {
        return metric;
    }

    @Override
    public int search(float[] query, int queryOffset, int k, int[] outIds, float[] outDistances) {
        int n = base.size();
        int dim = base.dim();
        float[] data = base.data();
        BoundedMaxHeap heap = new BoundedMaxHeap(Math.min(k, n));
        for (int i = 0, off = 0; i < n; i++, off += dim) {
            heap.offer(distance(query, queryOffset, off), i);
        }
        return heap.drainAscending(outIds, outDistances);
    }

    /**
     * Exact k-NN for a whole query set, parallelised across queries.
     *
     * @return ids and distances, each a flat {@code queries.size() * k} array, row-major,
     *         nearest first
     */
    public Exact searchAll(VectorDataset queries, int k, ProgressListener progress) {
        if (queries.dim() != base.dim()) {
            throw new IllegalArgumentException(
                    "query dim " + queries.dim() + " != base dim " + base.dim());
        }
        int nq = queries.size();
        int[] ids = new int[Math.multiplyExact(nq, k)];
        float[] distances = new float[ids.length];
        AtomicInteger done = new AtomicInteger();
        IntStream.range(0, nq).parallel().forEach(q -> {
            int[] localIds = new int[k];
            float[] localDistances = new float[k];
            search(queries.data(), queries.offset(q), k, localIds, localDistances);
            System.arraycopy(localIds, 0, ids, q * k, k);
            System.arraycopy(localDistances, 0, distances, q * k, k);
            if (progress != null) {
                int c = done.incrementAndGet();
                if (c % 500 == 0 || c == nq) {
                    progress.onProgress(c, nq);
                }
            }
        });
        return new Exact(ids, distances, k);
    }

    /** Distance from a query to one specific indexed vector, using this index's kernel. */
    public float distanceTo(float[] query, int queryOffset, int id) {
        return distance(query, queryOffset, base.offset(id));
    }

    /** The result of an exact sweep: {@code nq * k} ids with their distances. */
    public record Exact(int[] ids, float[] distances, int k) {
    }

    @Override
    public long estimatedBytes() {
        return base.rawBytes();
    }

    /** Reports how far a long-running exact sweep has got. */
    @FunctionalInterface
    public interface ProgressListener {
        void onProgress(int completed, int total);
    }
}
