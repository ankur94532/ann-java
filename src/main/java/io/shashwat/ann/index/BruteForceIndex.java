package io.shashwat.ann.index;

import io.shashwat.ann.distance.Metric;
import io.shashwat.ann.io.VectorDataset;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Exact k-NN by scanning every vector. This is the oracle: every approximate index is
 * validated against it, so it is written for obviousness rather than speed and uses the
 * scalar distance kernels only.
 */
public final class BruteForceIndex implements VectorIndex {

    private final VectorDataset base;
    private final Metric metric;

    public BruteForceIndex(VectorDataset base, Metric metric) {
        this.base = base;
        this.metric = metric;
    }

    @Override
    public String name() {
        return "brute-force(" + metric + ")";
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
            heap.offer(metric.scalar(query, queryOffset, data, off, dim), i);
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

    /** Distance from a query to one specific indexed vector. */
    public float distanceTo(float[] query, int queryOffset, int id) {
        return metric.scalar(query, queryOffset, base.data(), base.offset(id), base.dim());
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
