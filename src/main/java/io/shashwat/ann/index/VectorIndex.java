package io.shashwat.ann.index;

import io.shashwat.ann.distance.Metric;

/**
 * A searchable index over a fixed set of vectors.
 *
 * <p>Search writes into caller-supplied buffers rather than returning a new array: the
 * benchmark measures per-query latency, and an allocation per query would put GC noise
 * into every number.
 */
public interface VectorIndex {

    /** Short identifier including the parameters, e.g. {@code hnsw(M=16,efC=200,ef=64)}. */
    String name();

    int dim();

    /** Number of indexed vectors. */
    int size();

    Metric metric();

    /**
     * Finds the {@code k} nearest indexed vectors to {@code query[queryOffset ..
     * queryOffset+dim)}, writing their ids into {@code outIds} nearest-first and, when
     * {@code outDistances} is non-null, their distances alongside.
     *
     * @return how many results were written, which is {@code min(k, size())} unless the
     *         index prunes the search space and finds fewer
     */
    int search(float[] query, int queryOffset, int k, int[] outIds, float[] outDistances);

    /**
     * Bytes retained by the index structure itself. Whether this includes the base
     * vectors is defined per index and stated in PROTOCOL.md; the sweep harness also
     * measures retained heap independently.
     */
    long estimatedBytes();
}
