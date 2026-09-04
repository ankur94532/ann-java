package io.shashwat.ann.index;

/**
 * How HNSW picks which of a node's candidate neighbours to actually keep an edge to.
 *
 * <p>This is the one design choice that most changes the shape of the graph, so both
 * strategies are implemented and switchable, and the difference between them is one of
 * the Phase 6 ablations.
 */
public enum NeighbourSelection {

    /**
     * Keep the M nearest candidates.
     *
     * <p>Simple, and worse than it looks. In a clustered dataset the M nearest neighbours
     * of a node are all likely to be in the same cluster and in roughly the same direction,
     * so the node's edges are redundant: they all lead to the same region. A greedy search
     * arriving at that node has no edge that takes it anywhere new, and can only escape
     * the cluster by luck.
     */
    NEAREST_M,

    /**
     * The relative-neighbourhood heuristic from Algorithm 4 of the HNSW paper.
     *
     * <p>Walk the candidates nearest-first and keep a candidate {@code e} only if no
     * already-kept neighbour {@code r} is closer to {@code e} than {@code e} is to the
     * node itself. The rejected candidates are exactly the ones reachable in two short
     * hops via a neighbour already kept, so dropping the edge costs almost no reachability
     * and frees a slot in the degree budget for an edge in a direction not yet covered.
     *
     * <p>The effect is that each node's edges spread out over directions instead of
     * bunching up, long-range links across cluster boundaries survive pruning, and the
     * graph stays connected at a much lower degree.
     */
    HEURISTIC
}
