package io.shashwat.ann.index;

/** What the two HNSW implementations have in common, so the harness can drive either. */
public interface Hnsw extends VectorIndex {

    void buildAll(HnswIndex.ProgressListener progress);

    /** A view of the same graph searched at a different beam width. */
    Hnsw withEfSearch(int efSearch);

    HnswConfig config();

    int topLayer();

    int entryPoint();

    int levelOf(int node);

    int[] neighboursOf(int node, int layer);

    long edgeCount();

    long distanceComputations();

    default GraphDiagnostics diagnostics() {
        return GraphDiagnostics.of(size(), entryPoint(), node -> neighboursOf(node, 0));
    }
}
