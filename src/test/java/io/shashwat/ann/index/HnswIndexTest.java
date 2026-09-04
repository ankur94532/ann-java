package io.shashwat.ann.index;

import io.shashwat.ann.bench.Recall;
import io.shashwat.ann.distance.Metric;
import io.shashwat.ann.io.VectorDataset;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Correctness of the graph and of the search, on datasets small enough that the oracle
 * can be run for every query.
 */
class HnswIndexTest {

    private static final int DIM = 16;
    private static final int N = 4000;
    private static final int NQ = 200;
    private static final int K = 10;

    /**
     * Clustered rather than uniform: uniform random points in low dimensions are easy for
     * any graph index, and hide exactly the local-minimum failure that neighbour selection
     * exists to prevent.
     */
    private static VectorDataset clustered(long seed, int n, int dim) {
        Random rnd = new Random(seed);
        int clusters = 20;
        float[][] centres = new float[clusters][dim];
        for (float[] c : centres) {
            for (int i = 0; i < dim; i++) {
                c[i] = rnd.nextFloat() * 100;
            }
        }
        float[] data = new float[n * dim];
        for (int v = 0; v < n; v++) {
            float[] centre = centres[rnd.nextInt(clusters)];
            for (int i = 0; i < dim; i++) {
                data[v * dim + i] = centre[i] + (float) rnd.nextGaussian() * 3f;
            }
        }
        return new VectorDataset(data, n, dim);
    }

    private static HnswIndex build(VectorDataset base, HnswConfig config) {
        HnswIndex index = new HnswIndex(base, config);
        index.buildAll(null);
        return index;
    }

    @ParameterizedTest
    @EnumSource(NeighbourSelection.class)
    void findsMostTrueNeighbours(NeighbourSelection selection) {
        VectorDataset base = clustered(1, N, DIM);
        VectorDataset queries = clustered(2, NQ, DIM);
        HnswIndex index = build(base, HnswConfig.of(16, 200, 128).withSelection(selection));

        double recall = recallAgainstOracle(index, base, queries);
        assertTrue(recall > 0.95,
                selection + " recall was " + recall + ", expected > 0.95");
    }

    @Test
    void returnsResultsNearestFirst() {
        VectorDataset base = clustered(3, N, DIM);
        VectorDataset queries = clustered(4, 20, DIM);
        HnswIndex index = build(base, HnswConfig.of(16, 100, 64));

        int[] ids = new int[K];
        float[] distances = new float[K];
        for (int q = 0; q < queries.size(); q++) {
            int n = index.search(queries.data(), queries.offset(q), K, ids, distances);
            assertEquals(K, n);
            for (int i = 1; i < n; i++) {
                assertTrue(distances[i - 1] <= distances[i],
                        "results must be sorted by distance");
            }
            for (int i = 0; i < n; i++) {
                float expected = Metric.L2.scalar(queries.data(), queries.offset(q),
                        base.data(), base.offset(ids[i]), DIM);
                assertEquals(expected, distances[i], 1e-2f, "reported distance must be real");
            }
        }
    }

    @Test
    void respectsDegreeCaps() {
        VectorDataset base = clustered(5, N, DIM);
        HnswConfig config = HnswConfig.of(8, 100, 64);
        HnswIndex index = build(base, config);

        for (int node = 0; node < base.size(); node++) {
            for (int layer = 0; layer <= index.levelOf(node); layer++) {
                int degree = index.neighboursOf(node, layer).length;
                assertTrue(degree <= config.maxDegree(layer),
                        "node " + node + " layer " + layer + " has degree " + degree
                                + " > cap " + config.maxDegree(layer));
            }
        }
    }

    /**
     * Layer 0 must be one connected component reachable from the entry point. A search
     * always starts there, so anything in a separate component is unreachable at any
     * efSearch and is simply lost.
     */
    @ParameterizedTest
    @EnumSource(NeighbourSelection.class)
    void layerZeroIsReachableFromTheEntryPoint(NeighbourSelection selection) {
        VectorDataset base = clustered(6, N, DIM);
        HnswIndex index = build(base, HnswConfig.of(8, 100, 64).withSelection(selection));

        boolean[] seen = new boolean[base.size()];
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(index.entryPoint());
        seen[index.entryPoint()] = true;
        int reached = 1;
        while (!stack.isEmpty()) {
            for (int n : index.neighboursOf(stack.pop(), 0)) {
                if (!seen[n]) {
                    seen[n] = true;
                    reached++;
                    stack.push(n);
                }
            }
        }
        assertEquals(base.size(), reached,
                selection + ": " + (base.size() - reached) + " nodes unreachable in layer 0");
    }

    @Test
    void higherLayersContainFewerNodes() {
        VectorDataset base = clustered(7, N, DIM);
        HnswIndex index = build(base, HnswConfig.of(16, 100, 64));

        int[] perLayer = new int[index.topLayer() + 1];
        for (int node = 0; node < base.size(); node++) {
            for (int layer = 0; layer <= index.levelOf(node); layer++) {
                perLayer[layer]++;
            }
        }
        assertEquals(base.size(), perLayer[0]);
        for (int layer = 1; layer < perLayer.length; layer++) {
            assertTrue(perLayer[layer] < perLayer[layer - 1],
                    "layer " + layer + " (" + perLayer[layer] + ") must be smaller than layer "
                            + (layer - 1) + " (" + perLayer[layer - 1] + ")");
        }
    }

    /** More beam width must not find fewer true neighbours. */
    @Test
    void recallImprovesWithEfSearch() {
        VectorDataset base = clustered(8, N, DIM);
        VectorDataset queries = clustered(9, NQ, DIM);
        HnswConfig config = HnswConfig.of(8, 100, 1);

        HnswIndex narrow = build(base, config);
        double narrowRecall = recallAgainstOracle(narrow, base, queries);

        HnswIndex wide = build(base, config.withEfSearch(200));
        double wideRecall = recallAgainstOracle(wide, base, queries);

        assertTrue(wideRecall >= narrowRecall,
                "ef=200 recall " + wideRecall + " should be at least ef=1 recall " + narrowRecall);
        assertTrue(wideRecall > 0.9, "ef=200 recall was only " + wideRecall);
    }

    @Test
    void buildsAreReproducible() {
        VectorDataset base = clustered(10, 1500, DIM);
        HnswConfig config = HnswConfig.of(8, 100, 32);
        HnswIndex a = build(base, config);
        HnswIndex b = build(base, config);

        assertEquals(a.entryPoint(), b.entryPoint());
        assertEquals(a.topLayer(), b.topLayer());
        assertEquals(a.edgeCount(), b.edgeCount());
        for (int node = 0; node < base.size(); node++) {
            assertEquals(a.levelOf(node), b.levelOf(node), "level of node " + node);
            org.junit.jupiter.api.Assertions.assertArrayEquals(
                    a.neighboursOf(node, 0), b.neighboursOf(node, 0), "layer 0 of node " + node);
        }
    }

    @Test
    void handlesTinyAndSingletonIndexes() {
        VectorDataset one = new VectorDataset(new float[]{1, 2, 3, 4}, 1, 4);
        HnswIndex index = build(one, HnswConfig.of(4, 10, 10));
        int[] ids = new int[5];
        float[] distances = new float[5];
        assertEquals(1, index.search(new float[]{1, 2, 3, 4}, 0, 5, ids, distances));
        assertEquals(0, ids[0]);
        assertEquals(0f, distances[0]);
    }

    private static double recallAgainstOracle(HnswIndex index, VectorDataset base,
                                              VectorDataset queries) {
        BruteForceIndex oracle = new BruteForceIndex(base, Metric.L2);
        int[] found = new int[queries.size() * K];
        int[] truth = new int[queries.size() * K];
        int[] ids = new int[K];
        for (int q = 0; q < queries.size(); q++) {
            index.search(queries.data(), queries.offset(q), K, ids, null);
            System.arraycopy(ids, 0, found, q * K, K);
            oracle.search(queries.data(), queries.offset(q), K, ids, null);
            System.arraycopy(ids, 0, truth, q * K, K);
        }
        return Recall.mean(found, K, truth, K, queries.size(), K);
    }
}
