package io.shashwat.ann.index;

import io.shashwat.ann.io.VectorDataset;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The optimized index must be indistinguishable from the naive one, not merely similar.
 *
 * <p>Both draw levels from the same seeded RNG in the same order and order candidates by
 * the same total (distance, id) comparator, so they must produce byte-identical graphs and
 * byte-identical search results. Anything weaker would leave "is it faster?" and "is it
 * still the same algorithm?" tangled together, and the optimization table would be
 * measuring both at once.
 */
class FastHnswIndexTest {

    private static final int DIM = 16;
    private static final int N = 3000;
    private static final int K = 10;

    private static VectorDataset clustered(long seed, int n, int dim) {
        Random rnd = new Random(seed);
        int clusters = 15;
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

    @ParameterizedTest
    @EnumSource(NeighbourSelection.class)
    void buildsTheSameGraphAsTheNaiveIndex(NeighbourSelection selection) {
        VectorDataset base = clustered(21, N, DIM);
        HnswConfig config = HnswConfig.of(8, 100, 64).withSelection(selection);

        HnswIndex naive = new HnswIndex(base, config);
        naive.buildAll(null);
        FastHnswIndex fast = new FastHnswIndex(base, config);
        fast.buildAll(null);

        assertEquals(naive.entryPoint(), fast.entryPoint());
        assertEquals(naive.topLayer(), fast.topLayer());
        assertEquals(naive.edgeCount(), fast.edgeCount());
        for (int node = 0; node < base.size(); node++) {
            assertEquals(naive.levelOf(node), fast.levelOf(node), "level of node " + node);
            for (int layer = 0; layer <= naive.levelOf(node); layer++) {
                assertArrayEquals(naive.neighboursOf(node, layer), fast.neighboursOf(node, layer),
                        "node " + node + " layer " + layer);
            }
        }
    }

    @ParameterizedTest
    @EnumSource(NeighbourSelection.class)
    void returnsTheSameResultsAsTheNaiveIndex(NeighbourSelection selection) {
        VectorDataset base = clustered(22, N, DIM);
        VectorDataset queries = clustered(23, 200, DIM);
        HnswConfig config = HnswConfig.of(8, 100, 32).withSelection(selection);

        HnswIndex naive = new HnswIndex(base, config);
        naive.buildAll(null);
        FastHnswIndex fast = new FastHnswIndex(base, config);
        fast.buildAll(null);

        int[] naiveIds = new int[K];
        float[] naiveDistances = new float[K];
        int[] fastIds = new int[K];
        float[] fastDistances = new float[K];
        for (int q = 0; q < queries.size(); q++) {
            int a = naive.search(queries.data(), queries.offset(q), K, naiveIds, naiveDistances);
            int b = fast.search(queries.data(), queries.offset(q), K, fastIds, fastDistances);
            assertEquals(a, b, "result count for query " + q);
            assertArrayEquals(naiveIds, fastIds, "ids for query " + q);
            assertArrayEquals(naiveDistances, fastDistances, "distances for query " + q);
        }
    }

    @Test
    void efSearchViewSharesTheGraph() {
        VectorDataset base = clustered(24, 1500, DIM);
        FastHnswIndex fast = new FastHnswIndex(base, HnswConfig.of(8, 100, 16));
        fast.buildAll(null);

        FastHnswIndex wide = fast.withEfSearch(256);
        assertEquals(256, wide.config().efSearch());
        assertEquals(fast.edgeCount(), wide.edgeCount());
        assertEquals(fast.entryPoint(), wide.entryPoint());
        assertArrayEquals(fast.neighboursOf(0, 0), wide.neighboursOf(0, 0));
    }

    @Test
    void respectsDegreeCapsAndReportsExactMemory() {
        VectorDataset base = clustered(25, N, DIM);
        HnswConfig config = HnswConfig.of(8, 100, 64);
        FastHnswIndex fast = new FastHnswIndex(base, config);
        fast.buildAll(null);

        long upperEdges = 0;
        for (int node = 0; node < base.size(); node++) {
            for (int layer = 0; layer <= fast.levelOf(node); layer++) {
                int degree = fast.neighboursOf(node, layer).length;
                assertTrue(degree <= config.maxDegree(layer),
                        "node " + node + " layer " + layer + " degree " + degree);
                if (layer > 0) {
                    upperEdges += degree;
                }
            }
        }
        assertTrue(upperEdges > 0, "some upper-layer edges must exist");

        // Layer 0 arena + per-node level and offset arrays are a lower bound on the size.
        long layer0Bytes = (long) base.size() * (config.maxDegree(0) + 1) * Integer.BYTES;
        assertTrue(fast.estimatedBytes() > layer0Bytes,
                "estimate " + fast.estimatedBytes() + " must exceed the layer-0 arena "
                        + layer0Bytes);
    }
}
