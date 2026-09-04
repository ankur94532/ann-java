package io.shashwat.ann.index;

import io.shashwat.ann.bench.Recall;
import io.shashwat.ann.distance.Metric;
import io.shashwat.ann.io.VectorDataset;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class IvfPqIndexTest {

    private static final int DIM = 32;
    private static final int N = 20_000;
    private static final int NQ = 200;
    private static final int K = 10;

    private static VectorDataset clustered(long seed, int n, int dim) {
        Random rnd = new Random(seed);
        int clusters = 40;
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
                data[v * dim + i] = centre[i] + (float) rnd.nextGaussian() * 5f;
            }
        }
        return new VectorDataset(data, n, dim);
    }

    private static int[] exactTruth(VectorDataset base, VectorDataset queries) {
        return new BruteForceIndex(base, Metric.L2).searchAll(queries, K, null).ids();
    }

    private static double recall(IvfPqIndex index, VectorDataset queries, int[] truth) {
        int[] found = new int[queries.size() * K];
        int[] ids = new int[K];
        for (int q = 0; q < queries.size(); q++) {
            int n = index.search(queries.data(), queries.offset(q), K, ids, null);
            System.arraycopy(ids, 0, found, q * K, K);
            for (int i = n; i < K; i++) {
                found[q * K + i] = -1;
            }
        }
        return Recall.mean(found, K, truth, K, queries.size(), K);
    }

    /**
     * Phase 4's correctness requirement: recall must rise with {@code nprobe} and level off
     * at the ceiling that quantization error imposes, rather than reaching 1.0.
     *
     * <p>Monotonicity is not a theorem — adding a list adds candidates, and a badly
     * approximated impostor from the new list can push out a true neighbour — so the
     * assertion allows a small backward step while requiring the trend to hold.
     */
    @Test
    void recallRisesWithNprobeAndStopsBelowOne() {
        VectorDataset base = clustered(31, N, DIM);
        VectorDataset queries = clustered(32, NQ, DIM);
        int[] truth = exactTruth(base, queries);

        IvfPqIndex index = IvfPqIndex.build(base, IvfPqConfig.of(64, 8, 1), null);

        int[] nprobes = {1, 2, 4, 8, 16, 32, 64};
        double[] recalls = new double[nprobes.length];
        for (int i = 0; i < nprobes.length; i++) {
            recalls[i] = recall(index.withNprobe(nprobes[i]), queries, truth);
        }

        for (int i = 1; i < recalls.length; i++) {
            assertTrue(recalls[i] >= recalls[i - 1] - 0.01,
                    "recall fell from " + recalls[i - 1] + " at nprobe=" + nprobes[i - 1]
                            + " to " + recalls[i] + " at nprobe=" + nprobes[i]);
        }
        assertTrue(recalls[recalls.length - 1] > recalls[0] + 0.2,
                "recall barely moved across the nprobe range: "
                        + recalls[0] + " -> " + recalls[recalls.length - 1]);

        // nprobe == nlist scans everything, so whatever is missing is quantization error.
        assertTrue(recalls[recalls.length - 1] < 1.0,
                "exhaustive PQ search should not be exact; got "
                        + recalls[recalls.length - 1]);
    }

    /** More subvectors means a finer code, so a higher ceiling at the same nprobe. */
    @Test
    void largerMRaisesTheCeiling() {
        VectorDataset base = clustered(33, N, DIM);
        VectorDataset queries = clustered(34, NQ, DIM);
        int[] truth = exactTruth(base, queries);

        double coarse = recall(IvfPqIndex.build(base, IvfPqConfig.of(64, 4, 64), null),
                queries, truth);
        double fine = recall(IvfPqIndex.build(base, IvfPqConfig.of(64, 16, 64), null),
                queries, truth);

        assertTrue(fine > coarse,
                "m=16 recall " + fine + " should beat m=4 recall " + coarse);
    }

    @Test
    void everyVectorIsIndexedExactlyOnce() {
        VectorDataset base = clustered(35, 5000, DIM);
        IvfPqIndex index = IvfPqIndex.build(base, IvfPqConfig.of(32, 8, 4), null);

        int total = 0;
        for (int listSize : index.listSizes()) {
            total += listSize;
        }
        assertEquals(base.size(), total);
        assertEquals(base.size(), index.size());
    }

    @Test
    void isFarSmallerThanTheRawVectors() {
        VectorDataset base = clustered(36, N, DIM);
        IvfPqIndex index = IvfPqIndex.build(base, IvfPqConfig.of(64, 8, 4), null);

        long raw = index.rawBytes();
        long stored = index.estimatedBytes();
        assertTrue(stored * 8 < raw,
                "expected at least 8x compression: " + stored + " bytes vs raw " + raw);
    }

    @Test
    void returnsResultsNearestFirst() {
        VectorDataset base = clustered(37, 5000, DIM);
        VectorDataset queries = clustered(38, 20, DIM);
        IvfPqIndex index = IvfPqIndex.build(base, IvfPqConfig.of(32, 8, 8), null);

        int[] ids = new int[K];
        float[] distances = new float[K];
        for (int q = 0; q < queries.size(); q++) {
            int n = index.search(queries.data(), queries.offset(q), K, ids, distances);
            assertEquals(K, n);
            for (int i = 1; i < n; i++) {
                assertTrue(distances[i - 1] <= distances[i], "results must be sorted");
            }
        }
    }

    @Test
    void nprobeViewSharesTheBuiltIndex() {
        VectorDataset base = clustered(39, 4000, DIM);
        IvfPqIndex index = IvfPqIndex.build(base, IvfPqConfig.of(32, 8, 1), null);
        IvfPqIndex wide = index.withNprobe(16);

        assertEquals(16, wide.config().nprobe());
        assertEquals(index.estimatedBytes(), wide.estimatedBytes());
        assertEquals(index.size(), wide.size());
    }
}
