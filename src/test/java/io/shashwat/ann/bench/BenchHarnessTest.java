package io.shashwat.ann.bench;

import io.shashwat.ann.distance.Metric;
import io.shashwat.ann.index.BruteForceIndex;
import io.shashwat.ann.io.IntDataset;
import io.shashwat.ann.io.VectorDataset;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BenchHarnessTest {

    private static VectorDataset random(long seed, int n, int dim) {
        Random rnd = new Random(seed);
        float[] data = new float[n * dim];
        for (int i = 0; i < data.length; i++) {
            data[i] = rnd.nextFloat() * 50;
        }
        return new VectorDataset(data, n, dim);
    }

    @Test
    void exactIndexScoresPerfectRecallAgainstItsOwnGroundTruth() {
        VectorDataset base = random(1, 500, 8);
        VectorDataset queries = random(2, 30, 8);
        BruteForceIndex oracle = new BruteForceIndex(base, Metric.L2);
        int k = 10;
        BruteForceIndex.Exact exact = oracle.searchAll(queries, k, null);
        IntDataset truth = new IntDataset(exact.ids(), queries.size(), k);

        Measurement m = BenchHarness.measure("test", oracle, "exact", queries, truth, k, 3,
                1.25, 4096);

        assertEquals(1.0, m.recallAtK(), 1e-9);
        assertEquals(30, m.queries());
        assertEquals(3, m.runs());
        assertEquals(1.25, m.buildSeconds());
        assertEquals(500L * 8 * Float.BYTES, m.baseBytes(),
                "base_bytes must describe the indexed vectors, not the query set");
        assertTrue(m.meanLatencyUs() > 0);
        assertTrue(m.p95LatencyUs() >= m.meanLatencyUs() * 0.5,
                "p95 " + m.p95LatencyUs() + " should be in the same range as mean "
                        + m.meanLatencyUs());
    }

    /** An index that returns fewer than k results must be charged for the empty slots. */
    @Test
    void shortResultsCountAsMisses() {
        VectorDataset base = random(3, 200, 8);
        VectorDataset queries = random(4, 10, 8);
        BruteForceIndex oracle = new BruteForceIndex(base, Metric.L2);
        int k = 10;
        IntDataset truth = new IntDataset(oracle.searchAll(queries, k, null).ids(),
                queries.size(), k);

        Measurement m = BenchHarness.measure("test", new HalfResultIndex(oracle), "half",
                queries, truth, k, 1, 0, 0);
        assertEquals(0.5, m.recallAtK(), 1e-9);
    }

    @Test
    void csvRoundTripsThroughTheSink(@TempDir Path tmp) throws IOException {
        Path file = tmp.resolve("nested").resolve("results.csv");
        Measurement m = new Measurement("java", "SIFT1M", "hnsw", "M=16,efC=200,ef=64",
                10, 0.987654, 123.456, 234.567, 12.5, 1024, 2048, 10000, 3, "now");
        try (CsvSink sink = new CsvSink(file)) {
            sink.writeAll(List.of(m, m));
        }
        List<String> lines = Files.readAllLines(file);
        assertEquals(3, lines.size());
        assertEquals(Measurement.CSV_HEADER, lines.get(0));
        assertTrue(lines.get(1).contains("\"M=16,efC=200,ef=64\""),
                "params containing commas must be quoted: " + lines.get(1));
        assertTrue(lines.get(1).startsWith("java,SIFT1M,hnsw,"));
        assertTrue(lines.get(1).contains("0.987654"));
    }

    /** Returns only the first k/2 results, to exercise the short-result path. */
    private record HalfResultIndex(BruteForceIndex delegate)
            implements io.shashwat.ann.index.VectorIndex {

        @Override
        public String name() {
            return "half";
        }

        @Override
        public int dim() {
            return delegate.dim();
        }

        @Override
        public int size() {
            return delegate.size();
        }

        @Override
        public Metric metric() {
            return delegate.metric();
        }

        @Override
        public int search(float[] query, int queryOffset, int k, int[] outIds,
                          float[] outDistances) {
            delegate.search(query, queryOffset, k, outIds, outDistances);
            return k / 2;
        }

        @Override
        public long estimatedBytes() {
            return 0;
        }
    }
}
