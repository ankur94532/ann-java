package io.shashwat.ann.bench;

import io.shashwat.ann.index.VectorIndex;
import io.shashwat.ann.io.IntDataset;
import io.shashwat.ann.io.VectorDataset;

import java.time.Instant;
import java.util.Arrays;

/**
 * Runs one index over one query set exactly as PROTOCOL.md §4-§5 specify: a discarded
 * warm-up pass, then three measured single-threaded passes over the full query set, with
 * the median reported.
 */
public final class BenchHarness {

    private BenchHarness() {
    }

    public static Measurement measure(String dataset, VectorIndex index, String params,
                                      VectorDataset queries, IntDataset truth, int k,
                                      int runs, double buildSeconds, long indexBytes) {
        int nq = queries.size();
        int[] outIds = new int[k];
        float[] outDistances = new float[k];

        // Warm-up pass: separate from the measured set so no query is dropped, which
        // matters for GIST1M where the whole query set is only 1000 long.
        int warmupQueries = Math.min(1000, nq);
        for (int q = 0; q < warmupQueries; q++) {
            index.search(queries.data(), queries.offset(q), k, outIds, outDistances);
        }

        double[] meanUs = new double[runs];
        double[] p95Us = new double[runs];
        double[] recalls = new double[runs];

        int[] found = new int[Math.multiplyExact(nq, k)];
        long[] nanos = new long[nq];

        for (int run = 0; run < runs; run++) {
            Arrays.fill(found, -1);
            for (int q = 0; q < nq; q++) {
                long t0 = System.nanoTime();
                int n = index.search(queries.data(), queries.offset(q), k, outIds, outDistances);
                nanos[q] = System.nanoTime() - t0;
                System.arraycopy(outIds, 0, found, q * k, n);
                for (int i = n; i < k; i++) {
                    found[q * k + i] = -1; // a short result counts its empty slots as misses
                }
            }
            meanUs[run] = mean(nanos) / 1000.0;
            p95Us[run] = percentile(nanos, 0.95) / 1000.0;
            recalls[run] = Recall.mean(found, k, truth.data(), truth.dim(), nq, k);
        }

        // base_bytes is the raw size of the indexed vectors, matching FAISS's base.nbytes:
        // it is the denominator of the compression ratio in the memory plot.
        long baseBytes = (long) index.size() * index.dim() * Float.BYTES;
        return new Measurement(
                "java", dataset, index.name(), params, k,
                median(recalls), median(meanUs), median(p95Us),
                buildSeconds, indexBytes, baseBytes, nq, runs,
                Instant.now().toString());
    }

    /** Used-heap delta measurement, per PROTOCOL.md §7. */
    public static long usedHeapBytes() {
        Runtime rt = Runtime.getRuntime();
        for (int i = 0; i < 3; i++) {
            System.gc();
            try {
                Thread.sleep(120);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        return rt.totalMemory() - rt.freeMemory();
    }

    private static double mean(long[] values) {
        long sum = 0;
        for (long v : values) {
            sum += v;
        }
        return values.length == 0 ? 0 : (double) sum / values.length;
    }

    /** Nearest-rank percentile of a copy of {@code values}. */
    private static double percentile(long[] values, double fraction) {
        if (values.length == 0) {
            return 0;
        }
        long[] sorted = values.clone();
        Arrays.sort(sorted);
        int rank = (int) Math.ceil(fraction * sorted.length);
        return sorted[Math.min(sorted.length, Math.max(1, rank)) - 1];
    }

    private static double median(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int n = sorted.length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }
}
