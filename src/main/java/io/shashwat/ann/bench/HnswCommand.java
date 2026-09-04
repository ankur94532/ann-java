package io.shashwat.ann.bench;

import io.shashwat.ann.distance.Distance;
import io.shashwat.ann.index.HnswConfig;
import io.shashwat.ann.index.HnswIndex;
import io.shashwat.ann.index.NeighbourSelection;
import io.shashwat.ann.io.Datasets;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Builds one HNSW graph and measures it at a series of {@code efSearch} values.
 *
 * <p>{@code efSearch} is a search-time knob, so one build serves the whole sweep — which
 * is exactly why the recall/latency curve is cheap to produce and the recall/build-time
 * curve is not.
 */
public final class HnswCommand {

    private HnswCommand() {
    }

    public static int run(String[] args) {
        Args a = Args.parse(args);

        System.out.printf("kernel       : %s%n", Distance.kernelName());
        BenchData data = BenchData.load(a.dataset(), a.maxBase(), a.maxQueries(), a.k());
        System.out.printf("data         : %,d base x %d dims, %,d queries%n",
                data.base().size(), data.base().dim(), data.queries().size());

        HnswConfig config = new HnswConfig(a.m(), a.efConstruction(), a.efSearchValues()[0],
                a.selection(), io.shashwat.ann.distance.Metric.L2, 42L);
        System.out.printf("building     : M=%d efConstruction=%d selection=%s%n",
                config.m(), config.efConstruction(), config.selection());

        long heapBefore = BenchHarness.usedHeapBytes();
        HnswIndex index = new HnswIndex(data.base(), config);
        long t0 = System.nanoTime();
        index.buildAll((done, total) -> System.out.printf("\r  %,d / %,d inserted (%.0f%%)",
                done, total, 100.0 * done / total));
        double buildSeconds = (System.nanoTime() - t0) / 1e9;
        long heapAfter = BenchHarness.usedHeapBytes();

        System.out.printf("%nbuilt        : %.1fs, %,d directed edges, top layer %d, "
                        + "%,d distance computations%n",
                buildSeconds, index.edgeCount(), index.topLayer(),
                index.distanceComputations());
        System.out.printf("memory       : %.1f MiB retained heap delta, %.1f MiB by the "
                        + "index's own accounting%n",
                (heapAfter - heapBefore) / (1024.0 * 1024.0),
                index.estimatedBytes() / (1024.0 * 1024.0));
        System.out.println();

        List<Measurement> results = new ArrayList<>();
        try (CsvSink sink = a.csv() == null ? null : new CsvSink(a.csv())) {
            for (int efSearch : a.efSearchValues()) {
                HnswIndex configured = index.withEfSearch(efSearch);
                Measurement m = BenchHarness.measure(
                        data.label(), configured,
                        configured.config().shortName(),
                        data.queries(), data.groundTruth(), a.k(), a.runs(),
                        buildSeconds, index.estimatedBytes());
                System.out.println("  " + m);
                results.add(m);
                if (sink != null) {
                    sink.write(m);
                }
            }
            if (sink != null) {
                System.out.printf("%nwrote %d rows to %s%n", results.size(), sink.path());
            }
        }
        return 0;
    }

    private record Args(Datasets dataset, int m, int efConstruction, int[] efSearchValues,
                        NeighbourSelection selection, int k, int runs,
                        int maxBase, int maxQueries, Path csv) {

        static Args parse(String[] args) {
            Datasets dataset = Datasets.SIFT1M;
            int m = 16;
            int efConstruction = 200;
            int[] efSearch = {16, 32, 64, 128, 256, 512};
            NeighbourSelection selection = NeighbourSelection.HEURISTIC;
            int k = 10;
            int runs = 3;
            int maxBase = Integer.MAX_VALUE;
            int maxQueries = Integer.MAX_VALUE;
            Path csv = null;

            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--dataset" -> dataset = switch (args[++i].toLowerCase()) {
                        case "sift", "sift1m" -> Datasets.SIFT1M;
                        case "gist", "gist1m" -> Datasets.GIST1M;
                        default -> throw new IllegalArgumentException("unknown dataset " + args[i]);
                    };
                    case "--m" -> m = Integer.parseInt(args[++i]);
                    case "--efc" -> efConstruction = Integer.parseInt(args[++i]);
                    case "--ef" -> efSearch = parseInts(args[++i]);
                    case "--selection" -> selection = switch (args[++i].toLowerCase()) {
                        case "heuristic" -> NeighbourSelection.HEURISTIC;
                        case "nearestm", "nearest-m" -> NeighbourSelection.NEAREST_M;
                        default -> throw new IllegalArgumentException("unknown selection " + args[i]);
                    };
                    case "--k" -> k = Integer.parseInt(args[++i]);
                    case "--runs" -> runs = Integer.parseInt(args[++i]);
                    case "--base" -> maxBase = Integer.parseInt(args[++i]);
                    case "--queries" -> maxQueries = Integer.parseInt(args[++i]);
                    case "--csv" -> csv = Path.of(args[++i]);
                    default -> throw new IllegalArgumentException("unknown option " + args[i]);
                }
            }
            return new Args(dataset, m, efConstruction, efSearch, selection, k, runs,
                    maxBase, maxQueries, csv);
        }

        private static int[] parseInts(String csv) {
            String[] parts = csv.split(",");
            int[] out = new int[parts.length];
            for (int i = 0; i < parts.length; i++) {
                out[i] = Integer.parseInt(parts[i].trim());
            }
            return out;
        }
    }
}
