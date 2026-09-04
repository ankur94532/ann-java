package io.shashwat.ann.bench;

import io.shashwat.ann.distance.Distance;
import io.shashwat.ann.index.IvfPqConfig;
import io.shashwat.ann.index.IvfPqIndex;
import io.shashwat.ann.io.Datasets;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

/**
 * Builds one IVF-PQ index and measures it at a series of {@code nprobe} values.
 *
 * <p>Like {@code efSearch} for HNSW, {@code nprobe} is a search-time knob, so one build
 * serves the whole curve. Unlike HNSW, the build is dominated by k-means, which
 * PROTOCOL.md §6 requires to run single-threaded — at {@code nlist=4096} that is the most
 * expensive single step in this project.
 */
public final class IvfPqCommand {

    private IvfPqCommand() {
    }

    public static int run(String[] args) {
        Args a = Args.parse(args);

        System.out.printf("kernel       : %s%n", Distance.kernelName());
        BenchData data = BenchData.load(a.dataset(), a.maxBase(), a.maxQueries(), a.k());
        System.out.printf("data         : %,d base x %d dims, %,d queries%n",
                data.base().size(), data.base().dim(), data.queries().size());

        if (data.base().dim() % a.m() != 0) {
            System.err.printf("m=%d does not divide dim=%d%n", a.m(), data.base().dim());
            return 1;
        }

        IvfPqConfig config = new IvfPqConfig(a.nlist(), a.m(), a.nprobeValues()[0],
                a.trainIterations(), a.pointsPerCentroid(), a.init(),
                io.shashwat.ann.distance.Metric.L2, 42L);
        System.out.printf("building     : %s, %d Lloyd iterations%n",
                config.shortName(), config.trainIterations());

        long heapBefore = BenchHarness.usedHeapBytes();
        long t0 = System.nanoTime();
        IvfPqIndex index = IvfPqIndex.build(data.base(), config,
                message -> System.out.println("  " + message));
        double buildSeconds = (System.nanoTime() - t0) / 1e9;
        long heapAfter = BenchHarness.usedHeapBytes();

        int[] listSizes = index.listSizes();
        Arrays.sort(listSizes);
        System.out.printf("built        : %.1fs%n", buildSeconds);
        System.out.printf("lists        : %,d lists, sizes min/median/max = %,d/%,d/%,d%n",
                listSizes.length, listSizes[0], listSizes[listSizes.length / 2],
                listSizes[listSizes.length - 1]);
        System.out.printf("memory       : %.1f MiB index (%.1fx smaller than the %.1f MiB "
                        + "of raw vectors), %.1f MiB retained heap delta%n",
                index.estimatedBytes() / (1024.0 * 1024.0),
                index.rawBytes() / (double) index.estimatedBytes(),
                index.rawBytes() / (1024.0 * 1024.0),
                (heapAfter - heapBefore) / (1024.0 * 1024.0));
        System.out.println();

        List<Measurement> results = new ArrayList<>();
        try (CsvSink sink = a.csv() == null ? null : new CsvSink(a.csv())) {
            for (int nprobe : a.nprobeValues()) {
                if (nprobe > a.nlist()) {
                    continue;
                }
                IvfPqIndex configured = index.withNprobe(nprobe);
                if (a.phaseTiming()) {
                    configured.enablePhaseTiming(true);
                }
                Measurement m = BenchHarness.measure(
                        data.label(), configured, configured.config().shortName(),
                        data.queries(), data.groundTruth(), a.k(), a.runs(),
                        buildSeconds, index.estimatedBytes());
                System.out.println("  " + m);
                if (a.phaseTiming() && configured.timedQueries() > 0) {
                    double queries = configured.timedQueries();
                    double lut = configured.lutNanos() / queries / 1000.0;
                    double scan = configured.scanNanos() / queries / 1000.0;
                    System.out.printf("      phases: lookup-table %6.1f us (%.0f%%), "
                            + "list scan %6.1f us (%.0f%%)%n",
                            lut, 100 * lut / (lut + scan), scan, 100 * scan / (lut + scan));
                }
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

    private record Args(Datasets dataset, int nlist, int m, int[] nprobeValues,
                        int trainIterations, int pointsPerCentroid,
                        io.shashwat.ann.index.KMeans.Init init, int k, int runs,
                        int maxBase, int maxQueries, Path csv, boolean phaseTiming) {

        static Args parse(String[] args) {
            Datasets dataset = Datasets.SIFT1M;
            int nlist = 1024;
            int m = 16;
            int[] nprobe = {1, 4, 8, 16, 32, 64};
            int trainIterations = 25;
            int pointsPerCentroid = 256;
            io.shashwat.ann.index.KMeans.Init init =
                    io.shashwat.ann.index.KMeans.Init.KMEANS_PLUS_PLUS;
            boolean phaseTiming = false;
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
                    case "--nlist" -> nlist = Integer.parseInt(args[++i]);
                    case "--m" -> m = Integer.parseInt(args[++i]);
                    case "--nprobe" -> nprobe = parseInts(args[++i]);
                    case "--iterations" -> trainIterations = Integer.parseInt(args[++i]);
                    case "--points-per-centroid" -> pointsPerCentroid = Integer.parseInt(args[++i]);
                    case "--init" -> init = switch (args[++i].toLowerCase()) {
                        case "kmeans++", "kmeanspp" ->
                                io.shashwat.ann.index.KMeans.Init.KMEANS_PLUS_PLUS;
                        case "random" -> io.shashwat.ann.index.KMeans.Init.RANDOM_SAMPLE;
                        default -> throw new IllegalArgumentException("unknown init");
                    };
                    case "--phase-timing" -> phaseTiming = true;
                    case "--k" -> k = Integer.parseInt(args[++i]);
                    case "--runs" -> runs = Integer.parseInt(args[++i]);
                    case "--base" -> maxBase = Integer.parseInt(args[++i]);
                    case "--queries" -> maxQueries = Integer.parseInt(args[++i]);
                    case "--csv" -> csv = Path.of(args[++i]);
                    default -> throw new IllegalArgumentException("unknown option " + args[i]);
                }
            }
            return new Args(dataset, nlist, m, nprobe, trainIterations, pointsPerCentroid,
                    init, k, runs, maxBase, maxQueries, csv, phaseTiming);
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
