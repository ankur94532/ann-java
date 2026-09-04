package io.shashwat.ann.bench;

import io.shashwat.ann.distance.Distance;
import io.shashwat.ann.distance.Metric;
import io.shashwat.ann.index.FastHnswIndex;
import io.shashwat.ann.index.Hnsw;
import io.shashwat.ann.index.HnswConfig;
import io.shashwat.ann.index.IvfPqConfig;
import io.shashwat.ann.index.IvfPqIndex;
import io.shashwat.ann.index.NeighbourSelection;
import io.shashwat.ann.io.Datasets;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Phase 5.1: the full parameter sweep, one CSV row per configuration.
 *
 * <p>Resumable. Before each configuration the existing CSV is read and any row already
 * present for the same index and parameter string is skipped, so a sweep that runs for
 * hours can be interrupted and restarted without repeating work or duplicating rows.
 *
 * <p>Search-time knobs ({@code efSearch}, {@code nprobe}) are swept over a single build.
 * For IVF-PQ the coarse quantizer is additionally shared across the {@code m} values at a
 * given {@code nlist}, since they would train identical centroids; each configuration is
 * still charged the full training time in its reported {@code build_seconds}.
 */
public final class SweepCommand {

    private SweepCommand() {
    }

    public static int run(String[] args) {
        Args a = Args.parse(args);

        System.out.printf("kernel       : %s%n", Distance.kernelName());
        BenchData data = BenchData.load(a.dataset(), Integer.MAX_VALUE, a.maxQueries(), a.k());
        System.out.printf("data         : %,d base x %d dims, %,d queries, k=%d, %d runs%n",
                data.base().size(), data.base().dim(), data.queries().size(), a.k(), a.runs());
        System.out.printf("output       : %s%n%n", a.csv());

        Set<String> done = existingRows(a.csv());
        if (!done.isEmpty()) {
            System.out.printf("resuming: %d rows already present, they will be skipped%n%n",
                    done.size());
        }

        try (CsvSink sink = new CsvSink(a.csv())) {
            if (a.runHnsw()) {
                sweepHnsw(a, data, sink, done);
            }
            if (a.runIvfPq()) {
                sweepIvfPq(a, data, sink, done);
            }
        }
        System.out.printf("%nsweep complete: %s%n", a.csv());
        return 0;
    }

    private static void sweepHnsw(Args a, BenchData data, CsvSink sink, Set<String> done) {
        for (int m : a.mValues()) {
            for (int efConstruction : a.efcValues()) {
                if (allDone(done, "hnsw", a.efValues(),
                        ef -> "M=" + m + ",efC=" + efConstruction + ",ef=" + ef
                                + a.selectionSuffix())) {
                    System.out.printf("skip  hnsw M=%d efC=%d (all rows present)%n",
                            m, efConstruction);
                    continue;
                }
                HnswConfig config = new HnswConfig(m, efConstruction, a.efValues()[0],
                        a.selection(), Metric.L2, 42L);
                System.out.printf("build hnsw %s ...%n", config.shortName());

                Hnsw index = new FastHnswIndex(data.base(), config);
                long t0 = System.nanoTime();
                index.buildAll(null);
                double buildSeconds = (System.nanoTime() - t0) / 1e9;
                System.out.printf("      %.1fs, %,d edges, %s%n",
                        buildSeconds, index.edgeCount(), index.diagnostics());

                for (int efSearch : a.efValues()) {
                    Hnsw configured = index.withEfSearch(efSearch);
                    String params = configured.config().shortName();
                    if (done.contains(key("hnsw", params))) {
                        continue;
                    }
                    Measurement measurement = BenchHarness.measure(data.label(), configured,
                            params, data.queries(), data.groundTruth(), a.k(), a.runs(),
                            buildSeconds, index.estimatedBytes());
                    System.out.println("      " + measurement);
                    sink.write(measurement);
                    done.add(key("hnsw", params));
                }
            }
        }
    }

    private static void sweepIvfPq(Args a, BenchData data, CsvSink sink, Set<String> done) {
        for (int nlist : a.nlistValues()) {
            IvfPqIndex.CoarseQuantizer coarse = null;
            for (int pqM : a.pqMValues()) {
                if (data.base().dim() % pqM != 0) {
                    System.out.printf("skip  ivfpq m=%d (does not divide dim=%d)%n",
                            pqM, data.base().dim());
                    continue;
                }
                if (allDone(done, "ivfpq", a.nprobeValues(),
                        nprobe -> nprobe > nlist ? null
                                : "nlist=" + nlist + ",m=" + pqM + ",nprobe=" + nprobe)) {
                    System.out.printf("skip  ivfpq nlist=%d m=%d (all rows present)%n",
                            nlist, pqM);
                    continue;
                }
                IvfPqConfig config = new IvfPqConfig(nlist, pqM, a.nprobeValues()[0],
                        a.trainIterations(), a.pointsPerCentroid(), a.init(), Metric.L2, 42L);
                System.out.printf("build ivfpq %s ...%n", config.shortName());

                if (coarse == null) {
                    coarse = IvfPqIndex.trainCoarseQuantizer(data.base(), config,
                            message -> System.out.println("      " + message));
                    System.out.printf("      coarse quantizer trained in %.1fs "
                            + "(shared across the m values at this nlist)%n",
                            coarse.trainSeconds());
                }

                long t0 = System.nanoTime();
                IvfPqIndex index = IvfPqIndex.build(data.base(), config, coarse,
                        message -> System.out.println("      " + message));
                // The reused coarse quantizer is still charged in full, so build_seconds
                // is what a from-scratch build of this configuration would have cost.
                double buildSeconds = (System.nanoTime() - t0) / 1e9 + coarse.trainSeconds();
                System.out.printf("      %.1fs total, %.1f MiB (%.1fx smaller than raw)%n",
                        buildSeconds, index.estimatedBytes() / (1024.0 * 1024.0),
                        index.rawBytes() / (double) index.estimatedBytes());

                for (int nprobe : a.nprobeValues()) {
                    if (nprobe > nlist) {
                        continue;
                    }
                    IvfPqIndex configured = index.withNprobe(nprobe);
                    String params = configured.config().shortName();
                    if (done.contains(key("ivfpq", params))) {
                        continue;
                    }
                    Measurement measurement = BenchHarness.measure(data.label(), configured,
                            params, data.queries(), data.groundTruth(), a.k(), a.runs(),
                            buildSeconds, index.estimatedBytes());
                    System.out.println("      " + measurement);
                    sink.write(measurement);
                    done.add(key("ivfpq", params));
                }
            }
        }
    }

    private interface ParamNamer {
        String nameFor(int value);
    }

    private static boolean allDone(Set<String> done, String family, int[] values,
                                   ParamNamer namer) {
        for (int value : values) {
            String params = namer.nameFor(value);
            if (params != null && !done.contains(key(family, params))) {
                return false;
            }
        }
        return true;
    }

    private static String key(String family, String params) {
        return family + "|" + params;
    }

    /** Reads {@code index}/{@code params} pairs already in the CSV, for resumption. */
    private static Set<String> existingRows(Path csv) {
        Set<String> rows = new HashSet<>();
        if (!Files.isRegularFile(csv)) {
            return rows;
        }
        try {
            List<String> lines = Files.readAllLines(csv);
            for (String line : lines) {
                if (line.startsWith("harness,") || line.isBlank()) {
                    continue;
                }
                String[] fields = line.split(",", 5);
                if (fields.length < 4) {
                    continue;
                }
                String index = fields[2];
                String params = fields[3].replace("\"", "");
                // index is e.g. hnsw(M=16,efC=200,ef=64) or ivfpq(nlist=1024,...)
                String family = index.contains("(")
                        ? index.substring(0, index.indexOf('('))
                        : index;
                rows.add(key(family.toLowerCase(Locale.ROOT), params));
            }
        } catch (IOException e) {
            throw new UncheckedIOException("reading " + csv, e);
        }
        return rows;
    }

    private record Args(Datasets dataset, boolean runHnsw, boolean runIvfPq,
                        int[] mValues, int[] efcValues, int[] efValues,
                        int[] nlistValues, int[] pqMValues, int[] nprobeValues,
                        NeighbourSelection selection, int trainIterations,
                        int pointsPerCentroid, io.shashwat.ann.index.KMeans.Init init,
                        int k, int runs, int maxQueries, Path csv) {

        String selectionSuffix() {
            return selection == NeighbourSelection.HEURISTIC ? "" : ",sel=nearestM";
        }

        static Args parse(String[] args) {
            Datasets dataset = Datasets.SIFT1M;
            String which = "both";
            int[] mValues = {8, 16, 32};
            int[] efcValues = {100, 200, 400};
            int[] efValues = {16, 32, 64, 128, 256, 512};
            int[] nlistValues = {1024, 4096};
            int[] pqMValues = {8, 16, 32};
            int[] nprobeValues = {1, 4, 8, 16, 32, 64};
            NeighbourSelection selection = NeighbourSelection.HEURISTIC;
            int trainIterations = 25;
            int pointsPerCentroid = 256;
            io.shashwat.ann.index.KMeans.Init init =
                    io.shashwat.ann.index.KMeans.Init.KMEANS_PLUS_PLUS;
            int k = 10;
            int runs = 3;
            int maxQueries = Integer.MAX_VALUE;
            Path csv = null;

            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--dataset" -> dataset = switch (args[++i].toLowerCase(Locale.ROOT)) {
                        case "sift", "sift1m" -> Datasets.SIFT1M;
                        case "gist", "gist1m" -> Datasets.GIST1M;
                        default -> throw new IllegalArgumentException("unknown dataset " + args[i]);
                    };
                    case "--index" -> which = args[++i].toLowerCase(Locale.ROOT);
                    case "--m" -> mValues = parseInts(args[++i]);
                    case "--efc" -> efcValues = parseInts(args[++i]);
                    case "--ef" -> efValues = parseInts(args[++i]);
                    case "--nlist" -> nlistValues = parseInts(args[++i]);
                    case "--pq-m" -> pqMValues = parseInts(args[++i]);
                    case "--nprobe" -> nprobeValues = parseInts(args[++i]);
                    case "--selection" -> selection = switch (args[++i].toLowerCase(Locale.ROOT)) {
                        case "heuristic" -> NeighbourSelection.HEURISTIC;
                        case "nearestm", "nearest-m" -> NeighbourSelection.NEAREST_M;
                        default -> throw new IllegalArgumentException("unknown selection");
                    };
                    case "--iterations" -> trainIterations = Integer.parseInt(args[++i]);
                    case "--points-per-centroid" -> pointsPerCentroid = Integer.parseInt(args[++i]);
                    case "--init" -> init = switch (args[++i].toLowerCase(Locale.ROOT)) {
                        case "kmeans++", "kmeanspp" ->
                                io.shashwat.ann.index.KMeans.Init.KMEANS_PLUS_PLUS;
                        case "random" -> io.shashwat.ann.index.KMeans.Init.RANDOM_SAMPLE;
                        default -> throw new IllegalArgumentException("unknown init");
                    };
                    case "--k" -> k = Integer.parseInt(args[++i]);
                    case "--runs" -> runs = Integer.parseInt(args[++i]);
                    case "--queries" -> maxQueries = Integer.parseInt(args[++i]);
                    case "--csv" -> csv = Path.of(args[++i]);
                    default -> throw new IllegalArgumentException("unknown option " + args[i]);
                }
            }
            if (csv == null) {
                csv = Path.of("docs/results/java-"
                        + dataset.name().toLowerCase(Locale.ROOT) + ".csv");
            }
            boolean hnsw = which.equals("both") || which.equals("hnsw");
            boolean ivfpq = which.equals("both") || which.equals("ivfpq");
            if (!hnsw && !ivfpq) {
                throw new IllegalArgumentException("--index must be hnsw, ivfpq or both");
            }
            return new Args(dataset, hnsw, ivfpq, mValues, efcValues, efValues,
                    nlistValues, pqMValues, nprobeValues, selection, trainIterations,
                    pointsPerCentroid, init, k, runs, maxQueries, csv);
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
