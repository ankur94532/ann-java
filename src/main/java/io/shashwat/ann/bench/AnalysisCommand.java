package io.shashwat.ann.bench;

import io.shashwat.ann.distance.Distance;
import io.shashwat.ann.distance.Metric;
import io.shashwat.ann.index.FastHnswIndex;
import io.shashwat.ann.index.GraphDiagnostics;
import io.shashwat.ann.index.Hnsw;
import io.shashwat.ann.index.HnswConfig;
import io.shashwat.ann.index.IvfPqConfig;
import io.shashwat.ann.index.IvfPqIndex;
import io.shashwat.ann.index.NeighbourSelection;
import io.shashwat.ann.index.VectorIndex;
import io.shashwat.ann.io.Datasets;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Phase 6: per-query failure analysis.
 *
 * <p>A mean recall of 0.95 says nothing about whether the missing 5% is spread thinly
 * over every query or concentrated on a specific kind of query. This writes one CSV row
 * per query with the facts needed to tell those apart:
 *
 * <ul>
 *   <li><b>recall</b> for that query;
 *   <li><b>distance to its 10th true neighbour</b> — how isolated the query is. A query
 *       in a dense region has its ten neighbours packed close by; a query in a sparse
 *       region has to reach much further, and a graph search has correspondingly less
 *       signal to follow;
 *   <li><b>the spread of its true neighbourhood</b> ({@code d10 / d1}), which separates
 *       "far from everything" from "far from everything but with a tight cluster out
 *       there";
 *   <li><b>how many of its true neighbours are graph orphans</b>, for HNSW — neighbours
 *       that no search could return at any {@code efSearch}, because nothing points at
 *       them. Recall lost this way is structural, and no parameter fixes it.
 * </ul>
 */
public final class AnalysisCommand {

    private AnalysisCommand() {
    }

    public static int run(String[] args) {
        Args a = Args.parse(args);

        System.out.printf("kernel       : %s%n", Distance.kernelName());
        BenchData data = BenchData.load(a.dataset(), Integer.MAX_VALUE, a.maxQueries(), a.k());
        System.out.printf("data         : %,d base x %d dims, %,d queries%n",
                data.base().size(), data.base().dim(), data.queries().size());

        VectorIndex index;
        boolean[] orphan = null;
        String label;

        if (a.family().equals("hnsw")) {
            HnswConfig config = new HnswConfig(a.m(), a.efConstruction(), a.efSearch(),
                    a.selection(), Metric.L2, 42L);
            System.out.printf("building     : hnsw %s%n", config.shortName());
            Hnsw hnsw = new FastHnswIndex(data.base(), config);
            long t0 = System.nanoTime();
            hnsw.buildAll((done, total) -> System.out.printf("\r  %,d / %,d", done, total));
            System.out.printf("%nbuilt        : %.1fs%n", (System.nanoTime() - t0) / 1e9);
            GraphDiagnostics diagnostics = hnsw.diagnostics();
            System.out.printf("graph        : %s%n", diagnostics);
            orphan = orphanFlags(hnsw);
            index = hnsw;
            label = config.shortName();
        } else {
            IvfPqConfig config = new IvfPqConfig(a.nlist(), a.pqM(), a.nprobe(), 25, 256,
                    io.shashwat.ann.index.KMeans.Init.RANDOM_SAMPLE, Metric.L2, 42L);
            System.out.printf("building     : ivfpq %s%n", config.shortName());
            long t0 = System.nanoTime();
            index = IvfPqIndex.build(data.base(), config,
                    message -> System.out.println("  " + message));
            System.out.printf("built        : %.1fs%n", (System.nanoTime() - t0) / 1e9);
            label = config.shortName();
        }

        int k = a.k();
        int nq = data.queries().size();
        int[] found = new int[nq * k];
        int[] ids = new int[k];
        for (int q = 0; q < nq; q++) {
            int n = index.search(data.queries().data(), data.queries().offset(q), k, ids, null);
            System.arraycopy(ids, 0, found, q * k, k);
            for (int i = n; i < k; i++) {
                found[q * k + i] = -1;
            }
        }
        float[] perQuery = Recall.perQuery(found, k, data.groundTruth().data(),
                data.groundTruth().dim(), nq, k);
        System.out.printf("recall@%d     : %.4f mean%n", k, Recall.mean(perQuery));

        List<String> lines = new ArrayList<>(nq + 1);
        lines.add("query,recall,d1,dk,spread,orphan_neighbours");
        int queriesWithOrphanNeighbours = 0;
        for (int q = 0; q < nq; q++) {
            int truthBase = q * data.groundTruth().dim();
            int nearest = data.groundTruth().data()[truthBase];
            int kth = data.groundTruth().data()[truthBase + k - 1];
            double d1 = Math.sqrt(distance(data, q, nearest));
            double dk = Math.sqrt(distance(data, q, kth));
            int orphanNeighbours = 0;
            if (orphan != null) {
                for (int i = 0; i < k; i++) {
                    if (orphan[data.groundTruth().data()[truthBase + i]]) {
                        orphanNeighbours++;
                    }
                }
            }
            if (orphanNeighbours > 0) {
                queriesWithOrphanNeighbours++;
            }
            lines.add(String.format(Locale.ROOT, "%d,%.4f,%.4f,%.4f,%.4f,%d",
                    q, perQuery[q], d1, dk, d1 == 0 ? 1.0 : dk / d1, orphanNeighbours));
        }

        if (orphan != null) {
            System.out.printf("orphans      : %,d of %,d queries have at least one true "
                    + "neighbour that no search can reach%n", queriesWithOrphanNeighbours, nq);
        }

        try {
            if (a.csv().getParent() != null) {
                Files.createDirectories(a.csv().getParent());
            }
            Files.write(a.csv(), lines, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("writing " + a.csv(), e);
        }
        System.out.printf("wrote        : %s (%s, %s)%n", a.csv(), a.family(), label);
        return 0;
    }

    /** Marks every node with no incoming layer-0 edge. */
    private static boolean[] orphanFlags(Hnsw hnsw) {
        int n = hnsw.size();
        int[] inDegree = new int[n];
        for (int node = 0; node < n; node++) {
            for (int neighbour : hnsw.neighboursOf(node, 0)) {
                inDegree[neighbour]++;
            }
        }
        boolean[] orphan = new boolean[n];
        for (int node = 0; node < n; node++) {
            orphan[node] = inDegree[node] == 0 && node != hnsw.entryPoint();
        }
        return orphan;
    }

    private static float distance(BenchData data, int query, int baseId) {
        return Metric.L2.scalar(data.queries().data(), data.queries().offset(query),
                data.base().data(), data.base().offset(baseId), data.base().dim());
    }

    private record Args(Datasets dataset, String family, int m, int efConstruction,
                        int efSearch, NeighbourSelection selection, int nlist, int pqM,
                        int nprobe, int k, int maxQueries, Path csv) {

        static Args parse(String[] args) {
            Datasets dataset = Datasets.SIFT1M;
            String family = "hnsw";
            int m = 16;
            int efConstruction = 200;
            int efSearch = 64;
            NeighbourSelection selection = NeighbourSelection.HEURISTIC;
            int nlist = 1024;
            int pqM = 16;
            int nprobe = 16;
            int k = 10;
            int maxQueries = Integer.MAX_VALUE;
            Path csv = null;

            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--dataset" -> dataset = switch (args[++i].toLowerCase(Locale.ROOT)) {
                        case "sift", "sift1m" -> Datasets.SIFT1M;
                        case "gist", "gist1m" -> Datasets.GIST1M;
                        default -> throw new IllegalArgumentException("unknown dataset");
                    };
                    case "--index" -> family = args[++i].toLowerCase(Locale.ROOT);
                    case "--m" -> m = Integer.parseInt(args[++i]);
                    case "--efc" -> efConstruction = Integer.parseInt(args[++i]);
                    case "--ef" -> efSearch = Integer.parseInt(args[++i]);
                    case "--selection" -> selection = switch (args[++i].toLowerCase(Locale.ROOT)) {
                        case "heuristic" -> NeighbourSelection.HEURISTIC;
                        case "nearestm", "nearest-m" -> NeighbourSelection.NEAREST_M;
                        default -> throw new IllegalArgumentException("unknown selection");
                    };
                    case "--nlist" -> nlist = Integer.parseInt(args[++i]);
                    case "--pq-m" -> pqM = Integer.parseInt(args[++i]);
                    case "--nprobe" -> nprobe = Integer.parseInt(args[++i]);
                    case "--k" -> k = Integer.parseInt(args[++i]);
                    case "--queries" -> maxQueries = Integer.parseInt(args[++i]);
                    case "--csv" -> csv = Path.of(args[++i]);
                    default -> throw new IllegalArgumentException("unknown option " + args[i]);
                }
            }
            if (!family.equals("hnsw") && !family.equals("ivfpq")) {
                throw new IllegalArgumentException("--index must be hnsw or ivfpq");
            }
            if (csv == null) {
                csv = Path.of("docs/results/perquery-" + family + "-"
                        + dataset.name().toLowerCase(Locale.ROOT) + ".csv");
            }
            return new Args(dataset, family, m, efConstruction, efSearch, selection,
                    nlist, pqM, nprobe, k, maxQueries, csv);
        }
    }
}
