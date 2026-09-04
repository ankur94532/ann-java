package io.shashwat.ann.bench;

import io.shashwat.ann.distance.Metric;
import io.shashwat.ann.index.BruteForceIndex;
import io.shashwat.ann.io.Datasets;
import io.shashwat.ann.io.IntDataset;
import io.shashwat.ann.io.VectorDataset;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Checkpoint 1: runs the brute-force oracle over a dataset's full query set and checks it
 * against the ground truth shipped with the dataset.
 *
 * <p>The check is on <em>distances</em>, not ids. The datasets contain vectors that are
 * exactly equidistant from a query, so the identity of the k-th neighbour is not unique
 * and two correct exact searches can disagree about it. What must hold is that the sorted
 * distance sequence this search returns is identical to the sorted distance sequence of
 * the shipped ids — that is the well-defined statement of "the search is exact", and it
 * is what this command enforces.
 *
 * <p>The id-level recall is also reported, because it is the ceiling every approximate
 * index in this project is measured against.
 */
public final class OracleCommand {

    private OracleCommand() {
    }

    public static int run(String[] args) {
        Args a = Args.parse(args);
        Datasets dataset = a.dataset();
        if (!dataset.isAvailable()) {
            System.err.println(dataset + " is not present under " + Datasets.dataRoot()
                    + "; run the matching script under scripts/ first");
            return 1;
        }

        System.out.printf("dataset      : %s%n", dataset);
        long t0 = System.nanoTime();
        VectorDataset base = dataset.loadBase();
        VectorDataset queries = dataset.loadQueries(a.maxQueries());
        IntDataset truth = dataset.loadGroundTruth(a.maxQueries());
        System.out.printf("loaded       : %,d base x %d dims, %,d queries, gt width %d (%.1fs)%n",
                base.size(), base.dim(), queries.size(), truth.dim(), secondsSince(t0));

        BruteForceIndex oracle = new BruteForceIndex(base, Metric.L2);
        int k = a.k();
        int nq = queries.size();
        System.out.printf("scanning     : %,d queries x %,d vectors, k=%d, %d threads%n",
                nq, base.size(), k, Runtime.getRuntime().availableProcessors());

        long t1 = System.nanoTime();
        BruteForceIndex.Exact exact = oracle.searchAll(queries, k, (done, total) ->
                System.out.printf("\r  %,d / %,d queries (%.0f%%)", done, total,
                        100.0 * done / total));
        double scanSeconds = secondsSince(t1);
        System.out.printf("%ndone         : %.1fs (%,.0f distance computations/s)%n",
                scanSeconds, (double) nq * base.size() / scanSeconds);

        Verification v = verify(oracle, queries, exact, truth, k);
        double recall = Recall.mean(exact.ids(), k, truth.data(), truth.dim(), nq, k);

        System.out.println();
        System.out.printf("distances    : %,d / %,d queries reproduce the shipped distance "
                + "sequence exactly%n", v.distanceMatches(), nq);
        System.out.printf("ties         : %,d queries return a different id at some position "
                + "with an identical distance%n", v.tieOnlyDifferences());
        System.out.printf("errors       : %,d queries where a returned vector is genuinely "
                + "farther than a shipped one%n", v.realErrors());
        System.out.printf("recall@%-6d: %.6f  <- the ceiling for every index in this project%n",
                k, recall);

        if (v.realErrors() > 0) {
            System.err.println();
            System.err.println("FAIL: exact search returned a vector that is farther than one "
                    + "the shipped ground truth names. The loader or the scan is wrong.");
            System.err.println(v.firstErrorDetail());
            return 1;
        }
        System.out.println();
        System.out.println("OK: exact search reproduces the shipped ground truth up to ties.");
        return 0;
    }

    /**
     * Compares the sorted distance sequence of the returned ids with that of the shipped
     * ids, position by position, for every query.
     */
    private static Verification verify(BruteForceIndex oracle, VectorDataset queries,
                                       BruteForceIndex.Exact exact, IntDataset truth, int k) {
        int nq = queries.size();
        boolean[] distancesMatch = new boolean[nq];
        boolean[] idsMatch = new boolean[nq];
        String[] errorDetail = new String[nq];

        IntStream.range(0, nq).parallel().forEach(q -> {
            float[] mine = Arrays.copyOfRange(exact.distances(), q * k, q * k + k);
            float[] theirs = new float[k];
            for (int i = 0; i < k; i++) {
                theirs[i] = oracle.distanceTo(queries.data(), queries.offset(q),
                        truth.data()[q * truth.dim() + i]);
            }
            Arrays.sort(mine);
            Arrays.sort(theirs);
            distancesMatch[q] = Arrays.equals(mine, theirs);
            idsMatch[q] = Arrays.equals(
                    Arrays.copyOfRange(exact.ids(), q * k, q * k + k),
                    truth.row(q, k));
            if (!distancesMatch[q]) {
                errorDetail[q] = """
                        query %d
                          found ids  : %s
                          found dist : %s
                          truth ids  : %s
                          truth dist : %s"""
                        .formatted(q,
                                Arrays.toString(Arrays.copyOfRange(exact.ids(), q * k, q * k + k)),
                                Arrays.toString(mine),
                                Arrays.toString(truth.row(q, k)),
                                Arrays.toString(theirs));
            }
        });

        int distanceMatches = 0;
        int tieOnly = 0;
        int errors = 0;
        String firstError = "(none)";
        for (int q = 0; q < nq; q++) {
            if (distancesMatch[q]) {
                distanceMatches++;
                if (!idsMatch[q]) {
                    tieOnly++;
                }
            } else {
                if (errors == 0) {
                    firstError = errorDetail[q];
                }
                errors++;
            }
        }
        return new Verification(distanceMatches, tieOnly, errors, firstError);
    }

    private record Verification(int distanceMatches, int tieOnlyDifferences, int realErrors,
                                String firstErrorDetail) {
    }

    private static double secondsSince(long startNanos) {
        return (System.nanoTime() - startNanos) / 1e9;
    }

    /** {@code oracle [--dataset sift|gist] [--k 10] [--queries N]} */
    private record Args(Datasets dataset, int k, int maxQueries) {

        static Args parse(String[] args) {
            Datasets dataset = Datasets.SIFT1M;
            int k = 10;
            int maxQueries = Integer.MAX_VALUE;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--dataset" -> dataset = switch (args[++i].toLowerCase()) {
                        case "sift", "sift1m" -> Datasets.SIFT1M;
                        case "gist", "gist1m" -> Datasets.GIST1M;
                        default -> throw new IllegalArgumentException("unknown dataset " + args[i]);
                    };
                    case "--k" -> k = Integer.parseInt(args[++i]);
                    case "--queries" -> maxQueries = Integer.parseInt(args[++i]);
                    default -> throw new IllegalArgumentException("unknown option " + args[i]);
                }
            }
            return new Args(dataset, k, maxQueries);
        }
    }
}
