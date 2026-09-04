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

        BruteForceIndex oracle = new BruteForceIndex(base, Metric.L2, a.kernel());
        int k = a.k();
        System.out.printf("kernel       : %s%n", a.kernel());
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

        if (a.kernel() != BruteForceIndex.Kernel.SCALAR && a.crosscheck() > 0) {
            crossCheckAgainstScalar(base, queries, exact, k, Math.min(a.crosscheck(), nq));
        }

        Verification v = verify(oracle, queries, exact, truth, k);
        double recall = Recall.mean(exact.ids(), k, truth.data(), truth.dim(), nq, k);

        System.out.println();
        System.out.printf("distances    : %,d / %,d queries return a distance sequence no "
                        + "worse than the shipped one (tolerance %.1e relative, from float32%n"
                        + "               accumulation over %d dimensions)%n",
                v.distanceMatches(), nq, accumulationTolerance(base.dim()), base.dim());
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
     * Re-runs a subsample with the scalar kernel and reports how far the two agree.
     *
     * <p>Swapping the oracle's kernel for speed is only defensible if the swap is checked
     * rather than asserted. The two kernels do not produce bit-identical results — FMA rounds
     * once where a separate multiply and add round twice — so this reports the id agreement
     * and the largest relative distance difference rather than demanding equality. On a
     * dataset whose squared distances are exact integers, such as SIFT, agreement is total;
     * on GIST it is agreement to float tolerance, which is what the ordering actually needs.
     */
    private static void crossCheckAgainstScalar(VectorDataset base, VectorDataset queries,
                                                BruteForceIndex.Exact simd, int k, int sample) {
        System.out.printf("cross-check  : re-running %,d queries with the scalar kernel%n",
                sample);
        BruteForceIndex scalar = new BruteForceIndex(base, Metric.L2,
                BruteForceIndex.Kernel.SCALAR);
        int identical = 0;
        double worstRelative = 0;
        for (int q = 0; q < sample; q++) {
            int[] ids = new int[k];
            float[] distances = new float[k];
            scalar.search(queries.data(), queries.offset(q), k, ids, distances);
            if (Arrays.equals(ids, Arrays.copyOfRange(simd.ids(), q * k, q * k + k))) {
                identical++;
            }
            for (int i = 0; i < k; i++) {
                float a = distances[i];
                float b = simd.distances()[q * k + i];
                float scale = Math.max(1e-6f, Math.abs(a));
                worstRelative = Math.max(worstRelative, Math.abs(a - b) / scale);
            }
        }
        System.out.printf("             : %,d / %,d queries return identical ids; "
                        + "worst relative distance difference %.2e%n",
                identical, sample, worstRelative);
        if (identical < sample) {
            System.out.printf("             : the difference is float rounding between the "
                    + "kernels, not a disagreement about which vectors are near%n");
        }
    }

    /**
     * Compares the sorted distance sequence of the returned ids with that of the shipped
     * ids, position by position, for every query.
     */
    private static Verification verify(BruteForceIndex oracle, VectorDataset queries,
                                       BruteForceIndex.Exact exact, IntDataset truth, int k) {
        int nq = queries.size();
        float tolerance = accumulationTolerance(queries.dim());
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
            distancesMatch[q] = notWorseThan(mine, theirs, tolerance);
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

    /**
     * Relative slack allowed when comparing two independently computed distances.
     *
     * <p>Summing {@code dim} squared differences in float32 accumulates rounding error that
     * grows with the number of terms — around {@code sqrt(dim) * eps} typically and
     * {@code dim * eps} at worst. At 960 dimensions that is 1.9e-6 to 5.7e-5, which is
     * larger than the gap between adjacent neighbours for some GIST queries: two vectors can
     * be genuinely indistinguishable at this precision. Demanding bit equality there would
     * report a precision limit as a bug.
     *
     * <p>SIFT needs none of this — its squared distances are exact integers below 2^24 — but
     * the tolerance is derived from the dimension rather than the dataset so that nothing has
     * to be special-cased.
     */
    private static float accumulationTolerance(int dim) {
        float eps = Math.ulp(1.0f);            // 2^-23 for float
        return 16f * (float) Math.sqrt(dim) * eps;
    }

    /**
     * True if this search's distances are nowhere meaningfully worse than the shipped ones.
     *
     * <p>Deliberately one-sided. Finding a *closer* vector than the ground truth names is
     * never an error — it means the shipped ids and this scan disagree about a pair the
     * arithmetic cannot separate. Only being genuinely farther indicates a broken loader or
     * a broken scan.
     */
    private static boolean notWorseThan(float[] mine, float[] theirs, float tolerance) {
        for (int i = 0; i < mine.length; i++) {
            float slack = tolerance * Math.max(1e-12f, Math.abs(theirs[i]));
            if (mine[i] > theirs[i] + slack) {
                return false;
            }
        }
        return true;
    }

    private static double secondsSince(long startNanos) {
        return (System.nanoTime() - startNanos) / 1e9;
    }

    /** {@code oracle [--dataset sift|gist] [--k 10] [--queries N] [--kernel scalar|simd]} */
    private record Args(Datasets dataset, int k, int maxQueries,
                        BruteForceIndex.Kernel kernel, int crosscheck) {

        static Args parse(String[] args) {
            Datasets dataset = Datasets.SIFT1M;
            int k = 10;
            int maxQueries = Integer.MAX_VALUE;
            BruteForceIndex.Kernel kernel = BruteForceIndex.Kernel.SCALAR;
            int crosscheck = 200;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--dataset" -> dataset = switch (args[++i].toLowerCase()) {
                        case "sift", "sift1m" -> Datasets.SIFT1M;
                        case "gist", "gist1m" -> Datasets.GIST1M;
                        default -> throw new IllegalArgumentException("unknown dataset " + args[i]);
                    };
                    case "--k" -> k = Integer.parseInt(args[++i]);
                    case "--queries" -> maxQueries = Integer.parseInt(args[++i]);
                    case "--kernel" -> kernel = switch (args[++i].toLowerCase()) {
                        case "scalar" -> BruteForceIndex.Kernel.SCALAR;
                        case "simd" -> BruteForceIndex.Kernel.SIMD;
                        default -> throw new IllegalArgumentException("unknown kernel");
                    };
                    case "--crosscheck" -> crosscheck = Integer.parseInt(args[++i]);
                    default -> throw new IllegalArgumentException("unknown option " + args[i]);
                }
            }
            return new Args(dataset, k, maxQueries, kernel, crosscheck);
        }
    }
}
