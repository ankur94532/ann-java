package io.shashwat.ann.index;

import io.shashwat.ann.distance.Distance;

import java.util.Random;

/**
 * Lloyd's algorithm with k-means++ initialisation, over the flat vector layout.
 *
 * <p>Used twice by {@link IvfPqIndex}: once at full dimension to build the coarse
 * quantizer's {@code nlist} centroids, and once per subspace to build a 256-entry
 * codebook for the product quantizer.
 *
 * <p>Single-threaded, because PROTOCOL.md §6 fixes index construction as single-threaded
 * on both sides of the comparison. This is the slowest part of building an IVF-PQ index by
 * a wide margin and it is where most of the build-time plot comes from.
 */
public final class KMeans {

    private KMeans() {
    }

    /**
     * How the initial centroids are chosen.
     *
     * <p>This is a real difference from FAISS and not a detail: {@code IndexIVFPQ}'s
     * clustering defaults to {@code init_method = 0}, which seeds from randomly chosen
     * training points. Seeding better is a larger training budget, not a better algorithm,
     * so any recall comparison against FAISS has to say which of these was used.
     */
    public enum Init {
        /** FAISS's default: a random subsample of the training set. */
        RANDOM_SAMPLE,
        /** D^2 seeding. Costs one extra pass per centroid and usually converges better. */
        KMEANS_PLUS_PLUS
    }

    /**
     * @param centroids   flat {@code k * dim}
     * @param assignments nearest centroid of each training point
     * @param inertia     sum of squared distances to assigned centroids
     */
    public record Result(float[] centroids, int k, int dim, int[] assignments,
                         double inertia, int iterations) {
    }

    @FunctionalInterface
    public interface ProgressListener {
        void onIteration(int iteration, int maxIterations, double inertia);
    }

    public static Result fit(float[] data, int n, int dim, int k, int maxIterations,
                             long seed, ProgressListener progress) {
        return fit(data, n, dim, k, maxIterations, seed, Init.KMEANS_PLUS_PLUS, progress);
    }

    public static Result fit(float[] data, int n, int dim, int k, int maxIterations,
                             long seed, Init init, ProgressListener progress) {
        if (k <= 0 || k > n) {
            throw new IllegalArgumentException(
                    "k must be in [1, n]; got k=" + k + " with n=" + n);
        }
        Random rng = new Random(seed);
        float[] centroids = init == Init.KMEANS_PLUS_PLUS
                ? kmeansPlusPlusInit(data, n, dim, k, rng)
                : randomSampleInit(data, n, dim, k, rng);

        int[] assignments = new int[n];
        float[] bestDistance = new float[n];
        double inertia = Double.MAX_VALUE;
        int iteration = 0;

        for (; iteration < maxIterations; iteration++) {
            double newInertia = assign(data, n, dim, centroids, k, assignments, bestDistance);
            update(data, n, dim, centroids, k, assignments, bestDistance, rng);
            if (progress != null) {
                progress.onIteration(iteration + 1, maxIterations, newInertia);
            }
            // Lloyd's descends monotonically; when it stops moving there is nothing left.
            if (inertia - newInertia <= 1e-7 * Math.max(1.0, inertia)) {
                inertia = newInertia;
                iteration++;
                break;
            }
            inertia = newInertia;
        }
        // One last assignment so the returned labels match the returned centroids.
        inertia = assign(data, n, dim, centroids, k, assignments, bestDistance);
        return new Result(centroids, k, dim, assignments, inertia, iteration);
    }

    /**
     * k-means++ seeding: the first centre is uniform, and each next centre is drawn with
     * probability proportional to its squared distance from the centres already chosen.
     *
     * <p>Uniform seeding routinely drops several centres into the same dense region and
     * leaves whole clusters unrepresented, and Lloyd's cannot recover — it only ever moves
     * a centre within its own basin. For an inverted-file index that shows up directly as
     * recall loss: a query whose true neighbours sit in an under-covered region has them
     * spread across lists that {@code nprobe} never reaches.
     *
     * <p>The running-minimum trick keeps this to one pass per centre rather than k passes:
     * a new centre can only ever reduce a point's distance-to-nearest-centre, so the array
     * is updated against the newest centre alone.
     */
    private static float[] kmeansPlusPlusInit(float[] data, int n, int dim, int k, Random rng) {
        float[] centroids = new float[Math.multiplyExact(k, dim)];
        int first = rng.nextInt(n);
        System.arraycopy(data, first * dim, centroids, 0, dim);

        float[] closest = new float[n];
        double total = 0;
        for (int i = 0; i < n; i++) {
            closest[i] = Distance.l2Squared(data, i * dim, centroids, 0, dim);
            total += closest[i];
        }

        for (int c = 1; c < k; c++) {
            int chosen = sampleProportionalToDistance(closest, n, total, rng);
            System.arraycopy(data, chosen * dim, centroids, c * dim, dim);

            total = 0;
            for (int i = 0; i < n; i++) {
                float d = Distance.l2Squared(data, i * dim, centroids, c * dim, dim);
                if (d < closest[i]) {
                    closest[i] = d;
                }
                total += closest[i];
            }
        }
        return centroids;
    }

    /** FAISS's default seeding: k distinct training points, chosen uniformly. */
    private static float[] randomSampleInit(float[] data, int n, int dim, int k, Random rng) {
        float[] centroids = new float[Math.multiplyExact(k, dim)];
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        // Partial Fisher-Yates: only the first k positions need to be settled.
        for (int i = 0; i < k; i++) {
            int j = i + rng.nextInt(n - i);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
            System.arraycopy(data, order[i] * dim, centroids, i * dim, dim);
        }
        return centroids;
    }

    private static int sampleProportionalToDistance(float[] weights, int n, double total,
                                                    Random rng) {
        if (total <= 0) {
            // Every point coincides with a centre; any point will do.
            return rng.nextInt(n);
        }
        double target = rng.nextDouble() * total;
        double running = 0;
        for (int i = 0; i < n; i++) {
            running += weights[i];
            if (running >= target) {
                return i;
            }
        }
        return n - 1;
    }

    /** @return inertia */
    private static double assign(float[] data, int n, int dim, float[] centroids, int k,
                                 int[] assignments, float[] bestDistance) {
        double inertia = 0;
        for (int i = 0; i < n; i++) {
            int offset = i * dim;
            int best = 0;
            float bestD = Float.MAX_VALUE;
            for (int c = 0; c < k; c++) {
                float d = Distance.l2Squared(data, offset, centroids, c * dim, dim);
                if (d < bestD) {
                    bestD = d;
                    best = c;
                }
            }
            assignments[i] = best;
            bestDistance[i] = bestD;
            inertia += bestD;
        }
        return inertia;
    }

    /**
     * Moves each centroid to the mean of its members.
     *
     * <p>Empty clusters are re-seeded onto the training point currently farthest from its
     * own centroid. Leaving them empty would be worse than useless for an IVF index: an
     * empty list still costs a distance computation on every coarse search and can never
     * contribute a result, so it is pure overhead in the {@code nprobe} budget.
     */
    private static void update(float[] data, int n, int dim, float[] centroids, int k,
                               int[] assignments, float[] bestDistance, Random rng) {
        double[] sums = new double[k * dim];
        int[] counts = new int[k];
        for (int i = 0; i < n; i++) {
            int c = assignments[i];
            counts[c]++;
            int src = i * dim;
            int dst = c * dim;
            for (int j = 0; j < dim; j++) {
                sums[dst + j] += data[src + j];
            }
        }
        for (int c = 0; c < k; c++) {
            if (counts[c] == 0) {
                reseedEmptyCluster(data, n, dim, centroids, c, assignments, bestDistance, rng);
                continue;
            }
            int dst = c * dim;
            double inverse = 1.0 / counts[c];
            for (int j = 0; j < dim; j++) {
                centroids[dst + j] = (float) (sums[dst + j] * inverse);
            }
        }
    }

    private static void reseedEmptyCluster(float[] data, int n, int dim, float[] centroids,
                                           int cluster, int[] assignments,
                                           float[] bestDistance, Random rng) {
        int worst = -1;
        float worstDistance = -1;
        for (int i = 0; i < n; i++) {
            if (bestDistance[i] > worstDistance) {
                worstDistance = bestDistance[i];
                worst = i;
            }
        }
        if (worst < 0) {
            worst = rng.nextInt(n);
        }
        System.arraycopy(data, worst * dim, centroids, cluster * dim, dim);
        // Claim the point so a second empty cluster this round picks a different one.
        assignments[worst] = cluster;
        bestDistance[worst] = 0;
    }
}
