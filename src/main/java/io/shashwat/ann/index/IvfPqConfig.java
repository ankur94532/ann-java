package io.shashwat.ann.index;

import io.shashwat.ann.distance.Metric;

/**
 * Build- and search-time parameters for {@link IvfPqIndex}.
 *
 * @param nlist              number of coarse centroids, i.e. inverted lists. More lists
 *                           means shorter lists and less scanning per probe, but a more
 *                           expensive coarse search and a higher chance that a query's true
 *                           neighbours fall in a list {@code nprobe} does not reach.
 * @param m                  subvectors per code, so a code is {@code m} bytes. This is the
 *                           memory/accuracy dial: it sets both the compression ratio and
 *                           the recall ceiling.
 * @param nprobe             lists to scan per query. The recall/latency dial, changeable
 *                           after the build.
 * @param trainIterations    Lloyd iterations for both quantizers. FAISS's ProductQuantizer
 *                           defaults to 25; IndexIVFPQ's coarse clustering defaults to only
 *                           10, so scripts/faiss_bench.py raises it to 25 explicitly and
 *                           both sides train for the same number of iterations.
 * @param init               centroid seeding. FAISS defaults to a random subsample; this
 *                           defaults to k-means++, which is a larger training budget, so
 *                           the difference is measured as an ablation rather than assumed
 *                           to be free.
 * @param pointsPerCentroid  training-set size per centroid, capped against the base size.
 *                           256 also matches FAISS's default.
 */
public record IvfPqConfig(int nlist, int m, int nprobe, int trainIterations,
                          int pointsPerCentroid, KMeans.Init init, Metric metric, long seed) {

    public IvfPqConfig {
        if (nlist < 1) {
            throw new IllegalArgumentException("nlist must be positive");
        }
        if (m < 1) {
            throw new IllegalArgumentException("m must be positive");
        }
        if (nprobe < 1) {
            throw new IllegalArgumentException("nprobe must be positive");
        }
        if (nprobe > nlist) {
            throw new IllegalArgumentException(
                    "nprobe=" + nprobe + " cannot exceed nlist=" + nlist);
        }
        if (metric != Metric.L2) {
            throw new IllegalArgumentException(
                    "only L2 is implemented; the residual decomposition assumes it");
        }
    }

    public static IvfPqConfig of(int nlist, int m, int nprobe) {
        return new IvfPqConfig(nlist, m, nprobe, 25, 256,
                KMeans.Init.KMEANS_PLUS_PLUS, Metric.L2, 42L);
    }

    public IvfPqConfig withNprobe(int newNprobe) {
        return new IvfPqConfig(nlist, m, Math.min(newNprobe, nlist), trainIterations,
                pointsPerCentroid, init, metric, seed);
    }

    public IvfPqConfig withInit(KMeans.Init newInit) {
        return new IvfPqConfig(nlist, m, nprobe, trainIterations, pointsPerCentroid,
                newInit, metric, seed);
    }

    /** Training sample size for the coarse quantizer, capped at the base size. */
    public int coarseTrainingSize(int baseSize) {
        return (int) Math.min(baseSize, (long) nlist * pointsPerCentroid);
    }

    /** Training sample size for each product-quantizer subspace. */
    public int pqTrainingSize(int baseSize) {
        return (int) Math.min(baseSize,
                (long) ProductQuantizer.CENTROIDS_PER_SUBSPACE * pointsPerCentroid);
    }

    public String shortName() {
        return "nlist=" + nlist + ",m=" + m + ",nprobe=" + nprobe
                + (init == KMeans.Init.KMEANS_PLUS_PLUS ? "" : ",init=random");
    }
}
