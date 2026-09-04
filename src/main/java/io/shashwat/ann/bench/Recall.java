package io.shashwat.ann.bench;

/**
 * recall@k, defined once, for every index and both languages.
 *
 * <p>For one query it is {@code |returned_k ∩ true_k| / k}: an unordered set overlap, so
 * an index that finds all ten true neighbours but ranks them differently scores 1.0. The
 * reported figure is the mean over the whole query set. This is the definition FAISS's
 * own benchmarks use, which is what makes the comparison in the README legitimate.
 */
public final class Recall {

    private Recall() {
    }

    /**
     * Per-query recall@k.
     *
     * @param found        flat ids returned by the index, {@code nq * foundStride}
     * @param foundStride  row width of {@code found} (at least {@code k})
     * @param truth        flat exact ids, {@code nq * truthStride}
     * @param truthStride  row width of {@code truth} (at least {@code k}); SIFT ships 100
     * @param nq           number of queries
     * @param k            neighbours to score
     */
    public static float[] perQuery(int[] found, int foundStride,
                                   int[] truth, int truthStride,
                                   int nq, int k) {
        if (foundStride < k || truthStride < k) {
            throw new IllegalArgumentException(
                    "strides (" + foundStride + "," + truthStride + ") must be at least k=" + k);
        }
        float[] recalls = new float[nq];
        int[] sortedTruth = new int[k];
        for (int q = 0; q < nq; q++) {
            System.arraycopy(truth, q * truthStride, sortedTruth, 0, k);
            java.util.Arrays.sort(sortedTruth);
            int hits = 0;
            int foundBase = q * foundStride;
            for (int i = 0; i < k; i++) {
                if (java.util.Arrays.binarySearch(sortedTruth, found[foundBase + i]) >= 0) {
                    hits++;
                }
            }
            recalls[q] = hits / (float) k;
        }
        return recalls;
    }

    public static double mean(int[] found, int foundStride, int[] truth, int truthStride,
                              int nq, int k) {
        return mean(perQuery(found, foundStride, truth, truthStride, nq, k));
    }

    public static double mean(float[] values) {
        double sum = 0;
        for (float v : values) {
            sum += v;
        }
        return values.length == 0 ? 0 : sum / values.length;
    }

    /**
     * Fraction of queries whose returned ids match the exact ids in the same order. This
     * is stricter than recall and is only used to diagnose the oracle: two distinct
     * vectors at an identical distance may legitimately swap places.
     */
    public static double exactSequenceMatchRate(int[] found, int foundStride,
                                                int[] truth, int truthStride,
                                                int nq, int k) {
        int matches = 0;
        for (int q = 0; q < nq; q++) {
            boolean same = true;
            for (int i = 0; i < k && same; i++) {
                same = found[q * foundStride + i] == truth[q * truthStride + i];
            }
            if (same) {
                matches++;
            }
        }
        return nq == 0 ? 0 : matches / (double) nq;
    }
}
