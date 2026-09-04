package io.shashwat.ann.bench;

import io.shashwat.ann.distance.Metric;
import io.shashwat.ann.index.BruteForceIndex;
import io.shashwat.ann.io.Datasets;
import io.shashwat.ann.io.IntDataset;
import io.shashwat.ann.io.VectorDataset;

/**
 * Base vectors, queries and ground truth for a run.
 *
 * <p>The shipped ground-truth ids index the <em>full</em> base set, so as soon as the base
 * is subsampled for a development run they are meaningless and exact ground truth has to
 * be recomputed for the subset. That recomputation is why the dev path is honest rather
 * than merely fast, and it is also why PROTOCOL.md forbids subsampled numbers from
 * reaching the README: the task genuinely changes when the haystack shrinks.
 */
public record BenchData(Datasets dataset, VectorDataset base, VectorDataset queries,
                        IntDataset groundTruth, boolean subsampled) {

    public static BenchData load(Datasets dataset, int maxBase, int maxQueries, int k) {
        if (!dataset.isAvailable()) {
            throw new IllegalStateException(dataset + " is not present under "
                    + Datasets.dataRoot() + "; run the matching script under scripts/");
        }
        VectorDataset base = dataset.loadBase(maxBase);
        VectorDataset queries = dataset.loadQueries(maxQueries);
        boolean subsampled = base.size() < io.shashwat.ann.io.VecsReader
                .countVectors(dataset.basePath());

        IntDataset truth;
        if (subsampled) {
            System.out.printf("base is subsampled to %,d vectors; recomputing exact ground "
                    + "truth (the shipped ids index the full base)%n", base.size());
            long t0 = System.nanoTime();
            BruteForceIndex oracle = new BruteForceIndex(base, Metric.L2);
            int gtWidth = Math.max(k, 100);
            BruteForceIndex.Exact exact = oracle.searchAll(queries, gtWidth, null);
            truth = new IntDataset(exact.ids(), queries.size(), gtWidth);
            System.out.printf("ground truth recomputed in %.1fs%n",
                    (System.nanoTime() - t0) / 1e9);
        } else {
            truth = dataset.loadGroundTruth(maxQueries);
        }
        return new BenchData(dataset, base, queries, truth, subsampled);
    }

    public String label() {
        return subsampled ? dataset.name() + "[" + base.size() + "]" : dataset.name();
    }
}
