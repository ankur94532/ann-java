package io.shashwat.ann.index;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KMeansTest {

    @Test
    void recoversWellSeparatedClusters() {
        int dim = 8;
        int perCluster = 200;
        int clusters = 6;
        Random rnd = new Random(3);
        float[] data = new float[clusters * perCluster * dim];
        for (int c = 0; c < clusters; c++) {
            for (int i = 0; i < perCluster; i++) {
                int row = (c * perCluster + i) * dim;
                for (int j = 0; j < dim; j++) {
                    // Centres 1000 apart, noise of order 1: unmistakable.
                    data[row + j] = c * 1000f + (float) rnd.nextGaussian();
                }
            }
        }

        KMeans.Result result = KMeans.fit(data, clusters * perCluster, dim, clusters,
                50, 1L, null);

        for (int c = 0; c < clusters; c++) {
            Set<Integer> labels = new HashSet<>();
            for (int i = 0; i < perCluster; i++) {
                labels.add(result.assignments()[c * perCluster + i]);
            }
            assertEquals(1, labels.size(), "cluster " + c + " was split across " + labels);
        }
        Set<Integer> used = new HashSet<>();
        for (int a : result.assignments()) {
            used.add(a);
        }
        assertEquals(clusters, used.size(), "each true cluster should own one centroid");
    }

    /** k-means++ must not leave a centroid unused, and the reseeding must fix it if it does. */
    @Test
    void leavesNoEmptyCluster() {
        int n = 500;
        int dim = 4;
        int k = 64;
        Random rnd = new Random(9);
        float[] data = new float[n * dim];
        for (int i = 0; i < data.length; i++) {
            data[i] = rnd.nextFloat();
        }
        KMeans.Result result = KMeans.fit(data, n, dim, k, 25, 2L, null);

        int[] counts = new int[k];
        for (int a : result.assignments()) {
            counts[a]++;
        }
        for (int c = 0; c < k; c++) {
            assertTrue(counts[c] > 0, "centroid " + c + " owns nothing");
        }
    }

    @Test
    void inertiaDecreasesAndIsReproducible() {
        int n = 800;
        int dim = 6;
        Random rnd = new Random(4);
        float[] data = new float[n * dim];
        for (int i = 0; i < data.length; i++) {
            data[i] = rnd.nextFloat() * 20;
        }
        KMeans.Result few = KMeans.fit(data, n, dim, 4, 25, 5L, null);
        KMeans.Result many = KMeans.fit(data, n, dim, 32, 25, 5L, null);
        KMeans.Result repeat = KMeans.fit(data, n, dim, 32, 25, 5L, null);

        assertTrue(many.inertia() < few.inertia(),
                "more centroids must fit better: " + many.inertia() + " vs " + few.inertia());
        assertEquals(many.inertia(), repeat.inertia(), 1e-9,
                "the same seed must give the same clustering");
    }

    /**
     * Random-sample seeding must work, and must be reproducible, because it is the setting
     * that matches FAISS's default and so the setting any honest recall comparison uses.
     */
    @Test
    void randomSampleInitSeedsFromDistinctTrainingPoints() {
        int n = 400;
        int dim = 5;
        int k = 32;
        Random rnd = new Random(21);
        float[] data = new float[n * dim];
        for (int i = 0; i < data.length; i++) {
            data[i] = rnd.nextFloat() * 10;
        }

        KMeans.Result a = KMeans.fit(data, n, dim, k, 25, 3L, KMeans.Init.RANDOM_SAMPLE, null);
        KMeans.Result b = KMeans.fit(data, n, dim, k, 25, 3L, KMeans.Init.RANDOM_SAMPLE, null);
        assertEquals(a.inertia(), b.inertia(), 1e-9, "the same seed must reproduce");

        int[] counts = new int[k];
        for (int assignment : a.assignments()) {
            counts[assignment]++;
        }
        for (int c = 0; c < k; c++) {
            assertTrue(counts[c] > 0, "centroid " + c + " owns nothing");
        }

        KMeans.Result plusPlus = KMeans.fit(data, n, dim, k, 25, 3L,
                KMeans.Init.KMEANS_PLUS_PLUS, null);
        assertTrue(plusPlus.inertia() > 0 && a.inertia() > 0);
    }

    @Test
    void handlesKEqualToN() {
        int n = 20;
        int dim = 3;
        Random rnd = new Random(6);
        float[] data = new float[n * dim];
        for (int i = 0; i < data.length; i++) {
            data[i] = rnd.nextFloat();
        }
        KMeans.Result result = KMeans.fit(data, n, dim, n, 10, 7L, null);
        assertTrue(result.inertia() < 1e-6, "with k=n every point is its own centroid");
    }
}
