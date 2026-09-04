package io.shashwat.ann.io;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Checks the SIFT1M loader against values read out of the files by an independent
 * pure-Python {@code struct} reader, so a byte-order or stride bug in {@link VecsReader}
 * cannot validate itself. Skipped when the dataset has not been downloaded.
 */
class DatasetsTest {

    @Test
    void loadsSiftBase() {
        assumeTrue(Datasets.SIFT1M.isAvailable(), "SIFT1M not downloaded");
        VectorDataset base = Datasets.SIFT1M.loadBase();

        assertEquals(1_000_000, base.size());
        assertEquals(128, base.dim());
        assertEquals(128_000_000L, base.data().length);

        // Hand-checked prefix of vector 0.
        assertArrayEquals(
                new float[]{0, 16, 35, 5, 32, 31, 14, 10, 11, 78},
                java.util.Arrays.copyOf(base.vector(0), 10));

        // ... and of the last vector, which also proves the stride never drifts.
        assertArrayEquals(
                new float[]{114, 31, 0, 0, 0, 0, 0, 10},
                java.util.Arrays.copyOf(base.vector(999_999), 8));
    }

    @Test
    void loadsSiftQueries() {
        assumeTrue(Datasets.SIFT1M.isAvailable(), "SIFT1M not downloaded");
        VectorDataset queries = Datasets.SIFT1M.loadQueries();

        assertEquals(10_000, queries.size());
        assertEquals(128, queries.dim());
        assertArrayEquals(
                new float[]{1, 3, 11, 110, 62, 22, 4, 0, 43, 21},
                java.util.Arrays.copyOf(queries.vector(0), 10));
    }

    @Test
    void loadsSiftGroundTruth() {
        assumeTrue(Datasets.SIFT1M.isAvailable(), "SIFT1M not downloaded");
        IntDataset gt = Datasets.SIFT1M.loadGroundTruth();

        assertEquals(10_000, gt.size());
        assertEquals(100, gt.dim());
        assertArrayEquals(
                new int[]{932085, 934876, 561813, 708177, 706771,
                        695756, 435345, 701258, 455537, 872728},
                gt.row(0, 10));
    }

    @Test
    void capsLoadForDevelopmentRuns() {
        assumeTrue(Datasets.SIFT1M.isAvailable(), "SIFT1M not downloaded");
        VectorDataset base = Datasets.SIFT1M.loadBase(1000);
        assertEquals(1000, base.size());
        assertEquals(128 * 1000, base.data().length);
    }
}
