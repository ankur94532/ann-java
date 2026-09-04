package io.shashwat.ann.index;

import io.shashwat.ann.distance.ScalarDistance;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ProductQuantizerTest {

    private static float[] randomData(long seed, int n, int dim) {
        Random rnd = new Random(seed);
        float[] data = new float[n * dim];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) rnd.nextGaussian() * 10;
        }
        return data;
    }

    @Test
    void rejectsAnMThatDoesNotDivideDim() {
        assertThrows(IllegalArgumentException.class,
                () -> ProductQuantizer.train(new float[100], 10, 10, 3, 5, 1L, null));
    }

    /**
     * The lookup table is the whole mechanism: summing the {@code m} entries a code selects
     * must give exactly the squared distance from the query to the vector the code
     * reconstructs. If this drifts, every reported distance is wrong in a way no recall
     * number would explain.
     */
    @Test
    void lookupTableSumEqualsDistanceToTheReconstruction() {
        int dim = 32;
        int m = 8;
        int n = 2000;
        float[] data = randomData(11, n, dim);
        ProductQuantizer pq = ProductQuantizer.train(data, n, dim, m, 15, 1L, null);

        byte[] code = new byte[m];
        float[] reconstruction = new float[dim];
        float[] table = new float[pq.lookupTableSize()];
        float[] queries = randomData(12, 50, dim);

        for (int q = 0; q < 50; q++) {
            pq.computeLookupTable(queries, q * dim, table);
            for (int i = 0; i < 20; i++) {
                pq.encode(data, i * dim, code, 0);
                pq.decode(code, 0, reconstruction, 0);
                float expected = ScalarDistance.l2Squared(queries, q * dim,
                        reconstruction, 0, dim);
                float actual = pq.distanceFromTable(code, 0, table);
                assertEquals(expected, actual, 1e-2f * Math.max(1f, expected),
                        "query " + q + " vector " + i);
            }
        }
    }

    @Test
    void moreSubspacesReconstructMoreAccurately() {
        int dim = 32;
        int n = 3000;
        float[] data = randomData(13, n, dim);

        double coarse = meanReconstructionError(data, n, dim, 2);
        double fine = meanReconstructionError(data, n, dim, 16);

        assertTrue(fine < coarse,
                "m=16 error " + fine + " should beat m=2 error " + coarse);
    }

    @Test
    void encodesToExactlyMBytes() {
        int dim = 16;
        int m = 4;
        float[] data = randomData(14, 500, dim);
        ProductQuantizer pq = ProductQuantizer.train(data, 500, dim, m, 10, 1L, null);
        assertEquals(m, pq.m());
        assertEquals(dim / m, pq.subDim());
        assertEquals(m * 256, pq.lookupTableSize());

        byte[] codes = new byte[3 * m];
        for (int i = 0; i < 3; i++) {
            pq.encode(data, i * dim, codes, i * m);
        }
        assertEquals(3 * m, codes.length);
    }

    private static double meanReconstructionError(float[] data, int n, int dim, int m) {
        ProductQuantizer pq = ProductQuantizer.train(data, n, dim, m, 15, 1L, null);
        byte[] code = new byte[m];
        float[] reconstruction = new float[dim];
        double total = 0;
        for (int i = 0; i < n; i++) {
            pq.encode(data, i * dim, code, 0);
            pq.decode(code, 0, reconstruction, 0);
            total += ScalarDistance.l2Squared(data, i * dim, reconstruction, 0, dim);
        }
        return total / n;
    }
}
