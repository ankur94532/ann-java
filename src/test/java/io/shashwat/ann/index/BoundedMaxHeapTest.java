package io.shashwat.ann.index;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BoundedMaxHeapTest {

    @Test
    void sortableIntPreservesFloatOrder() {
        float[] values = {Float.NEGATIVE_INFINITY, -1e30f, -1f, -Float.MIN_VALUE, -0.0f,
                0.0f, Float.MIN_VALUE, 1f, 1e30f, Float.POSITIVE_INFINITY};
        for (int i = 1; i < values.length; i++) {
            int lo = Packing.sortableInt(values[i - 1]);
            int hi = Packing.sortableInt(values[i]);
            assertTrue(lo <= hi, values[i - 1] + " should sort before " + values[i]);
        }
    }

    @Test
    void packingRoundTrips() {
        long packed = Packing.pack(3.5f, 123456);
        assertEquals(3.5f, Packing.distance(packed));
        assertEquals(123456, Packing.id(packed));
    }

    @Test
    void packedLongsOrderByDistanceThenId() {
        assertTrue(Packing.pack(1.0f, 999) < Packing.pack(2.0f, 0));
        assertTrue(Packing.pack(1.0f, 5) < Packing.pack(1.0f, 6));
    }

    @Test
    void keepsTheSmallestK() {
        Random rnd = new Random(11);
        int n = 5000;
        int k = 17;
        float[] all = new float[n];
        BoundedMaxHeap heap = new BoundedMaxHeap(k);
        for (int i = 0; i < n; i++) {
            all[i] = rnd.nextFloat() * 1000;
            heap.offer(all[i], i);
        }
        float[] expected = all.clone();
        Arrays.sort(expected);

        int[] ids = new int[k];
        float[] dists = new float[k];
        assertEquals(k, heap.drainAscending(ids, dists));
        for (int i = 0; i < k; i++) {
            assertEquals(expected[i], dists[i], "position " + i);
            assertEquals(expected[i], all[ids[i]], "id at position " + i);
        }
    }

    @Test
    void worstDistanceIsInfiniteUntilFull() {
        BoundedMaxHeap heap = new BoundedMaxHeap(3);
        assertEquals(Float.POSITIVE_INFINITY, heap.worstDistance());
        heap.offer(1f, 0);
        heap.offer(2f, 1);
        assertFalse(heap.isFull());
        assertEquals(Float.POSITIVE_INFINITY, heap.worstDistance());
        heap.offer(3f, 2);
        assertTrue(heap.isFull());
        assertEquals(3f, heap.worstDistance());
        assertFalse(heap.offer(4f, 3), "a candidate worse than the worst kept is rejected");
        assertTrue(heap.offer(0.5f, 4));
        assertEquals(2f, heap.worstDistance());
    }

    @Test
    void handlesFewerOffersThanCapacity() {
        BoundedMaxHeap heap = new BoundedMaxHeap(10);
        heap.offer(2f, 20);
        heap.offer(1f, 10);
        assertEquals(2, heap.size());
        assertArrayEquals(new int[]{10, 20}, heap.drainIdsAscending());
    }

    @Test
    void clearResetsWithoutReallocating() {
        BoundedMaxHeap heap = new BoundedMaxHeap(2);
        heap.offer(1f, 1);
        heap.offer(2f, 2);
        heap.clear();
        assertEquals(0, heap.size());
        heap.offer(9f, 9);
        assertArrayEquals(new int[]{9}, heap.drainIdsAscending());
    }
}
