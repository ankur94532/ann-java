package io.shashwat.ann.index;

import org.junit.jupiter.api.Test;

import java.util.PriorityQueue;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LongMinHeapTest {

    @Test
    void popsInAscendingOrder() {
        Random rnd = new Random(2);
        LongMinHeap heap = new LongMinHeap(4);
        PriorityQueue<Long> reference = new PriorityQueue<>();
        for (int i = 0; i < 5000; i++) {
            float d = rnd.nextFloat() * 100;
            heap.push(d, i);
            reference.add(Packing.pack(d, i));
        }
        assertEquals(reference.size(), heap.size());
        long previous = Long.MIN_VALUE;
        while (!heap.isEmpty()) {
            long popped = heap.pop();
            assertEquals(reference.poll().longValue(), popped);
            assertTrue(popped >= previous, "heap must pop in ascending order");
            previous = popped;
        }
        assertTrue(reference.isEmpty());
    }

    /** Interleaved pushes and pops are what the search loop actually does. */
    @Test
    void handlesInterleavedPushAndPop() {
        Random rnd = new Random(5);
        LongMinHeap heap = new LongMinHeap(4);
        PriorityQueue<Long> reference = new PriorityQueue<>();
        for (int step = 0; step < 20_000; step++) {
            if (reference.isEmpty() || rnd.nextInt(3) != 0) {
                float d = rnd.nextFloat() * 1000;
                int id = rnd.nextInt(1 << 20);
                heap.push(d, id);
                reference.add(Packing.pack(d, id));
            } else {
                assertEquals(reference.peek().longValue(), heap.peek());
                assertEquals(reference.poll().longValue(), heap.pop());
            }
            assertEquals(reference.size(), heap.size());
        }
    }

    @Test
    void clearReusesTheArray() {
        LongMinHeap heap = new LongMinHeap(4);
        for (int i = 0; i < 100; i++) {
            heap.push(i, i);
        }
        int capacity = heap.capacity();
        heap.clear();
        assertEquals(0, heap.size());
        assertTrue(heap.isEmpty());
        heap.push(1f, 1);
        assertEquals(Packing.pack(1f, 1), heap.peek());
        assertEquals(capacity, heap.capacity(), "clear must not shrink or reallocate");
    }

    @Test
    void rejectsPopAndPeekOnAnEmptyHeap() {
        LongMinHeap heap = new LongMinHeap(4);
        assertThrows(IllegalStateException.class, heap::pop);
        assertThrows(IllegalStateException.class, heap::peek);
    }
}
