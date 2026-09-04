package io.shashwat.ann.index;

import java.util.Arrays;

/**
 * A growable min-heap of packed (distance, id) longs — the search frontier.
 *
 * <p>Unlike {@link BoundedMaxHeap} this one has no capacity limit: the frontier holds
 * every node discovered but not yet expanded, and how large that gets is a property of the
 * graph, not a parameter. It is reused across searches via {@link #clear()}, so a steady
 * state is reached after a handful of queries and the search loop stops allocating.
 */
public final class LongMinHeap {

    private long[] heap;
    private int size;

    public LongMinHeap(int initialCapacity) {
        this.heap = new long[Math.max(4, initialCapacity)];
    }

    public int size() {
        return size;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public void clear() {
        size = 0;
    }

    /** Largest number of entries held at once since construction, for diagnostics. */
    public int capacity() {
        return heap.length;
    }

    public void push(float distance, int id) {
        push(Packing.pack(distance, id));
    }

    public void push(long packed) {
        if (size == heap.length) {
            heap = Arrays.copyOf(heap, heap.length * 2);
        }
        int i = size++;
        while (i > 0) {
            int parent = (i - 1) >>> 1;
            if (heap[parent] <= packed) {
                break;
            }
            heap[i] = heap[parent];
            i = parent;
        }
        heap[i] = packed;
    }

    /** The smallest entry, without removing it. */
    public long peek() {
        if (size == 0) {
            throw new IllegalStateException("heap is empty");
        }
        return heap[0];
    }

    public long pop() {
        if (size == 0) {
            throw new IllegalStateException("heap is empty");
        }
        long top = heap[0];
        long last = heap[--size];
        if (size > 0) {
            int i = 0;
            int half = size >>> 1;
            while (i < half) {
                int child = (i << 1) + 1;
                int right = child + 1;
                if (right < size && heap[right] < heap[child]) {
                    child = right;
                }
                if (heap[child] >= last) {
                    break;
                }
                heap[i] = heap[child];
                i = child;
            }
            heap[i] = last;
        }
        return top;
    }
}
