package io.shashwat.ann.index;

import java.util.Arrays;

/**
 * Keeps the {@code k} smallest (distance, id) pairs it is offered.
 *
 * <p>Implemented as a max-heap over packed longs: the root is the worst of the current
 * best {@code k}, so deciding whether a candidate is worth keeping is one comparison
 * against {@code heap[0]}. That comparison is the innermost branch of every scan in this
 * codebase, which is why this is a primitive heap and not a {@code PriorityQueue}.
 */
public final class BoundedMaxHeap {

    private final long[] heap;
    private final int capacity;
    private int size;

    public BoundedMaxHeap(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive, got " + capacity);
        }
        this.capacity = capacity;
        this.heap = new long[capacity];
    }

    public int size() {
        return size;
    }

    public int capacity() {
        return capacity;
    }

    public boolean isFull() {
        return size == capacity;
    }

    public void clear() {
        size = 0;
    }

    /** Distance of the current worst kept entry; {@link Float#POSITIVE_INFINITY} if not full. */
    public float worstDistance() {
        return size < capacity ? Float.POSITIVE_INFINITY : Packing.distance(heap[0]);
    }

    /** @return true if the pair was kept. */
    public boolean offer(float distance, int id) {
        long packed = Packing.pack(distance, id);
        if (size < capacity) {
            heap[size] = packed;
            siftUp(size++);
            return true;
        }
        if (packed < heap[0]) {
            heap[0] = packed;
            siftDown(0);
            return true;
        }
        return false;
    }

    /**
     * Writes the kept ids, nearest first, into {@code outIds} (and distances into
     * {@code outDistances} when it is non-null). Destroys the heap.
     *
     * @return the number of entries written
     */
    public int drainAscending(int[] outIds, float[] outDistances) {
        Arrays.sort(heap, 0, size);
        for (int i = 0; i < size; i++) {
            outIds[i] = Packing.id(heap[i]);
            if (outDistances != null) {
                outDistances[i] = Packing.distance(heap[i]);
            }
        }
        return size;
    }

    /** Convenience for tests and cold paths. */
    public int[] drainIdsAscending() {
        int[] ids = new int[size];
        drainAscending(ids, null);
        return ids;
    }

    private void siftUp(int i) {
        long value = heap[i];
        while (i > 0) {
            int parent = (i - 1) >>> 1;
            if (heap[parent] >= value) {
                break;
            }
            heap[i] = heap[parent];
            i = parent;
        }
        heap[i] = value;
    }

    private void siftDown(int i) {
        long value = heap[i];
        int half = size >>> 1;
        while (i < half) {
            int child = (i << 1) + 1;
            int right = child + 1;
            if (right < size && heap[right] > heap[child]) {
                child = right;
            }
            if (heap[child] <= value) {
                break;
            }
            heap[i] = heap[child];
            i = child;
        }
        heap[i] = value;
    }
}
