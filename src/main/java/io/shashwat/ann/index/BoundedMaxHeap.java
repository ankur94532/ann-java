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
    private int capacity;
    private int size;

    public BoundedMaxHeap(int capacity) {
        this(capacity, capacity);
    }

    /**
     * A heap whose array is sized for {@code maxCapacity} but which can be
     * {@link #reset(int) reset} to any smaller capacity.
     *
     * <p>HNSW needs both {@code efConstruction} and {@code efSearch} widths from the same
     * object, and reallocating per search would put an allocation back into the timed loop
     * that the primitive heap exists to remove.
     */
    public static BoundedMaxHeap withMaxCapacity(int maxCapacity) {
        return new BoundedMaxHeap(maxCapacity, maxCapacity);
    }

    private BoundedMaxHeap(int capacity, int maxCapacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive, got " + capacity);
        }
        this.capacity = capacity;
        this.heap = new long[maxCapacity];
    }

    /** Empties the heap and sets a new capacity, reusing the existing array. */
    public void reset(int newCapacity) {
        if (newCapacity <= 0 || newCapacity > heap.length) {
            throw new IllegalArgumentException("capacity " + newCapacity
                    + " must be in [1, " + heap.length + "]");
        }
        this.capacity = newCapacity;
        this.size = 0;
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

    /**
     * The worst kept entry, still packed, so a caller holding another packed value can
     * compare the two on the full (distance, id) order with one long comparison.
     */
    public long worstPacked() {
        if (size == 0) {
            throw new IllegalStateException("heap is empty");
        }
        return heap[0];
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

    /**
     * Writes the kept entries into {@code out}, still packed, nearest first. Destroys the
     * heap. Leaving them packed lets the caller pass them straight to code that compares
     * on the (distance, id) order without unpacking and repacking.
     *
     * @return the number of entries written
     */
    public int drainAscendingPacked(long[] out) {
        Arrays.sort(heap, 0, size);
        System.arraycopy(heap, 0, out, 0, size);
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
