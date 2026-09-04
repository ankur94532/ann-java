package io.shashwat.ann.bench;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class RecallTest {

    @Test
    void perfectRecallWhenIdsMatch() {
        int[] found = {1, 2, 3, 4, 5, 6};
        int[] truth = {1, 2, 3, 4, 5, 6};
        assertEquals(1.0, Recall.mean(found, 3, truth, 3, 2, 3));
    }

    @Test
    void orderDoesNotMatter() {
        int[] found = {3, 1, 2};
        int[] truth = {1, 2, 3};
        assertEquals(1.0, Recall.mean(found, 3, truth, 3, 1, 3));
        assertEquals(0.0, Recall.exactSequenceMatchRate(found, 3, truth, 3, 1, 3));
    }

    @Test
    void countsPartialOverlap() {
        int[] found = {1, 2, 99, 98};
        int[] truth = {1, 2, 3, 4};
        assertEquals(0.5, Recall.mean(found, 4, truth, 4, 1, 4));
    }

    @Test
    void readsOnlyTheFirstKOfAWiderTruthRow() {
        // Ground-truth rows ship 100 wide; scoring k=2 must ignore columns 2 and 3.
        int[] found = {10, 11};
        int[] truth = {10, 11, 12, 13};
        assertEquals(1.0, Recall.mean(found, 2, truth, 4, 1, 2));

        int[] foundDeeper = {12, 13};
        assertEquals(0.0, Recall.mean(foundDeeper, 2, truth, 4, 1, 2));
    }

    @Test
    void averagesAcrossQueries() {
        int[] found = {1, 2, 7, 8};
        int[] truth = {1, 2, 3, 4};
        float[] perQuery = Recall.perQuery(found, 2, truth, 2, 2, 2);
        assertArrayEquals(new float[]{1.0f, 0.0f}, perQuery);
        assertEquals(0.5, Recall.mean(perQuery), 1e-9);
    }
}
