package io.shashwat.ann.index;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.function.IntFunction;

/**
 * Structural health of a layer-0 proximity graph.
 *
 * <p>Recall lost to graph structure cannot be bought back with {@code efSearch}, so it is
 * worth measuring separately from recall itself. Two distinct failures show up here.
 *
 * <ul>
 *   <li><b>Orphans</b> — nodes with no incoming edge. Nothing points at them, so no search
 *       can ever return them, at any beam width. They arise from the pruning heuristic:
 *       an outlier far from every cluster gets few edges of its own, and the back-edges
 *       pointing at it are the first thing dropped when its neighbours fill up. This is a
 *       property of HNSW, not a bug, and it puts a hard per-dataset floor under the recall
 *       loss.
 *   <li><b>Fragmentation</b> — nodes that have incoming edges but sit in a component the
 *       entry point cannot reach. This is the failure mode of nearest-M selection: every
 *       node spends its edge budget inside its own cluster, so the clusters stop being
 *       joined to each other and a search is confined to whichever one it lands in.
 * </ul>
 */
public record GraphDiagnostics(int nodes, int reachable, int orphans, int isolated,
                               double meanDegree, int minDegree, int maxDegree) {

    /**
     * @param neighbours layer-0 neighbour list of a node
     */
    public static GraphDiagnostics of(int nodes, int entryPoint, IntFunction<int[]> neighbours) {
        boolean[] seen = new boolean[nodes];
        int[] inDegree = new int[nodes];
        long degreeSum = 0;
        int minDegree = Integer.MAX_VALUE;
        int maxDegree = 0;

        for (int node = 0; node < nodes; node++) {
            int[] list = neighbours.apply(node);
            degreeSum += list.length;
            minDegree = Math.min(minDegree, list.length);
            maxDegree = Math.max(maxDegree, list.length);
            for (int n : list) {
                inDegree[n]++;
            }
        }

        int reachable = 0;
        if (nodes > 0 && entryPoint >= 0) {
            Deque<Integer> stack = new ArrayDeque<>();
            stack.push(entryPoint);
            seen[entryPoint] = true;
            reachable = 1;
            while (!stack.isEmpty()) {
                for (int n : neighbours.apply(stack.pop())) {
                    if (!seen[n]) {
                        seen[n] = true;
                        reachable++;
                        stack.push(n);
                    }
                }
            }
        }

        int orphans = 0;
        int isolated = 0;
        for (int node = 0; node < nodes; node++) {
            if (inDegree[node] == 0 && node != entryPoint) {
                orphans++;
            }
            if (!seen[node]) {
                isolated++;
            }
        }
        return new GraphDiagnostics(nodes, reachable, orphans, isolated,
                nodes == 0 ? 0 : (double) degreeSum / nodes,
                minDegree == Integer.MAX_VALUE ? 0 : minDegree, maxDegree);
    }

    public double reachableFraction() {
        return nodes == 0 ? 1.0 : reachable / (double) nodes;
    }

    /** Isolated nodes that are not orphans, i.e. genuine disconnected components. */
    public int fragmented() {
        return Math.max(0, isolated - orphans);
    }

    @Override
    public String toString() {
        return String.format(
                "layer 0: %,d nodes, %.4f reachable from the entry point, %,d orphans "
                        + "(no in-edge), %,d isolated, degree min/mean/max = %d/%.1f/%d",
                nodes, reachableFraction(), orphans, isolated, minDegree, meanDegree, maxDegree);
    }
}
