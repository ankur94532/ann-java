package io.shashwat.ann.index;

import io.shashwat.ann.distance.Distance;
import io.shashwat.ann.distance.Metric;
import io.shashwat.ann.io.VectorDataset;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;

/**
 * The optimized HNSW. Algorithmically identical to {@link HnswIndex} — same level draws,
 * same candidate ordering, same selection, so the two build the same graph — and different
 * only in how that graph is stored and traversed.
 *
 * <p>Optimizations land here one at a time, each with a measured before/after in
 * {@code docs/hnsw-optimization.md}.
 *
 * <p><b>Step 1: flat {@code int[]} arenas.</b> The neighbour lists were
 * {@code ArrayList<Integer>}: to read one edge, the CPU loads an {@code ArrayList}, then its
 * backing {@code Object[]}, then an {@code Integer} box, then the {@code int} inside it —
 * four dependent loads, on four unrelated cache lines, per edge. A search touches thousands
 * of edges, so nearly all of its time is spent waiting on pointer chases that carry 4 bytes
 * of payload each.
 *
 * <p>Here a node's whole neighbour list is a contiguous run of {@code int}s. Layer 0 is one
 * fixed-stride array indexed arithmetically ({@code node * stride}), so finding a node's
 * edges is a multiply, not a load, and the whole list arrives in one or two cache lines.
 * Upper layers are rarer, so they share a single packed arena with a per-node offset.
 *
 * <p><b>Step 2: a versioned stamp array for the visited set.</b> {@code HashSet<Integer>}
 * charged an allocation, a box, a hash and a probe for every node the beam looked at, and
 * a fresh {@code HashSet} per layer per query on top. Here the visited set is one
 * {@code int} per node holding the number of the search that last touched it; "have I seen
 * this?" is one array read and one compare, and clearing the set for the next search is
 * incrementing a counter rather than touching a million entries.
 *
 * <p>The array is 4 MiB at a million nodes and is allocated lazily, but insertion runs the
 * same layer search that queries do, so it is allocated during the build and does show up
 * in the build-time memory measurement. It is scratch rather than index and is excluded
 * from {@link #estimatedBytes()}; the 4 MiB gap between the two figures is this array.
 */
public final class FastHnswIndex implements Hnsw {

    private final VectorDataset base;
    private final HnswConfig config;
    private final int dim;

    /** Degree caps: {@code m0} at layer 0, {@code m} above. */
    private final int m;
    private final int m0;

    /** One slot for the degree, then the neighbour slots. */
    private final int stride0;
    private final int strideUpper;

    /** Layer 0, for every node: {@code [degree, n0, n1, ...]} at {@code node * stride0}. */
    private final int[] layer0;

    /** Top layer of each node. */
    private final int[] level;

    /** Start of a node's upper-layer block in {@link #upper}, or -1 if it has none. */
    private final int[] upperOffset;

    /** Packed blocks of {@code level} runs of {@code strideUpper} ints, one run per layer. */
    private int[] upper;
    private int upperUsed;

    private int entryPoint = -1;
    private int topLayer = -1;
    private int count;

    private final Random rng;
    private long distanceComputations;

    /**
     * Per-node stamp of the search that last visited it, and the current search's number.
     * A node counts as visited when its stamp equals {@link #visitGeneration}.
     */
    private int[] visitedStamp;
    private int visitGeneration;

    public FastHnswIndex(VectorDataset base, HnswConfig config) {
        this.base = base;
        this.config = config;
        this.dim = base.dim();
        this.m = config.m();
        this.m0 = config.maxDegree(0);
        this.stride0 = m0 + 1;
        this.strideUpper = m + 1;
        this.layer0 = new int[Math.multiplyExact(base.size(), stride0)];
        this.level = new int[base.size()];
        this.upperOffset = new int[base.size()];
        this.upper = new int[Math.max(1024, base.size() / 4)];
        this.rng = new Random(config.seed());
    }

    /** Shares the graph; only the search-time beam width differs. */
    private FastHnswIndex(FastHnswIndex source, int efSearch) {
        this.base = source.base;
        this.config = source.config.withEfSearch(efSearch);
        this.dim = source.dim;
        this.m = source.m;
        this.m0 = source.m0;
        this.stride0 = source.stride0;
        this.strideUpper = source.strideUpper;
        this.layer0 = source.layer0;
        this.level = source.level;
        this.upperOffset = source.upperOffset;
        this.upper = source.upper;
        this.upperUsed = source.upperUsed;
        this.entryPoint = source.entryPoint;
        this.topLayer = source.topLayer;
        this.count = source.count;
        this.rng = new Random(config.seed());
        // visitedStamp is deliberately not shared: it is per-search scratch, and two views
        // of the same graph must not be able to see each other's generation counter.
    }

    @Override
    public FastHnswIndex withEfSearch(int efSearch) {
        return new FastHnswIndex(this, efSearch);
    }

    @Override
    public void buildAll(HnswIndex.ProgressListener progress) {
        int n = base.size();
        for (int i = 0; i < n; i++) {
            add(i);
            if (progress != null && ((i + 1) % 10_000 == 0 || i + 1 == n)) {
                progress.onProgress(i + 1, n);
            }
        }
    }

    // ---------------------------------------------------------------- arena access

    private int[] arena(int layer) {
        return layer == 0 ? layer0 : upper;
    }

    /** Index of the slot holding {@code node}'s degree in {@code layer}. */
    private int degreeSlot(int node, int layer) {
        return layer == 0
                ? node * stride0
                : upperOffset[node] + (layer - 1) * strideUpper;
    }

    private int degree(int node, int layer) {
        return arena(layer)[degreeSlot(node, layer)];
    }

    private void allocateUpper(int node, int nodeLevel) {
        if (nodeLevel == 0) {
            upperOffset[node] = -1;
            return;
        }
        int needed = nodeLevel * strideUpper;
        if (upperUsed + needed > upper.length) {
            int target = Math.max(upper.length * 2, upperUsed + needed);
            int[] grown = new int[target];
            System.arraycopy(upper, 0, grown, 0, upperUsed);
            upper = grown;
        }
        upperOffset[node] = upperUsed;
        upperUsed += needed;
    }

    private void setNeighbours(int node, int layer, List<Integer> neighbourIds) {
        int[] arena = arena(layer);
        int slot = degreeSlot(node, layer);
        int cap = config.maxDegree(layer);
        int n = Math.min(cap, neighbourIds.size());
        arena[slot] = n;
        for (int i = 0; i < n; i++) {
            arena[slot + 1 + i] = neighbourIds.get(i);
        }
    }

    private void appendNeighbour(int node, int layer, int neighbour) {
        int[] arena = arena(layer);
        int slot = degreeSlot(node, layer);
        arena[slot + 1 + arena[slot]] = neighbour;
        arena[slot]++;
    }

    // ---------------------------------------------------------------- insertion

    public void add(int id) {
        if (id != count) {
            throw new IllegalArgumentException("expected id " + count + ", got " + id);
        }
        int nodeLevel = randomLevel();
        level[id] = nodeLevel;
        allocateUpper(id, nodeLevel);
        count++;

        if (entryPoint == -1) {
            entryPoint = id;
            topLayer = nodeLevel;
            return;
        }

        int queryOffset = base.offset(id);
        int current = entryPoint;
        for (int layer = topLayer; layer > nodeLevel; layer--) {
            current = greedyDescend(base.data(), queryOffset, current, layer);
        }

        List<Integer> entryPoints = new ArrayList<>(1);
        entryPoints.add(current);
        for (int layer = Math.min(topLayer, nodeLevel); layer >= 0; layer--) {
            List<Candidate> found = searchLayer(base.data(), queryOffset, entryPoints,
                    config.efConstruction(), layer);
            List<Integer> selected = select(queryOffset, found, config.maxDegree(layer));
            link(id, selected, layer);

            entryPoints = new ArrayList<>(found.size());
            for (Candidate c : found) {
                entryPoints.add(c.id());
            }
        }

        if (nodeLevel > topLayer) {
            topLayer = nodeLevel;
            entryPoint = id;
        }
    }

    private void link(int id, List<Integer> selected, int layer) {
        setNeighbours(id, layer, selected);

        int cap = config.maxDegree(layer);
        int[] arena = arena(layer);
        for (int neighbour : selected) {
            int slot = degreeSlot(neighbour, layer);
            int deg = arena[slot];
            if (deg < cap) {
                appendNeighbour(neighbour, layer, id);
                continue;
            }
            // Full: re-select over the existing neighbours plus the new edge.
            int neighbourOffset = base.offset(neighbour);
            List<Candidate> candidates = new ArrayList<>(deg + 1);
            for (int i = 0; i < deg; i++) {
                int other = arena[slot + 1 + i];
                candidates.add(new Candidate(other, distance(base.data(), neighbourOffset, other)));
            }
            candidates.add(new Candidate(id, distance(base.data(), neighbourOffset, id)));
            candidates.sort(NEAREST_FIRST);
            setNeighbours(neighbour, layer, select(neighbourOffset, candidates, cap));
        }
    }

    private int randomLevel() {
        double u = 1.0 - rng.nextDouble();
        return (int) Math.floor(-Math.log(u) * config.levelMultiplier());
    }

    // ---------------------------------------------------------------- search

    @Override
    public int search(float[] query, int queryOffset, int k, int[] outIds, float[] outDistances) {
        if (entryPoint == -1) {
            return 0;
        }
        int current = entryPoint;
        for (int layer = topLayer; layer > 0; layer--) {
            current = greedyDescend(query, queryOffset, current, layer);
        }

        List<Integer> entryPoints = new ArrayList<>(1);
        entryPoints.add(current);
        List<Candidate> found = searchLayer(query, queryOffset, entryPoints,
                Math.max(config.efSearch(), k), 0);

        int n = Math.min(k, found.size());
        for (int i = 0; i < n; i++) {
            outIds[i] = found.get(i).id();
            if (outDistances != null) {
                outDistances[i] = found.get(i).distance();
            }
        }
        return n;
    }

    /** Walks downhill from {@code start} in one layer until no neighbour is closer. */
    private int greedyDescend(float[] query, int queryOffset, int start, int layer) {
        int current = start;
        float currentDistance = distance(query, queryOffset, current);
        boolean improved = true;
        while (improved) {
            improved = false;
            int[] arena = arena(layer);
            int slot = degreeSlot(current, layer);
            int deg = arena[slot];
            for (int i = 1; i <= deg; i++) {
                int neighbour = arena[slot + i];
                float d = distance(query, queryOffset, neighbour);
                if (d < currentDistance) {
                    currentDistance = d;
                    current = neighbour;
                    improved = true;
                }
            }
        }
        return current;
    }

    /**
     * Marks {@code node} visited for the current search.
     *
     * @return true if it had not been visited yet
     */
    private boolean visit(int node) {
        if (visitedStamp[node] == visitGeneration) {
            return false;
        }
        visitedStamp[node] = visitGeneration;
        return true;
    }

    /**
     * Starts a new search generation. On the one iteration in two billion where the counter
     * would wrap back onto a live stamp, the array is cleared and numbering restarts.
     */
    private void newVisitGeneration() {
        if (visitedStamp == null) {
            visitedStamp = new int[Math.max(1, base.size())];
        }
        if (visitGeneration == Integer.MAX_VALUE) {
            java.util.Arrays.fill(visitedStamp, 0);
            visitGeneration = 0;
        }
        visitGeneration++;
    }

    private List<Candidate> searchLayer(float[] query, int queryOffset,
                                        List<Integer> entryPoints, int ef, int layer) {
        newVisitGeneration();
        PriorityQueue<Candidate> candidates = new PriorityQueue<>(NEAREST_FIRST);
        PriorityQueue<Candidate> results = new PriorityQueue<>(NEAREST_FIRST.reversed());

        for (int entry : entryPoints) {
            if (visit(entry)) {
                float d = distance(query, queryOffset, entry);
                candidates.add(new Candidate(entry, d));
                results.add(new Candidate(entry, d));
            }
        }
        while (results.size() > ef) {
            results.poll();
        }

        while (!candidates.isEmpty()) {
            Candidate nearest = candidates.poll();
            if (results.size() >= ef && NEAREST_FIRST.compare(nearest, results.peek()) > 0) {
                break;
            }
            int[] arena = arena(layer);
            int slot = degreeSlot(nearest.id(), layer);
            int deg = arena[slot];
            for (int i = 1; i <= deg; i++) {
                int neighbour = arena[slot + i];
                if (!visit(neighbour)) {
                    continue;
                }
                float d = distance(query, queryOffset, neighbour);
                if (results.size() < ef || d < results.peek().distance()) {
                    Candidate c = new Candidate(neighbour, d);
                    candidates.add(c);
                    results.add(c);
                    if (results.size() > ef) {
                        results.poll();
                    }
                }
            }
        }

        List<Candidate> out = new ArrayList<>(results);
        out.sort(NEAREST_FIRST);
        return out;
    }

    // ---------------------------------------------------------------- neighbour selection

    private List<Integer> select(int queryOffset, List<Candidate> candidates, int max) {
        return switch (config.selection()) {
            case NEAREST_M -> {
                int n = Math.min(max, candidates.size());
                List<Integer> kept = new ArrayList<>(n);
                for (int i = 0; i < n; i++) {
                    kept.add(candidates.get(i).id());
                }
                yield kept;
            }
            case HEURISTIC -> {
                List<Integer> kept = new ArrayList<>(max);
                for (Candidate candidate : candidates) {
                    if (kept.size() >= max) {
                        break;
                    }
                    int candidateOffset = base.offset(candidate.id());
                    boolean redundant = false;
                    for (int keptId : kept) {
                        if (distance(base.data(), candidateOffset, keptId) < candidate.distance()) {
                            redundant = true;
                            break;
                        }
                    }
                    if (!redundant) {
                        kept.add(candidate.id());
                    }
                }
                yield kept;
            }
        };
    }

    // ---------------------------------------------------------------- helpers

    private float distance(float[] query, int queryOffset, int node) {
        distanceComputations++;
        return Distance.compute(config.metric(), query, queryOffset,
                base.data(), base.offset(node), dim);
    }

    private record Candidate(int id, float distance) {
    }

    private static final Comparator<Candidate> NEAREST_FIRST =
            Comparator.comparingDouble(Candidate::distance).thenComparingInt(Candidate::id);

    // ---------------------------------------------------------------- reporting

    @Override
    public String name() {
        return "hnsw(" + config.shortName() + ")";
    }

    @Override
    public int dim() {
        return dim;
    }

    @Override
    public int size() {
        return count;
    }

    @Override
    public Metric metric() {
        return config.metric();
    }

    @Override
    public HnswConfig config() {
        return config;
    }

    @Override
    public int topLayer() {
        return topLayer;
    }

    @Override
    public int entryPoint() {
        return entryPoint;
    }

    @Override
    public long distanceComputations() {
        return distanceComputations;
    }

    @Override
    public void resetDistanceComputations() {
        distanceComputations = 0;
    }

    @Override
    public int levelOf(int node) {
        return level[node];
    }

    @Override
    public int[] neighboursOf(int node, int layer) {
        if (layer > level[node]) {
            return new int[0];
        }
        int[] arena = arena(layer);
        int slot = degreeSlot(node, layer);
        int deg = arena[slot];
        int[] out = new int[deg];
        System.arraycopy(arena, slot + 1, out, 0, deg);
        return out;
    }

    @Override
    public long edgeCount() {
        long edges = 0;
        for (int node = 0; node < count; node++) {
            for (int layer = 0; layer <= level[node]; layer++) {
                edges += degree(node, layer);
            }
        }
        return edges;
    }

    /**
     * The arenas plus the two per-node int arrays. Exact, because everything the index owns
     * is a primitive array of known length — which is itself one of the benefits of the flat
     * representation over the boxed one.
     */
    @Override
    public long estimatedBytes() {
        return (long) layer0.length * Integer.BYTES
                + (long) upperUsed * Integer.BYTES
                + (long) level.length * Integer.BYTES
                + (long) upperOffset.length * Integer.BYTES;
    }
}
