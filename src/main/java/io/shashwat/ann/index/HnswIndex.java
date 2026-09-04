package io.shashwat.ann.index;

import io.shashwat.ann.distance.Distance;
import io.shashwat.ann.io.VectorDataset;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

/**
 * Hierarchical Navigable Small World graph (Malkov &amp; Yashunin, 2016).
 *
 * <p><b>This is the correctness reference — deliberately the naive implementation.</b>
 * Boxed neighbour lists, a {@link HashSet} for the visited set, and a
 * {@link PriorityQueue} of objects for both heaps. It is the "before" column of the
 * optimization table in {@code docs/hnsw-optimization.md}; every change in
 * {@link FastHnswIndex} is measured against the numbers this class produces, so it has to
 * exist and has to stay correct.
 *
 * <p>The structure is a stack of proximity graphs. Every node is in layer 0; a node is
 * additionally in layers 1..L for an L drawn from an exponential distribution, so each
 * layer up holds roughly 1/M of the layer below. A search enters at the top, where the
 * graph is tiny and the hops are long, and greedily walks downhill; each descent lands it
 * in the right neighbourhood of the next layer down, so the work is logarithmic in n
 * rather than linear.
 */
public final class HnswIndex implements VectorIndex {

    private final VectorDataset base;
    private final HnswConfig config;
    private final int dim;

    /** {@code graph.get(node).get(layer)} is node's neighbour list in that layer. */
    private final List<List<List<Integer>>> graph;

    /** Highest layer each node belongs to. */
    private final List<Integer> nodeLevel;

    private int entryPoint = -1;
    private int topLayer = -1;
    private int count;

    private final Random rng;

    /** Counters for the optimization write-up; not used by the algorithm. */
    private long distanceComputations;

    public HnswIndex(VectorDataset base, HnswConfig config) {
        this(base, config, new ArrayList<>(base.size()), new ArrayList<>(base.size()));
    }

    private HnswIndex(VectorDataset base, HnswConfig config,
                      List<List<List<Integer>>> graph, List<Integer> nodeLevel) {
        this.base = base;
        this.config = config;
        this.dim = base.dim();
        this.graph = graph;
        this.nodeLevel = nodeLevel;
        this.rng = new Random(config.seed());
    }

    /** Inserts every vector of the base set, in file order. */
    public void buildAll(ProgressListener progress) {
        int n = base.size();
        for (int i = 0; i < n; i++) {
            add(i);
            if (progress != null && ((i + 1) % 10_000 == 0 || i + 1 == n)) {
                progress.onProgress(i + 1, n);
            }
        }
    }

    // ---------------------------------------------------------------- insertion

    /**
     * Inserts base vector {@code id}, which must be the next id in order.
     *
     * <p>Two phases. First a cheap greedy descent from the entry point down to the layer
     * just above the new node's own level, which costs a handful of distance computations
     * per layer and exists only to find a good starting point. Then, from the new node's
     * level down to 0, a full {@code efConstruction}-wide beam search per layer, whose
     * results are the candidate neighbours to link to.
     */
    public void add(int id) {
        if (id != count) {
            throw new IllegalArgumentException("expected id " + count + ", got " + id);
        }
        int level = randomLevel();

        List<List<Integer>> layers = new ArrayList<>(level + 1);
        for (int l = 0; l <= level; l++) {
            layers.add(new ArrayList<>());
        }
        graph.add(layers);
        nodeLevel.add(level);
        count++;

        if (entryPoint == -1) {
            entryPoint = id;
            topLayer = level;
            return;
        }

        int queryOffset = base.offset(id);
        int current = entryPoint;
        float currentDistance = distance(queryOffset, current);

        // Phase 1: greedy descent through the layers this node will not join.
        for (int layer = topLayer; layer > level; layer--) {
            boolean improved = true;
            while (improved) {
                improved = false;
                for (int neighbour : neighbours(current, layer)) {
                    float d = distance(queryOffset, neighbour);
                    if (d < currentDistance) {
                        currentDistance = d;
                        current = neighbour;
                        improved = true;
                    }
                }
            }
        }

        // Phase 2: beam search and link, from this node's own level down to 0.
        List<Integer> entryPoints = new ArrayList<>();
        entryPoints.add(current);
        for (int layer = Math.min(topLayer, level); layer >= 0; layer--) {
            List<Candidate> found = searchLayer(queryOffset, entryPoints,
                    config.efConstruction(), layer);
            List<Integer> selected = select(queryOffset, found, config.maxDegree(layer));
            link(id, selected, layer);

            entryPoints = new ArrayList<>(found.size());
            for (Candidate c : found) {
                entryPoints.add(c.id());
            }
        }

        if (level > topLayer) {
            topLayer = level;
            entryPoint = id;
        }
    }

    /**
     * Adds the edges {@code id -> selected} and each {@code selected -> id}, then repairs
     * any neighbour that has gone over its degree budget.
     *
     * <p>The edges have to go both ways or the graph is not navigable: an edge that only
     * points away from an old node means a search that reaches the old node can never
     * discover the new one. Adding the back-edge is what can push an existing node over
     * its cap, and the repair runs the same selection strategy over that node's full
     * neighbour list — not a "drop the farthest" rule, because under the heuristic the
     * farthest neighbour may be the only edge covering its direction.
     */
    private void link(int id, List<Integer> selected, int layer) {
        List<Integer> own = neighbours(id, layer);
        own.addAll(selected);

        int cap = config.maxDegree(layer);
        for (int neighbour : selected) {
            List<Integer> theirs = neighbours(neighbour, layer);
            theirs.add(id);
            if (theirs.size() > cap) {
                int neighbourOffset = base.offset(neighbour);
                List<Candidate> candidates = new ArrayList<>(theirs.size());
                for (int other : theirs) {
                    candidates.add(new Candidate(other, distance(neighbourOffset, other)));
                }
                candidates.sort(NEAREST_FIRST);
                List<Integer> kept = select(neighbourOffset, candidates, cap);
                theirs.clear();
                theirs.addAll(kept);
            }
        }
    }

    /**
     * Draws a node's top layer from a decaying exponential:
     * {@code floor(-ln(U(0,1]) * mL)} with {@code mL = 1/ln(M)}.
     *
     * <p>With that scale the probability of reaching layer l falls by a factor of M each
     * level, so layer l holds about {@code n / M^l} nodes and the number of layers is
     * {@code ~log_M(n)}. The layers are then a coarse-to-fine index over the same points,
     * which is the whole reason the descent costs a logarithmic number of hops.
     */
    private int randomLevel() {
        double u = 1.0 - rng.nextDouble(); // in (0, 1], so the log is finite
        return (int) Math.floor(-Math.log(u) * config.levelMultiplier());
    }

    // ---------------------------------------------------------------- search

    @Override
    public int search(float[] query, int queryOffset, int k, int[] outIds, float[] outDistances) {
        if (entryPoint == -1) {
            return 0;
        }
        int current = entryPoint;
        float currentDistance = distance(query, queryOffset, current);

        for (int layer = topLayer; layer > 0; layer--) {
            boolean improved = true;
            while (improved) {
                improved = false;
                for (int neighbour : neighbours(current, layer)) {
                    float d = distance(query, queryOffset, neighbour);
                    if (d < currentDistance) {
                        currentDistance = d;
                        current = neighbour;
                        improved = true;
                    }
                }
            }
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

    private List<Candidate> searchLayer(int queryOffset, List<Integer> entryPoints,
                                        int ef, int layer) {
        return searchLayer(base.data(), queryOffset, entryPoints, ef, layer);
    }

    /**
     * Best-first search of one layer, returning the {@code ef} closest nodes it found,
     * nearest first.
     *
     * <p>Two heaps. {@code candidates} is a min-heap of nodes discovered but not yet
     * expanded — the frontier, popped nearest-first so the search always follows the most
     * promising direction. {@code results} is a max-heap capped at {@code ef} holding the
     * best answers seen so far, keyed so that its root is the <em>worst</em> of them,
     * which makes "is this new node good enough to keep?" a single comparison.
     *
     * <p><b>Termination.</b> The loop stops as soon as the nearest unexpanded candidate is
     * farther away than the worst kept result. At that moment every remaining candidate is
     * also farther (the frontier is popped in increasing distance), and the graph edges are
     * assumed to lead only to points not much nearer, so no unexpanded branch can improve
     * the result set — continuing would only re-confirm what is already held. This is the
     * approximation in HNSW: the assumption is a property of a well-built proximity graph,
     * not a proof, and it is exactly why recall is not 1.0.
     *
     * <p>So {@code ef} is not "how many results to return" — it is how much worse than the
     * current best a node may be and still be worth exploring. A larger {@code ef} keeps a
     * worse worst-result, which raises the bar for termination, which lets the search push
     * through a ridge of slightly-worse nodes to reach a better basin behind it. That is
     * the whole recall/latency dial.
     */
    private List<Candidate> searchLayer(float[] query, int queryOffset,
                                        List<Integer> entryPoints, int ef, int layer) {
        Set<Integer> visited = new HashSet<>();
        PriorityQueue<Candidate> candidates = new PriorityQueue<>(NEAREST_FIRST);
        PriorityQueue<Candidate> results = new PriorityQueue<>(NEAREST_FIRST.reversed());

        for (int entry : entryPoints) {
            if (visited.add(entry)) {
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
            for (int neighbour : neighbours(nearest.id(), layer)) {
                if (!visited.add(neighbour)) {
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
            case NEAREST_M -> selectNearest(candidates, max);
            case HEURISTIC -> selectHeuristic(queryOffset, candidates, max);
        };
    }

    /** Keep the {@code max} nearest candidates. {@code candidates} is nearest-first. */
    private static List<Integer> selectNearest(List<Candidate> candidates, int max) {
        int n = Math.min(max, candidates.size());
        List<Integer> kept = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            kept.add(candidates.get(i).id());
        }
        return kept;
    }

    /**
     * Algorithm 4 of the HNSW paper: keep a candidate only if it is closer to the node
     * being linked than to anything already kept.
     *
     * <p>Read the rejection test as a question about redundancy. If some already-kept
     * neighbour {@code r} sits closer to candidate {@code e} than {@code e} sits to the
     * node {@code q}, then {@code e} is in {@code r}'s neighbourhood, and a search that
     * reaches {@code q} can get to {@code e} by hopping to {@code r} first. Spending one
     * of {@code q}'s few edge slots on {@code e} buys a shortcut that was already there.
     * Spending it instead on a candidate that no kept neighbour covers buys reachability
     * in a direction that had none.
     *
     * <p>The geometric statement is that the kept set approximates a relative neighbourhood
     * graph: for every kept edge {@code (q,e)} the lens-shaped region between {@code q} and
     * {@code e} contains no other kept neighbour. Edges therefore fan out over directions
     * instead of clumping, and — the part that matters most — an edge that leaves a dense
     * cluster survives, because inside the cluster there is nothing closer to the far
     * endpoint than the far endpoint is to {@code q}.
     *
     * <p>Without it (see {@link NeighbourSelection#NEAREST_M}) a node in a dense cluster
     * spends all M edges on cluster-mates. Greedy search entering that cluster finds no
     * edge that reduces the distance and stops in a local minimum, so recall collapses on
     * exactly the clustered data that ANN is used for. The failure is not gradual: it shows
     * up as a hard recall ceiling that more {@code efSearch} cannot lift, because the
     * problem is that the necessary edge does not exist, not that the beam is too narrow.
     */
    private List<Integer> selectHeuristic(int queryOffset, List<Candidate> candidates, int max) {
        List<Integer> kept = new ArrayList<>(max);
        for (Candidate candidate : candidates) {
            if (kept.size() >= max) {
                break;
            }
            int candidateOffset = base.offset(candidate.id());
            boolean redundant = false;
            for (int keptId : kept) {
                if (distance(candidateOffset, keptId) < candidate.distance()) {
                    redundant = true;
                    break;
                }
            }
            if (!redundant) {
                kept.add(candidate.id());
            }
        }
        return kept;
    }

    // ---------------------------------------------------------------- helpers

    private List<Integer> neighbours(int node, int layer) {
        List<List<Integer>> layers = graph.get(node);
        return layer < layers.size() ? layers.get(layer) : List.of();
    }

    private float distance(int queryOffset, int node) {
        return distance(base.data(), queryOffset, node);
    }

    private float distance(float[] query, int queryOffset, int node) {
        distanceComputations++;
        return Distance.compute(config.metric(), query, queryOffset,
                base.data(), base.offset(node), dim);
    }

    /**
     * Ordering is by distance and then by id, never by distance alone. The tie-break is
     * not cosmetic: it makes insertion fully deterministic, so the naive and optimized
     * implementations build the same graph and the optimization table measures speed
     * rather than an accidental difference in graph quality.
     */
    private record Candidate(int id, float distance) {
    }

    private static final Comparator<Candidate> NEAREST_FIRST =
            Comparator.comparingDouble(Candidate::distance).thenComparingInt(Candidate::id);

    // ---------------------------------------------------------------- reporting

    @Override
    public String name() {
        return "hnsw-naive(" + config.shortName() + ")";
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
    public io.shashwat.ann.distance.Metric metric() {
        return config.metric();
    }

    public HnswConfig config() {
        return config;
    }

    /**
     * A view of this graph searched at a different beam width. The graph itself is shared,
     * not copied — {@code efSearch} is a search-time parameter only, which is what lets one
     * build serve a whole recall/latency curve.
     */
    public HnswIndex withEfSearch(int efSearch) {
        HnswIndex view = new HnswIndex(base, config.withEfSearch(efSearch), graph, nodeLevel);
        view.entryPoint = entryPoint;
        view.topLayer = topLayer;
        view.count = count;
        return view;
    }

    public int topLayer() {
        return topLayer;
    }

    public int entryPoint() {
        return entryPoint;
    }

    public long distanceComputations() {
        return distanceComputations;
    }

    /** Highest layer {@code node} belongs to. */
    public int levelOf(int node) {
        return nodeLevel.get(node);
    }

    /** Copy of a node's neighbour list in one layer. For tests and diagnostics only. */
    public int[] neighboursOf(int node, int layer) {
        List<Integer> list = neighbours(node, layer);
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }

    /** Total number of directed edges, and the number in layer 0. */
    public long edgeCount() {
        long edges = 0;
        for (List<List<Integer>> layers : graph) {
            for (List<Integer> list : layers) {
                edges += list.size();
            }
        }
        return edges;
    }

    /**
     * Very rough: 16 bytes of {@code Integer} per edge plus ~40 of {@code ArrayList}
     * overhead per list. The naive representation's footprint is not worth measuring
     * carefully — that it is roughly an order of magnitude off the flat version is the
     * entire point of replacing it.
     */
    @Override
    public long estimatedBytes() {
        long lists = 0;
        for (List<List<Integer>> layers : graph) {
            lists += layers.size();
        }
        return edgeCount() * 20 + lists * 48;
    }

    @FunctionalInterface
    public interface ProgressListener {
        void onProgress(int completed, int total);
    }
}
