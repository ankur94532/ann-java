package io.shashwat.ann.index;

import io.shashwat.ann.distance.Metric;

/**
 * Build- and search-time parameters for {@link HnswIndex}.
 *
 * @param m              degree budget at layers above 0. Layer 0 gets {@code 2*m}, which
 *                       is where nearly all the search work happens and where the extra
 *                       connectivity pays for itself.
 * @param efConstruction width of the beam used while inserting. Larger means each new
 *                       node sees more of the graph before choosing its edges, so the
 *                       edges are better; it costs build time, not query time.
 * @param efSearch       width of the beam used while querying. This is the recall/latency
 *                       dial and the only parameter that can be changed after the build.
 * @param selection      which neighbour-selection strategy to use.
 * @param metric         distance metric.
 * @param seed           seed for the level-assignment RNG; fixed so builds reproduce.
 */
public record HnswConfig(int m, int efConstruction, int efSearch,
                         NeighbourSelection selection, Metric metric, long seed) {

    public HnswConfig {
        if (m < 2) {
            throw new IllegalArgumentException("m must be at least 2, got " + m);
        }
        if (efConstruction < 1) {
            throw new IllegalArgumentException("efConstruction must be positive");
        }
        if (efSearch < 1) {
            throw new IllegalArgumentException("efSearch must be positive");
        }
    }

    public static HnswConfig of(int m, int efConstruction, int efSearch) {
        return new HnswConfig(m, efConstruction, efSearch,
                NeighbourSelection.HEURISTIC, Metric.L2, 42L);
    }

    public HnswConfig withEfSearch(int newEfSearch) {
        return new HnswConfig(m, efConstruction, newEfSearch, selection, metric, seed);
    }

    public HnswConfig withSelection(NeighbourSelection newSelection) {
        return new HnswConfig(m, efConstruction, efSearch, newSelection, metric, seed);
    }

    /** Degree cap for a given layer. */
    public int maxDegree(int layer) {
        return layer == 0 ? 2 * m : m;
    }

    /**
     * The {@code mL} of the paper. Levels are drawn from an exponential distribution
     * scaled by this, which makes layer sizes decay by a factor of {@code m} per level and
     * so gives the descent through the layers a cost logarithmic in the number of nodes.
     */
    public double levelMultiplier() {
        return 1.0 / Math.log(m);
    }

    public String shortName() {
        return "M=" + m + ",efC=" + efConstruction + ",ef=" + efSearch
                + (selection == NeighbourSelection.HEURISTIC ? "" : ",sel=nearestM");
    }
}
