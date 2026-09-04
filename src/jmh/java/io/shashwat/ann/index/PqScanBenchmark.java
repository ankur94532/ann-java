package io.shashwat.ann.index;

import io.shashwat.ann.distance.ScalarDistance;
import io.shashwat.ann.distance.SimdDistance;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * The IVF-PQ inner loops, isolated.
 *
 * <p>An IVF-PQ query rebuilds a {@code m x 256} lookup table once per probed list, because
 * the query residual depends on which centroid's list is being scanned. That table costs
 * {@code m * 256} squared-distance computations over subvectors of {@code dim/m}
 * dimensions — 4 or 8 elements at the sizes this project sweeps.
 *
 * <p>Those are very short vectors, and {@code docs/kernels.md} already measured a fixed
 * prologue/epilogue cost of roughly 4 ns for the four-accumulator SIMD kernel: zeroing four
 * accumulators, an add tree, and a horizontal {@code reduceLanes}. At 4 elements of actual
 * work that overhead cannot possibly amortise, so this benchmark asks whether calling the
 * SIMD kernel here is a pessimisation rather than an optimisation.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "-Xmx2g"})
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
public class PqScanBenchmark {

    private static final int KSUB = ProductQuantizer.CENTROIDS_PER_SUBSPACE;
    private static final int DIM = 128;

    /** Subvectors per code; dim/m is then 16, 8, 4 and 2 elements. */
    @Param({"8", "16", "32", "64"})
    public int m;

    private int subDim;
    private float[] codebooks;
    private float[] codebooksT;
    private float[] query;
    private float[] table;

    @Setup
    public void setup() {
        Random rnd = new Random(4242 + m);
        subDim = DIM / m;
        codebooks = new float[m * KSUB * subDim];
        for (int i = 0; i < codebooks.length; i++) {
            codebooks[i] = rnd.nextFloat() * 40 - 20;
        }
        query = new float[DIM];
        for (int i = 0; i < DIM; i++) {
            query[i] = rnd.nextFloat() * 40 - 20;
        }
        table = new float[m * KSUB];

        // Dimension-major within each subspace: (j*subDim + d)*KSUB + c.
        codebooksT = new float[m * subDim * KSUB];
        for (int j = 0; j < m; j++) {
            for (int c = 0; c < KSUB; c++) {
                for (int d = 0; d < subDim; d++) {
                    codebooksT[(j * subDim + d) * KSUB + c] =
                            codebooks[(j * KSUB + c) * subDim + d];
                }
            }
        }
    }

    /**
     * The shipped implementation: vectorised across the codeword axis, so the vector width
     * no longer has to divide the subvector length and there is one horizontal reduction
     * per lane-width of codewords instead of one per codeword.
     */
    @Benchmark
    public float[] lookupTableTransposed() {
        var species = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;
        int lanes = species.length();
        for (int j = 0; j < m; j++) {
            int base = j * subDim * KSUB;
            int row = j * KSUB;
            int src = j * subDim;
            for (int c = 0; c < KSUB; c += lanes) {
                var acc = jdk.incubator.vector.FloatVector.zero(species);
                for (int d = 0; d < subDim; d++) {
                    var diff = jdk.incubator.vector.FloatVector
                            .fromArray(species, codebooksT, base + d * KSUB + c)
                            .sub(query[src + d]);
                    acc = diff.fma(diff, acc);
                }
                acc.intoArray(table, row + c);
            }
        }
        return table;
    }

    /** What IvfPqIndex does today: the SIMD kernel, once per codeword. */
    @Benchmark
    public float[] lookupTableSimd() {
        for (int j = 0; j < m; j++) {
            int src = j * subDim;
            int book = j * KSUB * subDim;
            int row = j * KSUB;
            for (int c = 0; c < KSUB; c++) {
                table[row + c] = SimdDistance.l2SquaredUnrolled(query, src, codebooks,
                        book + c * subDim, subDim);
            }
        }
        return table;
    }

    /** The same thing through the plain scalar kernel. */
    @Benchmark
    public float[] lookupTableScalar() {
        for (int j = 0; j < m; j++) {
            int src = j * subDim;
            int book = j * KSUB * subDim;
            int row = j * KSUB;
            for (int c = 0; c < KSUB; c++) {
                table[row + c] = ScalarDistance.l2Squared(query, src, codebooks,
                        book + c * subDim, subDim);
            }
        }
        return table;
    }

    /**
     * Scalar, but with the query subvector hoisted into locals and the codebook walked
     * sequentially, so the inner loop has no repeated index arithmetic and reads the
     * codebook straight down.
     */
    @Benchmark
    public float[] lookupTableScalarHoisted() {
        for (int j = 0; j < m; j++) {
            int row = j * KSUB;
            int book = j * KSUB * subDim;
            int src = j * subDim;
            for (int c = 0, off = book; c < KSUB; c++, off += subDim) {
                float sum = 0;
                for (int d = 0; d < subDim; d++) {
                    float diff = query[src + d] - codebooks[off + d];
                    sum += diff * diff;
                }
                table[row + c] = sum;
            }
        }
        return table;
    }
}
