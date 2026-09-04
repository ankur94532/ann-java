package io.shashwat.ann.distance;

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
 * Scalar vs Vector-API distance kernels at the two dimensions the project benchmarks:
 * 128 (SIFT) and 960 (GIST).
 *
 * <p>Two residency regimes are measured, because they answer different questions.
 *
 * <ul>
 *   <li><b>pair</b> — the same two vectors every call, resident in L1. This is the
 *       arithmetic throughput of the kernel with the memory system taken out of the
 *       picture, and it is the number the "SIMD speedup" claim refers to.
 *   <li><b>scan</b> — one query against a 4 MiB block of base vectors, which does not fit
 *       in L2. This is what an index actually does, and the gap between the two regimes is
 *       the honest answer to "how much does the SIMD kernel buy an index?".
 * </ul>
 *
 * <p>Run with {@code ./gradlew jmh -PjmhArgs="DistanceBenchmark -rf json -rff
 * build/jmh-distance.json"}.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "-Xmx2g"})
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
public class DistanceBenchmark {

    /** 4 MiB of base vectors, comfortably larger than this machine's L2. */
    private static final int SCAN_BYTES = 4 << 20;

    @Param({"128", "960"})
    public int dim;

    private float[] a;
    private float[] b;

    private float[] query;
    private float[] block;
    private int blockVectors;

    @Setup
    public void setup() {
        Random rnd = new Random(20240904L + dim);
        a = randomVector(rnd, dim);
        b = randomVector(rnd, dim);

        query = randomVector(rnd, dim);
        blockVectors = Math.max(1, SCAN_BYTES / (dim * Float.BYTES));
        block = randomVector(rnd, blockVectors * dim);
    }

    private static float[] randomVector(Random rnd, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() * 255f;
        }
        return v;
    }

    // ---- L1-resident pair: kernel arithmetic throughput ------------------------------

    @Benchmark
    public float pairScalarL2() {
        return ScalarDistance.l2Squared(a, 0, b, 0, dim);
    }

    @Benchmark
    public float pairSimdL2() {
        return SimdDistance.l2Squared(a, 0, b, 0, dim);
    }

    @Benchmark
    public float pairSimdL2Unrolled() {
        return SimdDistance.l2SquaredUnrolled(a, 0, b, 0, dim);
    }

    @Benchmark
    public float pairScalarInnerProduct() {
        return ScalarDistance.innerProduct(a, 0, b, 0, dim);
    }

    @Benchmark
    public float pairSimdInnerProduct() {
        return SimdDistance.innerProduct(a, 0, b, 0, dim);
    }

    // ---- Streaming scan: what an index actually does ---------------------------------

    @Benchmark
    public float scanScalarL2() {
        float best = Float.MAX_VALUE;
        for (int i = 0, off = 0; i < blockVectors; i++, off += dim) {
            float d = ScalarDistance.l2Squared(query, 0, block, off, dim);
            if (d < best) {
                best = d;
            }
        }
        return best;
    }

    @Benchmark
    public float scanSimdL2Unrolled() {
        float best = Float.MAX_VALUE;
        for (int i = 0, off = 0; i < blockVectors; i++, off += dim) {
            float d = SimdDistance.l2SquaredUnrolled(query, 0, block, off, dim);
            if (d < best) {
                best = d;
            }
        }
        return best;
    }
}
