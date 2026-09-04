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
 *   <li><b>scanL2</b> — one query against a 4 MiB block. That is far past this machine's
 *       128 KiB L1 but comfortably inside its 16 MiB shared L2, so it measures a kernel fed
 *       from L2.
 *   <li><b>scanRam</b> — the same against a 64 MiB block, which cannot live in any cache
 *       on this machine and so is fed from DRAM. This is the regime a brute-force scan over
 *       SIFT1M (512 MiB) actually runs in, and the gap between it and the pair benchmark is
 *       the honest answer to "how much does the SIMD kernel buy an index?".
 * </ul>
 *
 * <p>Run with {@code ./gradlew jmh -PjmhArgs="DistanceBenchmark -rf json -rff
 * build/jmh-distance.json"}.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Fork(value = 1, jvmArgsAppend = {"--add-modules", "jdk.incubator.vector", "-Xmx4g"})
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
public class DistanceBenchmark {

    /** Past L1 (128 KiB here), inside the 16 MiB shared L2. */
    private static final int SCAN_L2_BYTES = 4 << 20;

    /** Past any cache on this machine, so the loads come from DRAM. */
    private static final int SCAN_RAM_BYTES = 64 << 20;

    @Param({"128", "960"})
    public int dim;

    private float[] a;
    private float[] b;

    private float[] query;
    private float[] l2Block;
    private int l2BlockVectors;
    private float[] ramBlock;
    private int ramBlockVectors;

    @Setup
    public void setup() {
        Random rnd = new Random(20240904L + dim);
        a = randomVector(rnd, dim);
        b = randomVector(rnd, dim);

        query = randomVector(rnd, dim);
        l2BlockVectors = Math.max(1, SCAN_L2_BYTES / (dim * Float.BYTES));
        l2Block = randomVector(rnd, l2BlockVectors * dim);
        ramBlockVectors = Math.max(1, SCAN_RAM_BYTES / (dim * Float.BYTES));
        ramBlock = randomVector(rnd, ramBlockVectors * dim);
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
    public float scanL2ScalarL2() {
        return scanScalar(l2Block, l2BlockVectors);
    }

    @Benchmark
    public float scanL2SimdL2Unrolled() {
        return scanSimd(l2Block, l2BlockVectors);
    }

    @Benchmark
    public float scanRamScalarL2() {
        return scanScalar(ramBlock, ramBlockVectors);
    }

    @Benchmark
    public float scanRamSimdL2Unrolled() {
        return scanSimd(ramBlock, ramBlockVectors);
    }

    private float scanScalar(float[] block, int vectors) {
        float best = Float.MAX_VALUE;
        for (int i = 0, off = 0; i < vectors; i++, off += dim) {
            float d = ScalarDistance.l2Squared(query, 0, block, off, dim);
            if (d < best) {
                best = d;
            }
        }
        return best;
    }

    private float scanSimd(float[] block, int vectors) {
        float best = Float.MAX_VALUE;
        for (int i = 0, off = 0; i < vectors; i++, off += dim) {
            float d = SimdDistance.l2SquaredUnrolled(query, 0, block, off, dim);
            if (d < best) {
                best = d;
            }
        }
        return best;
    }
}
