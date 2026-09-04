package io.shashwat.ann.bench;

/**
 * One configuration's measured result. The field order is the CSV column order, and the
 * Python FAISS harness emits the identical header.
 */
public record Measurement(
        String harness,
        String dataset,
        String index,
        String params,
        int k,
        double recallAtK,
        double meanLatencyUs,
        double p95LatencyUs,
        double buildSeconds,
        long indexBytes,
        long baseBytes,
        int queries,
        int runs,
        String timestamp) {

    public static final String CSV_HEADER =
            "harness,dataset,index,params,k,recall_at_k,mean_latency_us,p95_latency_us,"
                    + "build_seconds,index_bytes,base_bytes,queries,runs,timestamp";

    public String toCsvRow() {
        return String.join(",",
                harness,
                dataset,
                index,
                "\"" + params + "\"",
                Integer.toString(k),
                String.format("%.6f", recallAtK),
                String.format("%.3f", meanLatencyUs),
                String.format("%.3f", p95LatencyUs),
                String.format("%.3f", buildSeconds),
                Long.toString(indexBytes),
                Long.toString(baseBytes),
                Integer.toString(queries),
                Integer.toString(runs),
                timestamp);
    }

    @Override
    public String toString() {
        return String.format("%-28s recall=%.4f  mean=%8.1f us  p95=%8.1f us  "
                        + "build=%7.1fs  index=%6.1f MiB",
                params, recallAtK, meanLatencyUs, p95LatencyUs, buildSeconds,
                indexBytes / (1024.0 * 1024.0));
    }
}
