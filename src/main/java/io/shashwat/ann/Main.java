package io.shashwat.ann;

import io.shashwat.ann.bench.HnswCommand;
import io.shashwat.ann.bench.IvfPqCommand;
import io.shashwat.ann.bench.OracleCommand;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

/**
 * Entry point. With no arguments it prints the build and environment banner that the
 * benchmark protocol requires to be recorded alongside every result set.
 */
public final class Main {

    public static final String VERSION = "ann-java 0.1.0";

    private Main() {
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println(VERSION);
            System.out.print(environmentBanner());
            return;
        }
        String[] rest = Arrays.copyOfRange(args, 1, args.length);
        int exit = switch (args[0]) {
            case "version" -> {
                System.out.println(VERSION);
                yield 0;
            }
            case "env" -> {
                System.out.print(environmentBanner());
                yield 0;
            }
            case "oracle" -> OracleCommand.run(rest);
            case "hnsw" -> HnswCommand.run(rest);
            case "ivfpq" -> IvfPqCommand.run(rest);
            default -> {
                System.err.println("unknown command: " + args[0]);
                System.err.println(usage());
                yield 2;
            }
        };
        if (exit != 0) {
            System.exit(exit);
        }
    }

    private static String usage() {
        return """
                usage: Main <command> [options]
                  version                        print the build version
                  env                            print the hardware/JVM banner
                  oracle [--dataset sift|gist] [--k 10] [--queries N]
                                                 exact search, validated against shipped ground truth
                  hnsw [--dataset sift|gist] [--impl fast|naive] [--m 16] [--efc 200] [--ef 16,32,...]
                       [--selection heuristic|nearestM] [--base N] [--queries N]
                       [--runs 3] [--csv path]
                                                 build one HNSW graph and sweep efSearch
                  ivfpq [--dataset sift|gist] [--nlist 1024] [--m 16] [--nprobe 1,4,8,...]
                        [--base N] [--queries N] [--runs 3] [--csv path]
                                                 build one IVF-PQ index and sweep nprobe
                """;
    }

    /**
     * The hardware/JVM facts PROTOCOL.md requires to be reported with any benchmark run.
     */
    public static String environmentBanner() {
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        Runtime rt = Runtime.getRuntime();
        return """
                jvm            : %s %s (%s)
                os             : %s %s %s
                cores          : %d
                max heap       : %.1f GiB
                float species  : %s (%d lanes, %d bits)
                """.formatted(
                System.getProperty("java.vm.name"),
                System.getProperty("java.version"),
                System.getProperty("java.vendor"),
                System.getProperty("os.name"),
                System.getProperty("os.version"),
                System.getProperty("os.arch"),
                rt.availableProcessors(),
                rt.maxMemory() / (double) (1L << 30),
                species,
                species.length(),
                species.vectorBitSize());
    }
}
