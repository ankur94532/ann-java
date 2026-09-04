package io.shashwat.ann;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

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
            System.out.println(environmentBanner());
            return;
        }
        switch (args[0]) {
            case "env" -> System.out.println(environmentBanner());
            case "version" -> System.out.println(VERSION);
            default -> {
                System.err.println("unknown command: " + args[0]);
                System.err.println("usage: Main [version|env]");
                System.exit(2);
            }
        }
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
