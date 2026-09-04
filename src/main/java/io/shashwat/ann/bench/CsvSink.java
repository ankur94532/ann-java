package io.shashwat.ann.bench;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;

/** Appends {@link Measurement} rows to a CSV, writing the header if the file is new. */
public final class CsvSink implements AutoCloseable {

    private final Path path;

    public CsvSink(Path path) {
        this.path = path;
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            if (!Files.exists(path)) {
                Files.writeString(path, Measurement.CSV_HEADER + System.lineSeparator(),
                        StandardCharsets.UTF_8, StandardOpenOption.CREATE);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("preparing " + path, e);
        }
    }

    public void write(Measurement measurement) {
        try {
            Files.writeString(path, measurement.toCsvRow() + System.lineSeparator(),
                    StandardCharsets.UTF_8, StandardOpenOption.APPEND);
        } catch (IOException e) {
            throw new UncheckedIOException("appending to " + path, e);
        }
    }

    public void writeAll(List<Measurement> measurements) {
        measurements.forEach(this::write);
    }

    public Path path() {
        return path;
    }

    @Override
    public void close() {
        // Each write is flushed on its own; nothing is buffered across a crash.
    }
}
