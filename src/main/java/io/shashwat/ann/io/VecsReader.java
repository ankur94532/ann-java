package io.shashwat.ann.io;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Readers for the TEXMEX {@code .fvecs} / {@code .ivecs} binary formats.
 *
 * <p>Both formats are a bare sequence of records with no file header. Each record is a
 * 4-byte little-endian dimension {@code d} followed by {@code d} values — float32 for
 * fvecs, int32 for ivecs. The vector count is therefore implied by the file length:
 * {@code n = fileSize / (4 + 4*d)}.
 */
public final class VecsReader {

    /** Read buffer target size; rounded down to a whole number of records. */
    private static final int CHUNK_BYTES = 8 << 20;

    private VecsReader() {
    }

    public static VectorDataset readFvecs(Path path) {
        return readFvecs(path, Integer.MAX_VALUE);
    }

    /**
     * Reads at most {@code maxVectors} vectors from an fvecs file.
     *
     * @param maxVectors cap on how many vectors to load, for quick development runs.
     *                   Final benchmark numbers must use the whole file.
     */
    public static VectorDataset readFvecs(Path path, int maxVectors) {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            Layout layout = layoutOf(ch, path);
            int n = (int) Math.min(layout.count(), maxVectors);
            float[] out = new float[Math.multiplyExact(n, layout.dim())];
            readRecords(ch, layout, n, (chunk, records, dim, outOffset) -> {
                FloatBuffer fb = chunk.asFloatBuffer();
                IntBuffer ib = chunk.asIntBuffer();
                int off = outOffset;
                for (int r = 0; r < records; r++) {
                    int base = r * (dim + 1);
                    checkHeader(ib.get(base), dim, path);
                    fb.position(base + 1);
                    fb.get(out, off, dim);
                    off += dim;
                }
            });
            return new VectorDataset(out, n, layout.dim());
        } catch (IOException e) {
            throw new UncheckedIOException("reading " + path, e);
        }
    }

    public static IntDataset readIvecs(Path path) {
        return readIvecs(path, Integer.MAX_VALUE);
    }

    public static IntDataset readIvecs(Path path, int maxVectors) {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            Layout layout = layoutOf(ch, path);
            int n = (int) Math.min(layout.count(), maxVectors);
            int[] out = new int[Math.multiplyExact(n, layout.dim())];
            readRecords(ch, layout, n, (chunk, records, dim, outOffset) -> {
                IntBuffer ib = chunk.asIntBuffer();
                int off = outOffset;
                for (int r = 0; r < records; r++) {
                    int base = r * (dim + 1);
                    checkHeader(ib.get(base), dim, path);
                    ib.position(base + 1);
                    ib.get(out, off, dim);
                    off += dim;
                }
            });
            return new IntDataset(out, n, layout.dim());
        } catch (IOException e) {
            throw new UncheckedIOException("reading " + path, e);
        }
    }

    /** Reads the leading dimension word and derives the record count from the file size. */
    private static Layout layoutOf(FileChannel ch, Path path) throws IOException {
        long size = ch.size();
        if (size < 4) {
            throw new IOException(path + " is too short to be a .fvecs/.ivecs file");
        }
        ByteBuffer head = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        while (head.hasRemaining()) {
            if (ch.read(head) < 0) {
                throw new IOException("unexpected EOF reading dimension of " + path);
            }
        }
        int dim = head.flip().getInt();
        if (dim <= 0 || dim > 1 << 20) {
            throw new IOException(path + " declares an implausible dimension " + dim
                    + " (wrong file, or wrong byte order)");
        }
        long recordBytes = 4L + 4L * dim;
        if (size % recordBytes != 0) {
            throw new IOException(path + " size " + size + " is not a multiple of the "
                    + recordBytes + "-byte record implied by d=" + dim);
        }
        return new Layout(dim, size / recordBytes, recordBytes);
    }

    /** Streams the file in whole-record chunks, handing each chunk to {@code sink}. */
    private static void readRecords(FileChannel ch, Layout layout, int wanted, ChunkSink sink)
            throws IOException {
        int recordsPerChunk = Math.max(1, (int) (CHUNK_BYTES / layout.recordBytes()));
        int chunkBytes = (int) (recordsPerChunk * layout.recordBytes());
        ByteBuffer buf = ByteBuffer.allocateDirect(chunkBytes).order(ByteOrder.LITTLE_ENDIAN);

        ch.position(0);
        int done = 0;
        while (done < wanted) {
            int records = Math.min(recordsPerChunk, wanted - done);
            int bytes = (int) (records * layout.recordBytes());
            buf.clear().limit(bytes);
            while (buf.hasRemaining()) {
                if (ch.read(buf) < 0) {
                    throw new IOException("unexpected EOF after " + done + " records");
                }
            }
            buf.flip();
            sink.accept(buf, records, layout.dim(), done * layout.dim());
            done += records;
        }
    }

    private static void checkHeader(int declared, int dim, Path path) {
        if (declared != dim) {
            throw new UncheckedIOException(new IOException(
                    path + " has a ragged record: expected d=" + dim + ", found " + declared));
        }
    }

    /** Number of vectors in a vecs file, without reading the payload. */
    public static long countVectors(Path path) {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            return layoutOf(ch, path).count();
        } catch (IOException e) {
            throw new UncheckedIOException("reading " + path, e);
        }
    }

    /** Dimension declared by the first record of a vecs file. */
    public static int dimensionOf(Path path) {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            return layoutOf(ch, path).dim();
        } catch (IOException e) {
            throw new UncheckedIOException("reading " + path, e);
        }
    }

    public static boolean exists(Path path) {
        return Files.isRegularFile(path);
    }

    private record Layout(int dim, long count, long recordBytes) {
    }

    @FunctionalInterface
    private interface ChunkSink {
        void accept(ByteBuffer chunk, int records, int dim, int outOffset);
    }
}
