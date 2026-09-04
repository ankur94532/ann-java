package io.shashwat.ann.io;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Format-level tests that build their own vecs files, so they run without the datasets.
 */
class VecsReaderTest {

    @TempDir
    Path tmp;

    @Test
    void roundTripsFvecs() throws IOException {
        Random rnd = new Random(7);
        int n = 37;
        int d = 13; // deliberately not a multiple of any SIMD lane count
        float[] expected = new float[n * d];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = rnd.nextFloat() * 200 - 100;
        }
        Path f = tmp.resolve("t.fvecs");
        writeFvecs(f, expected, n, d);

        VectorDataset ds = VecsReader.readFvecs(f);
        assertEquals(n, ds.size());
        assertEquals(d, ds.dim());
        assertArrayEquals(expected, ds.data());
        assertEquals(d, ds.offset(1));
    }

    @Test
    void roundTripsIvecs() throws IOException {
        int n = 5;
        int d = 100;
        int[] expected = new int[n * d];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = i * 31 - 7;
        }
        Path f = tmp.resolve("t.ivecs");
        writeIvecs(f, expected, n, d);

        IntDataset ds = VecsReader.readIvecs(f);
        assertEquals(n, ds.size());
        assertEquals(d, ds.dim());
        assertArrayEquals(expected, ds.data());
        assertArrayEquals(new int[]{expected[d], expected[d + 1]}, ds.row(1, 2));
    }

    @Test
    void honoursMaxVectors() throws IOException {
        int n = 40;
        int d = 8;
        float[] all = new float[n * d];
        for (int i = 0; i < all.length; i++) {
            all[i] = i;
        }
        Path f = tmp.resolve("cap.fvecs");
        writeFvecs(f, all, n, d);

        VectorDataset ds = VecsReader.readFvecs(f, 9);
        assertEquals(9, ds.size());
        assertEquals(d, ds.dim());
        for (int i = 0; i < 9 * d; i++) {
            assertEquals(all[i], ds.data()[i]);
        }
    }

    /** Chunking must not corrupt records that straddle a read-buffer boundary. */
    @Test
    void readsFilesLargerThanOneChunk() throws IOException {
        int d = 64;
        int n = 40_000; // 40000 * 260 bytes = ~10 MB, more than the 8 MB read chunk
        float[] all = new float[n * d];
        for (int i = 0; i < all.length; i++) {
            all[i] = i % 977;
        }
        Path f = tmp.resolve("big.fvecs");
        writeFvecs(f, all, n, d);

        VectorDataset ds = VecsReader.readFvecs(f);
        assertEquals(n, ds.size());
        assertArrayEquals(all, ds.data());
    }

    @Test
    void rejectsTruncatedFile() throws IOException {
        Path f = tmp.resolve("bad.fvecs");
        writeFvecs(f, new float[]{1, 2, 3, 4}, 1, 4);
        byte[] bytes = Files.readAllBytes(f);
        Files.write(f, java.util.Arrays.copyOf(bytes, bytes.length - 3));

        assertThrows(RuntimeException.class, () -> VecsReader.readFvecs(f));
    }

    @Test
    void reportsLayoutWithoutReadingPayload() throws IOException {
        Path f = tmp.resolve("layout.fvecs");
        writeFvecs(f, new float[3 * 6], 3, 6);
        assertEquals(3, VecsReader.countVectors(f));
        assertEquals(6, VecsReader.dimensionOf(f));
    }

    static void writeFvecs(Path path, float[] data, int n, int d) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(n * (4 + 4 * d)).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            buf.putInt(d);
            for (int j = 0; j < d; j++) {
                buf.putFloat(data[i * d + j]);
            }
        }
        Files.write(path, buf.array());
    }

    static void writeIvecs(Path path, int[] data, int n, int d) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(n * (4 + 4 * d)).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            buf.putInt(d);
            for (int j = 0; j < d; j++) {
                buf.putInt(data[i * d + j]);
            }
        }
        Files.write(path, buf.array());
    }
}
