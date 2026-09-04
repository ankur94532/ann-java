package io.shashwat.ann.io;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Locations of the benchmark datasets and lazy loaders for their parts.
 *
 * <p>The datasets live under {@code data/} and are not committed; fetch them with
 * {@code scripts/download_sift.sh} and {@code scripts/download_gist.sh}. The root can be
 * overridden with the {@code ann.data.dir} system property.
 */
public enum Datasets {

    SIFT1M("sift", "sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs", 128),
    GIST1M("gist", "gist_base.fvecs", "gist_query.fvecs", "gist_groundtruth.ivecs", 960);

    private final String dir;
    private final String baseFile;
    private final String queryFile;
    private final String groundTruthFile;
    private final int dim;

    Datasets(String dir, String baseFile, String queryFile, String groundTruthFile, int dim) {
        this.dir = dir;
        this.baseFile = baseFile;
        this.queryFile = queryFile;
        this.groundTruthFile = groundTruthFile;
        this.dim = dim;
    }

    public static Path dataRoot() {
        return Paths.get(System.getProperty("ann.data.dir", "data"));
    }

    public Path basePath() {
        return dataRoot().resolve(dir).resolve(baseFile);
    }

    public Path queryPath() {
        return dataRoot().resolve(dir).resolve(queryFile);
    }

    public Path groundTruthPath() {
        return dataRoot().resolve(dir).resolve(groundTruthFile);
    }

    /** The dimension this dataset is documented to have; loaders assert against it. */
    public int dim() {
        return dim;
    }

    public boolean isAvailable() {
        return Files.isRegularFile(basePath())
                && Files.isRegularFile(queryPath())
                && Files.isRegularFile(groundTruthPath());
    }

    public VectorDataset loadBase() {
        return loadBase(Integer.MAX_VALUE);
    }

    public VectorDataset loadBase(int maxVectors) {
        return checkDim(VecsReader.readFvecs(basePath(), maxVectors), basePath());
    }

    public VectorDataset loadQueries() {
        return loadQueries(Integer.MAX_VALUE);
    }

    public VectorDataset loadQueries(int maxVectors) {
        return checkDim(VecsReader.readFvecs(queryPath(), maxVectors), queryPath());
    }

    /** Exact 100-NN ids per query, as shipped with the dataset. */
    public IntDataset loadGroundTruth() {
        return loadGroundTruth(Integer.MAX_VALUE);
    }

    public IntDataset loadGroundTruth(int maxQueries) {
        return VecsReader.readIvecs(groundTruthPath(), maxQueries);
    }

    private VectorDataset checkDim(VectorDataset ds, Path path) {
        if (ds.dim() != dim) {
            throw new IllegalStateException(
                    path + " has dim " + ds.dim() + ", expected " + dim + " for " + name());
        }
        return ds;
    }
}
