# ann-java

Approximate nearest-neighbour indexes (HNSW, IVF-PQ) written from scratch in Java 21,
benchmarked against FAISS on SIFT1M and GIST1M.

Work in progress. See [PROTOCOL.md](PROTOCOL.md) for the frozen benchmark protocol.

## Build

```
./gradlew build
./gradlew run          # prints version + environment banner
```

Requires JDK 21 (Gradle resolves a toolchain). The Vector API is an incubator module,
so every JVM invocation needs `--add-modules jdk.incubator.vector`; the build files do
this for you.
