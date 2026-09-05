#!/usr/bin/env bash
# Downloads and unpacks GIST1M (960-dim, the Phase 6 hard case) into data/gist/.
# ~2.6 GB compressed, 5.8 GB unpacked. The extra is gist_learn.fvecs (1.9 GB), a
# separate training set the tarball ships and this project never reads: the PQ
# codebooks are trained on residuals sampled from the base set itself.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
TARBALL="$DEST/gist.tar.gz"

mkdir -p "$DEST"

if [ -f "$DEST/gist/gist_base.fvecs" ]; then
  echo "GIST1M already present at $DEST/gist — nothing to do."
  exit 0
fi

echo "Downloading $URL (~2.6 GB) ..."
curl -L --fail --retry 3 -C - -o "$TARBALL" "$URL"

echo "Unpacking ..."
tar -xzf "$TARBALL" -C "$DEST"
rm -f "$TARBALL"

ls -la "$DEST/gist"
echo "Done."
