#!/usr/bin/env bash
# Downloads and unpacks SIFT1M (Jegou et al., TEXMEX corpus) into data/sift/.
#
# Contents after unpacking:
#   sift_base.fvecs       1,000,000 x 128 float32
#   sift_query.fvecs         10,000 x 128 float32
#   sift_learn.fvecs        100,000 x 128 float32
#   sift_groundtruth.ivecs   10,000 x 100 int32   (exact 100-NN ids into base)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
TARBALL="$DEST/sift.tar.gz"

mkdir -p "$DEST"

if [ -f "$DEST/sift/sift_base.fvecs" ]; then
  echo "SIFT1M already present at $DEST/sift — nothing to do."
  exit 0
fi

echo "Downloading $URL (~168 MB) ..."
curl -L --fail --retry 3 -C - -o "$TARBALL" "$URL"

echo "Unpacking ..."
tar -xzf "$TARBALL" -C "$DEST"
rm -f "$TARBALL"

ls -la "$DEST/sift"
echo "Done."
