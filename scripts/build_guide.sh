#!/usr/bin/env bash
# Renders docs/guide/guide.html to a PDF using headless Chrome.
#
# Chrome rather than pandoc/weasyprint because it is already present on most machines and
# because the guide uses inline SVG diagrams and CSS paged-media rules that it handles
# faithfully. The @page rules in the HTML control margins and page size.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"

# The guide's measured figures are generated from docs/results/. Refuse to print a PDF from
# an HTML file whose generated regions disagree with the data.
if ! python3 "$ROOT/scripts/readme_tables.py" --check; then
  echo "Generated regions are stale; run scripts/readme_tables.py first." >&2
  exit 1
fi

if [ ! -x "$CHROME" ]; then
  echo "Chrome not found at: $CHROME" >&2
  echo "Set CHROME=/path/to/chrome (or google-chrome / chromium on Linux)." >&2
  exit 1
fi

"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$ROOT/docs/guide/ann-java-guide.pdf" \
  "file://$ROOT/docs/guide/guide.html"

echo "wrote docs/guide/ann-java-guide.pdf"
