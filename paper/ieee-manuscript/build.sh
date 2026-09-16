#!/usr/bin/env bash
# Build both manuscripts. Pass "en" or "zh" to build only one.
#
#   bash build.sh        # both
#   bash build.sh en     # English submission version only
#   bash build.sh zh     # Chinese verification version only
#
# XeLaTeX is required for both: the Chinese version needs xeCJK, and under
# XeLaTeX the IEEEtran Times fallback must be replaced explicitly (see main-en.tex).
set -euo pipefail

cd "$(dirname "$0")"

# Pin the embedded PDF creation date so repeated builds are byte-identical.
# Without this every build rewrites the PDFs with a new timestamp and the
# working tree looks dirty even though nothing actually changed.
export SOURCE_DATE_EPOCH=1700000000
export FORCE_SOURCE_DATE=1

if ! command -v xelatex >/dev/null 2>&1; then
  echo "xelatex not found. Install TeX Live (or MiKTeX) and retry." >&2
  exit 1
fi

build() {
  local job="$1"
  local errors

  xelatex -interaction=nonstopmode "${job}.tex" >/dev/null
  bibtex "${job}" >/dev/null
  xelatex -interaction=nonstopmode "${job}.tex" >/dev/null
  xelatex -interaction=nonstopmode "${job}.tex" >/dev/null

  errors=$(grep -cE "^! " "${job}.log" || true)
  if [ "${errors}" -ne 0 ]; then
    echo "FAILED ${job}: ${errors} LaTeX error(s)" >&2
    grep -E "^! " "${job}.log" >&2
    exit 1
  fi

  local pages overfull
  pages=$(pdfinfo "${job}.pdf" | awk '/^Pages:/ {print $2}')
  overfull=$(grep -c 'Overfull' "${job}.log" || true)
  echo "Built ${job}.pdf (${pages} pages, ${overfull} overfull box(es))."

  if [ "${pages}" -gt 6 ]; then
    echo "  WARNING: the 6-page limit is exceeded." >&2
  fi
}

case "${1:-both}" in
  en)   build main-en ;;
  zh)   build main-zh ;;
  both) build main-en; build main-zh ;;
  *)    echo "usage: bash build.sh [en|zh]" >&2; exit 2 ;;
esac
