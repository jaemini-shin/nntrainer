#!/usr/bin/env bash
##
# @file  gbs-cache-prune.sh
# @brief Keep only the newest version of each RPM in the GBS package cache.
#
# GBS never purges superseded packages from ~/GBS-ROOT/local/cache. Across the
# daily restore -> download -> save cycle the cache therefore grows without
# bound (observed on CI: aarch64 268MB fresh -> 2.6GB accumulated, pushing the
# repo over the 10GB Actions-cache limit and triggering eviction thrash).
#
# This keeps, per package (name+arch), only the most-recently-downloaded .rpm
# and deletes older versions, bounding the cache to ~one snapshot (~270MB/arch).
# A wrongly-pruned package only costs a re-download on the next run, never a
# build failure, so erring toward pruning is safe.
set -u

CACHE="${1:-$HOME/GBS-ROOT/local/cache}"

if [ ! -d "$CACHE" ]; then
  echo "gbs-cache-prune: no cache dir at $CACHE, nothing to do"
  exit 0
fi

echo "== gbs package cache before prune =="
du -sh "$CACHE" 2>/dev/null || true
echo "rpm files: $(find "$CACHE" -name '*.rpm' 2>/dev/null | wc -l)"

# newest first (by mtime); keep first occurrence of each package key, drop rest
find "$CACHE" -type f -name '*.rpm' -printf '%T@\t%p\n' 2>/dev/null \
  | sort -rn \
  | awk -F'\t' '
      {
        path = $2
        n = split(path, seg, "/"); base = seg[n]
        key = base
        # strip trailing -<version>-<release>.<arch>.rpm to group all versions
        sub(/-[^-]+-[^-]+\.[^.]+\.rpm$/, "", key)
        if (key in seen) { print path }   # older duplicate -> prune
        else             { seen[key] = 1 }
      }' \
  | while IFS= read -r f; do rm -f "$f"; done

echo "== gbs package cache after prune =="
du -sh "$CACHE" 2>/dev/null || true
echo "rpm files: $(find "$CACHE" -name '*.rpm' 2>/dev/null | wc -l)"
