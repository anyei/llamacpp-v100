#!/usr/bin/env bash
# Prune the oldest entries from a worker's RPC tensor cache (cache-rpc/rpc) until a
# target number of gigabytes has been freed. The cache is content-addressed and
# rebuildable: deleting an entry only forces a re-stream on a future load (the
# verify-on-read gate keeps correctness), and a loaded model holds its tensors in
# RAM, not from these files - so this is safe to run while the worker is serving.
# Oldest-first by mtime = superseded model / -ts-split generations go first.
#
# Usage:
#   scripts/prune-rpc-cache.sh [--apply] [DIR] [TARGET_GB]
#     DIR        cache dir (default: ./cache-rpc/rpc)
#     TARGET_GB  gigabytes to reclaim (default: 50)
#   Without --apply it only prints the plan (dry run). Files are usually root-owned;
#   run with sudo to actually delete: sudo scripts/prune-rpc-cache.sh --apply
set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then APPLY=1; shift; fi
DIR="${1:-./cache-rpc/rpc}"
TARGET_GB="${2:-50}"
TARGET_BYTES=$(( TARGET_GB * 1024 * 1024 * 1024 ))

if [[ ! -d "$DIR" ]]; then
  echo "error: cache dir '$DIR' not found (pass it as the first non-flag arg)" >&2
  exit 1
fi

total_before=$(du -sb "$DIR" 2>/dev/null | cut -f1)
echo "cache dir : $DIR"
echo "current   : $(( total_before / 1024 / 1024 / 1024 )) GiB"
echo "target    : free ${TARGET_GB} GiB (oldest first)"
echo

# oldest-first: "<mtime-epoch> <size-bytes> <path>", sorted ascending by mtime.
# Only regular files; skip the modelidx-* sidecars (cheap, and dropping them just
# re-hashes a --model-dir on next start - keep them to avoid that).
mapfile -t rows < <(
  find "$DIR" -maxdepth 1 -type f ! -name 'modelidx-*' -printf '%T@ %s %p\n' \
    | sort -n
)

acc=0
n=0
del_list=()
oldest=""; newest=""
for row in "${rows[@]}"; do
  [[ $acc -ge $TARGET_BYTES ]] && break
  epoch=${row%% *}; rest=${row#* }; size=${rest%% *}; path=${rest#* }
  acc=$(( acc + size ))
  n=$(( n + 1 ))
  del_list+=("$path")
  [[ -z "$oldest" ]] && oldest=$epoch
  newest=$epoch
done

if [[ $n -eq 0 ]]; then
  echo "nothing to prune."
  exit 0
fi

echo "would remove : $n files, $(( acc / 1024 / 1024 / 1024 )) GiB"
[[ -n "$oldest" ]] && echo "mtime range  : $(date -d "@$oldest" '+%Y-%m-%d %H:%M') -> $(date -d "@$newest" '+%Y-%m-%d %H:%M')"
echo "would leave  : $(( (total_before - acc) / 1024 / 1024 / 1024 )) GiB"
echo

if [[ $APPLY -eq 0 ]]; then
  echo "DRY RUN - re-run with --apply (and sudo if the files are root-owned) to delete."
  exit 0
fi

printf '%s\0' "${del_list[@]}" | xargs -0 rm -f
echo "done. cache now: $(( $(du -sb "$DIR" | cut -f1) / 1024 / 1024 / 1024 )) GiB"
