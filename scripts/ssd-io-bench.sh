#!/usr/bin/env bash
# SSD IO benchmark for ssd-streaming feasibility (docs/ssd-streaming-plan.md).
# Measures the two access patterns that bound streamed-MoE decode:
#   1. sequential O_DIRECT read  (load / prefill batching)
#   2. random expert-slice-sized O_DIRECT reads, 1/2/4/8 threads (the miss path)
# O_DIRECT bypasses the page cache, so results are honest without root.
#
# Usage:  ./ssd-io-bench.sh [big_file_on_the_target_drive]
#   Point it at any large file (>= 8 GB; a GGUF is perfect). Without an
#   argument it creates an 8 GB test file in the current directory.
#
# Reference (V100 box NVMe, the plan-doc baseline): seq ~1.6 GB/s;
# random 2 MiB: 1.67 GB/s @1t -> ~2.7 GB/s @2+t (saturates at 2 threads).
# A PCIe Gen-4 drive should show ~2-4x these numbers and keep scaling to 8t.
set -u

F=${1:-./ssd-io-bench.testfile}
if [ ! -f "$F" ]; then
    echo "creating 8 GiB test file at $F ..."
    dd if=/dev/urandom of="$F" bs=1M count=8192 status=progress conv=fsync
fi
SZ=$(stat -c %s "$F")
[ "$SZ" -lt $((2*1024*1024*1024)) ] && { echo "file too small (<2 GiB), use a bigger one"; exit 1; }
echo "target: $F ($((SZ/1024/1024/1024)) GiB)  drive: $(df --output=source "$F" | tail -1)"
echo

if command -v fio >/dev/null 2>&1; then
    echo "== using fio =="
    echo "-- sequential read, 1 MiB blocks, O_DIRECT --"
    fio --name=seq --filename="$F" --readonly --rw=read --bs=1M --direct=1 \
        --time_based --runtime=10 --ioengine=psync --group_reporting 2>/dev/null \
        | grep -E "READ:|bw=" | head -2
    for BS in 2M 7M; do
        for NJ in 1 2 4 8; do
            echo "-- random read, bs=$BS, threads=$NJ, O_DIRECT --"
            fio --name=rnd --filename="$F" --readonly --rw=randread --bs=$BS --direct=1 \
                --numjobs=$NJ --time_based --runtime=10 --ioengine=psync --group_reporting 2>/dev/null \
                | grep -E "READ:" | head -1
        done
    done
else
    echo "== fio not found, dd fallback (apt install fio for better numbers) =="
    echo "-- sequential read, O_DIRECT, 4 GiB --"
    dd if="$F" of=/dev/null bs=1M count=4096 iflag=direct 2>&1 | tail -1
    # random reads: N parallel workers, each does 64 reads of 2 MiB at random offsets
    for NJ in 1 2 4 8; do
        START=$(date +%s.%N)
        for w in $(seq 1 $NJ); do
            (
             for i in $(seq 1 64); do
                 OFF=$(( (RANDOM * 32768 + RANDOM) % (SZ / 1048576 - 2) ))
                 dd if="$F" of=/dev/null bs=2M count=1 skip=$((OFF/2)) iflag=direct 2>/dev/null
             done
            ) &
        done
        wait
        END=$(date +%s.%N)
        MB=$((NJ * 64 * 2))
        echo "-- random 2 MiB x64/worker, threads=$NJ: $(echo "$MB $START $END" | awk '{printf "%.0f MB/s", $1/($3-$2)}') --"
    done
fi

echo
echo "interpretation: cold streamed-MoE decode is IO-bound at ~(random-read BW / 1.83 GB)"
echo "tokens/s for DeepSeek-81B-class models (1.83 GB cold expert traffic per token);"
echo "the RAM/VRAM caches multiply from there. Set LLAMA_SSD_STREAM_READ_THREADS to the"
echo "thread count where your drive's random-read BW stops scaling."
