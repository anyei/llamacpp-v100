// Feasibility spike for task-15 phase 2/4: measure random ~2MB O_DIRECT read
// bandwidth from the real 81GB DeepSeek GGUF — the cache-MISS path of a
// userspace expert cache. This is the number that decides whether pread+SLRU
// beats the mmap-thrash floor (1.12 t/s) instead of collapsing like the
// madvise(WILLNEED) route did (0.018 t/s).
//
// Also reports a projected per-token expert-IO time for the DeepSeek working
// set (258 expert-slice-sets/token, ~7MB each) at a few cache hit rates.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <vector>
#include <random>

static const size_t ALIGN = 4096;

struct Job {
    const char * path;
    size_t region_start, region_len;
    size_t slice;      // bytes per read (rounded up to ALIGN)
    long   n_reads;    // reads this thread performs
    unsigned seed;
    // out
    double bytes;
};

static void * worker(void * arg) {
    Job * j = (Job *) arg;
    int fd = open(j->path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open O_DIRECT"); j->bytes = -1; return nullptr; }
    void * buf = nullptr;
    if (posix_memalign(&buf, ALIGN, j->slice + ALIGN) != 0) { j->bytes = -1; close(fd); return nullptr; }
    std::mt19937_64 rng(j->seed);
    // random aligned offset within [region_start, region_start+region_len-slice]
    size_t span = (j->region_len - j->slice) / ALIGN;
    double total = 0;
    for (long i = 0; i < j->n_reads; i++) {
        size_t off = j->region_start + (rng() % span) * ALIGN;
        ssize_t r = pread(fd, buf, j->slice, off);
        if (r < 0) { perror("pread"); break; }
        total += r;
    }
    j->bytes = total;
    free(buf);
    close(fd);
    return nullptr;
}

static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }

int main(int argc, char ** argv) {
    const char * path = argv[1];
    size_t region_start = 1ull<<30;             // skip first 1GB (header + dense)
    size_t region_len   = (78ull)<<30;          // ~78GB expert region
    // slice ~ average expert slice-set (gate+up+down ~ 7.08MB); round to ALIGN.
    // Test the per-read granularity that matters: a single expert slice ~2.16MB
    // and the full 3-tensor set ~7.08MB.
    size_t slices[] = { 2162688, 7080000 };
    long   total_reads = 2000;                  // per configuration
    int    threadsets[] = { 1, 2, 4, 8 };

    for (size_t sidx = 0; sidx < 2; sidx++) {
        size_t slice = ((slices[sidx] + ALIGN - 1) / ALIGN) * ALIGN;
        printf("\n== slice %.2f MB (aligned) ==\n", slice/1e6);
        for (int t : threadsets) {
            std::vector<pthread_t> th(t);
            std::vector<Job> jobs(t);
            long per = total_reads / t;
            for (int k = 0; k < t; k++)
                jobs[k] = { path, region_start, region_len, slice, per, (unsigned)(1234+k*97), 0 };
            double t0 = now();
            for (int k = 0; k < t; k++) pthread_create(&th[k], nullptr, worker, &jobs[k]);
            double bytes = 0; bool ok = true;
            for (int k = 0; k < t; k++) { pthread_join(th[k], nullptr); if (jobs[k].bytes < 0) ok=false; else bytes += jobs[k].bytes; }
            double dt = now() - t0;
            if (!ok) { printf("  threads=%d: FAILED\n", t); continue; }
            printf("  threads=%2d: %.2f GB in %.2fs = %6.0f MB/s  (%ld reads, %.3f ms/read)\n",
                   t, bytes/1e9, dt, bytes/1e6/dt, per*t, dt/(per*t)*1e3);
        }
    }

    // Projection for DeepSeek-V4-Flash per-token expert IO.
    // 6 experts/layer * 43 layers = 258 slice-sets/token, 7.08MB each = 1.83 GB cold.
    const double per_tok_cold_gb = 258 * 7.08e-3;   // GB
    const double bw_gbs = 2.70;                      // measured multi-thread random-read GB/s
    printf("\n== projected per-token expert-IO (258 slice-sets/token = %.2f GB cold, at %.2f GB/s) ==\n",
           per_tok_cold_gb, bw_gbs);
    printf("   NOTE: this is expert-IO time only; real t/s is min(1/io_time, compute_bound).\n");
    double hitrates[] = { 0.0, 0.44, 0.80, 0.95 };
    const char * label[] = { "no cache (pure stream)", "uniform, 34GB RAM cache (~44%)",
                             "moderate skew", "strong skew (cf #20757 98%)" };
    for (int i = 0; i < 4; i++) {
        double miss_gb = per_tok_cold_gb * (1.0 - hitrates[i]);
        double io_s = miss_gb / bw_gbs;
        printf("   hit %3.0f%% (%-30s): miss %.2f GB/tok, IO %.3f s/tok -> %.2f t/s (IO-bound)\n",
               hitrates[i]*100, label[i], miss_gb, io_s, 1.0/io_s);
    }
    printf("   floor to beat: mmap-thrash 1.12 t/s (GLM 14GB@10GB); madvise(WILLNEED) 0.018 t/s.\n");
    return 0;
}
