#pragma once

// mmap streaming director (task 15 phase 1) - explicit residency steering for
// models larger than RAM. Enable with LLAMA_SSD_STREAMING=1; window size via
// LLAMA_SSD_STREAM_BUDGET (MiB, default 4096). See docs/ssd-streaming-plan.md.

struct common_params;
struct ggml_tensor;

// installs the streaming eval callback if LLAMA_SSD_STREAMING is set;
// returns true when streaming is active
bool common_ssd_streaming_init(common_params & params);

bool common_ssd_streaming_cb(struct ggml_tensor * t, bool ask, void * user_data);
