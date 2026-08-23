# Interesting tools (not currently in any critical path)

- [gigatoken](https://github.com/marcelroed/gigatoken) — Rust BPE tokenizer, ~1000x HF tokenizers via SIMD + caching + multi-core CPU encode (~24 GB/s on big EPYC). Not relevant to distributed inference (llama.cpp has its own C++ tokenizer; tokenization is off the serving critical path). Worth revisiting only if we ever bulk-tokenize large corpora offline (e.g. large-scale synthetic traffic generation for expert profiling).
