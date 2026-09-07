from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

import gguf
import numpy as np

from .base import ModelBase
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin
from .qwen3vl import Qwen3VLVisionModel


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    """Qwen3.8-Flash-Next.

    Shares the Qwen3.5 gated delta net and interleaved mrope, and adds three things:
    hyper-connections in place of every layer norm, QSA sparse attention on the full
    attention layers, and PLE n-gram hash embeddings on a single layer.
    """

    model_arch = gguf.MODEL_ARCH.QWEN4EXP

    # Target conversions always drop the MTP block (the bundled-in-target form
    # needs a nextn layer in the target graph, which the port does not build);
    # --mtp exports the head as its own file, reusing the target embed/lm_head
    # (mtp_use_dedicated_embeddings is false) - vLLM drops the head entirely.
    supports_mtp_export = True
    no_mtp = True
    _original_block_count: int | None = None

    @staticmethod
    def _text_hparams(hparams: dict) -> dict:
        # the multimodal checkpoint nests the text keys until base.py merges them
        return {**hparams, **hparams.get("text_config", {})}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mtp_only:
            # the head file carries the MTP layer at index num_hidden_layers
            hp = self._text_hparams(self.hparams)
            self.block_count = hp["num_hidden_layers"] + hp.get("mtp_num_hidden_layers", 1)
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        # shards held only until the row stride is known, normally none
        self._ple_pending: dict[int, Tensor] = {}
        self._ple_shard_rows: dict[int, int] = {}
        self._ple_row_dim: int | None = None
        self._ple_rows_per_shard: int | None = None
        self._ple_map: np.memmap | None = None
        self._ple_path: Path | None = None

    def index_tensors(self, remote_hf_model_id: str | None = None):
        type(self)._original_block_count = self._text_hparams(self.hparams)["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if name.startswith("model.mtp."):
            name = name.replace("model.", "", 1)
        if name.startswith("mtp."):
            if not cls.mtp_only:
                return None
            assert cls._original_block_count is not None
            bc = cls._original_block_count
            parts = name.split(".", 3)
            # the head's own final mixer takes the model-level mixer slot
            if parts[1] == "hyper_connection_mixer":
                name = "model." + name[len("mtp."):]
            elif len(parts) == 4 and parts[1] == "layers" and parts[2].isdecimal():
                name = f"model.layers.{bc + int(parts[2])}.{parts[3]}"
            elif len(parts) == 3:
                remapper = {
                    "fc_embedding":          "fc_embedding",
                    "fc_hidden":             "fc_hidden",
                    "pre_fc_norm_embedding": "enorm",
                    "pre_fc_norm_hidden":    "hnorm",
                }
                if parts[1] not in remapper:
                    raise ValueError(f"unexpected MTP tensor {name!r}")
                name = f"model.layers.{bc}.{remapper[parts[1]]}.{parts[2]}"
            return super().filter_tensors((name, gen))
        if cls.mtp_only:
            # the head reuses the target's embeddings and lm head; the 48 main
            # layers, the PLE table and its hash constants all stay out
            if name not in ("model.embed_tokens.weight",
                            "model.language_model.embed_tokens.weight",
                            "lm_head.weight"):
                return None
        return super().filter_tensors(item)

    def _read_hash_constants(self, suffix: str) -> list[int]:
        """Read an int64 PLE constant straight from the checkpoint.

        prepare_tensors() casts every non-float dtype to float32 before
        modify_tensors() sees it (base.py), which would silently round these
        45-bit multipliers. Reading the lazy tensor here bypasses that.
        """
        for name, gen in self.model_tensors.items():
            if name.endswith(suffix):
                t = gen()
                if t.dtype != torch.int64:
                    t = t.to(torch.int64)
                return [int(x) for x in t.tolist()]
        raise ValueError(f"PLE constant {suffix!r} missing from the checkpoint")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_hyper_connection_count(hp["hc_count"])
        self.gguf_writer.add_hyper_connection_low_rank(hp["hc_lowrank"])

        n_layer = hp["num_hidden_layers"]
        self.gguf_writer.add_indexer_head_count(hp["indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["indexer_budget"])
        ratio = hp["indexer_compress_ratio"]

        if self.mtp_only:
            # the head file: only the nextn slots exist, and the MTP layer is
            # always full attention (config `mtp.layer_types`); no PLE
            n_mtp = self.hparams.get("mtp_num_hidden_layers", 1)
            self.gguf_writer.add_attention_compress_ratios([0] * n_layer + [ratio] * n_mtp)
            self.gguf_writer.add_nextn_predict_layers(n_mtp)
            return

        layer_types = hp["layer_types"]
        self.gguf_writer.add_attention_compress_ratios(
            [ratio if layer_types[i] == "full_attention" else 0 for i in range(n_layer)]
        )

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones
        ple_layers = [i - 1 for i in hp["ple_layer_ids"]]
        if not ple_layers:
            return
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(hp["ngram_size"])
        self.gguf_writer.add_ple_heads_per_ngram(hp["heads_per_ngram"])
        self.gguf_writer.add_ple_conv_kernel(hp["ple_conv_kernel_size"])
        self.gguf_writer.add_ple_eos_token_id(self._eos_token_id())
        # an image is decoded as an embeddings-only batch, so the graph has no placeholder
        # ids to hash; carry the id and let it stand in for those positions
        _img = self._image_token_id()
        if _img is not None:
            self.gguf_writer.add_ple_image_token_id(int(_img))
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        self.gguf_writer.add_ple_layer_multipliers(
            self._read_hash_constants("ple_embedding.layer_multipliers"))
        self.gguf_writer.add_ple_head_offsets(
            self._read_hash_constants("ple_embedding.ngram_heads_offsets"))
        self.gguf_writer.add_ple_head_vocab_sizes(
            self._read_hash_constants("ple_embedding.ngram_heads_vocab_sizes"))

    def _image_token_id(self) -> int | None:
        # base.py merges text_config into the root of hparams, where image_token_id already is
        img = self.hparams.get("image_token_id")
        return None if img is None else int(img)

    def _eos_token_id(self) -> int:
        eos = self.hparams.get("eos_token_id")
        if isinstance(eos, list):
            # the PLE hash resets n-grams on the primary EOS
            return int(eos[-1])
        if eos is None:
            raise ValueError("eos_token_id is required: the PLE hash resets its n-grams on it")
        return int(eos)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # int64 hash constants must stay exact; 1-D tensors force F32, so use KV
        if name.endswith("ple_embedding.layer_multipliers"):
            self._ple_multipliers = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_offsets"):
            self._ple_head_offsets = [int(x) for x in data_torch.tolist()]
            return []
        if name.endswith("ple_embedding.ngram_heads_vocab_sizes"):
            self._ple_head_vocab_sizes = [int(x) for x in data_torch.tolist()]
            return []

        if ".ngram_embedding.shard_" in name:
            return self._place_ple_shard(data_torch, name)

        # one projection feeds indexer q and k; split it, as minimax-m3 does
        if ".indexer.index_qk_proj.weight" in name:
            n_q = self.hparams["indexer_n_heads"] * self.hparams["indexer_head_dim"]
            q = data_torch[:n_q]
            k = data_torch[n_q:]
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid, ".weight"), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid, ".weight"), k),
            ]

        # Gemma zero-centred gammas the inherited norm.weight rule misses
        if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight",
                          ".indexer.q_layernorm.weight", ".indexer.k_layernorm.weight")):
            return [(self.map_tensor_name(name), data_torch + 1)]

        if name.endswith(".ple.conv1d.weight"):
            return [(self.map_tensor_name(name), data_torch.squeeze())]

        return super().modify_tensors(data_torch, name, bid)

    # -- the PLE table ----------------------------------------------------
    #
    # The 128 shards concatenate into one enormous tensor, which peaks near 300 GB of RSS.
    # Each shard is written straight into a memory-mapped file at its final row offset and
    # then dropped, so only one shard is resident. The file is removed after the write.
    # It holds float32 because base.py has already cast the shards to it.

    def _place_ple_shard(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:

        idx = int(name.rpartition(".shard_")[2].partition(".")[0])
        n_parts = self.hparams["split_ngram_parts"]
        rows, row_dim = int(data_torch.shape[0]), int(data_torch.shape[-1])

        self._ple_row_dim = row_dim
        self._ple_shard_rows[idx] = rows

        if self._ple_map is None:
            if idx == n_parts - 1 and n_parts > 1:
                # the last shard can be short, so it cannot set the stride
                # this happens only if the checkpoint yields the shards out of order
                self._ple_pending[idx] = data_torch
                return []
            self._ple_rows_per_shard = rows
            self._ple_path = self.fname_out.parent / f".{self.fname_out.stem}.ple.tmp"
            self._ple_map = np.memmap(
                self._ple_path, dtype=np.float32, mode="w+",
                shape=(n_parts * rows, row_dim))

        for i, held in list(self._ple_pending.items()):
            self._ple_pending.pop(i)
            self._write_ple_shard(i, held)
        self._write_ple_shard(idx, data_torch)

        if len(self._ple_shard_rows) < n_parts:
            return []

        total = sum(self._ple_shard_rows.values())
        table = self._finish_ple_table(total)

        gguf_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD]
        return [(gguf_name + ".weight", table)]

    def _write_ple_shard(self, idx: int, shard: Tensor) -> None:
        # the caller opens the map and fixes the stride before the first write
        assert self._ple_map is not None and self._ple_rows_per_shard is not None

        rows = int(shard.shape[0])
        if idx != self.hparams["split_ngram_parts"] - 1 and rows != self._ple_rows_per_shard:
            raise ValueError(
                f"PLE shard {idx} has {rows} rows, expected {self._ple_rows_per_shard}; "
                "shards other than the last must be uniform for direct placement"
            )

        start = idx * self._ple_rows_per_shard
        # the shard is still lazy here; force it, so exactly one shard is resident
        from .base import LazyTorchTensor

        eager = LazyTorchTensor.to_eager(shard).to(torch.float32).contiguous()
        self._ple_map[start:start + rows] = eager.numpy()
        del eager

    def _finish_ple_table(self, total_rows: int):
        # only reached once every shard has been written, so the map is open
        assert self._ple_map is not None and self._ple_path is not None
        assert self._ple_row_dim is not None

        self._ple_map.flush()
        del self._ple_map
        self._ple_map = None

        # trim the tail if the last shard came up short of a full stride
        want = total_rows * self._ple_row_dim * 4
        if self._ple_path.stat().st_size != want:
            with open(self._ple_path, "r+b") as f:
                f.truncate(want)

        raw = np.memmap(self._ple_path, dtype=np.float32, mode="r+",
                        shape=(total_rows, self._ple_row_dim))
        return torch.from_numpy(np.asarray(raw))

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)
        if not self.mtp_only or not from_dir:
            return
        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._ple_pending:
            raise ValueError(
                f"unprocessed PLE embedding shards: {sorted(self._ple_pending)}"
            )

    def write(self):
        try:
            super().write()
        finally:
            if self._ple_path is not None and self._ple_path.exists():
                self._ple_path.unlink()


@ModelBase.register("Qwen4ExpForConditionalGeneration")
class Qwen4ExpVisionModel(Qwen3VLVisionModel):
    """The vision tower is an unmodified Qwen3-VL ViT."""
