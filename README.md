# serve-leanstral-2603 — how to serve **Leanstral-2603 (119B-A6B, MLA)** efficiently (any GPU)

*A GPU-agnostic operations manual for one model. The reference tok/s is from one rig; the **recipe applies to any GPU**
— figure out your VRAM, then follow it.*

- **Architecture:** MoE + DeepSeek-MLA (119B total / 6.5B active, 128 routed +1 shared experts)
- **Fits in VRAM?** NO — Q4 ≈72 GB exceeds consumer VRAM; must offload experts to RAM

## Recipe
**Minimize the damage:** **`-fit on`** (auto-decides GPU vs CPU placement) `-fa on` + q8 KV. Do NOT pass `-ngl 99` (it aborts `-fit on`); do NOT blanket `--n-cpu-moe 40` (leaves the GPU ~87% idle → near-zero speed). `-fit on` keeps all active params (attn/shared/dense/KV) on GPU, spills only cold routed experts to RAM. MLA = tiny KV cache, so context is cheap.

**Serving flags (llama.cpp):**
```
-fit on -fa on --cache-type-k q8_0 --cache-type-v q8_0 --jinja --chat-template-file chat_template.jinja
```

## Reference throughput
0.94 → **22.4 tok/s (24×), 7/7 clean** just by switching the broken `--n-cpu-moe 40` to `-fit on`. ~34 reported on a 192 GB-RAM symmetric-2×24 GB rig — the gap is RAM bandwidth, not config.

## Failures → fixes
- `--n-cpu-moe 40` → 0.94 tok/s (over-offloaded).
- `ik_llama.cpp` (native `-mla`) **fatal-aborts** on mainline GGUFs: *'incompatible DeepSeek model'* (`llama-hparams.cpp`) — needs an **ik_llama-requantized** model (from Q8/safetensors).
- vLLM/SGLang are Mistral's official path but target the NVFP4 *base* model (not this fine-tune) on multi-GPU.

## Verdict
Usable at the `-fit on` config on any 48 GB-class setup. More RAM bandwidth → more speed.

---
## The one decision: does it FIT in your VRAM?
Estimate size ≈ params × bytes/weight (Q4≈0.5, Q8≈1, FP16≈2 B/param) + KV + ~2–3 GB overhead.
- **Fits** → full GPU residency, **no offload**, single card if it fits on one → *bandwidth-bound, fast.*
- **Doesn't fit** → offload experts to RAM (use `-fit on`), keep the active path on GPU → *RAM-bandwidth-bound, slower.*

## Measure honestly
Use the server's **`/completion` decode timings** (`predicted_per_second`), greedy, cache off, multiple workloads —
NOT short OpenAI wall-time (it understates decode). See `bench_decode.py`.

## Files
- `REPORT.md` — the detailed benchmark (throughput · config · tuning-research+sources · analysis · failures), if present.
- `bench_decode.py` — honest decode-tok/s measurement (`/completion` timings).
- `mctl-entry.json` — the `models.json` snippet for the [mctl](https://github.com/) switcher.

*Part of a per-model serving-playbook set. Cross-model comparison: see the `_ALL-MODELS-SUMMARY.md` overview.*
