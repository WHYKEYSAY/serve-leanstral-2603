# leanstral-2603 (119B-A6B, deepseek2/MLA) — team-urgent (Mohan / Lean 4)

**From unusable to usable: 0.94 → 21.3 tok/s (22.6×)** via `-fit on`. Mistral's Lean-4 theorem-proving MoE.
The win was entirely **config** — the hardware was never the limit.

## Decode throughput
| Engine | Config | Decode tok/s | Workloads |
|---|---|---|---|
| **llama.cpp (mainline b9733)** | `-fit on -fa on` q8-KV | **21.3** (code 20.0 / prose 22.6) | 2/7 (bench_decode) — *full 7-workload pending a GPU window* |
| llama.cpp (broken first config) | `--n-cpu-moe 40 --tensor-split 46,4` | 0.94 | — |
| **ik_llama.cpp** | `--fit -mla3 -fmoe -rtr` | **n/a — INCOMPATIBLE** (see Failures) | — |
| vLLM / SGLang | NVFP4 base, TP≥2 | n/a | base model ≠ fine-tune, >48G — Mistral's path, not ours |

## Serving configuration (the working one)
| Param | Value |
|---|---|
| Backend | llama.cpp mainline (build b9733) |
| Quant | jackcloudman **Q4_K_M GGUF, 72 GB** (only Q4/Q8 exist; no sub-Q4) |
| Placement | **`-fit on`** auto: all active params (attn/dense/shared-expert/KV) on GPU, routed experts spill to CPU |
| VRAM | 5090 31.5 GB + 5080 13.9 GB = **45.4 GB on GPU**; ~27 GB cold experts on CPU RAM |
| KV / ctx / fa | `--cache-type-k/v q8_0` · 8192 · `-fa on` |
| Port | :8005 via `mctl switch leanstral` |

## Tuning-research (sources + chosen config + why)
Multi-agent deep-research (95 agents, adversarially verified; raw in `leanstral-deep-research-raw.json`).
**Sources:** Mistral `Leanstral-2603` + `Mistral-Small-4-119B-2603-NVFP4` cards; vLLM recipes; jackcloudman GGUF card
(**~34 tok/s on 2×4090 via `-fit on`** — the key number); the llama.cpp MoE-offload guides (DocShotgun); ik_llama docs.
**Architecture:** 119B total / **6.5B active**, DeepSeek **MLA**, **128 routed + 1 shared experts, 4 active/token** (~5.5% hot).
**Chosen config + why:** `-fit on`, NO manual `-ngl`/`--n-cpu-moe`/`--tensor-split`. The whole point: keep every
*always-active* param on GPU and offload **only** routed-expert FFN. My first config did the opposite — `-ngl 99`
*aborts* `-fit on`, and `--n-cpu-moe 40` blanket-paged ~all experts to CPU (GPU 87 % idle → 0.94).
**Spec-decode:** no Leanstral MTP/draft found; the MLA path (ik_llama) was the intended lever.

## Analysis — why 21.3, and the limiter
Leanstral (72 G) **doesn't fit** 48 G VRAM → it's on the **bad side of the VRAM-fit cliff**: ~27 G of cold experts
live in CPU RAM, read per token for the active 4 experts. **Limiter = CPU-RAM bandwidth** (our 90 GB DDR5-6000 ≈
80 GB/s). The jackcloudman card's ~34 was on **192 GB RAM + symmetric 2×4090**; our 90 GB + asymmetric 32+16
explains the 21 vs 34 gap — not config (we now match their `-fit on` recipe). The lever to close it (ik_llama's
faster MLA+MoE CPU kernels) is blocked — see Failures.

### Empirical tuning ladder
| Config | Result |
|---|---|
| `--n-cpu-moe 40 --tensor-split 46,4 --no-mmap` | loads, **0.94** (GPU 87 % idle) |
| `--n-cpu-moe 24 / 28 --tensor-split 34,12 / 30,12` | **5080 OOM** (16.8 / 16.1 G on 15 G free) |
| **`-fit on -fa q8KV` (best)** | **21.3**, 45 G on GPU, coherent Lean-4 ✓ |

## Failures (with the actual error)
- **ik_llama.cpp = incompatible with this GGUF.** Built clean (`--mla3 --fmoe --fit --run-time-repack` all present),
  recognizes deepseek2/MLA (`key/value_length_mla=128`), but **fatal-aborts at `llama-hparams.cpp:1086`**:
  *"Detected incompatible DeepSeek model without a known way to fix it — make your own ik_llama.cpp compatible model
  or ask the provider."* The mainline GGUF's MLA head dims fail ik_llama's check (`n_head_kv%4 / n_embd_head_k /
  n_embd_head_v / n_rot!=64`). **Fix = re-quantize Leanstral with ik_llama's own tooling** (from Q8 126 GB or the
  original safetensors) — a large job deferred; 21.3 is usable now.
- First config's 0.94 was a config error (`-ngl 99` aborting `-fit on` + blanket `--n-cpu-moe 40`), not hardware.

## Verdict
✅ **USABLE for the team at 21.3 tok/s** (mainline `-fit on`, coherent Lean-4, 22.6× the broken baseline). Servable
on-demand via `mctl switch leanstral` on :8005. **To reach ~34:** ik_llama requant (deferred) or more RAM bandwidth.
**Disk: KEPT** (team-urgent + one of the two keepers). Remaining §E item: full 7-workload bench (have 2/7).

_2026-06-27 · deep-research + empirical tuning + ik_llama build · keyingd · `2026-06-27-benchmark-research-log.md` §7._
