# leanstral-2603 (119B-A6B, deepseek2/MLA) — team-urgent (Mohan / Lean 4)

**From unusable to usable: 0.94 → 22.4 tok/s (23.8×)** via `-fit on`. Mistral's Lean-4 theorem-proving MoE.
The win was entirely **config** — the hardware was never the limit. **Now 7/7 clean.**

## Decode throughput (7 workloads, `-fit on`)
| Workload | decode tok/s |
|---|---|
| Math / reasoning | **24.2** |
| Summarization | 24.2 |
| JSON / structured | 24.0 |
| Translation (multilingual) | 24.0 |
| Chat / dialogue | 23.0 |
| Code generation | 21.1 |
| Free-form prose | 16.2 |
| **Average** | **~22.4** (prefill ~21.7) |

## Engine comparison
| Engine | Config | Decode tok/s | Status |
|---|---|---|---|
| **llama.cpp (mainline b9733)** | `-fit on -fa on` q8-KV | **22.4** (7/7 clean) | ✅ working daily |
| llama.cpp (broken first config) | `--n-cpu-moe 40 --tensor-split 46,4` | 0.94 | config error |
| **ik_llama.cpp** | `--fit -mla3 -fmoe -rtr` | **n/a — INCOMPATIBLE** (see Failures) | blocked |
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
✅ **USABLE for the team at 22.4 tok/s, 7/7 clean** (mainline `-fit on`, coherent Lean-4, 23.8× the broken baseline).
Servable on-demand via `mctl switch leanstral` on :8005. **To reach ~34:** ik_llama requant (deferred) or more RAM
bandwidth. **Disk: KEPT** (team-urgent + one of the two keepers). **§E: complete — the 7-workload bench now lands
(earlier batch-switch run failed to allocate the 5080 buffer mid-switch; a clean single-model restart fixed it,
confirming it was a VRAM-not-cleared-on-switch artifact, not a model problem).**

_2026-06-27 · deep-research + empirical tuning + ik_llama build + 7/7 clean rerun · keyingd · `2026-06-27-benchmark-research-log.md` §7._

## Update 2026-06-30 — the quant-to-VRAM lever DOUBLES it: 22.4 → 43.2 tok/s
| Config | Decode | VRAM/RAM | Note |
|---|---|---|---|
| mainline `-fit on` Q4_K_M (72G) | 22.4 | 45G/27G | original |
| **mainline `-fit on` Q3_K_S (48G)** | **43.2** | **44G/9G** | **~1.9× — RAM offload 27→9G** |
| ik_llama (any -mla) | ❌ crash | — | `llama-hparams.cpp:1086` rejects deepseek2 GGUF (needs ik_llama-requant) |

**Finding:** Leanstral is only 6.5B-active — shrinking Q4→Q3 (self-quantized via llama-quantize) cut the RAM-resident experts from 27G→9G, nearly eliminating the slow-RAM bottleneck → **doubled decode**. ik_llama is a dead-end (hparams crash, MLA-independent). The lever is QUANT, on mainline llama.cpp. (Q3-from-Q4 is a lossy speed proof; redo from Q8 for a quality keeper.) Next: IQ2 to fit fully in 48G VRAM.

### Quant ladder — the peak is Q3_K_S
| Quant | Size | VRAM/RAM | Decode | |
|---|---|---|---|---|
| Q4_K_M | 72G | 45/27 | 22.4 | original |
| **Q3_K_S** | 48G | 44/9 | **43.2** | 🏆 PEAK (~1.9×) |
| Q2_K | 41G | 43/2 | ~37 (declining) | past sweet spot — lossier compute + degenerate output offset the RAM savings |
**Conclusion:** the optimal is **Q3_K_S = 43.2 tok/s** — shrinking further (Q2) doesn't help once it's mostly VRAM-resident. The quant-to-VRAM lever DOUBLED Leanstral; ik_llama remains a dead-end (deepseek2 hparams crash). For a quality keeper (Lean-4), the Q3_K_S should be re-quantized cleanly from Q8 (not the lossy Q4→Q3 used for this speed proof).

### FINAL: clean Q3_K_S (from Q8) = 49.5 tok/s — the production keeper (2.2×)
| Config | Decode | Quality | |
|---|---|---|---|
| Q4_K_M (orig) | 22.4 | full | baseline |
| lossy Q3 (from Q4) | 43.2 | degraded | speed proof |
| **clean Q3_K_S (from Q8)** | **49.5** | **coherent (Lean-4 OK)** | 🏆 **KEEPER, 2.2×** |
The clean-from-Q8 Q3 is both faster (49.5 vs 43.2 — cleaner tensor quant → more efficient compute) AND coherent. **Leanstral optimized: 22.4 → 49.5 tok/s, quality-preserving.** Serve for Mohan via `-fit on` on the clean Q3. VRAM 44G + RAM 9G. ik_llama remains dead-end (deepseek2 hparams crash); SGLang/vLLM can't offload → can't fit (only IQ2 fits VRAM, quality-broken).

### FINAL ENGINE VERDICT — all 3 alt-engines fail the deepseek2 GGUF; llama.cpp is the only path
| Engine | Leanstral GGUF result |
|---|---|
| **mainline llama.cpp** | ✅ **49.5 tok/s** (clean Q3, coherent) — the ONLY working engine (has RAM-offload) |
| ik_llama.cpp | ❌ crash `llama-hparams.cpp:1086` (deepseek2 rejected, MLA-independent) |
| **SGLang** | ❌ `ValueError: GGUF model with architecture deepseek2 is not supported yet` |
| vLLM | ❌ can't load GGUF at all + won't bind headless on this rig |

**Definitive:** mainline llama.cpp's CPU-RAM expert-offload is the ONLY viable Leanstral path on a 48GB rig — the alt-engines have no offload AND reject the deepseek2 GGUF arch. **49.5 tok/s (2.2× the original 22.4) is the ceiling**, via the quant-to-VRAM lever (Q4→clean Q3 from Q8, coherent). Dig closed.
