#!/usr/bin/env python3
"""bench_decode.py — RIGOROUS throughput: uses llama-server /completion native timings
to separate DECODE tok/s from prompt-processing, + records VRAM. Addresses the
'numbers too low / missing metrics' critique (short-gen OpenAI wall-time understated decode).

  bench_decode.py --port 8011 --label "Qwen3.6-35B Q4 full-5090"
"""
import argparse, json, time, urllib.request, subprocess
from pathlib import Path

PROMPTS = {
    "code": "Write a complete, well-commented Python implementation of an LRU cache class with get/put O(1), plus a red-black tree insert. Be thorough.",
    "prose": "Write a detailed 500-word essay on how cities shape the people who live in them.",
}

def completion(port, prompt, n=400):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r.get("timings", {})

def vram():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        return [int(x) for x in out.split()]
    except Exception:
        return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=str(Path.home() / "bench" / "results_rigorous.jsonl"))
    a = ap.parse_args()
    print(f"=== {a.label} (:{a.port}) ===")
    rec = {"label": a.label, "port": a.port, "vram_mib": vram(), "workloads": {}}
    decs = []
    for name, prompt in PROMPTS.items():
        try:
            t = completion(a.port, prompt)
            d = round(t.get("predicted_per_second", 0), 1)
            p = round(t.get("prompt_per_second", 0), 1)
            rec["workloads"][name] = {"decode_tok_s": d, "prompt_tok_s": p, "n": t.get("predicted_n")}
            decs.append(d)
            print(f"  {name:6}  decode {d:6.1f} tok/s   prompt {p:7.1f} tok/s   ({t.get('predicted_n')} tok)")
        except Exception as e:
            rec["workloads"][name] = {"error": str(e)[:80]}
            print(f"  {name:6}  FAIL: {str(e)[:70]}")
    rec["decode_tok_s_avg"] = round(sum(decs) / len(decs), 1) if decs else None
    print(f"  VRAM used: {rec['vram_mib']} MiB   |   AVG decode {rec['decode_tok_s_avg']} tok/s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
