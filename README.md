# KAIROS-Lite: Weakly-Supervised Selective Acceptance & Calibration for LMs

## Executive Summary

**KAIROS-Lite** is a compact, reproducible system for **quality-aware selection** of language-model outputs under **coverage control**. It wraps a GPT-2 generator with a lightweight assessor that:

* pools hidden states into a latent vector,
* adds simple text features (sentences/length/repetition, optional inv-perplexity & prompt-continuation **coherence**),
* learns a tiny MLP head from **weak labels** (perplexity + min-sentences),
* **calibrates** scores (temperature or **isotonic**),
* gates outputs to hit a **target coverage**, and
* reports **risk–coverage**, **reliability**, **ECE/Brier**, bootstrap CIs, and a **cluster-preserving permutation (“WOW”)** lift test.

This repo also includes strong **coverage-matched baselines** (PPL\@cov, entropy\@cov, maxP\@cov, Platt-scaled heuristics, majority-of-k) and exports **CSV/JSON** artifacts plus plots for audit.

> **Safety note.** The controller abstains by design. Weak labels are **not ground truth**; use domain-appropriate GT and audits prior to deployment.

---

## Positioning & Applications

### Metacognitive Research — Teaching LMs to Monitor Themselves

A practical sandbox for **selective generation** and **post-hoc calibration** with statistical checks (bootstrap CIs, permutation lift). Useful for studying how confidence surrogates (latent+text features) correlate with quality proxies and GT when available.

### Hybrid Human–AI Select-or-Escalate

When quality matters more than coverage, accept **only** high-confidence candidates and escalate the rest. Tune **target coverage** to your operational budget; plug in GT tasks (QA/sum/safety) to measure selective accuracy.

---

## System Architecture Overview

### Core Components

* **TinyGen** — GPT-2 wrapper (any HuggingFace GPT-2 variant) that samples continuations and returns: text, perplexity, token entropies, mean max-prob, pooled hidden state, and a **prompt↔continuation cosine** coherence signal.
* **AssessorLite** — 64-d latent + features → **MLP head** → probability; supports **temperature** or **isotonic** calibration.
* **Coverage Gate** — selects using calibrated/uncal/raw score at a **target coverage**, with exact tie-fraction handling.
* **Baselines & Stats** — PPL/entropy/maxP/length @cov, Platt on heuristics, majority-of-k, **ECE/Brier/AUC**, CI via **iid or cluster bootstrap**, and **WOW** (cluster-preserving permutation) lift.
* **Reporter** — tables, plots (reliability, risk–coverage, score CDFs, ROC/PR), **CSV/JSON** summaries.

```
┌─────────────────────────────── KAIROS-Lite Pipeline ──────────────────────────────┐
│  Prompts ──► TinyGen (GPT-2) ─► Features & Latent ─► AssessorLite ─► Calibrator   │
│                                                                │                  │
│                                                                └─► Coverage Gate  │
│                                                                                   │
│  Baselines @cov (PPL, entropy, maxP, length, Platt, majority-of-k)                │
│  Stats: ECE/Brier/AUC • (Cluster) Bootstrap CIs • WOW permutation lift            │
│  Outputs: tables, plots, CSVs, JSON                                               │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
pip install torch transformers numpy matplotlib
```

> GPU is optional; CUDA is auto-detected.

### File Formats

* `data/prompts.txt` — one prompt per line (comments with `#` are ignored).
* *(Optional GT)* `task_data.jsonl` — JSONL with `{"prompt": ..., "answers": [...]}` (QA) or `{"prompt": ..., "references": [...]}` (summ) or `{"prompt": ..., "label": 0/1}` (safety/label).

### CLI Commands

#### Demo (single-seed, tables + plots + CSV/JSON)

```bash
python main.py demo \
  --prompts_file data/prompts.txt \
  --prompts_mode random \
  --primary_metric indep \
  --model gpt2-large \
  --gen_temp 0.70 \
  --k 5 \
  --select_by uncal \
  --coherence_feat \
  --calibrator isotonic \
  --weak_min_sents 2 \
  --weak_cap 55 \
  --cal_split 0.30 \
  --target_cov 0.40 \
  --n 60 \
  --n_boot 200 \
  --perm_reps 500 \
  --seed 43 \
  --out out_kairos_lite_explainer_ultra \
  --save_json
```

#### Benchmark (multi-seed CSV)

```bash
python main.py bench \
  --model gpt2-large \
  --n 40 --k 5 --cal_split 0.30 \
  --seeds 41 42 43 44 45 \
  --out out_kairos_lite_bench \
  --n_boot 1500
```

*(GT evaluation)* add e.g. `--task_type qa --task_data data/qa.jsonl --qa_f1_accept 0.5`.

---

## Generator (TinyGen) — what “ALETHIA” is in the sister repo

This repo does **not** ship the full ALETHIA agent; instead it uses **TinyGen**, a lean GPT-2 wrapper that:

* encodes pooled continuation states → **64-d latent** via a small MLP,
* returns **perplexity**, **mean token entropy**, **mean max-prob**, and an optional **coherence** feature (prompt vs continuation cosine),
* supports a QA short-answer post-processor (`Q:/A:` formatting + first-clause clipping).

---

## KAIROS: Metacognitive Assessment Layer

### Purpose

Estimate the probability that a candidate is “OK” under **weak labels** (perplexity < cap & ≥ min sentences) or **GT** when available.

### Assessment Dimensions

* **Latent** pooled representation (64-d by default)
* **Text features:** sentence count, normalized length, repetition ratio
* **Optional features:** inverse perplexity, **coherence** (prompt↔continuation cosine)

### Calibration Mechanisms

* **Temperature** or **Isotonic** calibration on a **cal-holdout** split.
* **Coverage gate** with **exact tie-fraction** so eval coverage tracks the target.

---

## Benchmark Results

### Demo run (seed 43; GPT-2-large; PRIMARY = `indep`)

**Target coverage:** 0.40 (picked on cal-hold)

| Arm                         | Coverage | Sel. Quality (PRIMARY) | Coverage Error | Notes                          |
| --------------------------- | :------: | :--------------------: | :------------: | ------------------------------ |
| **baseline (accept-all)**   |   1.000  |          0.870         |      0.600     | no selection                   |
| ppl\@cov (holdout)          |   0.219  |          0.885         |      0.181     | threshold on −perplexity       |
| **length\@cov (holdout)**   |   0.481  |        **0.896**       |      0.081     | word-count heuristic           |
| entropy\@cov (holdout)      |   0.281  |          0.865         |      0.119     | lower token entropy is better  |
| maxp\@cov (holdout)         |   0.248  |          0.854         |      0.152     | higher mean max-prob           |
| Platt(PPL)@cov              |   0.595  |          0.860         |      0.195     | 1-D logistic on PPL            |
| Platt(Len)@cov              |   0.719  |          0.869         |      0.319     | 1-D logistic on sentence count |
| majority-of-k (PPL)@cov     |   0.095  |          0.860         |      0.305     | prompt-level accept            |
| majority-of-k (Entropy)@cov |   0.167  |          0.862         |      0.233     | prompt-level accept            |
| assessor (uncal select)     |   0.271  |          0.874         |      0.129     | selects by uncal score         |
| **assessor (cal select)**   |   0.281  |        **0.891**       |      0.119     | isotonic-calibrated            |

Additional stats (assessor, cal-select):

* **ECE vs weak:** 0.052 → **0.034** (uncal→cal), **Brier:** 0.020 → **0.011**
* **AUC vs weak:** **0.952**
* **WOW (cluster-preserving) lift:** **+0.014** \[ +0.001, +0.025 ] (p = 0.032)

Artifacts:

* Plots in `out_kairos_lite_explainer_ultra/demo/*.png`
* Tables in `out_kairos_lite_explainer_ultra/demo/metrics_by_arm.csv`
* Summary JSON in `out_kairos_lite_explainer_ultra/demo/summary.json`

> *Repro note.* This demo run uses **proxies** (no GT). Enable `--task_type` + `--task_data` to evaluate against ground truth (EM/F1, ROUGE-L, or labels) and report **ECE/AUC vs GT**.

---

## Metacognition and Self-Awareness

This lite repo does **not** maintain a persistent self-model. A simple **coherence** feature (prompt↔continuation cosine) can be toggled via `--coherence_feat` and used by the assessor. For a broader proof of concept agentic implementation of this module please see [Alethia](https://cosmonaught.github.io/alethia/).

---

## Quality Control

### Validation Pipeline (weak labels & proxies)

* **Weak label**: `perplexity < --weak_cap` **and** `count_sents ≥ --weak_min_sents` (QA uses safer sentence counting for short answers).
* **Proxies**:

  * `quality_proxy_main(text, perp)` mixes inverse perplexity & sentence structure,
  * `quality_proxy_indep(text)` is independent of perplexity/length (punctuation/casing/cleanliness, long-word share, 3-gram diversity, reading ease).

### Statistics & Reporting

* **ECE/Brier/AUC** vs weak (and **vs GT** when provided)
* **(Cluster) bootstrap CIs** for coverage/quality
* **WOW permutation** lift preserving per-prompt accept counts
* **Risk–coverage** and **reliability** plots

---

## Output Examples

The demo prints a few accepted/rejected examples with:

```
[prob=... | sel=... (uncal) | primary=... | weak_ok=... | accepted=... | perp=...] <first 220 chars>...
```

You’ll also get reliability curves, risk–coverage, score CDFs, ROC/PR in `out/*/demo/`.

---

## Evaluation Metrics

### Coverage–Quality Trade-off

Tune `--target_cov` and select by `--select_by {raw,uncal,cal}`. Gate selection uses exact **tie-fraction** to match the target on cal-hold.

### Calibration

* **Temperature** or **isotonic** calibration,
* **ECE** (binning), **Brier score**, and **reliability** plots.

---

## File Structure

```
.
├── main.py                      # Everything: generator, assessor, baselines, stats, CLI
├── data/
│   └── prompts.txt              # Example prompts (one per line)
└── out_kairos_lite_explainer_ultra/
    └── demo/
        ├── *.png                # reliability, risk–coverage, ROC/PR, score CDFs
        ├── metrics_by_arm.csv   # per-arm table
        ├── wow_and_corr.csv     # WOW lift + proxy correlations
        └── summary.json         # full run summary
```

---

## Advanced Configuration

**Core**

* `--model {gpt2,gpt2-medium,gpt2-large,...}` — HF GPT-2 family
* `--k` — candidates per prompt (default 5)
* `--gen_temp` — sampling temperature
* `--n` — number of prompts; `--prompts_file/--prompts_mode {random,head}`

**Assessor & Calibration**

* `--latent_dim` — latent size (default 64)
* `--select_by {raw,uncal,cal}` — score used for gating
* `--cal_split` — fraction of prompts used for calibration/training
* `--cal_holdout_frac` — fraction of cal used for holdout (calibrator + tau)
* `--calibrator {temp,isotonic}` — post-hoc calibration
* `--target_cov` — desired coverage on cal-hold
* `--weak_cap` / `--weak_min_sents` — weak label definition
* `--coherence_feat` — include prompt↔continuation cosine as a feature
* `--ablate_invperp` / `--run_ablation` — remove inv-perp and/or run ablation arm

**Ground Truth (optional)**

* `--task_type {qa,summ,safety,label}`
* `--task_data path/to/*.jsonl`
* `--qa_f1_accept` / `--summ_f1_accept` — GT label threshold

**Statistics & Outputs**

* `--n_boot` / `--alpha` — bootstrap CIs
* `--perm_reps` — WOW permutations
* `--cluster_bootstrap` — prompt-cluster bootstrap instead of iid
* `--examples_k` — how many examples to print
* `--save_json` — write `summary.json`
* `--out` — output directory
* `--seed` — RNG seed

---

## Future Directions

1. **Ground-truth tasks by default** (QA/ROUGE/safety) with selective accuracy vs coverage.
2. **Conformal coverage control** to better generalize target coverage from holdout→eval.
3. **Model breadth** (LLama-style decoders, instruction-tuned variants) and cross-dataset validation.
4. **Feature ablations** (latent-only vs features-only), calibrator comparisons, and multi-coverage sweeps.
5. **Richer proxies** (task-aware, hallucination checks) and human evals.

---

## License

**MIT** — see `LICENSE`.
