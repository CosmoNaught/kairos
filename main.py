import re, time, math, json, random, argparse, hashlib, csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ------------------------------ utils ----------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info():
    print("[INFO] ===============================================")
    print(f"[INFO] Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA:    {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")
        print(f"[INFO] GPU:     {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] CUDA not available; running on CPU")
    print("[INFO] ===============================================", flush=True)


def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- ROBUST sentence count (esp. for QA short answers) ---
def count_sents(text: str) -> int:
    t = (text or '')
    n = len(re.findall(r'[^.!?]+[.!?]', t))
    if n == 0:
        # treat a non-empty, punctuationless short answer as 1 sentence (QA-safe)
        return 1 if re.search(r"[A-Za-z]", t) else 0
    return n

# word-length helper for length@cov baseline
_WORD_RE = re.compile(r"[A-Za-z']+")
def _word_len(s: str) -> int:
    return len(_WORD_RE.findall(s or ""))



def inv_perp_feature(perp: float) -> float:
    return 1.0 / (1.0 + max(0.0, perp) / 30.0)


# ---------------- Quality proxies (kept) -----------------
# [KEEP] Main proxy remains for sanity checks only (not primary by default)
def sent_feature(text: str) -> float:
    return min(1.0, count_sents(text) / 3.0)


def quality_proxy_main(text: str, perp: float) -> float:
    return float(np.clip(0.65 * inv_perp_feature(perp) + 0.35 * sent_feature(text), 0.0, 1.0))

# [PRIMARY if no GT] Independent proxy (no perplexity/length)
VOWELS = set("aeiouy")


def _syllable_estimate(word: str) -> int:
    w = word.lower()
    if not w:
        return 1
    groups, prev_v = 0, False
    for ch in w:
        v = (ch in VOWELS)
        if v and not prev_v:
            groups += 1
        prev_v = v
    if w.endswith('e') and groups > 1:
        groups -= 1
    return max(1, groups)


def quality_proxy_indep(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    end_ok = 1.0 if t[-1] in ".!?" else 0.0
    sents = re.split(r'(?<=[.!?])\s+', t)
    caps = [1.0 if (s and s.lstrip()[0:1].isupper()) else 0.0 for s in sents if s]
    caps_ok = float(np.mean(caps)) if caps else 0.0
    clean = sum(ch.isalpha() or ch.isspace() or ch in ",.;:'\"-()" for ch in t)
    clean_ratio = clean / max(1, len(t))
    toks = [w for w in re.findall(r"[A-Za-z']+", t)]
    long_share = sum(len(w) >= 6 for w in toks) / max(1, len(toks))
    long_score = 1.0 - abs(long_share - 0.4) / 0.4
    long_score = float(np.clip(long_score, 0.0, 1.0))
    if len(toks) >= 3:
        trigs = [tuple(toks[i:i+3]) for i in range(len(toks) - 2)]
        trig_div = len(set(trigs)) / max(1, len(trigs))
    else:
        trig_div = 0.0
    words = len(toks); sents_n = max(1, len(sents)); syll = sum(_syllable_estimate(w) for w in toks)
    asl = words / sents_n; asw = syll / max(1, words)
    flesch_like = 206.835 - 1.015 * asl - 84.6 * asw
    read_norm = float(np.clip((flesch_like + 50.0) / 170.0, 0.0, 1.0))
    score = 0.20 * end_ok + 0.20 * caps_ok + 0.20 * clean_ratio + 0.15 * long_score + 0.15 * trig_div + 0.10 * read_norm
    return float(np.clip(score, 0.0, 1.0))


# ---------------- GT task evaluation [ADD] ------------------------------------

# Helpers for tokenization agnostic normalization, EM/F1, and LCS (Rouge-L-like)
_non_word = re.compile(r"\W+")

def _norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = _non_word.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def _tokenize(s: str) -> List[str]:
    return _norm_text(s).split()

def _em(pred: str, refs: List[str]) -> int:
    p = _norm_text(pred)
    return int(any(p == _norm_text(r) for r in refs))

def _f1(pred: str, refs: List[str]) -> float:
    p_toks = _tokenize(pred)
    if not p_toks: return 0.0
    best = 0.0
    for r in refs:
        r_toks = _tokenize(r)
        if not r_toks:
            best = max(best, 0.0); continue
        common = {}
        for t in p_toks:
            common[t] = min(p_toks.count(t), r_toks.count(t))
        num_same = sum(common.values())
        if num_same == 0:
            best = max(best, 0.0); continue
        precision = num_same / len(p_toks)
        recall = num_same / len(r_toks)
        f1 = (2 * precision * recall) / max(1e-9, (precision + recall))
        best = max(best, f1)
    return float(best)

def _lcs_length(a: List[str], b: List[str]) -> int:
    # classic DP for Rouge-L-like
    m, n = len(a), len(b)
    dp = [0]*(n+1)
    for i in range(1, m+1):
        prev = 0
        for j in range(1, n+1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    return dp[-1]

def _rouge_l_f1(pred: str, refs: List[str]) -> float:
    p = _tokenize(pred)
    best = 0.0
    for r in refs:
        rtk = _tokenize(r)
        if not rtk or not p:
            continue
        lcs = _lcs_length(p, rtk)
        prec = lcs / len(p)
        rec = lcs / len(rtk)
        f1 = (2*prec*rec) / max(1e-9, (prec + rec))
        best = max(best, f1)
    return float(best)

# Loader for task datasets (JSONL)
def load_task_data(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not path: return None
    p = Path(path)
    if not p.exists():
        print(f"[WARN] task_data path not found: {p}");
        return None
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                rows.append(json.loads(s))
            except Exception:
                pass
    if not rows:
        print(f"[WARN] task_data is empty or failed to parse: {p}")
    return rows


# ---------------- Correlations (kept) -----------------

def pearson_corr(a: List[float], b: List[float]) -> float:
    x = np.asarray(a, float); y = np.asarray(b, float)
    if len(x) < 2:
        return float('nan')
    x = x - x.mean(); y = y - y.mean()
    den = float(np.sqrt((x**2).sum()) * np.sqrt((y**2).sum()))
    return float((x * y).sum() / den) if den > 0 else float('nan')


def spearman_corr(a: List[float], b: List[float]) -> float:
    x = np.asarray(a, float); y = np.asarray(b, float)
    if len(x) < 2:
        return float('nan')
    def rank_avg(z):
        order = z.argsort(kind='mergesort')
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(z), dtype=float)
        uniq, inv, counts = np.unique(z, return_inverse=True, return_counts=True)
        for k, c in enumerate(counts):
            if c > 1:
                idx = np.where(inv == k)[0]
                ranks[idx] = ranks[idx].mean()
        return ranks
    return pearson_corr(rank_avg(x), rank_avg(y))


# ---------------- Bootstraps (iid + prompt-cluster) [ADD] --------------------

def bootstrap_ci(vals: List[float], iters=2000, alpha=0.05) -> Tuple[float, float]:
    if not vals:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(123)
    arr = np.array(vals, float); means = []
    for _ in range(iters):
        s = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(s)))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)

# [STATS] cluster bootstrap resamples prompts; all k candidates of a prompt move together
def cluster_bootstrap_ci(vals: List[float], clusters: List[Any], iters=2000, alpha=0.05) -> Tuple[float, float]:
    if not vals:
        return (float('nan'), float('nan'))
    vals = np.asarray(vals, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    rng = np.random.default_rng(123)
    means = []
    for _ in range(iters):
        sel = rng.choice(uniq, size=len(uniq), replace=True)
        m = np.isin(clusters, sel)
        vs = vals[m]
        if len(vs) == 0:
            means.append(float('nan'))
        else:
            means.append(float(vs.mean()))
    means = np.array(means, float)
    lo, hi = np.nanpercentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ECE/Brier (kept)
def ece_brier(preds: List[float], labels: List[int], n_bins=20) -> Tuple[float, float]:
    preds = np.asarray(preds, float); labels = np.asarray(labels, int)
    bins = np.linspace(0.0, 1.0, n_bins + 1); ece = 0.0; N = len(preds)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1] if i < n_bins - 1 else 1.0 + 1e-9
        m = (preds >= lo) & (preds < hi)
        if m.any():
            ece += (m.sum() / N) * abs(labels[m].mean() - preds[m].mean())
    brier = float(np.mean((preds - labels) ** 2))
    return float(ece), float(brier)


# ---- stats helpers ----
DEFAULT_Z = 1.959963984540054  # 95% normal quantile
def wilson_ci(p_hat: float, n: int, z: float = DEFAULT_Z) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denom
    rad = z * math.sqrt((p_hat * (1 - p_hat)) / n + (z ** 2) / (4 * n ** 2)) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))


# Average-rank AUC (proper tie handling)
def auc_roc(scores: List[float], labels: List[int]) -> float:
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos = s[y == 1]; neg = s[y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = s.argsort(kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    for k, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == k)[0]
            ranks[idx] = ranks[idx].mean()
    r_pos = ranks[y == 1].sum()
    auc = (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def bootstrap_ci_idx(N: int, stat_fn, iters: int = 2000, alpha: float = 0.05, seed: int = 123):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(iters):
        idx = rng.choice(N, size=N, replace=True)
        vals.append(float(stat_fn(idx)))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def ece_delta_bootstrap(preds_uncal, preds_cal, labels, iters=2000, alpha=0.05):
    p_u = np.asarray(preds_uncal, float)
    p_c = np.asarray(preds_cal, float)
    y = np.asarray(labels, int)
    N = len(y)

    def stat(idx):
        e_u, _ = ece_brier(p_u[idx].tolist(), y[idx].tolist(), n_bins=15)
        e_c, _ = ece_brier(p_c[idx].tolist(), y[idx].tolist(), n_bins=15)
        return e_u - e_c

    lo, hi = bootstrap_ci_idx(N, stat, iters=iters, alpha=alpha)
    return float(stat(np.arange(N))), float(lo), float(hi)


def permutation_pvalue_two_sided(diffs: np.ndarray) -> float:
    return float(2.0 * min(np.mean(diffs <= 0.0), np.mean(diffs >= 0.0)))


def stable_jitter(s: str, scale: float = 1e-6) -> float:
    h = hashlib.sha1(s.encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "big") / float(1 << 64)
    return (v + 1e-12) * scale


# ------------------------- prompts (kept) ------------------------------------
PROMPTS_DEFAULT = [
    "The fundamental principles of machine learning include",
    "We evaluate the effect of temperature on decoding by",
    "The limitations of small language models arise when",
    "A concise summary of the core assumptions behind transformers is",
    "In safety-constrained generation, the main trade-offs are",
    "Future directions for calibration in LMs include",
    "An experiment that reveals overconfidence would be",
    "The relationship between perplexity and answer quality is",
    "The role of coherence in judging acceptability is",
    "A minimal controller for accept/abstain can be designed by",
]


def load_prompts(n: int, seed: int, prompts_file: Optional[str], mode: str = "random") -> List[str]:
    set_seeds(seed)
    lines: List[str] = []
    if prompts_file and Path(prompts_file).exists():
        try:
            with open(prompts_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        lines.append(s)
        except Exception:
            lines = []
    if not lines:
        base = PROMPTS_DEFAULT[:]
        while len(base) < n:
            base.append(random.choice(PROMPTS_DEFAULT))
        random.shuffle(base)
        return base[:n]

    if len(lines) <= n:
        return lines[:]
    if mode == "head":
        return lines[:n]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lines), size=n, replace=False)
    return [lines[i] for i in idx]

def _format_prompt(pr: str, task_type: Optional[str]) -> str:
    """Light prompt formatting to steer models for specific tasks."""
    if task_type == "qa":
        pr = pr.strip()
        # If the file already contains 'Q:' or 'A:', don't double-wrap.
        if not pr.lower().startswith("q:"):
            return f"Q: {pr}\nA:"
    return pr

def _max_new_tokens_for_task(task_type: Optional[str]) -> int:
    # Keep QA tight so models are more likely to emit just the answer.
    return 24 if task_type == "qa" else 60

def _postprocess_qa_short(text: str) -> str:
    # Keep only first line → first clause; strip leading articles.
    t = text.splitlines()[0]
    t = re.split(r'[.!?]', t)[0]
    t = re.sub(r'^(the|a|an)\s+', '', t.strip(), flags=re.I)
    # ensure terminal punctuation so it counts as a sentence for weak labels
    if t and t[-1] not in ".!?":
        t = t + "."
    return t


# ------------------------- generator wrapper (extended) ----------------------
class TinyGen(nn.Module):
    def __init__(self, model_name="gpt2", latent_dim=64):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name)
        self.tok = GPT2Tokenizer.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token
        self.hidden_dim = self.gpt.config.hidden_size
        self.latent_dim = latent_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.latent_dim), nn.Tanh()
        )
        self.to(DEVICE).eval()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[INFO] Loaded {model_name} | params: {total_params/1e6:.1f}M", flush=True)

    @torch.inference_mode()
    def _continuation_perplexity_and_token_stats(self, gen_ids: torch.Tensor, input_len: int) -> Tuple[float, float, float]:
        """
        [CHANGE] Returns (perplexity, mean_token_entropy, mean_max_prob) over continuation.
        """
        ids = gen_ids.unsqueeze(0)
        T = ids.size(1); start = max(1, int(input_len))
        if T - start < 2:
            return 100.0, 10.0, 0.0
        logits = self.gpt(ids).logits  # [1,T,V]
        shift_logits = logits[:, start - 1:T - 1, :].contiguous()
        shift_labels = ids[:, start:T].contiguous()
        # PPL
        loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                               shift_labels.reshape(-1), reduction='mean')
        perp = float(torch.exp(loss).item())
        # Tokenwise entropy / max-prob
        with torch.no_grad():
            probs = F.softmax(shift_logits, dim=-1)  # [1,L,V]
            logp = torch.log(probs + 1e-9)
            ent = -(probs * logp).sum(dim=-1).mean().item()
            maxp = probs.max(dim=-1).values.mean().item()
        return perp, float(ent), float(maxp)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens=60, temperature=0.9) -> Dict[str, Any]:
        ids = self.tok(prompt, return_tensors="pt", truncation=True, max_length=180).to(DEVICE)
        t0 = time.time()
        out = self.gpt.generate(
            input_ids=ids["input_ids"],
            attention_mask=ids.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            min_new_tokens=max(24, int(0.5 * max_new_tokens)),
            do_sample=True,
            temperature=float(temperature),
            top_p=0.92, top_k=50,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            return_dict_in_generate=True,
        )
        gen_ids = out.sequences[0]
        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        if text and text[-1] not in ".!?":
            text += "."
        perp, mean_ent, mean_maxp = self._continuation_perplexity_and_token_stats(gen_ids, input_len=ids["input_ids"].shape[1])

        # Pool hidden states
        with torch.no_grad():
            hs_all = self.gpt(gen_ids.unsqueeze(0), output_hidden_states=True).hidden_states[-1]  # [1, T, H]
            start = ids["input_ids"].shape[1]
            if hs_all.size(1) > start:
                pooled = hs_all[:, start:, :].mean(dim=1).squeeze(0)  # [H]
            else:
                pooled = hs_all[:, -1:, :].mean(dim=1).squeeze(0)
            z = self.state_encoder(pooled)  # [D]

            # OPTIONAL coherence
            if start < hs_all.size(1):
                prompt_vec = hs_all[:, max(0, start-32):start, :].mean(dim=1)
                cont_vec   = hs_all[:, start:, :].mean(dim=1)
                cos = F.cosine_similarity(prompt_vec, cont_vec, dim=-1)
                coherence = float((cos.clamp(-1, 1).item() + 1.0) * 0.5)
            else:
                coherence = 0.5

        return {
            "text": text,
            "perplexity": float(perp),
            "mean_entropy": float(mean_ent),        # [ADD]
            "mean_maxp": float(mean_maxp),          # [ADD]
            "latent": z,
            "latency": time.time() - t0,
            "coherence": coherence,
            "n_new_tokens": int(len(gen_ids) - ids["input_ids"].shape[1])  # [FIX] correct token length diff
        }


# ----------------------------- assessor (lite) (kept + options) --------------

def mlp(d_in: int) -> nn.Module:
    m = nn.Sequential(
        nn.Linear(d_in, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1)
    ).to(DEVICE)
    for layer in m:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    m.eval(); return m


class AssessorLite:
    def __init__(self, latent_dim=64, weak_cap=55.0, weak_min_sents=3, calibrator: str = "temp",
                 use_invperp: bool = False, use_coherence: bool = False):  # [CHANGE] default inv-perp OFF
        # [CHANGE] features = latent + sent/len/rep + optional(inv-perp) + optional(coherence)
        self.use_invperp = bool(use_invperp)
        self.use_coherence = bool(use_coherence)
        extra = (1 if use_invperp else 0) + (1 if use_coherence else 0)
        self.d_in = latent_dim + (3 + extra)
        self.head = mlp(self.d_in)
        self.temp = 1.0
        self.calibrated = False
        self.weak_cap = float(weak_cap)
        self.weak_min_sents = int(weak_min_sents)
        self.calibrator = calibrator  # "temp" or "isotonic"
        self.iso_x = None
        self.iso_z = None

    def weak_label(self, perp: float, text: str) -> int:
        return int((perp < self.weak_cap) and (count_sents(text) >= self.weak_min_sents))

    def _len_feat(self, text: str) -> float:
        return min(1.0, max(0.0, len((text or "").split()) / 60.0))

    def _rep_feat(self, text: str) -> float:
        toks = (text or "").split()
        if len(toks) < 3:
            return 0.0
        bigrams = [tuple(toks[i:i + 2]) for i in range(len(toks) - 1)]
        ub = len(set(bigrams)); tb = max(1, len(bigrams))
        rep = 1.0 - (ub / tb)
        return float(np.clip(rep, 0.0, 1.0))

    def _feats(self, z: torch.Tensor, perp: float, text: str, coh: Optional[float] = None) -> torch.Tensor:
        feats = [
            inv_perp_feature(perp) if self.use_invperp else None,
            sent_feature(text),
            self._len_feat(text),
            self._rep_feat(text),
            (float(coh) if (self.use_coherence and (coh is not None)) else None)
        ]
        feats = [f for f in feats if f is not None]
        f = torch.tensor(feats, device=DEVICE)
        return torch.cat([z, f], dim=0)

    def train_head(self, Z, PERP, TEXTS, COH=None, lr=1e-3, epochs=120, rebalance=True, target_pos=0.5):
        if not Z:
            self.calibrated = False; self.temp = 1.0; return

        def build_labels(cap, min_s):
            return [int((p < cap) and (count_sents(t) >= min_s)) for p, t in zip(PERP, TEXTS)]

        X = [self._feats(z, p, t, (COH[i] if COH is not None else None)) for i, (z, p, t) in enumerate(zip(Z, PERP, TEXTS))]
        Y = build_labels(self.weak_cap, self.weak_min_sents)

        if rebalance and (all(y == 0 for y in Y) or all(y == 1 for y in Y)):
            cap_bal = float(np.quantile(np.asarray(PERP, float), target_pos))
            Y = build_labels(cap_bal, self.weak_min_sents)
            if all(y == 0 for y in Y) or all(y == 1 for y in Y):
                Y = build_labels(cap_bal, self.weak_min_sents + 1)
            if all(y == 0 for y in Y) or all(y == 1 for y in Y):
                self.head.eval(); self.calibrated = False; self.temp = 1.0; return

        X = torch.stack(X, dim=0).to(DEVICE)
        Yt = torch.tensor(Y, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        self.head.train()
        opt = torch.optim.Adam(self.head.parameters(), lr=lr, weight_decay=1e-5)
        for _ in range(epochs):
            pred = torch.sigmoid(self.head(X))
            loss = F.binary_cross_entropy(pred, Yt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            opt.step()
        self.head.eval()

    @torch.inference_mode()
    def _head_logits_and_labels(self, Z, PERP, TEXTS, COH):
        X = [self._feats(z, p, t, c) for z, p, t, c in zip(Z, PERP, TEXTS, COH)]
        X = torch.stack(X, dim=0).to(DEVICE)
        logits = self.head(X).squeeze(1).detach().cpu().numpy().tolist()
        labels = [self.weak_label(p, t) for p, t in zip(PERP, TEXTS)]
        return logits, labels

    def _fit_isotonic(self, probs: np.ndarray, labels: np.ndarray):
        o = np.argsort(probs)
        x_s, y_s = probs[o], labels[o].astype(float)
        vals, wts = [], []
        for yi in y_s:
            vals.append(float(yi)); wts.append(1.0)
            while len(vals) >= 2 and vals[-2] > vals[-1]:
                new_w = wts[-2] + wts[-1]
                new_v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / new_w
                vals[-2:] = [new_v]; wts[-2:] = [new_w]
        z = np.empty_like(y_s, dtype=float)
        k = 0
        for v, w in zip(vals, wts):
            cnt = int(round(w)); z[k:k + cnt] = v; k += cnt
        return x_s, z

    def _fit_temperature(self, logits: np.ndarray, labels: np.ndarray):
        logit_t = torch.tensor(logits, dtype=torch.float32, device=DEVICE)
        label_t = torch.tensor(labels, dtype=torch.float32, device=DEVICE)
        x = torch.tensor(0.0, requires_grad=True, device=DEVICE)  # T = exp(x)
        opt = torch.optim.Adam([x], lr=0.1)
        for _ in range(300):
            opt.zero_grad()
            T = torch.exp(x)
            q = torch.sigmoid(logit_t / T)
            loss = F.binary_cross_entropy(q, label_t)
            loss.backward(); opt.step()
        return float(torch.exp(x).detach().cpu().item())

    def fit_calibrator(self, ZH, PH, TH, CH):
        logits, labels = self._head_logits_and_labels(ZH, PH, TH, CH)
        if not (any(labels) and any(1 - l for l in labels)):
            self.calibrated = False; self.temp = 1.0; self.iso_x = None; self.iso_z = None
            return
        if self.calibrator == "isotonic":
            p = 1.0 / (1.0 + np.exp(-np.asarray(logits, float)))
            self.iso_x, self.iso_z = self._fit_isotonic(p, np.asarray(labels, float))
            self.calibrated = True
        else:
            self.temp = self._fit_temperature(np.asarray(logits, float), np.asarray(labels, float))
            self.calibrated = True

    @torch.inference_mode()
    def score_raw(self, z: torch.Tensor, perp: float, text: str, kind: str = "raw", coh: Optional[float] = None) -> float:
        x = self._feats(z, perp, text, coh).unsqueeze(0)
        logit = self.head(x).squeeze().item()
        if kind == "raw":
            return float(logit)
        if kind == "uncal":
            return float(1 / (1 + math.exp(-logit)))
        if not self.calibrated:
            return float(1 / (1 + math.exp(-logit)))
        if self.calibrator == "isotonic" and self.iso_x is not None:
            p = 1.0 / (1.0 + math.exp(-logit))
            if p <= float(self.iso_x[0]):
                return float(self.iso_z[0])
            if p >= float(self.iso_x[-1]):
                return float(self.iso_z[-1])
            j = int(np.searchsorted(self.iso_x, p, side="right")) - 1
            return float(self.iso_z[j])
        zt = logit / max(1e-6, self.temp)
        return float(1 / (1 + math.exp(-zt)))


# ------------------------------ coverage control (kept) ----------------------

def choose_cut_with_tie_fraction(preds: List[float], target_cov: float) -> Tuple[float, float]:
    s = np.sort(np.asarray(preds, float))
    N = len(s)
    if N == 0:
        return 1.0, 0.0
    target_cov = float(np.clip(target_cov, 0.0, 1.0))
    if target_cov <= 0.0:
        return float(np.nextafter(s.max(), 2.0)), 0.0
    if target_cov >= 1.0:
        return float(np.nextafter(s.min(), -1.0)), 1.0
    k = int(math.ceil(target_cov * N))
    k = max(1, min(N, k))
    tau_cut = s[N - k]
    num_strict = int(np.sum(s > tau_cut))
    num_equal = int(np.sum(s == tau_cut))
    need_from_eq = k - num_strict
    eq_frac = 0.0 if num_equal == 0 else float(np.clip(need_from_eq / num_equal, 0.0, 1.0))
    return float(tau_cut), eq_frac


def accept_with_tie_break(conf: float, tau_cut: float, eq_frac: float, rng: np.random.Generator, jitter: float = 0.0) -> int:
    if (conf + jitter) > tau_cut:
        return 1
    if (conf + jitter) < tau_cut or eq_frac <= 0.0:
        return 0
    return int(rng.random() < eq_frac)


# ------------------------------ evaluation data model ------------------------
@dataclass
class Sample:
    arm: str
    prompt: str
    text: str
    conf: float
    conf_uncal: float
    sel_score: float
    sel_kind: str
    weak_ok: int
    accepted: int
    quality_main: float
    quality_indep: float
    latency: float
    perp: float
    mean_entropy: float
    mean_maxp: float
    # Ground-truth fields [ADD]
    gt_score: Optional[float] = None
    gt_label: Optional[int] = None


# ------------------------------ primary metric chooser [ADD] -----------------

def sample_primary_value(s: Sample, primary: str) -> float:
    """
    primary in {"indep","gt","main"}
    """
    if primary == "gt" and (s.gt_score is not None):
        return float(s.gt_score)
    if primary == "indep":
        return float(s.quality_indep)
    if primary == "main":
        return float(s.quality_main)
    # fallback order: gt -> indep -> main
    return float(s.gt_score if s.gt_score is not None else s.quality_indep if s.quality_indep is not None else s.quality_main)


# ------------------------------ metrics summarizer (extended) ----------------

def _mean_or_nan(xs: List[float]) -> float:
    if not xs: return float('nan')
    return float(np.mean(xs))

def summarize(samples: List[Sample],
              *,
              target_cov: Optional[float] = None,
              n_boot: int = 2000,
              alpha: float = 0.05,
              primary_metric: str = "indep",
              cluster_bootstrap: bool = False) -> Dict[str, float]:
    N = len(samples)
    coverage = np.mean([s.accepted for s in samples]) if samples else 0.0

    # Primary selective quality (NaN when no accepts) [FIX nicer handling]
    prim_vals = [sample_primary_value(s, primary_metric) for s in samples if s.accepted]
    sel_quality_primary = _mean_or_nan(prim_vals)

    # Legacy + independent for reporting:
    sel_quality_main = _mean_or_nan([s.quality_main for s in samples if s.accepted])
    sel_quality_indep = _mean_or_nan([s.quality_indep for s in samples if s.accepted])

    lat = _mean_or_nan([s.latency for s in samples])
    preds_cal = [s.conf for s in samples]
    preds_uncal = [s.conf_uncal for s in samples]
    labels_weak = [s.weak_ok for s in samples]

    # Calibration vs weak labels (kept)
    ece_cal_w, brier_cal_w = ece_brier(preds_cal, labels_weak, n_bins=20)
    ece_unc_w, brier_unc_w = ece_brier(preds_uncal, labels_weak, n_bins=20)
    auc_w = auc_roc(preds_cal, labels_weak)

    # Calibration vs GT labels if present [ADD]
    gt_present = any(s.gt_label is not None for s in samples)
    if gt_present:
        labels_gt = [int(s.gt_label) for s in samples]
        ece_cal_gt, brier_cal_gt = ece_brier(preds_cal, labels_gt, n_bins=20)
        ece_unc_gt, brier_unc_gt = ece_brier(preds_uncal, labels_gt, n_bins=20)
        auc_gt = auc_roc(preds_cal, labels_gt)
    else:
        ece_cal_gt = ece_unc_gt = brier_cal_gt = brier_unc_gt = float('nan')
        auc_gt = float('nan')

    # CIs (iid or cluster) for coverage and selected qualities
    accs = [s.accepted for s in samples]
    clusters = [s.prompt for s in samples]
    if cluster_bootstrap:
        cov_lo, cov_hi = cluster_bootstrap_ci(accs, clusters, iters=n_boot, alpha=alpha)
        selp_lo, selp_hi = cluster_bootstrap_ci([sample_primary_value(s, primary_metric) for s in samples if s.accepted] or [],
                                                [s.prompt for s in samples if s.accepted], iters=n_boot, alpha=alpha)
        selq_lo, selq_hi = cluster_bootstrap_ci([s.quality_main for s in samples if s.accepted] or [],
                                                [s.prompt for s in samples if s.accepted], iters=n_boot, alpha=alpha)
        selqi_lo, selqi_hi = cluster_bootstrap_ci([s.quality_indep for s in samples if s.accepted] or [],
                                                  [s.prompt for s in samples if s.accepted], iters=n_boot, alpha=alpha)
    else:
        cov_lo, cov_hi = bootstrap_ci(accs or [], iters=n_boot, alpha=alpha)
        selp_lo, selp_hi = bootstrap_ci([sample_primary_value(s, primary_metric) for s in samples if s.accepted] or [], iters=n_boot, alpha=alpha)
        selq_lo, selq_hi = bootstrap_ci([s.quality_main for s in samples if s.accepted] or [], iters=n_boot, alpha=alpha)
        selqi_lo, selqi_hi = bootstrap_ci([s.quality_indep for s in samples if s.accepted] or [], iters=n_boot, alpha=alpha)

    lat_lo, lat_hi = bootstrap_ci([s.latency for s in samples] or [], iters=n_boot, alpha=alpha)

    tgt = target_cov if target_cov is not None else coverage
    cov_err = abs(coverage - tgt)
    if samples and (target_cov is not None):
        rng = np.random.default_rng(123)
        boot = []
        arr = np.array(accs, float)
        for _ in range(n_boot):
            idx = rng.choice(len(arr), size=len(arr), replace=True)
            cov_b = float(arr[idx].mean())
            boot.append(abs(cov_b - target_cov))
        cov_err_ci_lo, cov_err_ci_hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]).tolist()
    else:
        cov_err_ci_lo, cov_err_ci_hi = 0.0, 0.0

    return {
        "total": N,
        "coverage": coverage, "coverage_ci_lo": cov_lo, "coverage_ci_hi": cov_hi,
        "selective_quality_primary": sel_quality_primary, "selective_quality_primary_ci_lo": selp_lo, "selective_quality_primary_ci_hi": selp_hi,
        "selective_quality": sel_quality_main, "selective_quality_ci_lo": selq_lo, "selective_quality_ci_hi": selq_hi,
        "selective_quality_indep": sel_quality_indep, "selective_quality_indep_ci_lo": selqi_lo, "selective_quality_indep_ci_hi": selqi_hi,
        "mean_latency_s": lat, "mean_latency_s_ci_lo": lat_lo, "mean_latency_s_ci_hi": lat_hi,
        # [FIX] Correct Brier storage
        "ece_uncal_weak": ece_unc_w, "ece_cal_weak": ece_cal_w, "brier_uncal_weak": brier_unc_w, "brier_cal_weak": brier_cal_w, "auc_weak": auc_w,
        "ece_uncal_gt": ece_unc_gt, "ece_cal_gt": ece_cal_gt, "brier_uncal_gt": brier_unc_gt, "brier_cal_gt": brier_cal_gt, "auc_gt": auc_gt,
        "cov_err": cov_err, "cov_err_ci_lo": cov_err_ci_lo, "cov_err_ci_hi": cov_err_ci_hi,
        "primary_metric": primary_metric
    }


def _fmt_val(x: float) -> str:
    return "—" if (x is None or (isinstance(x, float) and (math.isnan(x)))) else f"{x:.3f}"

def print_table(tag: str, row: Dict[str, float], *, prob_metrics: bool = True):
    print(f"\n=== {tag} ===")
    print("metric\tvalue")
    print(f"coverage\t{row['coverage']:.3f} [{_fmt_val(row['coverage_ci_lo'])},{_fmt_val(row['coverage_ci_hi'])}]")
    print(f"sel_quality(PRIMARY={row.get('primary_metric','indep')})\t{_fmt_val(row['selective_quality_primary'])} [{_fmt_val(row['selective_quality_primary_ci_lo'])},{_fmt_val(row['selective_quality_primary_ci_hi'])}]")
    print(f"sel_quality(main)\t{_fmt_val(row['selective_quality'])} [{_fmt_val(row['selective_quality_ci_lo'])},{_fmt_val(row['selective_quality_ci_hi'])}]")
    print(f"sel_quality(indep)\t{_fmt_val(row['selective_quality_indep'])} [{_fmt_val(row['selective_quality_indep_ci_lo'])},{_fmt_val(row['selective_quality_indep_ci_hi'])}]")
    print(f"latency_s\t{_fmt_val(row['mean_latency_s'])} [{_fmt_val(row['mean_latency_s_ci_lo'])},{_fmt_val(row['mean_latency_s_ci_hi'])}]")
    if prob_metrics:
        print(f"ECE_weak (uncal → cal)\t{_fmt_val(row['ece_uncal_weak'])} → {_fmt_val(row['ece_cal_weak'])} | Brier_weak {_fmt_val(row['brier_uncal_weak'])} → {_fmt_val(row['brier_cal_weak'])}")
        if not math.isnan(row.get("ece_cal_gt", float('nan'))):
            print(f"ECE_GT   (uncal → cal)\t{_fmt_val(row['ece_uncal_gt'])} → {_fmt_val(row['ece_cal_gt'])} | Brier_GT {_fmt_val(row['brier_uncal_gt'])} → {_fmt_val(row['brier_cal_gt'])}")
        print(f"AUC (vs weak)\t{_fmt_val(row['auc_weak'])}")
        if not math.isnan(row.get("auc_gt", float('nan'))):
            print(f"AUC (vs GT)\t{_fmt_val(row['auc_gt'])}")
    else:
        print("ECE/AUC\t—")
    if 'cov_err' in row:
        print(f"Coverage error |cov−target|\t{row['cov_err']:.3f} [{_fmt_val(row['cov_err_ci_lo'])},{_fmt_val(row['cov_err_ci_hi'])}]")


# ------------------------------- NEW PLOTS (extended) ------------------------

def _roc_pr(labels: np.ndarray, scores: np.ndarray):
    # [FIX] threshold sorted array to be consistent with y-ordering
    order = np.argsort(scores)
    s = scores[order]; y = labels[order]
    uniq = np.unique(s)
    tpr, fpr, prec, rec = [], [], [], []
    P = y.sum(); N = len(y) - P
    for t in np.r_[uniq, [uniq[-1] + 1e-9]]:
        m = s >= t  # [FIX]
        TP = (y[m] == 1).sum(); FP = (y[m] == 0).sum()
        FN = P - TP
        tpr.append(TP / max(1, P))
        fpr.append(FP / max(1, N))
        p = TP / max(1, TP + FP); r = TP / max(1, TP + FN)
        prec.append(p); rec.append(r)
    return np.array(fpr), np.array(tpr), np.array(rec), np.array(prec)


def plot_all(out: Path, samples: List[Sample], *, title_prefix: str = "", primary_metric: str = "indep"):
    out.mkdir(parents=True, exist_ok=True)
    lbl = np.array([s.weak_ok for s in samples], int)
    lbl_gt = np.array([(-1 if s.gt_label is None else s.gt_label) for s in samples], int)
    prob_u = np.array([s.conf_uncal for s in samples], float)
    prob_c = np.array([s.conf for s in samples], float)
    q_primary = np.array([sample_primary_value(s, primary_metric) for s in samples], float)

    # 1) Reliability (20 bins) — annotate PRIMARY
    def rel_bins(p):
        edges = np.linspace(0, 1, 21)
        xs, ys, ns = [], [], []
        for i in range(20):
            lo, hi = edges[i], edges[i + 1] if i < 19 else 1.0 + 1e-9
            m = (p >= lo) & (p < hi)
            if m.any():
                xs.append(float(p[m].mean())); ys.append(float(lbl[m].mean())); ns.append(int(m.sum()))
        return np.array(xs), np.array(ys), np.array(ns)

    for tag, p in [("uncal", prob_u), ("cal", prob_c)]:
        xs, ys, ns = rel_bins(p)
        plt.figure()
        plt.plot(xs, ys, marker='o', label=f"{tag}")
        plt.plot([0, 1], [0, 1], '--', linewidth=1.0)
        plt.xlabel("Predicted probability"); plt.ylabel("Empirical weak-OK rate")
        plt.title(f"{title_prefix}Reliability ({tag}) • PRIMARY={primary_metric}")
        plt.tight_layout(); plt.savefig(out / f"reliability_{tag}.png", dpi=160); plt.close()

    # 2) Risk–coverage on PRIMARY (fine sweep)
    taus = np.linspace(0.02, 0.98, 49)
    covs, risks = [], []
    base_q = float(np.nanmean(q_primary))
    for t in taus:
        m = prob_c >= t
        cov = float(m.mean()); covs.append(cov)
        sq = float(np.nanmean(q_primary[m])) if m.any() else float('nan')
        risks.append(1.0 - sq if not math.isnan(sq) else float('nan'))
    plt.figure(); plt.plot(covs, risks, marker='o')
    if not math.isnan(base_q):
        plt.axhline(1.0 - base_q, linestyle='--', linewidth=1.0, label='random@cov')
    plt.xlabel("Coverage"); plt.ylabel("Selective risk (1 - PRIMARY)")
    plt.title(f"{title_prefix}Risk–coverage (cal) • PRIMARY={primary_metric}")
    if not math.isnan(base_q):
        plt.legend()
    plt.tight_layout()
    plt.savefig(out / "risk_coverage_primary.png", dpi=160); plt.close()

    # 3) Score CDFs (uncal vs cal)
    for tag, p in [("uncal", prob_u), ("cal", prob_c)]:
        xs = np.sort(p)
        ys = np.linspace(0, 1, len(xs))
        plt.figure(); plt.plot(xs, ys)
        plt.xlabel("Score"); plt.ylabel("CDF")
        plt.title(f"{title_prefix}Score CDF: {tag} • PRIMARY={primary_metric}")
        plt.tight_layout(); plt.savefig(out / f"score_cdf_{tag}.png", dpi=160); plt.close()

    # 4) ROC & PR (cal) vs weak labels (kept)
    fpr, tpr, rec, prec = _roc_pr(lbl, prob_c)
    plt.figure(); plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{title_prefix}ROC (cal vs weak) • PRIMARY={primary_metric}"); plt.tight_layout()
    plt.savefig(out / "roc_cal_weak.png", dpi=160); plt.close()

    plt.figure(); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title_prefix}PR (cal vs weak) • PRIMARY={primary_metric}"); plt.tight_layout()
    plt.savefig(out / "pr_cal_weak.png", dpi=160); plt.close()

    # 5) [ADD] Selective accuracy vs coverage when GT labels exist
    if np.any(lbl_gt >= 0):
        gt = lbl_gt.copy()
        covs2, accs2 = [], []
        for t in taus:
            m = prob_c >= t
            covs2.append(float(m.mean()))
            accs2.append(float(gt[m].mean()) if m.any() else float('nan'))
        plt.figure(); plt.plot(covs2, accs2, marker='o')
        plt.xlabel("Coverage"); plt.ylabel("Selective accuracy (GT)")
        plt.title(f"{title_prefix}Selective accuracy vs coverage (cal) • PRIMARY={primary_metric}")
        plt.tight_layout(); plt.savefig(out / "sel_acc_vs_cov_gt.png", dpi=160); plt.close()


# ------------------------------ CSV writers (extended) -----------------------

def write_csv(path: Path, header: List[str], rows: List[List[Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_metrics_csv(out_dir: Path, table: Dict[str, Dict[str, float]], wow: Dict[str, Any], corr: Dict[str, float]):
    header = [
        "arm", "total", "coverage", "coverage_ci_lo", "coverage_ci_hi",
        "primary_metric", "sel_quality_primary", "sel_quality_primary_ci_lo", "sel_quality_primary_ci_hi",
        "sel_quality_main", "sel_quality_main_ci_lo", "sel_quality_main_ci_hi",
        "sel_quality_indep", "sel_quality_indep_ci_lo", "sel_quality_indep_ci_hi",
        "latency_s", "latency_s_ci_lo", "latency_s_ci_hi",
        "ece_uncal_weak", "ece_cal_weak", "brier_uncal_weak", "brier_cal_weak", "auc_weak",
        "ece_uncal_gt", "ece_cal_gt", "brier_uncal_gt", "brier_cal_gt", "auc_gt",
        "cov_err", "cov_err_ci_lo", "cov_err_ci_hi"
    ]
    rows = []
    for arm, row in table.items():
        rows.append([
            arm, int(row.get("total", 0)),
            row.get("coverage", 0.0), row.get("coverage_ci_lo", float('nan')), row.get("coverage_ci_hi", float('nan')),
            row.get("primary_metric", "indep"),
            row.get("selective_quality_primary", float('nan')), row.get("selective_quality_primary_ci_lo", float('nan')), row.get("selective_quality_primary_ci_hi", float('nan')),
            row.get("selective_quality", float('nan')), row.get("selective_quality_ci_lo", float('nan')), row.get("selective_quality_ci_hi", float('nan')),
            row.get("selective_quality_indep", float('nan')), row.get("selective_quality_indep_ci_lo", float('nan')), row.get("selective_quality_indep_ci_hi", float('nan')),
            row.get("mean_latency_s", float('nan')), row.get("mean_latency_s_ci_lo", float('nan')), row.get("mean_latency_s_ci_hi", float('nan')),
            row.get("ece_uncal_weak", float('nan')), row.get("ece_cal_weak", float('nan')), row.get("brier_uncal_weak", float('nan')), row.get("brier_cal_weak", float('nan')), row.get("auc_weak", float('nan')),
            row.get("ece_uncal_gt", float('nan')), row.get("ece_cal_gt", float('nan')), row.get("brier_uncal_gt", float('nan')), row.get("brier_cal_gt", float('nan')), row.get("auc_gt", float('nan')),
            row.get("cov_err", 0.0), row.get("cov_err_ci_lo", 0.0), row.get("cov_err_ci_hi", 0.0),
        ])
    write_csv(out_dir / "metrics_by_arm.csv", header, rows)

    header2 = ["lift_primary", "lift_ci_lo", "lift_ci_hi", "p_value", "ece_delta", "ece_delta_ci_lo", "ece_delta_ci_hi", "pearson_main_indep", "spearman_main_indep"]
    rows2 = [[
        wow.get("lift", 0.0), wow.get("lift_ci", [float('nan'), float('nan')])[0], wow.get("lift_ci", [float('nan'), float('nan')])[1], wow.get("pval", 1.0),
        wow.get("ece_delta", float('nan')), wow.get("ece_delta_ci", [float('nan'), float('nan')])[0], wow.get("ece_delta_ci", [float('nan'), float('nan')])[1],
        corr.get("pearson", float('nan')), corr.get("spearman", float('nan'))
    ]]
    write_csv(out_dir / "wow_and_corr.csv", header2, rows2)


# ------------------------------- NEW BASELINES @cov [ADD] --------------------

def baseline_platt(scores_hold: List[float], labels_hold: List[int]) -> Callable[[float], float]:
    """
    Fit 1D Platt scaling (logistic on a single score) on holdout.
    Returns a function mapping score -> prob.
    """
    x = torch.tensor(scores_hold, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    y = torch.tensor(labels_hold, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    w = torch.zeros((1,1), requires_grad=True, device=DEVICE)
    b = torch.zeros((1,), requires_grad=True, device=DEVICE)
    opt = torch.optim.Adam([w,b], lr=0.1)
    for _ in range(400):
        z = x @ w + b
        p = torch.sigmoid(z)
        loss = F.binary_cross_entropy(p, y)
        opt.zero_grad(); loss.backward(); opt.step()
    def mapper(s: float) -> float:
        with torch.no_grad():
            z = torch.tensor([[s]], dtype=torch.float32, device=DEVICE) @ w + b
            return float(torch.sigmoid(z).item())
    return mapper


# [NEW BASELINE] Majority-of-k (prompt-level acceptance) ----------------------
def majority_of_k_at_cov(samples: List[Sample], hold_scores: List[float], target_cov: float, m_ratio: float = 0.6, kind: str = "ppl") -> List[Sample]:
    """
    Accept or reject an entire prompt depending on how many of its k candidates exceed a holdout threshold.
    - hold_scores are used to choose the tau/eq on CAL-HOLD (scalar per candidate in holdout).
    - kind is only for labeling.
    """
    tau, eq = choose_cut_with_tie_fraction(hold_scores, target_cov)
    by_prompt: Dict[str, List[Sample]] = {}
    for s in samples:
        by_prompt.setdefault(s.prompt, []).append(s)
    out = []
    rng = np.random.default_rng(0)
    for prompt, group in by_prompt.items():
        # derive candidate scores for this prompt
        if kind == "ppl":
            cand_scores = [-g.perp for g in group]
        elif kind == "entropy":
            cand_scores = [-g.mean_entropy for g in group]
        else:
            cand_scores = [g.sel_score for g in group]
        m_needed = int(math.ceil(m_ratio * len(group)))
        strict = sum(sc > tau for sc in cand_scores)
        ties = sum(sc == tau for sc in cand_scores)
        passes = strict + rng.binomial(ties, eq)
        accept_prompt = int(passes >= m_needed)
        for g in group:
            ng = Sample(**{**g.__dict__})
            ng.arm = f"majority_{kind}@cov"
            ng.accepted = accept_prompt
            out.append(ng)
    return out


# ------------------------------- runs ----------------------------------------

def run_split(args, gen: TinyGen, assessor: AssessorLite, prompts: List[str],
              tau: Any, calibrated: bool, select_by: str, tag_label: str,
              task_examples: Optional[Dict[str, Dict[str, Any]]] = None,
              primary_metric: str = "indep") -> Tuple[List[Sample], Dict[str, float]]:
    samples = []
    if isinstance(tau, tuple):
        tau_cut, eq_frac, rng_gate = tau
    else:
        tau_cut, eq_frac, rng_gate = float(tau), 0.0, np.random.default_rng(args.seed + (1 if calibrated else 0))
    tag = tag_label
    print(f"[EVAL:{tag}] generating {len(prompts)}×{args.k} items...", flush=True)
    for i, pr in enumerate(prompts, 1):
        gt = task_examples.get(pr) if task_examples else None
        for kk in range(args.k):
            res = gen.generate(
                _format_prompt(pr, args.task_type),
                max_new_tokens=_max_new_tokens_for_task(args.task_type),
                temperature=args.gen_temp
            )
            text = res["text"]
            if args.task_type == "qa":
                text = _postprocess_qa_short(text)
            perp, z, lat, coh = res["perplexity"], res["latent"], res["latency"], res["coherence"]
            conf_uncal = assessor.score_raw(z, perp, text, kind="uncal", coh=coh)
            conf_cal = assessor.score_raw(z, perp, text, kind="cal", coh=coh)
            sel_score = assessor.score_raw(z, perp, text, kind=("raw" if select_by == "raw" else select_by), coh=coh)
            weak = assessor.weak_label(perp, text)
            accepted = accept_with_tie_break(sel_score, tau_cut, eq_frac, rng_gate, jitter=stable_jitter(pr + str(kk)))

            # Compute GT metrics if available [ADD]
            gt_score = None; gt_label = None
            if gt:
                if args.task_type == "qa":
                    refs = gt.get("answers", []) or []
                    gt_em = _em(text, refs); gt_f1 = _f1(text, refs)
                    gt_score = float(gt_f1)
                    gt_label = int(gt_em or (gt_f1 >= args.qa_f1_accept))
                elif args.task_type == "summ":
                    refs = gt.get("references", []) or ([gt["reference"]] if "reference" in gt else [])
                    gt_score = float(_rouge_l_f1(text, refs))
                    gt_label = int(gt_score >= args.summ_f1_accept)
                elif args.task_type in ("safety","label"):
                    lab = gt.get("label", None)
                    if lab is not None:
                        gt_label = int(lab)
                        gt_score = float(gt_label)

            samples.append(Sample(
                arm=tag, prompt=pr, text=text, conf=conf_cal, conf_uncal=conf_uncal,
                sel_score=sel_score, sel_kind=("raw" if select_by == "raw" else select_by),
                weak_ok=weak, accepted=accepted,
                quality_main=quality_proxy_main(text, perp),
                quality_indep=quality_proxy_indep(text),
                latency=lat, perp=perp,
                mean_entropy=res["mean_entropy"], mean_maxp=res["mean_maxp"],
                gt_score=gt_score, gt_label=gt_label
            ))
        if i % 20 == 0 or i == len(prompts):
            print(f"[EVAL:{tag}] {i}/{len(prompts)} done", flush=True)
    row = summarize(samples, target_cov=args.target_cov, n_boot=args.n_boot, alpha=args.alpha,
                    primary_metric=primary_metric, cluster_bootstrap=args.cluster_bootstrap)
    return samples, row


def build_task_index(task_rows: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Dict[str, Any]]]:
    if not task_rows: return None
    idx = {}
    for r in task_rows:
        pr = r.get("prompt", None)
        if pr is None: continue
        idx[pr] = r
    return idx


def cluster_preserving_wow(samples: List[Sample], primary_metric: str, rng: np.random.Generator, reps: int = 4000) -> Tuple[float, Tuple[float,float], float]:
    """
    [STATS] WOW using PRIMARY metric with per-prompt permutation that preserves accepted count per prompt.
    """
    # group by prompt
    byp: Dict[str, List[Sample]] = {}
    for s in samples:
        byp.setdefault(s.prompt, []).append(s)
    # actual selected quality
    def sel_quality(ss: List[Sample]) -> float:
        vals = [sample_primary_value(x, primary_metric) for x in ss if x.accepted]
        return float(np.mean(vals)) if vals else float('nan')

    actual = sel_quality(samples)
    # For each prompt, count accepts
    acc_counts = {p: sum(x.accepted for x in xs) for p, xs in byp.items()}
    # Permute within prompt
    perm_vals = []
    for _ in range(reps):
        sim_sel = []
        for p, xs in byp.items():
            k = len(xs); a = acc_counts[p]
            idx = rng.choice(k, size=a, replace=False) if a > 0 else []
            marks = np.zeros(k, dtype=int); marks[idx] = 1
            for j, x in enumerate(xs):
                ss = Sample(**{**x.__dict__})
                ss.accepted = int(marks[j])
                sim_sel.append(ss)
        perm_vals.append(sel_quality(sim_sel))
    perm = np.asarray(perm_vals, float)
    lift = actual - float(np.nanmean(perm))
    ci_lo, ci_hi = np.nanpercentile(actual - perm, [2.5, 97.5]).tolist()
    pval = permutation_pvalue_two_sided(actual - perm)
    return float(lift), (float(ci_lo), float(ci_hi)), float(pval)


def run_demo(args):
    set_seeds(args.seed)
    print_device_info()
    out = Path(args.out); (out / "demo").mkdir(parents=True, exist_ok=True)

    # [SAFETY] brief note
    print("[SAFETY] This controller abstains; failures can include rejecting valid content or accepting harmful/inaccurate outputs.\n"
          "         Use domain-appropriate ground truth and audits for deployment.", flush=True)

    # Prompts & optional task data
    task_rows = load_task_data(args.task_data)
    if task_rows:
        prompts_all = [r.get("prompt","") for r in task_rows if "prompt" in r][:args.n]  # [FIX] alignment to GT
    else:
        prompts_all = load_prompts(args.n, args.seed, args.prompts_file, args.prompts_mode)
    task_index = build_task_index(task_rows) if task_rows else None

    gen = TinyGen(model_name=args.model, latent_dim=args.latent_dim)

    assessor = AssessorLite(latent_dim=args.latent_dim, weak_cap=args.weak_cap, weak_min_sents=args.weak_min_sents,
                            calibrator=args.calibrator, use_invperp=not args.ablate_invperp,
                            use_coherence=args.coherence_feat)
    assessor_abl = None
    if args.run_ablation:
        assessor_abl = AssessorLite(latent_dim=args.latent_dim, weak_cap=args.weak_cap, weak_min_sents=args.weak_min_sents,
                                    calibrator=args.calibrator, use_invperp=False,
                                    use_coherence=False)

    # split cal/eval
    n_cal = max(8, int(len(prompts_all) * args.cal_split))
    cal_prompts = prompts_all[:n_cal]
    eval_prompts = prompts_all[n_cal:]

    # split cal into train/holdout
    n_tr = max(4, int((1.0 - args.cal_holdout_frac) * len(cal_prompts)))
    cal_train = cal_prompts[:n_tr]
    cal_hold = cal_prompts[n_tr:]

    # ---- CAL-TRAIN: train head (k candidates each)
    Z, P, T, C = [], [], [], []
    t_train0 = time.time()
    print(f"[CAL-TRAIN] generating {len(cal_train)}×{args.k} items...", flush=True)
    for i, pr in enumerate(cal_train, 1):
        for kk in range(args.k):
            res = gen.generate(
                _format_prompt(pr, args.task_type),
                max_new_tokens=_max_new_tokens_for_task(args.task_type),
                temperature=args.gen_temp
            )
            Z.append(res["latent"]); P.append(res["perplexity"]); T.append(res["text"]); C.append(res["coherence"])
        if i % 10 == 0 or i == len(cal_train):
            print(f"[CAL-TRAIN] {i}/{len(cal_train)} done", flush=True)
    assessor.train_head(Z, P, T, C)
    if assessor_abl is not None:
        assessor_abl.train_head(Z, P, T, C)
    train_time = time.time() - t_train0

    # ---- CAL-HOLD: fit calibrator and choose tau on selection score
    ZH, PH, TH, CH, EH, MPH = [], [], [], [], [], []  # [FIX] collect entropy & maxp on holdout
    print(f"[CAL-HOLD] generating {len(cal_hold)}×{args.k} items...", flush=True)
    for i, pr in enumerate(cal_hold, 1):
        for kk in range(args.k):
            r = gen.generate(
                _format_prompt(pr, args.task_type),
                max_new_tokens=_max_new_tokens_for_task(args.task_type),
                temperature=args.gen_temp
            )
            if args.task_type == "qa":
                r["text"] = _postprocess_qa_short(r["text"])
            ZH.append(r["latent"]); PH.append(r["perplexity"]); TH.append(r["text"]); CH.append(r["coherence"])
            EH.append(r["mean_entropy"]); MPH.append(r["mean_maxp"])  # [FIX]
        if i % 10 == 0 or i == len(cal_hold):
            print(f"[CAL-HOLD] {i}/{len(cal_hold)} done", flush=True)
    t_cal0 = time.time()
    assessor.fit_calibrator(ZH, PH, TH, CH)
    if assessor_abl is not None:
        assessor_abl.fit_calibrator(ZH, PH, TH, CH)
    cal_time = time.time() - t_cal0
    print(f"[CAL] mode={assessor.calibrator} | temp={getattr(assessor,'temp',1.0):.3f} | calibrated={assessor.calibrated}", flush=True)
    if assessor_abl is not None:
        print(f"[CAL-ABL] mode={assessor_abl.calibrator} | temp={getattr(assessor_abl,'temp',1.0):.3f} | calibrated={assessor_abl.calibrated}", flush=True)

    sel_kind = ("raw" if args.select_by == "raw" else args.select_by)
    cal_scores = [assessor.score_raw(z, p, t, kind=sel_kind, coh=c) for z, p, t, c in zip(ZH, PH, TH, CH)]
    tau_cut, eq_frac = choose_cut_with_tie_fraction(cal_scores, args.target_cov)
    s = np.asarray(cal_scores, float)
    cal_cov = (float(np.sum(s > tau_cut)) + eq_frac * float(np.sum(s == tau_cut))) / max(1.0, float(len(s)))
    rng_gate = np.random.default_rng(args.seed + 123)

    # ---- Baseline: accept-all (eval, k candidates each)
    base_samples = []
    total_new_tokens = 0
    print(f"[EVAL:baseline] generating {len(eval_prompts)}×{args.k} items...", flush=True)
    for i, pr in enumerate(eval_prompts, 1):
        gt = task_index.get(pr) if task_index else None
        for kk in range(args.k):
            r = gen.generate(
                _format_prompt(pr, args.task_type),
                max_new_tokens=_max_new_tokens_for_task(args.task_type),
                temperature=args.gen_temp
            )
            if args.task_type == "qa":
                r["text"] = _postprocess_qa_short(r["text"])
            total_new_tokens += r.get("n_new_tokens", 0)
            gt_score = None; gt_label = None
            if gt:
                if args.task_type == "qa":
                    refs = gt.get("answers", []) or []
                    em = _em(r["text"], refs); f1 = _f1(r["text"], refs)
                    gt_score = float(f1); gt_label = int(em or (f1 >= args.qa_f1_accept))
                elif args.task_type == "summ":
                    refs = gt.get("references", []) or ([gt["reference"]] if "reference" in gt else [])
                    rl = _rouge_l_f1(r["text"], refs)
                    gt_score = float(rl); gt_label = int(rl >= args.summ_f1_accept)
                elif args.task_type in ("safety","label"):
                    lab = gt.get("label", None)
                    if lab is not None:
                        gt_label = int(lab); gt_score = float(gt_label)

            base_samples.append(Sample(
                arm="baseline", prompt=pr, text=r["text"], conf=1.0, conf_uncal=1.0,
                sel_score=1.0, sel_kind="—",
                weak_ok=assessor.weak_label(r["perplexity"], r["text"]),
                accepted=1,
                quality_main=quality_proxy_main(r["text"], r["perplexity"]),
                quality_indep=quality_proxy_indep(r["text"]),
                latency=r["latency"], perp=r["perplexity"],
                mean_entropy=r["mean_entropy"], mean_maxp=r["mean_maxp"],
                gt_score=gt_score, gt_label=gt_label
            ))
        if i % 20 == 0 or i == len(eval_prompts):
            print(f"[EVAL:baseline] {i}/{len(eval_prompts)} done", flush=True)

    # Primary metric choice
    primary_metric = (args.primary_metric if args.primary_metric in ("gt","indep","main") else
                      ("gt" if task_index is not None else "indep"))

    base_row = summarize(base_samples, target_cov=args.target_cov, n_boot=args.n_boot, alpha=args.alpha,
                         primary_metric=primary_metric, cluster_bootstrap=args.cluster_bootstrap)

    # ---- Assessor arms (coverage-exact gate)
    task_index_eval = task_index
    uncal_samples, uncal_row = run_split(args, gen, assessor, eval_prompts,
                                         tau=(tau_cut, eq_frac, rng_gate), calibrated=False,
                                         select_by=args.select_by, tag_label="assessor_uncal_select",
                                         task_examples=task_index_eval, primary_metric=primary_metric)
    cal_samples, cal_row = run_split(args, gen, assessor, eval_prompts,
                                     tau=(tau_cut, eq_frac, rng_gate), calibrated=True,
                                     select_by=args.select_by, tag_label="assessor_cal_select",
                                     task_examples=task_index_eval, primary_metric=primary_metric)

    # ---- Strong baselines @ same target coverage (decided on CAL-HOLD)
    # PPL@cov
    ppl_cut, ppl_eq = choose_cut_with_tie_fraction([-p for p in PH], args.target_cov)
    rng_ppl = np.random.default_rng(args.seed + 321)
    ppl_samples = []
    print(f"[EVAL:ppl@cov] scoring {len(base_samples)} items...", flush=True)
    for s0 in base_samples:
        score = -s0.perp
        acc = accept_with_tie_break(score, ppl_cut, ppl_eq, rng_ppl, jitter=stable_jitter(s0.prompt + "p"))
        ns = Sample(**{**s0.__dict__})
        ns.arm = "ppl_at_cov"; ns.sel_score = score; ns.sel_kind = "ppl@cov"; ns.accepted = acc
        ppl_samples.append(ns)
    ppl_row = summarize(ppl_samples, target_cov=args.target_cov, n_boot=args.n_boot, alpha=args.alpha,
                        primary_metric=primary_metric, cluster_bootstrap=args.cluster_bootstrap)

    # Length@cov  (use WORD COUNT to avoid degenerate QA baselines)
    len_hold_scores = [_word_len(t) for t in TH]
    len_cut, len_eq = choose_cut_with_tie_fraction(len_hold_scores, args.target_cov)
    rng_len = np.random.default_rng(args.seed + 654)
    len_samples = []
    print(f"[EVAL:length@cov] scoring {len(base_samples)} items...", flush=True)
    for s0 in base_samples:
        score = _word_len(s0.text)
        acc = accept_with_tie_break(score, len_cut, len_eq, rng_len, jitter=stable_jitter(s0.prompt + "l"))
        ns = Sample(**{**s0.__dict__})
        ns.arm = "length_at_cov"; ns.sel_score = score; ns.sel_kind = "len@cov"; ns.accepted = acc
        len_samples.append(ns)
    len_row = summarize(len_samples, target_cov=args.target_cov, n_boot=args.n_boot, alpha=args.alpha,
                        primary_metric=primary_metric, cluster_bootstrap=args.cluster_bootstrap)

    # Entropy@cov (lower entropy = more confident) [FIX: cut on CAL-HOLD EH]
    ent_cut, ent_eq = choose_cut_with_tie_fraction([-e for e in EH], args.target_cov)
    rng_ent = np.random.default_rng(args.seed + 777)
    ent_samples = []
    print(f"[EVAL:entropy@cov] scoring {len(base_samples)} items...", flush=True)
    for s0 in base_samples:
        score = -s0.mean_entropy
        acc = accept_with_tie_break(score, ent_cut, ent_eq, rng_ent, jitter=stable_jitter(s0.prompt + "e"))
        ns = Sample(**{**s0.__dict__})
        ns.arm = "entropy_at_cov"; ns.sel_score = score; ns.sel_kind = "entropy@cov"; ns.accepted = acc
        ent_samples.append(ns)
    ent_row = summarize(ent_samples, target_cov=args.target_cov, n_boot=args.n_boot, alpha=args.alpha,
                        primary_metric=primary_metric, cluster_bootstrap=args.cluster_bootstrap)

    # MaxP@cov (higher mean max prob = more confident) [FIX: cut on CAL-HOLD MPH]
    maxp_cut, maxp_eq = choose_cut_with_tie_fraction(MPH, args.target_cov)
    rng_mp = np.random.default_rng(args.seed + 778)
    mp_samples = []
    print(f"[EVAL:maxp@cov] scoring {len(base_samples)} items...", flush=True)
    for s0 in base_samples:
        score = s0.mean_maxp
        acc = accept_with_tie_break(score, maxp_cut, maxp_eq, rng_mp, jitter=stable_jitter(s0.prompt + "m"))
        ns = Sample(**{**s0.__dict__})
        ns.arm = "maxp_at_cov"; ns.sel_score = score; ns.sel_kind = "maxp@cov"; ns.accepted = acc
        mp_samples.append(ns)
    mp_row = summarize(mp_samples, target_cov=args.target_cov, n_boot=args.n_boot, alpha=args.alpha,
                       primary_metric=primary_metric, cluster_bootstrap=args.cluster_bootstrap)

    # Platt on PPL/Length (calibrators atop fixed heuristic)
    ppl_platt = baseline_platt([-p for p in PH], [int(p < args.weak_cap) for p in PH])
    len_platt = baseline_platt([count_sents(t) for t in TH], [int(count_sents(t) >= args.weak_min_sents) for t in TH])
    ppl_probs_hold = [ppl_platt(-p) for p in PH]
    tau_ppl_prob, eq_ppl_prob = choose_cut_with_tie_fraction(ppl_probs_hold, args.target_cov)
    rng_pp = np.random.default_rng(args.seed + 879)
    pplp_samples = []
    for s0 in base_samples:
        prob = ppl_platt(-s0.perp)
        acc = accept_with_tie_break(prob, tau_ppl_prob, eq_ppl_prob, rng_pp, jitter=stable_jitter(s0.prompt + "pp"))
        ns = Sample(**{**s0.__dict__}); ns.arm = "platt_ppl@cov"; ns.sel_score = prob; ns.sel_kind = "platt_ppl"; ns.accepted = acc
        pplp_samples.append(ns)
    pplp_row = summarize(pplp_samples, target_cov=args.target_cov, n_boot=args.n_boot, alpha=args.alpha,
                         primary_metric=primary_metric, cluster_bootstrap=args.cluster_bootstrap)

    len_probs_hold = [len_platt(count_sents(t)) for t in TH]
    tau_len_prob, eq_len_prob = choose_cut_with_tie_fraction(len_probs_hold, args.target_cov)
    rng_lp = np.random.default_rng(args.seed + 880)
    lenp_samples = []
    for s0 in base_samples:
        prob = len_platt(count_sents(s0.text))
        acc = accept_with_tie_break(prob, tau_len_prob, eq_len_prob, rng_lp, jitter=stable_jitter(s0.prompt + "lp"))
        ns = Sample(**{**s0.__dict__}); ns.arm = "platt_len@cov"; ns.sel_score = prob; ns.sel_kind = "platt_len"; ns.accepted = acc
        lenp_samples.append(ns)
    lenp_row = summarize(lenp_samples, target_cov=args.target_cov, n_boot=args.n_boot, alpha=args.alpha,
                         primary_metric=primary_metric, cluster_bootstrap=args.cluster_bootstrap)

    # [NEW BASELINE] Majority-of-k (prompt-level) using PPL and Entropy variants
    majk_ppl_samples = majority_of_k_at_cov(base_samples, hold_scores=[-p for p in PH], target_cov=args.target_cov, m_ratio=0.6, kind="ppl")
    majk_ppl_row = summarize(majk_ppl_samples, target_cov=args.target_cov, n_boot=args.n_boot, alpha=args.alpha,
                             primary_metric=primary_metric, cluster_bootstrap=args.cluster_bootstrap)

    majk_ent_samples = majority_of_k_at_cov(base_samples, hold_scores=[-e for e in EH], target_cov=args.target_cov, m_ratio=0.6, kind="entropy")
    majk_ent_row = summarize(majk_ent_samples, target_cov=args.target_cov, n_boot=args.n_boot, alpha=args.alpha,
                             primary_metric=primary_metric, cluster_bootstrap=args.cluster_bootstrap)

    # ---- Optional ablation arm (no inv-perp in features)
    abl_row = None
    if assessor_abl is not None:
        abl_samples, abl_row = run_split(args, gen, assessor_abl, eval_prompts,
                                         tau=(tau_cut, eq_frac, rng_gate), calibrated=True,
                                         select_by=args.select_by, tag_label="assessor_cal_no_invperp",
                                         task_examples=task_index_eval, primary_metric=primary_metric)

    # ---- Reporting
    print("\n>>> DEMO (single seed) <<<")
    src = "task_data" if task_index else (args.prompts_file if args.prompts_file else "built-in prompts")
    print(f"seed={args.seed} | n={args.n} | k={args.k} | cal_split={args.cal_split:.2f} "
          f"({len(cal_train)} train / {len(cal_hold)} holdout) | "
          f"target_cov={args.target_cov:.2f} | select_by={args.select_by} | "
          f"calibrator={args.calibrator} | tau_cut={tau_cut:.6f} | eq_frac={eq_frac:.3f} | cal_cov={cal_cov:.3f} | prompts={src}")
    print(f"[COST] train_time={train_time:.2f}s | calib_time={cal_time:.2f}s | gen_new_tokens≈{total_new_tokens}", flush=True)

    print_table("Baseline (accept-all)", base_row, prob_metrics=False)
    print_table("PPL-top-k@cov (holdout)", ppl_row, prob_metrics=False)
    print_table("Length-top-k@cov (holdout)", len_row, prob_metrics=False)
    print_table("Entropy@cov (holdout)", ent_row, prob_metrics=False)
    print_table("MaxP@cov (holdout)", mp_row, prob_metrics=False)
    print_table("Platt(PPL)@cov (holdout)", pplp_row, prob_metrics=False)
    print_table("Platt(Len)@cov (holdout)", lenp_row, prob_metrics=False)
    print_table("Majority-of-k (PPL)@cov", majk_ppl_row, prob_metrics=False)
    print_table("Majority-of-k (Entropy)@cov", majk_ent_row, prob_metrics=False)
    print_table("Assessor (uncalibrated select-by)", uncal_row, prob_metrics=True)
    print_table("Assessor (calibrated select-by)", cal_row, prob_metrics=True)
    if abl_row is not None:
        print_table("Assessor (calibrated, no inv-perp)", abl_row, prob_metrics=True)

    # ---- Correlation between proxies (kept)
    qm = [s.quality_main for s in cal_samples]
    qi = [s.quality_indep for s in cal_samples]
    p_r = pearson_corr(qm, qi); s_r = spearman_corr(qm, qi)
    print(f"\n[QUALITY PROXIES] Pearson r(main,indep)={p_r:.3f} | Spearman ρ={s_r:.3f}")

    # ---- WOW block (PRIMARY metric, cluster-preserving permutation)
    rng = np.random.default_rng(args.seed + 999)
    lift, (lift_ci_lo, lift_ci_hi), pval = cluster_preserving_wow(cal_samples, primary_metric, rng, reps=args.perm_reps)
    delta_ece, d_lo, d_hi = ece_delta_bootstrap(
        [s.conf_uncal for s in cal_samples],
        [s.conf for s in cal_samples],
        [s.weak_ok for s in cal_samples],
        iters=args.n_boot, alpha=args.alpha
    )

    print("\n--- WOW (statistical checks; PRIMARY metric) ---")
    p_str = ("< 1e-3" if pval < 1e-3 else f"= {pval:.3g}")
    print(f"Lift@cov (selQ_primary − cluster-random@same-cov): {lift:+.3f} [{lift_ci_lo:+.3f},{lift_ci_hi:+.3f}]  (p {p_str})")
    print(f"ECEΔ (uncal→cal vs weak) bootstrap CI: {delta_ece:+.3f} [{d_lo:+.3f},{d_hi:+.3f}]")
    if not math.isnan(cal_row.get("auc_gt", float('nan'))):
        print(f"AUC vs GT: {cal_row['auc_gt']:.3f}")

    # ---- Plots (primary)
    plot_all(out / "demo", cal_samples, title_prefix="assessor-cal • ", primary_metric=primary_metric)

    # ---- CSV outputs
    metrics_by_arm = {
        "baseline": base_row,
        "ppl_at_cov": ppl_row,
        "len_at_cov": len_row,
        "entropy_at_cov": ent_row,
        "maxp_at_cov": mp_row,
        "platt_ppl_at_cov": pplp_row,
        "platt_len_at_cov": lenp_row,
        "majority_ppl_at_cov": majk_ppl_row,
        "majority_entropy_at_cov": majk_ent_row,
        "assessor_uncal": uncal_row,
        "assessor_cal": cal_row
    }
    if abl_row is not None:
        metrics_by_arm["assessor_cal_no_invperp"] = abl_row

    wow_dict = {
        "lift": lift, "lift_ci": [lift_ci_lo, lift_ci_hi], "pval": float(pval),
        "ece_delta": delta_ece, "ece_delta_ci": [d_lo, d_hi]
    }
    corr_dict = {"pearson": p_r, "spearman": s_r}

    write_metrics_csv(out / "demo", metrics_by_arm, wow_dict, corr_dict)

    # ---- Snippets (few eval examples)
    print("\n--- A few eval examples (assessor-cal; PRIMARY metric shown) ---")
    for s in cal_samples[:min(args.examples_k, len(cal_samples))]:
        txt = (s.text.strip().replace("\n", " "))
        if len(txt) > 220:
            txt = txt[:220] + "..."
        prim = sample_primary_value(s, primary_metric)
        print(f"[prob={s.conf:.2f} | sel={s.sel_score:.2f} ({s.sel_kind}) | primary={prim if not math.isnan(prim) else float('nan'):.2f} | weak_ok={s.weak_ok} | accepted={s.accepted} | perp={s.perp:.1f}] {txt}")

    if args.save_json:
        j = {
            "seed": args.seed, "n": args.n, "k": args.k, "cal_split": args.cal_split,
            "cal_train": len(cal_train), "cal_holdout": len(cal_hold), "select_by": args.select_by,
            "target_cov": args.target_cov, "tau_cut": tau_cut, "eq_frac": eq_frac, "cal_cov": cal_cov,
            "primary_metric": primary_metric,
            "baseline": base_row, "ppl_at_cov": ppl_row, "len_at_cov": len_row,
            "entropy_at_cov": ent_row, "maxp_at_cov": mp_row, "platt_ppl_at_cov": pplp_row, "platt_len_at_cov": lenp_row,
            "majority_ppl_at_cov": majk_ppl_row, "majority_entropy_at_cov": majk_ent_row,
            "assessor_uncal": uncal_row, "assessor_cal": cal_row,
            "quality_proxy_corr": {"pearson": p_r, "spearman": s_r},
            "wow": {
                "lift": lift, "lift_ci": [lift_ci_lo, lift_ci_hi], "pval": float(pval),
                "ece_delta": delta_ece, "ece_delta_ci": [d_lo, d_hi]
            },
            "cost": {"train_time_s": train_time, "cal_time_s": cal_time, "gen_new_tokens": total_new_tokens}
        }
        if abl_row is not None:
            j["assessor_cal_no_invperp"] = abl_row
        with open(out / "demo" / "summary.json", "w", encoding="utf-8") as f:
            json.dump(j, f, indent=2)
        print(f"[INFO] Wrote {out/'demo'/'summary.json'}", flush=True)


def run_bench(args):
    set_seeds(args.seed)
    print_device_info()
    out = Path(args.out); (out / "bench").mkdir(parents=True, exist_ok=True)
    if args.seeds:
        seeds = [int(s) for s in args.seeds]
    elif args.seed_range and len(args.seed_range) == 2:
        a, b = int(args.seed_range[0]), int(args.seed_range[1])
        seeds = list(range(min(a, b), max(a, b) + 1))
    else:
        seeds = [int(args.seed)]

    agg = []
    for sd in seeds:
        set_seeds(sd)
        prompts_all = load_prompts(args.n, sd, args.prompts_file, args.prompts_mode)
        gen = TinyGen(model_name=args.model, latent_dim=args.latent_dim)
        assessor = AssessorLite(latent_dim=args.latent_dim, weak_cap=args.weak_cap, weak_min_sents=args.weak_min_sents,
                                calibrator=args.calibrator, use_invperp=not args.ablate_invperp,
                                use_coherence=args.coherence_feat)
        n_cal = max(8, int(len(prompts_all) * args.cal_split))
        cal_prompts = prompts_all[:n_cal]
        eval_prompts = prompts_all[n_cal:]
        n_tr = max(4, int((1.0 - args.cal_holdout_frac) * len(cal_prompts)))
        cal_train = cal_prompts[:n_tr]
        cal_hold = cal_prompts[n_tr:]

        Z, P, T, C = [], [], [], []
        for pr in cal_train:
            for kk in range(args.k):
                r = gen.generate(
                _format_prompt(pr, args.task_type),
                max_new_tokens=_max_new_tokens_for_task(args.task_type),
                temperature=args.gen_temp
                )
                Z.append(r["latent"]); P.append(r["perplexity"]); T.append(r["text"]); C.append(r["coherence"])
        assessor.train_head(Z, P, T, C)

        ZH, PH, TH, CH = [], [], [], []
        for pr in cal_hold:
            for kk in range(args.k):
                r = gen.generate(
                _format_prompt(pr, args.task_type),
                max_new_tokens=_max_new_tokens_for_task(args.task_type),
                temperature=args.gen_temp
                )
                ZH.append(r["latent"]); PH.append(r["perplexity"]); TH.append(r["text"]); CH.append(r["coherence"])
        assessor.fit_calibrator(ZH, PH, TH, CH)

        sel_kind = ("raw" if args.select_by == "raw" else args.select_by)
        cal_scores = [assessor.score_raw(z, p, t, kind=sel_kind, coh=c) for z, p, t, c in zip(ZH, PH, TH, CH)]
        tau_cut, eq_frac = choose_cut_with_tie_fraction(cal_scores, args.target_cov)
        rng_gate = np.random.default_rng(sd + 123)

        eval_samples = []
        for pr in eval_prompts:
            for kk in range(args.k):
                r = gen.generate(
                _format_prompt(pr, args.task_type),
                max_new_tokens=_max_new_tokens_for_task(args.task_type),
                temperature=args.gen_temp
                )
                z, p, t, coh = r["latent"], r["perplexity"], r["text"], r["coherence"]
                conf_cal = assessor.score_raw(z, p, t, kind="cal", coh=coh)
                conf_unc = assessor.score_raw(z, p, t, kind="uncal", coh=coh)
                sel = assessor.score_raw(z, p, t, kind=sel_kind, coh=coh)
                acc = accept_with_tie_break(sel, tau_cut, eq_frac, rng_gate, jitter=stable_jitter(pr + str(kk)))
                eval_samples.append(Sample("assessor_cal", pr, t, conf_cal, conf_unc, sel, sel_kind,
                                           assessor.weak_label(p, t), acc,
                                           quality_proxy_main(t, p), quality_proxy_indep(t), r["latency"], p,
                                           mean_entropy=r["mean_entropy"], mean_maxp=r["mean_maxp"]))
        row = summarize(eval_samples, target_cov=args.target_cov, n_boot=args.n_boot, alpha=args.alpha,
                        primary_metric=("gt" if args.primary_metric=="gt" else "indep"),
                        cluster_bootstrap=args.cluster_bootstrap)
        agg.append({
            "seed": sd,
            "tau_cut": tau_cut, "eq_frac": eq_frac,
            "assessor_cov": row["coverage"],
            "assessor_quality_primary": row["selective_quality_primary"],
            "ece_uncal_weak": row["ece_uncal_weak"], "ece_cal_weak": row["ece_cal_weak"],
        })
        print(f"[seed {sd}] cov={row['coverage']:.3f} | selQ(PRIMARY)={_fmt_val(row['selective_quality_primary'])} "
              f"| ECE(weak) {row['ece_uncal_weak']:.3f}→{row['ece_cal_weak']:.3f}", flush=True)

    # CSV bench
    header = ["seed", "tau_cut", "eq_frac", "assessor_cov", "assessor_quality_primary", "ece_uncal_weak", "ece_cal_weak"]
    rows = [[r["seed"], r["tau_cut"], r["eq_frac"], r["assessor_cov"], r["assessor_quality_primary"], r["ece_uncal_weak"], r["ece_cal_weak"]] for r in agg]
    write_csv(out / "bench" / "across_seeds.csv", header, rows)

    if args.save_json:
        with open(out / "bench" / "across_seeds.json", "w", encoding='utf-8') as f:
            json.dump(agg, f, indent=2)
        print(f"[INFO] Wrote {out/'bench'/'across_seeds.json'}", flush=True)


# --------------------------------- CLI ---------------------------------------

def main():
    ap = argparse.ArgumentParser(description="KAIROS-lite: weakly-supervised assessor + calibration (leakage-safe, GT, cluster stats)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--model", default="gpt2")
    parent.add_argument("--seed", type=int, default=43)
    parent.add_argument("--n", type=int, default=40)
    parent.add_argument("--k", type=int, default=5, help="number of candidates per prompt (k>=1)")
    parent.add_argument("--cal_split", type=float, default=0.30, help="fraction for calibration/training")
    parent.add_argument("--cal_holdout_frac", type=float, default=0.50, help="fraction of cal used as holdout for calibrator & tau")
    parent.add_argument("--select_by", type=str, default="cal", choices=["raw", "uncal", "cal"],
                        help="which score to rank/select by (default: calibrated prob)")
    parent.add_argument("--target_cov", type=float, default=0.40, help="target coverage on CAL-HOLD")
    parent.add_argument("--weak_cap", type=float, default=55.0)
    parent.add_argument("--weak_min_sents", type=int, default=3)
    parent.add_argument("--gen_temp", type=float, default=0.9)
    parent.add_argument("--calibrator", type=str, default="temp", choices=["temp", "isotonic"])
    parent.add_argument("--prompts_file", type=str, default=None, help="optional path to prompts.txt (one prompt per line)")
    parent.add_argument("--prompts_mode", type=str, default="random", choices=["random", "head"], help="if a file is given: sample randomly or take head")
    parent.add_argument("--n_boot", type=int, default=1500)
    parent.add_argument("--alpha", type=float, default=0.05)
    parent.add_argument("--perm_reps", type=int, default=6000)
    parent.add_argument("--examples_k", type=int, default=5)
    parent.add_argument("--save_json", action="store_true")
    parent.add_argument("--out", type=str, default="out_kairos_lite")
    parent.add_argument("--ablate_invperp", action="store_true", help="Remove inv-perp from features (main assessor)")
    parent.add_argument("--run_ablation", action="store_true", help="Also run separate arm with inv-perp/coherence removed")
    parent.add_argument("--coherence_feat", action="store_true", help="Add prompt–continuation cosine as an assessor feature")
    parent.add_argument("--cluster_bootstrap", action="store_true", help="Use prompt-cluster bootstrap for CIs")  # [ADD]
    parent.add_argument("--primary_metric", type=str, default="indep", choices=["indep","gt","main"], help="Primary metric for reporting")  # [ADD]
    parent.add_argument("--latent_dim", type=int, default=64, help="Assessor latent dim ablation")  # [ABLT]

    # GT task args
    parent.add_argument("--task_type", type=str, default=None, choices=["qa","summ","safety","label"], help="Enable ground-truth evaluation")
    parent.add_argument("--task_data", type=str, default=None, help="Path to JSONL with prompts + GT")
    parent.add_argument("--qa_f1_accept", type=float, default=0.5, help="QA F1 threshold for GT label=1")
    parent.add_argument("--summ_f1_accept", type=float, default=0.25, help="Summ ROUGE-L(F1) threshold for GT label=1")

    p_demo = sub.add_parser("demo", parents=[parent], help="single-seed demo with tables + plots + CSV/JSON")
    p_bench = sub.add_parser("bench", parents=[parent], help="seed-sweep benchmark with across-seed CSV")
    p_bench.add_argument("--seeds", nargs="+", type=int, default=[])
    p_bench.add_argument("--seed_range", nargs=2, type=int, default=None)

    args = ap.parse_args()
    # Make QA weak labels sane (short answers count as one sentence)
    if args.task_type == "qa" and args.weak_min_sents > 1:
        args.weak_min_sents = 1
    if args.cmd == "demo":
        run_demo(args)
    elif args.cmd == "bench":
        run_bench(args)


if __name__ == "__main__":
    main()
