#!/usr/bin/env python3
"""One-shot builder for pure_models.ipynb Phase 3 runner."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "pure_models.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md("""# JMMR Phase 3 — Multi-seed / Ablation / m-sweep (Colab)

**Manuscript ID:** JMMR-2606-1990

## Session plan (GPU limits)

| Session | `CONFIG["mode"]` | Datasets | Notes |
|---------|------------------|----------|-------|
| 1 | `"main"` | imdb, twitter | Table 3 — Std / Diff / DE / DE-best / jDE |
| 2 | `"ablation"` | imdb, twitter | Table A1 — 8 variants |
| 3 | `"m_sweep"` | imdb, twitter | Fig A1 — val-only m selection |

- Set `CONFIG["seeds"]` each session (e.g. `[0,1,2,3,4]` or `list(range(10))`).
- Completed runs are **skipped** via JSON checkpoints in `output_dir`.
- Phase 4 (BERT, GitHub) is **separate** — no re-run needed after this notebook.

## Before running

1. Upload CSVs to Google Drive.
2. Update `CONFIG["data_paths"]` and `CONFIG["output_dir"]`.
3. Runtime → **T4 GPU**.
4. After Colab: copy `results/` to repo `Related Files/experiments/phase3/` and run `import_results_to_latex.py`.
"""),
    code("""# Colab setup — run this cell first
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "termcolor", "scikit-learn", "seaborn", "scipy"])
"""),
    code("""# ========== CONFIG — adjust seeds / mode / paths here ==========
CONFIG = {
    "seeds": list(range(5)),
    "mode": "main",
    "datasets": ["imdb", "twitter"],
    "epochs": 5,
    "batch_size": 32,
    "lr": 1e-3,
    "embed_dim": 64,
    "num_heads": 1,
    "num_layers": 2,
    "hidden_dim": 128,
    "max_len": 300,
    "mutation_factor": 0.5,
    "crossover_rate": 0.9,               # DE/best/1/bin (Table notation CR)
    "jde_F_init": 0.5,
    "jde_tau1": 0.1,
    "jde_tau2": 0.1,
    "m_sweep_values": [0.1, 0.3, 0.5, 0.7, 0.9],
    "val_ratio": 0.1,
    "early_stop_patience": 2,
    "twitter_max_train": 200_000,
    "twitter_test_size": 20_000,
    "output_dir": "/content/drive/MyDrive/JMMR_Phase3/results",
    "data_paths": {
        "imdb": "/content/drive/MyDrive/IMDB Dataset.csv",
        "yelp": "/content/drive/MyDrive/yelp.csv",
        "twitter": "/content/drive/MyDrive/twitter.csv",
    },
}

MAIN_MODELS = ["standard", "de_full", "de_best", "jde", "diff"]
ABLATION_MODELS = [
    "standard", "diff", "de_full", "de_best", "jde",
    "de_avg3", "de_learned", "de_sum", "de_nobase", "de_indepV",
]
"""),
    code("""import copy
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(colored(f"Device: {device}", "blue"))
if torch.cuda.is_available():
    print(colored(f"GPU: {torch.cuda.get_device_name(0)}", "green"))

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
"""),
    code("""# ========== Data loading (official splits) ==========

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_imdb_official(path: str):
    df = pd.read_csv(path)
    review_col = "review" if "review" in df.columns else df.columns[0]
    label_col = "sentiment" if "sentiment" in df.columns else "label"
    df = df.copy()
    df["text"] = df[review_col].map(clean_text)
    if df[label_col].dtype == object:
        df["label"] = df[label_col].map({"positive": 1, "negative": 0})
    else:
        df["label"] = df[label_col].astype(int)
    df = df.dropna(subset=["text", "label"])
    n = len(df)
    if n >= 50000:
        train_df = df.iloc[:25000]
        test_df = df.iloc[25000:50000]
    else:
        split = int(n * 0.8)
        train_df, test_df = df.iloc[:split], df.iloc[split:]
    return train_df["text"].tolist(), train_df["label"].tolist(), test_df["text"].tolist(), test_df["label"].tolist()


def load_yelp_official(path: str):
    df = pd.read_csv(path)
    text_col = "review_text" if "review_text" in df.columns else "text"
    if "stars" in df.columns:
        df = df[df["stars"].isin([1, 2, 4, 5])].copy()
        df["label"] = (df["stars"] >= 4).astype(int)
    elif "class_index" in df.columns:
        df = df[df["class_index"].isin([1, 2])].copy()
        df["label"] = df["class_index"].map({2: 1, 1: 0})
    else:
        raise ValueError("Yelp CSV needs 'stars' or 'class_index' column")
    df["text"] = df[text_col].map(clean_text)
    df = df.dropna(subset=["text", "label"])
    n = len(df)
    split = int(n * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    return train_df["text"].tolist(), train_df["label"].tolist(), test_df["text"].tolist(), test_df["label"].tolist()


def load_twitter_official(path: str, max_train: int, test_size: int, seed: int):
    df = pd.read_csv(path)
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else "sentiment"
    df = df.copy()
    df["text"] = df[text_col].map(clean_text)
    df["label"] = df[label_col].astype(int)
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin([0, 1])]
    test_size = min(test_size, max(1000, int(0.1 * len(df))))
    train_pool, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    if len(train_pool) > max_train:
        train_df, _ = train_test_split(
            train_pool, train_size=max_train, random_state=seed, stratify=train_pool["label"]
        )
    else:
        train_df = train_pool
    return (
        train_df["text"].tolist(), train_df["label"].tolist(),
        test_df["text"].tolist(), test_df["label"].tolist(),
    )


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=300):
        unk = vocab["<unk>"]
        self.texts = [
            torch.tensor([vocab.get(t, unk) for t in txt.split()][:max_len], dtype=torch.long)
            for txt in texts
        ]
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def build_vocab(train_texts):
    counter = Counter(tok for txt in train_texts for tok in txt.split())
    vocab = {w: i + 1 for i, (w, _) in enumerate(counter.most_common())}
    vocab["<unk>"] = 0
    return vocab


def prepare_splits(dataset_name, cfg, seed):
    path = cfg["data_paths"][dataset_name]
    if dataset_name == "imdb":
        tr_x, tr_y, te_x, te_y = load_imdb_official(path)
    elif dataset_name == "yelp":
        tr_x, tr_y, te_x, te_y = load_yelp_official(path)
    elif dataset_name == "twitter":
        tr_x, tr_y, te_x, te_y = load_twitter_official(
            path, cfg["twitter_max_train"], cfg["twitter_test_size"], seed
        )
    else:
        raise ValueError(dataset_name)

    tr_x, va_x, tr_y, va_y = train_test_split(
        tr_x, tr_y, test_size=cfg["val_ratio"], random_state=seed, stratify=tr_y
    )
    vocab = build_vocab(tr_x)
    max_len = cfg["max_len"]
    loaders = {
        "train": DataLoader(TextDataset(tr_x, tr_y, vocab, max_len), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn),
        "val": DataLoader(TextDataset(va_x, va_y, vocab, max_len), batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(TextDataset(te_x, te_y, vocab, max_len), batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn),
    }
    return loaders, len(vocab)


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float)
    return texts.to(device), labels.to(device)
"""),
]

# Models cell - large
MODELS_CODE = r'''
# ========== Models ==========

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=1, hidden_dim=128, num_layers=2, max_len=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x) + self.pos_embed[:, :x.size(1)]
        x = self.encoder(x).mean(dim=1)
        return self.classifier(x)


class DEAttention(nn.Module):
    """DE attention with ablation modes."""
    MODES = ("de_full", "de_avg3", "de_learned", "de_sum", "de_nobase", "de_indepV")

    def __init__(self, embed_dim, num_heads, mutation_factor=0.5, mode="de_full", independent_v=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mutation_factor = mutation_factor
        self.mode = mode
        self.independent_v = independent_v or mode == "de_indepV"

        self.q_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.k_proj = nn.Linear(embed_dim, embed_dim * 3)
        if self.independent_v:
            self.v_proj = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        if mode == "de_learned":
            self.stream_logits = nn.Parameter(torch.zeros(3))

    def _split_heads(self, t, n_heads):
        b, s, _ = t.size()
        return t.view(b, s, n_heads, self.head_dim).transpose(1, 2)

    def _attn_scores(self, q, k, seq_len):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        return scores.masked_fill(mask, float("-inf"))

    def _combine(self, s1, s2, s3):
        m = self.mutation_factor
        if self.mode == "de_full":
            return s1 + m * (s2 - s3)
        if self.mode == "de_avg3":
            return (s1 + s2 + s3) / 3.0
        if self.mode == "de_learned":
            w = F.softmax(self.stream_logits, dim=0)
            return w[0] * s1 + w[1] * s2 + w[2] * s3
        if self.mode == "de_sum":
            return s1 + s2 + s3
        if self.mode == "de_nobase":
            return m * (s2 - s3)
        if self.mode == "de_indepV":
            return s1 + m * (s2 - s3)
        raise ValueError(self.mode)

    def _apply_values(self, attn, v_parts, s1, s2, s3):
        if self.independent_v:
            o1 = torch.matmul(s1, v_parts[0])
            o2 = torch.matmul(s2, v_parts[1])
            o3 = torch.matmul(s3, v_parts[2])
            if self.mode == "de_indepV":
                m = self.mutation_factor
                return o1 + m * (o2 - o3)
            return (o1 + o2 + o3) / 3.0
        return torch.matmul(attn, v_parts)

    def forward(self, x):
        b, seq_len, _ = x.size()
        qs = self.q_proj(x).chunk(3, dim=-1)
        ks = self.k_proj(x).chunk(3, dim=-1)
        qs = [self._split_heads(q, self.num_heads) for q in qs]
        ks = [self._split_heads(k, self.num_heads) for k in ks]

        raw = [self._attn_scores(q, k, seq_len) for q, k in zip(qs, ks)]
        s1, s2, s3 = [F.softmax(r, dim=-1) for r in raw]
        attn = self._combine(s1, s2, s3)

        if self.independent_v:
            v_chunks = self.v_proj(x).chunk(3, dim=-1)
            v_parts = [self._split_heads(v, self.num_heads) for v in v_chunks]
            out = self._apply_values(attn, v_parts, s1, s2, s3)
        else:
            v = self._split_heads(self.v_proj(x), self.num_heads)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(b, seq_len, self.embed_dim)
        return self.norm(self.out_proj(out))


class DETransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=1, hidden_dim=128, num_layers=2,
                 max_len=300, mutation_factor=0.5, mode="de_full"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.layers = nn.ModuleList([
            nn.Sequential(
                DEAttention(embed_dim, num_heads, mutation_factor, mode=mode),
                nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim), nn.LayerNorm(embed_dim),
            ) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x) + self.pos_embed[:, :x.size(1)]
        for layer in self.layers:
            x = x + layer(x)
        return self.classifier(x.mean(dim=1))


# --- DE/best/1/bin and jDE (ported from typeb_just_token3_.ipynb) ---

def _causal_mask(seq_len, device):
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


class DEBest1BinAttention(nn.Module):
    """DE/best/1/bin: mutant around best stream + binomial crossover."""

    def __init__(self, embed_dim, num_heads, mutation_factor=0.5, crossover_rate=0.9, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.k_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.best_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def _heads(self, t, b, s):
        return t.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _scores(self, q, k, seq_len):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        return scores.masked_fill(_causal_mask(seq_len, q.device), float("-inf"))

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).chunk(3, dim=-1)
        k = self.k_proj(x).chunk(3, dim=-1)
        v = self.v_proj(x)
        q_best = self.best_proj(x)
        k_best = self.best_proj(x)

        q = [self._heads(qi, b, s) for qi in q]
        k = [self._heads(ki, b, s) for ki in k]
        q_best = self._heads(q_best, b, s)
        k_best = self._heads(k_best, b, s)
        v = self._heads(v, b, s)

        attn_best = F.softmax(self._scores(q_best, k_best, s), dim=-1)
        attn_r1 = F.softmax(self._scores(q[0], k[0], s), dim=-1)
        attn_r2 = F.softmax(self._scores(q[1], k[1], s), dim=-1)

        attn_mut = attn_best + self.mutation_factor * (attn_r1 - attn_r2)
        rand_mask = (torch.rand_like(attn_mut) < self.crossover_rate).float()
        attn = rand_mask * attn_mut + (1 - rand_mask) * attn_best

        attn = F.relu(attn)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, self.embed_dim)
        out = self.out_proj(out) + self.residual(x)
        return self.norm(out)


class JDEAttention(nn.Module):
    """Self-adaptive jDE: S1 + F*(S2-S3) with per-forward F adaptation."""

    def __init__(self, embed_dim, num_heads, F_init=0.5, tau1=0.1, tau2=0.1, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.tau1 = tau1
        self.tau2 = tau2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.k_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.F = nn.Parameter(torch.tensor(float(F_init)))

    def _heads(self, t, b, s):
        return t.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def _scores(self, q, k, seq_len):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        return scores.masked_fill(_causal_mask(seq_len, q.device), float("-inf"))

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).chunk(3, dim=-1)
        k = self.k_proj(x).chunk(3, dim=-1)
        v = self.v_proj(x)
        q = [self._heads(qi, b, s) for qi in q]
        k = [self._heads(ki, b, s) for ki in k]
        v = self._heads(v, b, s)

        s1 = F.softmax(self._scores(q[0], k[0], s), dim=-1)
        s2 = F.softmax(self._scores(q[1], k[1], s), dim=-1)
        s3 = F.softmax(self._scores(q[2], k[2], s), dim=-1)

        r1 = torch.rand(1, device=x.device)
        r2 = torch.rand(1, device=x.device)
        with torch.no_grad():
            if r1 < self.tau1:
                self.F.data = 0.1 + 0.9 * torch.rand(1, device=x.device)
            if r2 < self.tau2:
                self.F.data = self.F.data * 0.9 + 0.1 * torch.rand(1, device=x.device)

        attn = s1 + self.F * (s2 - s3)
        attn = F.relu(attn)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, self.embed_dim)
        return self.norm(self.out_proj(out) + x)


class DEExtendedEncoderLayer(nn.Module):
    """Pre-norm encoder for DE/best and jDE (attention blocks include internal residual)."""

    def __init__(self, embed_dim, num_heads, hidden_dim, attention):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DEBestTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=1, hidden_dim=128, num_layers=2,
                 max_len=300, mutation_factor=0.5, crossover_rate=0.9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.layers = nn.ModuleList([
            DEExtendedEncoderLayer(
                embed_dim, num_heads, hidden_dim,
                DEBest1BinAttention(embed_dim, num_heads, mutation_factor, crossover_rate),
            ) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x) + self.pos_embed[:, :x.size(1)]
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x.mean(dim=1))


class JDETransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=1, hidden_dim=128, num_layers=2,
                 max_len=300, F_init=0.5, tau1=0.1, tau2=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.layers = nn.ModuleList([
            DEExtendedEncoderLayer(
                embed_dim, num_heads, hidden_dim,
                JDEAttention(embed_dim, num_heads, F_init=F_init, tau1=tau1, tau2=tau2),
            ) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x) + self.pos_embed[:, :x.size(1)]
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x.mean(dim=1))


class DiffAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, layer_idx):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = max(1, embed_dim // num_heads // 2)
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, seq_len, _ = x.size()
        q = self.q_proj(x).view(b, seq_len, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.num_heads, 2 * self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = F.softmax(attn.masked_fill(mask, float("-inf")), dim=-1)
        lam = torch.exp((self.lambda_q1 * self.lambda_k1).sum()) - torch.exp((self.lambda_q2 * self.lambda_k2).sum()) + self.lambda_init
        attn = attn.view(b, self.num_heads, 2, seq_len, seq_len)
        attn = attn[:, :, 0] - lam * attn[:, :, 1]
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, seq_len, self.embed_dim)
        return self.norm(self.out_proj(out))


class DiffTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=1, hidden_dim=128, num_layers=2, max_len=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.layers = nn.ModuleList([
            nn.Sequential(
                DiffAttention(embed_dim, num_heads, i + 1),
                nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim), nn.LayerNorm(embed_dim),
            ) for i in range(num_layers)
        ])
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x) + self.pos_embed[:, :x.size(1)]
        for layer in self.layers:
            x = x + layer(x)
        return self.classifier(x.mean(dim=1))


def build_model(name, vocab_size, cfg, mutation_factor=None):
    kw = dict(
        vocab_size=vocab_size,
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        max_len=cfg["max_len"],
    )
    m = mutation_factor if mutation_factor is not None else cfg["mutation_factor"]
    if name == "standard":
        return StandardTransformer(**kw).to(device)
    if name == "diff":
        return DiffTransformer(**kw).to(device)
    if name == "de_best":
        return DEBestTransformer(
            **kw, mutation_factor=m, crossover_rate=cfg.get("crossover_rate", 0.9),
        ).to(device)
    if name == "jde":
        return JDETransformer(
            **kw,
            F_init=cfg.get("jde_F_init", 0.5),
            tau1=cfg.get("jde_tau1", 0.1),
            tau2=cfg.get("jde_tau2", 0.1),
        ).to(device)
    if name.startswith("de_"):
        return DETransformer(**kw, mutation_factor=m, mode=name).to(device)
    raise ValueError(f"Unknown model: {name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
'''

CELLS.append(code(MODELS_CODE))

TRAIN_CODE = r'''
# ========== Training / evaluation ==========

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def evaluate_loader(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for texts, labels in loader:
            probs = model(texts).squeeze(-1)
            pred = (probs > 0.5).long().cpu().numpy()
            ys.extend(labels.long().cpu().numpy().tolist())
            ps.extend(pred.tolist())
    return compute_metrics(ys, ps)


def train_one_run(model, loaders, cfg):
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.BCELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for texts, labels in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{cfg['epochs']}", leave=False):
            optimizer.zero_grad()
            out = model(texts).squeeze(-1)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()

        val_m = evaluate_loader(model, loaders["val"])
        history.append({"epoch": epoch + 1, "train_loss": running / max(1, len(loaders["train"])), **{f"val_{k}": v for k, v in val_m.items()}})
        if val_m["acc"] > best_val:
            best_val = val_m["acc"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                break

    model.load_state_dict(best_state)
    test_m = evaluate_loader(model, loaders["test"])
    return {
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "test": test_m,
        "history": history,
    }


def result_path(cfg, dataset, model, seed, suffix=""):
    fname = f"{dataset}_{model}_seed{seed}{suffix}.json"
    return Path(cfg["output_dir"]) / fname


def load_result(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_result(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
'''

CELLS.append(code(TRAIN_CODE))

RUNNERS_CODE = r'''
# ========== Experiment runners ==========

def run_single(dataset, model_name, seed, cfg, mutation_factor=None, suffix=""):
    path = result_path(cfg, dataset, model_name, seed, suffix)
    cached = load_result(path)
    if cached:
        print(colored(f"SKIP (cached): {path.name}", "cyan"))
        return cached

    set_seed(seed)
    loaders, vocab_size = prepare_splits(dataset, cfg, seed)
    model = build_model(model_name, vocab_size, cfg, mutation_factor=mutation_factor)
    n_params = count_params(model)

    t0 = time.time()
    fit = train_one_run(model, loaders, cfg)
    train_min = (time.time() - t0) / 60.0

    payload = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "mutation_factor": mutation_factor if mutation_factor is not None else cfg["mutation_factor"],
        "params": n_params,
        "train_minutes": train_min,
        "best_val_acc": fit["best_val_acc"],
        "best_epoch": fit["best_epoch"],
        **{f"test_{k}": v for k, v in fit["test"].items()},
    }
    save_result(path, payload)
    print(colored(
        f"OK {dataset}/{model_name}/seed{seed} — test_acc={payload['test_acc']:.4f} f1={payload['test_f1_macro']:.4f}",
        "green",
    ))
    return payload


def run_main(cfg):
    rows = []
    for ds in cfg["datasets"]:
        for model in MAIN_MODELS:
            for seed in cfg["seeds"]:
                rows.append(run_single(ds, model, seed, cfg))
    return rows


def run_ablation(cfg):
    rows = []
    for ds in cfg["datasets"]:
        for model in ABLATION_MODELS:
            for seed in cfg["seeds"]:
                rows.append(run_single(ds, model, seed, cfg))
    return rows


def run_m_sweep(cfg):
    rows = []
    for ds in cfg["datasets"]:
        for seed in cfg["seeds"]:
            path = result_path(cfg, ds, "de_full", seed, suffix="_msweep")
            cached = load_result(path)
            if cached:
                print(colored(f"SKIP (cached): {path.name}", "cyan"))
                rows.append(cached)
                continue

            set_seed(seed)
            loaders, vocab_size = prepare_splits(ds, cfg, seed)
            best_m, best_val = None, -1.0
            best_state = None
            per_m = []

            for m in cfg["m_sweep_values"]:
                model = build_model("de_full", vocab_size, cfg, mutation_factor=m)
                fit = train_one_run(model, loaders, cfg)
                val_acc = fit["best_val_acc"]
                per_m.append({"m": m, "val_acc": val_acc})
                if val_acc > best_val:
                    best_val = val_acc
                    best_m = m
                    best_state = copy.deepcopy(model.state_dict())

            model = build_model("de_full", vocab_size, cfg, mutation_factor=best_m)
            model.load_state_dict(best_state)
            test_m = evaluate_loader(model, loaders["test"])

            payload = {
                "dataset": ds,
                "model": "de_full",
                "seed": seed,
                "best_m_val": best_m,
                "val_acc_at_best_m": best_val,
                "m_sweep_val": per_m,
                **{f"test_{k}": v for k, v in test_m.items()},
            }
            save_result(path, payload)
            rows.append(payload)
            print(colored(f"m-sweep {ds} seed{seed}: best_m={best_m}, test_acc={payload['test_acc']:.4f}", "green"))
    return rows


def rows_to_summary(rows, group_cols=("dataset", "model")):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metrics = [c for c in df.columns if c.startswith("test_")]
    agg = df.groupby(list(group_cols))[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in agg.columns]
    return agg


def paired_ttests(cfg):
    """DE vs Standard and DE vs Diff on test_acc across seeds."""
    out_dir = Path(cfg["output_dir"])
    records = []
    for ds in cfg["datasets"]:
        for baseline, label in [("standard", "DE vs Standard"), ("diff", "DE vs Diff")]:
            de_vals, base_vals = [], []
            for seed in cfg["seeds"]:
                de = load_result(out_dir / f"{ds}_de_full_seed{seed}.json")
                ba = load_result(out_dir / f"{ds}_{baseline}_seed{seed}.json")
                if de and ba:
                    de_vals.append(de["test_acc"])
                    base_vals.append(ba["test_acc"])
            if len(de_vals) >= 2:
                diff = np.array(de_vals) - np.array(base_vals)
                t, p = stats.ttest_rel(de_vals, base_vals)
                ci_low, ci_high = stats.t.interval(
                    0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff)
                )
                records.append({
                    "dataset": ds,
                    "comparison": label,
                    "p_value": float(p),
                    "mean_diff": float(diff.mean()),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_seeds": len(de_vals),
                })
    return pd.DataFrame(records)


def aggregate_results(cfg):
    out = Path(cfg["output_dir"])
    json_files = list(out.glob("*.json"))
    if not json_files:
        print("No JSON results to aggregate.")
        return

    main_rows = [load_result(p) for p in out.glob("*_seed*.json") if "_msweep" not in p.name and load_result(p)]
    main_rows = [r for r in main_rows if r and r.get("model") in MAIN_MODELS]

    abl_rows = [load_result(p) for p in out.glob("*_seed*.json") if "_msweep" not in p.name and load_result(p)]
    abl_rows = [r for r in abl_rows if r and r.get("model") in ABLATION_MODELS]

    ms_rows = [load_result(p) for p in out.glob("*_msweep.json")]
    ms_rows = [r for r in ms_rows if r]

    if main_rows:
        sm = rows_to_summary(main_rows)
        sm.to_csv(out / "summary_main.csv", index=False)
        print(f"Wrote {out / 'summary_main.csv'}")

    if abl_rows:
        sa = rows_to_summary(abl_rows)
        sa.to_csv(out / "summary_ablation.csv", index=False)
        print(f"Wrote {out / 'summary_ablation.csv'}")

    if ms_rows:
        ms_df = pd.DataFrame(ms_rows)
        ms_df.to_csv(out / "summary_m_sweep.csv", index=False)
        print(f"Wrote {out / 'summary_m_sweep.csv'}")

    stats_df = paired_ttests(cfg)
    if not stats_df.empty:
        stats_df.to_csv(out / "stats_paired_ttest.csv", index=False)
        print(f"Wrote {out / 'stats_paired_ttest.csv'}")

    return stats_df


def plot_m_sweep(cfg):
    path = Path(cfg["output_dir"]) / "summary_m_sweep.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "m_sweep_val" not in df.columns:
        return
    # per-m val curves stored inside each JSON; replot from raw jsons
    records = []
    for p in Path(cfg["output_dir"]).glob("*_msweep.json"):
        r = load_result(p)
        for item in r.get("m_sweep_val", []):
            records.append({"dataset": r["dataset"], "seed": r["seed"], "m": item["m"], "val_acc": item["val_acc"]})
    if not records:
        return
    pdf = pd.DataFrame(records)
    agg = pdf.groupby(["dataset", "m"])["val_acc"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(8, 4))
    for ds in agg["dataset"].unique():
        sub = agg[agg["dataset"] == ds]
        plt.errorbar(sub["m"], sub["mean"], yerr=sub["std"], marker="o", label=ds)
    plt.xlabel("mutation factor m")
    plt.ylabel("validation accuracy")
    plt.title("m-sweep (val only)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(cfg["output_dir"]) / "fig_a1_m_sweep.png", dpi=150)
    plt.show()


def run_experiments(cfg):
    mode = cfg["mode"]
    print(colored(f"\\n=== Phase 3 runner — mode={mode} ===", "magenta", attrs=["bold"]))
    if mode in ("main", "all"):
        run_main(cfg)
    if mode in ("ablation", "all"):
        run_ablation(cfg)
    if mode in ("m_sweep", "all"):
        run_m_sweep(cfg)
    aggregate_results(cfg)
    plot_m_sweep(cfg)
    print(colored("\\nDone. Copy results/ to repo phase3/ and run import_results_to_latex.py", "green"))
'''

CELLS.append(code(RUNNERS_CODE))

CELLS.append(code("""# ========== Run ==========
# Change CONFIG["mode"] and CONFIG["seeds"] before each Colab session.
run_experiments(CONFIG)
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []},
    },
    "cells": CELLS,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} ({len(CELLS)} cells)")
