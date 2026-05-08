# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Baseline leaderboard, SIRA comparison charts, and per-query failure reports.

Data
~~~~
Published baseline numbers on BEIR (10 datasets) and TREC-DL 2019.

Agentic methods
    All agentic methods (HyDE, CoT, Search-R1) use **Qwen2.5-3B-Instruct**.

Non-agentic baselines (official checkpoints)
    - SPARTA : ``BeIR/sparta-msmarco-distilbert-base-v1``
    - SPLADE : ``naver/splade-cocondenser-ensembledistil``
    - DocT5  : ``doc2query/msmarco-t5-base-v1``
    - E5     : ``intfloat/e5-base-v2``

Search agent QA accuracy (NQ & HotpotQA, best reported)
    E-GRPO, HiPRAG, SSP, A²Search, TIPS, Search-R1

Usage::

    source sandbox.sh
    python scripts/failure_report.py                     # summary + plots + report + browse
    python scripts/failure_report.py --force              # regenerate report even if up-to-date
    python scripts/failure_report.py --complete           # also generate complete-pipeline charts
    python scripts/failure_report.py --quick               # leaderboard + plots only (fast, no per-query analysis)
    python scripts/failure_report.py --no-browse          # skip TUI browser, just generate report
    python scripts/failure_report.py --browse             # TUI browser only (auto-regenerates if stale)
    python scripts/failure_report.py --browse --k 50

Charts (to ``analysis/plots/``)::

    render:recall10-comparison.pdf          — all 10 BEIR datasets (small multiples)
    render:recall10-comparison-complete.pdf — complete-pipeline datasets only
    render:recall10-stages.pdf              — pipeline progression, small multiples
    render:recall10-stages-complete.pdf     — pipeline progression, complete only
    render:qe-recall-vs-sota.pdf            — QE R@50/200 vs best baseline R@10
    render:qe-recall-vs-sota-complete.pdf   — QE R@50/200 vs SOTA, complete only

Statistics (to ``analysis/``)::

    analyze:baseline-leaderboard.md   — BEIR averages + SIRA pipeline progression

Failure report outputs::

    {db_root}/reports/failures-k{k}.json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

STAGE_ORDER = ["baseline", "doc-enrich", "query-enrich", "rerank"]

# ---------------------------------------------------------------------------
# Baseline scores — each list is aligned with ``DATASETS``
# ---------------------------------------------------------------------------

MetricName = Literal["Recall@10", "Recall@50", "Recall@100", "NDCG@10", "NDCG@50", "NDCG@100"]

DATASETS: list[str] = [
    "Arguana",
    "Climate-Fever",
    "CQADupStack",
    "Fever",
    "FIQA",
    "HotpotQA",
    "NQ",
    "Quora",
    "SCIDOCS",
    "SciFact",
    "TREC-DL 2019",
]

BEIR_DATASETS: list[str] = [d for d in DATASETS if d != "TREC-DL 2019"]

METHODS: list[str] = [
    "HyDE",
    "SPARTA",
    "DocT5",
    "CoT",
    "BM25",
    "Search-R1",
    "E5",
    "SPLADE",
    "GrepRAG",
    "ShellAgent",
]

# fmt: off
RECALL_10: dict[str, list[float]] = {
    "HyDE":      [0.7091, 0.2598, 0.3299, 0.7132, 0.2845, 0.5051, 0.4918, 0.5151, 0.1530, 0.8344, 0.1047],
    "SPARTA":    [0.6181, 0.1050, 0.3100, 0.7246, 0.2450, 0.5356, 0.5584, 0.7445, 0.1303, 0.7084, 0.1327],
    "DocT5":     [0.7824, 0.1761, 0.4339, 0.6769, 0.3397, 0.6527, 0.5068, 0.8975, 0.1663, 0.8270, 0.1456],
    "CoT":       [0.7752, 0.1867, 0.4020, 0.6765, 0.3086, 0.5789, 0.4961, 0.8704, 0.1595, 0.7961, 0.1278],
    "BM25":      [0.7738, 0.1764, 0.4163, 0.6747, 0.3198, 0.6141, 0.4543, 0.9014, 0.1636, 0.8078, 0.1274],
    "Search-R1": [0.7774, 0.1884, 0.3995, 0.6587, 0.3191, 0.5988, 0.4863, 0.8609, 0.1580, 0.8201, 0.1313],
    "E5":        [0.7909, 0.2899, 0.5138, 0.9109, 0.4697, 0.7276, 0.7877, 0.9428, 0.1962, 0.8489, 0.1721],
    "SPLADE":    [0.8137, 0.2881, 0.4924, 0.8954, 0.4139, 0.7027, 0.7381, 0.9206, 0.1654, 0.8230, 0.1765],
    "GrepRAG":   [0.5746, 0.0176, 0.2400, 0.1635, 0.1009, 0.3434, 0.1628, 0.5105, 0.1027, 0.5883, 0.0000],
    "ShellAgent": [0.2263, 0.0305, 0.2084, 0.2557, 0.1035, 0.4327, 0.1884, 0.3330, 0.0843, 0.6685, 0.0000],
}

RECALL_100: dict[str, list[float]] = {
    "HyDE":      [0.9531, 0.4963, 0.5280, 0.8591, 0.5408, 0.6928, 0.7675, 0.7480, 0.3640, 0.9349, 0.3889],
    "SPARTA":    [0.8947, 0.2271, 0.4648, 0.8488, 0.4400, 0.6776, 0.7869, 0.8954, 0.3016, 0.8647, 0.3833],
    "DocT5":     [0.9644, 0.3739, 0.5905, 0.8626, 0.5845, 0.7879, 0.7967, 0.9768, 0.3703, 0.9177, 0.4947],
    "CoT":       [0.9673, 0.3853, 0.5742, 0.8572, 0.5729, 0.7297, 0.7861, 0.9668, 0.3472, 0.9103, 0.4620],
    "BM25":      [0.9630, 0.3728, 0.5794, 0.8609, 0.5534, 0.7694, 0.7465, 0.9768, 0.3630, 0.9127, 0.4377],
    "Search-R1": [0.9644, 0.3933, 0.5658, 0.8497, 0.5634, 0.7486, 0.7752, 0.9627, 0.3447, 0.9093, 0.4565],
    "E5":        [0.9737, 0.5230, 0.7007, 0.9613, 0.7204, 0.8490, 0.9483, 0.9941, 0.4184, 0.9467, 0.5385],
    "SPLADE":    [0.9829, 0.5209, 0.6939, 0.9542, 0.6316, 0.8176, 0.9295, 0.9865, 0.3734, 0.9320, 0.5521],
    "GrepRAG":   [0.7952, 0.0746, 0.3487, 0.3987, 0.2568, 0.5623, 0.4349, 0.5600, 0.2600, 0.8179, 0.0000],
    "ShellAgent": [0.4261, 0.0909, 0.2779, 0.4207, 0.1903, 0.5525, 0.3167, 0.3886, 0.1623, 0.7958, 0.0000],
}

RECALL_50: dict[str, list[float]] = {
    "GrepRAG":   [0.7573, 0.0479, 0.3161, 0.3100, 0.2044, 0.4989, 0.3497, 0.5527, 0.2065, 0.7579, 0.0000],
    "ShellAgent": [0.4226, 0.0839, 0.2763, 0.4165, 0.1841, 0.5513, 0.3120, 0.3873, 0.1590, 0.7908, 0.0000],
}

NDCG_50: dict[str, list[float]] = {
    "GrepRAG":   [0.3979, 0.0208, 0.2230, 0.1308, 0.1041, 0.3349, 0.1332, 0.4671, 0.1301, 0.4539, 0.0000],
    "ShellAgent": [0.1580, 0.0352, 0.1728, 0.1691, 0.0911, 0.3764, 0.1339, 0.2617, 0.1014, 0.4684, 0.0000],
}

NDCG_10: dict[str, list[float]] = {
    "HyDE":      [0.4366, 0.2004, 0.2463, 0.5507, 0.2223, 0.4451, 0.3315, 0.3924, 0.1402, 0.6565, 0.4949],
    "SPARTA":    [0.3890, 0.0852, 0.2497, 0.6101, 0.1925, 0.5132, 0.3983, 0.6294, 0.1272, 0.5894, 0.6189],
    "DocT5":     [0.4946, 0.1381, 0.3667, 0.5122, 0.2731, 0.6299, 0.3302, 0.7960, 0.1589, 0.6920, 0.5533],
    "CoT":       [0.4951, 0.1471, 0.3354, 0.4932, 0.2486, 0.5595, 0.3168, 0.7608, 0.1518, 0.6647, 0.4935],
    "BM25":      [0.4874, 0.1372, 0.3481, 0.5036, 0.2532, 0.5851, 0.2916, 0.8055, 0.1565, 0.6791, 0.4722],
    "Search-R1": [0.4911, 0.1487, 0.3282, 0.4824, 0.2530, 0.5713, 0.3110, 0.7529, 0.1511, 0.6987, 0.5142],
    "E5":        [0.5323, 0.2397, 0.4196, 0.8096, 0.3932, 0.6905, 0.5835, 0.8648, 0.1855, 0.7156, 0.7119],
    "SPLADE":    [0.5253, 0.2293, 0.4083, 0.7933, 0.3478, 0.6869, 0.5369, 0.8344, 0.1586, 0.7025, 0.7352],
    "GrepRAG":   [0.3555, 0.0122, 0.2048, 0.0971, 0.0763, 0.2927, 0.0908, 0.4557, 0.0921, 0.4129, 0.0000],
    "ShellAgent": [0.1114, 0.0206, 0.1558, 0.1298, 0.0686, 0.3417, 0.1036, 0.2476, 0.0727, 0.4374, 0.0000],
}

NDCG_100: dict[str, list[float]] = {
    "HyDE":      [0.4904, 0.2685, 0.2919, 0.5837, 0.2889, 0.4928, 0.3925, 0.4465, 0.2117, 0.6791, 0.4608],
    "SPARTA":    [0.4492, 0.1178, 0.2846, 0.6372, 0.2440, 0.5492, 0.4490, 0.6656, 0.1847, 0.6230, 0.5152],
    "DocT5":     [0.5364, 0.1929, 0.4034, 0.5534, 0.3364, 0.6645, 0.3952, 0.8171, 0.2276, 0.7133, 0.5522],
    "CoT":       [0.5393, 0.2024, 0.3750, 0.5332, 0.3178, 0.5979, 0.3810, 0.7858, 0.2150, 0.6904, 0.5020],
    "BM25":      [0.5305, 0.1915, 0.3857, 0.5447, 0.3148, 0.6246, 0.3562, 0.8257, 0.2235, 0.7036, 0.4777],
    "Search-R1": [0.5342, 0.2057, 0.3665, 0.5248, 0.3177, 0.6096, 0.3748, 0.7793, 0.2142, 0.7188, 0.5060],
    "E5":        [0.5734, 0.3067, 0.4638, 0.8220, 0.4610, 0.7217, 0.6217, 0.8796, 0.2614, 0.7376, 0.6571],
    "SPLADE":    [0.5649, 0.2965, 0.4557, 0.8069, 0.4067, 0.7164, 0.5815, 0.8524, 0.2286, 0.7248, 0.6723],
    "GrepRAG":   [0.4041, 0.0265, 0.2290, 0.1456, 0.1150, 0.3475, 0.1478, 0.4686, 0.1448, 0.4642, 0.0000],
    "ShellAgent": [0.1587, 0.0368, 0.1731, 0.1699, 0.0925, 0.3766, 0.1347, 0.2619, 0.1024, 0.4693, 0.0000],
}
# fmt: on

ALL_METRICS: dict[MetricName, dict[str, list[float]]] = {
    "Recall@10": RECALL_10,
    "Recall@50": RECALL_50,
    "Recall@100": RECALL_100,
    "NDCG@10": NDCG_10,
    "NDCG@50": NDCG_50,
    "NDCG@100": NDCG_100,
}

# ---------------------------------------------------------------------------
# Search Agent QA accuracy (%) — NQ, HotpotQA
# ---------------------------------------------------------------------------

QA_DATASETS: list[str] = ["NQ", "HotpotQA"]

# fmt: off
QA_ACCURACY: dict[str, list[float]] = {
    "Search-R1":  [48.0, 43.3],
    "TIPS":       [43.38, 42.95],
    "A²Search":   [51.4, 49.5],
    "E-GRPO":     [55.8, 69.0],
    "SSP":        [62.6, 62.8],
    "HiPRAG":     [71.2, 62.4],
}
# fmt: on

QA_REFERENCES: dict[str, dict[str, str]] = {
    "Search-R1":  {"arxiv": "2503.09516", "model": "Qwen2.5-7B (RL)"},
    "TIPS":       {"arxiv": "2603.22293", "model": "Qwen2.5-7B-Instruct"},
    "A²Search":   {"arxiv": "2510.07958", "model": "Qwen2.5-7B-Instruct"},
    "E-GRPO":     {"arxiv": "2510.24694", "model": "Qwen2.5-7B-Instruct, Web Env"},
    "SSP":        {"arxiv": "2510.18821", "model": "Qwen2.5-32B-Instruct"},
    "HiPRAG":     {"arxiv": "2510.07794", "model": "Qwen2.5-7B-Instruct (GRPO)"},
}

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DB_ROOT = os.environ.get("SIRA_DB_ROOT", "./experiments")
PLOTS_DIR = Path(__file__).resolve().parent.parent / "analysis" / "plots"

SLUG_MAP: dict[str, str] = {
    "Arguana": "arguana",
    "Climate-Fever": "climate-fever",
    "CQADupStack": "cqadupstack",
    "Fever": "fever",
    "FIQA": "fiqa",
    "HotpotQA": "hotpotqa",
    "NQ": "nq",
    "Quora": "quora",
    "SCIDOCS": "scidocs",
    "SciFact": "scifact",
}

SHOW_METHODS = [
    "BM25",
    "DocT5",
    "HyDE",
    "CoT",
    "Search-R1",
    "SPARTA",
    "SPLADE",
    "E5",
    "GrepRAG",
    "ShellAgent",
    "SIRA",
]

BAR_COLORS: dict[str, str] = {
    "BM25": "#78909c",
    "DocT5": "#039be5",
    "HyDE": "#3949ab",
    "CoT": "#43a047",
    "Search-R1": "#7cb342",
    "SPARTA": "#fb8c00",
    "SPLADE": "#e64a19",
    "E5": "#d32f2f",
    "GrepRAG": "#00acc1",
    "ShellAgent": "#26a69a",
    "SIRA": "#7b1fa2",
}

LINE_COLORS: dict[str, str] = {
    "Arguana": "#1565c0",
    "Climate-Fever": "#00838f",
    "CQADupStack": "#c62828",
    "Fever": "#ef6c00",
    "FIQA": "#d84315",
    "HotpotQA": "#2e7d32",
    "NQ": "#283593",
    "Quora": "#6a1b9a",
    "SCIDOCS": "#558b2f",
    "SciFact": "#ad1457",
}

LINE_MARKERS: dict[str, str] = {
    "Arguana": "s",
    "Climate-Fever": "P",
    "CQADupStack": "v",
    "Fever": "X",
    "FIQA": "D",
    "HotpotQA": "p",
    "NQ": "h",
    "Quora": "*",
    "SCIDOCS": "^",
    "SciFact": "o",
}

SIRA_STAGE_LABELS: dict[str, str] = {
    "baseline": "base",
    "doc-enrich": "doc",
    "query-enrich": "qry",
    "rerank": "rrk",
}


def _apply_chart_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 16,
            "axes.facecolor": "#fafbfd",
            "figure.facecolor": "white",
            "axes.edgecolor": "#c0c8d4",
            "axes.linewidth": 0.8,
            "grid.color": "#dde3ec",
            "grid.linewidth": 0.6,
            "xtick.color": "#3a4a5c",
            "ytick.color": "#3a4a5c",
            "text.color": "#1e2a3a",
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "axes.titlesize": 26,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
        }
    )


def _style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c0c8d4")
    ax.spines["bottom"].set_color("#c0c8d4")
    ax.tick_params(axis="both", which="major", length=4, width=0.8)


# ---------------------------------------------------------------------------
# Leaderboard — DataFrame helpers
# ---------------------------------------------------------------------------


def get_dataframe(metric: MetricName) -> pd.DataFrame:
    return pd.DataFrame(ALL_METRICS[metric], index=DATASETS).T


def get_beir_averages(metric: MetricName) -> pd.Series:
    df = get_dataframe(metric)[BEIR_DATASETS]
    return df.mean(axis=1).rename(f"{metric} (BEIR avg)")


# ---------------------------------------------------------------------------
# Leaderboard — SIRA results loading
# ---------------------------------------------------------------------------


def collect_sira_results(
    db_root: str = DB_ROOT,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    """Returns (best_r10, per_stage_r10, per_stage_all_metrics).

    per_stage_all_metrics: {dataset: {stage: {metric_name: value}}}
    """
    from concurrent.futures import ThreadPoolExecutor

    tasks = [
        (name, slug, st, os.path.join(db_root, slug, "eval", st, "best.json"))
        for name, slug in SLUG_MAP.items()
        for st in STAGE_ORDER
    ]

    def _read(t: tuple) -> tuple[str, str, dict | None]:
        name, _slug, stage, fp = t
        data = _load_json(fp)
        return name, stage, data

    per_stage: dict[str, dict[str, float]] = {n: {} for n in SLUG_MAP}
    per_stage_all: dict[str, dict[str, dict[str, float]]] = {n: {} for n in SLUG_MAP}
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        for name, stage, data in pool.map(_read, tasks):
            if data is not None:
                if "Recall@10" in data:
                    per_stage[name][stage] = data["Recall@10"]
                per_stage_all[name][stage] = data

    best_r10 = {n: list(sv.values())[-1] for n, sv in per_stage.items() if sv}
    return best_r10, per_stage, per_stage_all


def print_summary(
    sira_best: dict[str, float] | None = None,
    per_stage: dict[str, dict[str, float]] | None = None,
) -> None:
    for metric_name in ALL_METRICS:
        avg = get_beir_averages(metric_name)  # type: ignore[arg-type]
        print(f"\n=== {avg.name} ===")
        for method, value in avg.items():
            print(f"  {method:<12s} {value:.4f}")

    if sira_best is None or per_stage is None:
        sira_best, per_stage, per_stage_all = collect_sira_results()
    if not per_stage:
        return

    print("\n" + "=" * 60)
    print("SIRA Pipeline — Recall@10 per stage")
    print("=" * 60)
    header = f"  {'Dataset':<16s}"
    for st in STAGE_ORDER:
        header += f"  {st:<14s}"
    print(header)
    print("  " + "-" * (16 + 16 * len(STAGE_ORDER)))
    for name in SLUG_MAP:
        stages = per_stage.get(name, {})
        if not stages:
            continue
        row = f"  {name:<16s}"
        for st in STAGE_ORDER:
            v = stages.get(st)
            row += f"  {v:<14.4f}" if v is not None else f"  {'-':<14s}"
        print(row)

    complete = [d for d in SLUG_MAP if len(per_stage.get(d, {})) == len(STAGE_ORDER)]
    if complete:
        print(f"\n  Complete pipelines: {', '.join(complete)}")
        for d in complete:
            best = list(per_stage[d].values())[-1]
            baseline_others = [RECALL_10[m][DATASETS.index(d)] for m in METHODS]
            beat = best > max(baseline_others)
            print(
                f"    {d}: SIRA={best:.4f}  best-baseline={max(baseline_others):.4f}"
                f"  {'✓ SIRA wins' if beat else '✗ baseline wins'}"
            )


def save_leaderboard_md(
    sira_best: dict[str, float] | None = None,
    per_stage: dict[str, dict[str, float]] | None = None,
    per_stage_all: dict[str, dict[str, dict[str, float]]] | None = None,
) -> None:
    """Save baseline leaderboard and SIRA pipeline stats as markdown."""
    if sira_best is None or per_stage is None or per_stage_all is None:
        sira_best, per_stage, per_stage_all = collect_sira_results()

    lines: list[str] = []
    lines.append("# Baseline Leaderboard & SIRA Pipeline\n")

    # --- Dataset metadata ---
    lines.append("## Dataset Overview\n")
    lines.append("| Dataset | Corpus | Queries | Qrels | Avg Labels/Query |")
    lines.append("|---------|-------:|--------:|------:|-----------------:|")
    for name, slug in SLUG_MAP.items():
        ds_dir = os.path.join(DB_ROOT, slug)
        corpus_path = os.path.join(ds_dir, "raw", "corpus.jsonl")
        queries_path = os.path.join(ds_dir, "raw", "queries-test.jsonl")
        qrels_path = os.path.join(ds_dir, "raw", "qrels-test.jsonl")
        try:
            n_corpus = sum(1 for _ in open(corpus_path))
            n_queries = sum(1 for _ in open(queries_path))
            from collections import Counter
            qc: Counter[str] = Counter()
            for line in open(qrels_path):
                qc[json.loads(line)["query-id"]] += 1
            n_qrels = sum(qc.values())
            avg_lab = n_qrels / len(qc) if qc else 0
            lines.append(
                f"| {name} | {n_corpus:,} | {n_queries:,} | {n_qrels:,} | {avg_lab:.2f} |"
            )
        except Exception:
            pass
    lines.append("")

    # --- BEIR averages ---
    for metric_name in ALL_METRICS:
        avg = get_beir_averages(metric_name)  # type: ignore[arg-type]
        lines.append(f"## {avg.name}\n")
        lines.append("| Method | Score |")
        lines.append("|--------|------:|")
        for method, value in avg.sort_values(ascending=False).items():
            lines.append(f"| {method} | {value:.4f} |")
        lines.append("")

    # --- Per-dataset per-metric tables ---
    for metric_name, metric_dict in ALL_METRICS.items():
        lines.append(f"## {metric_name} — Per Dataset\n")
        methods_with_data = [m for m in METHODS if m in metric_dict]
        header = "| Dataset |"
        sep = "|---------|"
        for m in methods_with_data:
            header += f" {m} |"
            sep += "------:|"
        lines.append(header)
        lines.append(sep)
        for ds_name in BEIR_DATASETS:
            di = DATASETS.index(ds_name)
            row = f"| {ds_name} |"
            for m in methods_with_data:
                v = metric_dict[m][di]
                row += f" {v:.4f} |" if v > 0 else " — |"
            lines.append(row)
        lines.append("")

    # --- SIRA Pipeline stages (all metrics) ---
    if per_stage_all:
        sira_metrics = ["Recall@10", "Recall@50", "Recall@100", "NDCG@10", "NDCG@50", "NDCG@100"]
        for metric in sira_metrics:
            lines.append(f"## SIRA Pipeline — {metric} per stage\n")
            header = "| Dataset |"
            sep = "|---------|"
            for st in STAGE_ORDER:
                header += f" {st} |"
                sep += "------:|"
            lines.append(header)
            lines.append(sep)
            for name in SLUG_MAP:
                stages = per_stage_all.get(name, {})
                if not stages:
                    continue
                row = f"| {name} |"
                for st in STAGE_ORDER:
                    v = stages.get(st, {}).get(metric)
                    row += f" {v:.4f} |" if v is not None else " — |"
                lines.append(row)
            lines.append("")

        complete = [d for d in SLUG_MAP if len(per_stage.get(d, {})) == len(STAGE_ORDER)]
        if complete:
            lines.append("## Complete Pipelines — SIRA vs Best Baseline\n")
            lines.append("| Dataset | SIRA | Best Baseline | Winner |")
            lines.append("|---------|-----:|--------------:|--------|")
            for d in complete:
                best = list(per_stage[d].values())[-1]
                baseline_others = [RECALL_10[m][DATASETS.index(d)] for m in METHODS]
                best_bl = max(baseline_others)
                winner = "SIRA" if best > best_bl else "Baseline"
                lines.append(f"| {d} | {best:.4f} | {best_bl:.4f} | {winner} |")
            lines.append("")

    # --- Search Agent QA accuracy ---
    lines.append("## Search Agent QA Accuracy (%) — NQ & HotpotQA\n")
    lines.append("| Paper | Model | NQ | HotpotQA |")
    lines.append("|-------|-------|---:|---------:|")
    for method in sorted(QA_ACCURACY, key=lambda m: -QA_ACCURACY[m][0]):
        nq, hqa = QA_ACCURACY[method]
        ref = QA_REFERENCES.get(method, {})
        model = ref.get("model", "")
        lines.append(f"| {method} | {model} | {nq:.1f} | {hqa:.1f} |")
    lines.append("")

    out = PLOTS_DIR.parent / "analyze:baseline-leaderboard.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Leaderboard — Charts
# ---------------------------------------------------------------------------


def plot_rerank_comparison(
    datasets: list[str],
    sira_best: dict[str, float],
    per_stage: dict[str, dict[str, float]],
    suffix: str = "",
    title_extra: str = "",
    metric: MetricName = "Recall@10",
) -> None:
    import matplotlib.pyplot as plt

    _apply_chart_style()
    n_d, n_m = len(datasets), len(SHOW_METHODS)

    width = max(22, len(datasets) * 2.6)
    fig, ax = plt.subplots(figsize=(width, 10))
    _style_ax(ax)

    bar_w, gap = 0.09, 0.35

    metric_dict = ALL_METRICS[metric]
    for j, method in enumerate(SHOW_METHODS):
        if method != "SIRA" and method not in metric_dict:
            continue
        vals = []
        for d in datasets:
            if method == "SIRA":
                vals.append(sira_best.get(d, 0))
            else:
                vals.append(metric_dict[method][DATASETS.index(d)])

        xs = [i * (n_m * bar_w + gap) + j * bar_w for i in range(n_d)]
        is_sira = method == "SIRA"
        ax.bar(
            xs,
            vals,
            width=bar_w,
            label=method,
            color=BAR_COLORS[method],
            edgecolor="#6a1b9a" if is_sira else "white",
            linewidth=2.5 if is_sira else 0.5,
            zorder=10 if is_sira else 5,
            alpha=0.9,
        )

        if is_sira:
            for i, d in enumerate(datasets):
                sv = vals[i]
                if sv == 0:
                    continue
                others = [
                    metric_dict[m][DATASETS.index(d)]
                    for m in SHOW_METHODS
                    if m != "SIRA" and m in metric_dict
                ]
                stages = per_stage.get(d, {})
                last_stage = list(stages.keys())[-1] if stages else ""
                tag = SIRA_STAGE_LABELS.get(last_stage, "")
                y_off = 0.012
                if sv > max(others):
                    ax.text(
                        xs[i],
                        sv + y_off,
                        f"★\n{tag}",
                        ha="center",
                        fontsize=18,
                        color="#6a1b9a",
                        fontweight="bold",
                        zorder=11,
                        linespacing=0.8,
                    )
                else:
                    ax.text(
                        xs[i],
                        sv + y_off,
                        tag,
                        ha="center",
                        fontsize=16,
                        color="#6a1b9a",
                        zorder=11,
                    )

    tick_pos = [i * (n_m * bar_w + gap) + (n_m - 1) * bar_w / 2 for i in range(n_d)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(datasets, fontsize=20, fontweight="bold", rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel(metric, fontsize=22, fontweight="bold")
    title = f"{metric} — SIRA (best stage) vs. Baselines"
    if title_extra:
        title += f" — {title_extra}"
    ax.set_title(title, fontsize=24, fontweight="bold", pad=16, color="#1e2a3a")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=6,
        fontsize=16,
        framealpha=0.95,
        edgecolor="#c0c8d4",
        fancybox=True,
    )

    metric_slug = metric.lower().replace("@", "")
    out = PLOTS_DIR / f"render:{metric_slug}-comparison{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_pipeline_stages(
    datasets: list[str],
    per_stage: dict[str, dict[str, float]],
    suffix: str = "",
    title_extra: str = "",
    metric: MetricName = "Recall@10",
    per_stage_all: dict[str, dict[str, dict[str, float]]] | None = None,
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    _apply_chart_style()
    stage_labels = ["BM25", "DocE", "QE", "Post"]
    x_pos = np.arange(len(STAGE_ORDER))

    # Use per_stage_all if available, otherwise fall back to per_stage (Recall@10 only)
    def _get_val(d: str, sk: str) -> float:
        if per_stage_all and d in per_stage_all and sk in per_stage_all[d]:
            return per_stage_all[d][sk].get(metric, np.nan)
        if metric == "Recall@10":
            return per_stage.get(d, {}).get(sk, np.nan)
        return np.nan

    active = [
        d
        for d in datasets
        if not all(np.isnan(_get_val(d, sk)) for sk in STAGE_ORDER)
    ]
    n = len(active)
    if n == 0:
        return

    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 8, nrows * 8),
        squeeze=False,
    )

    for idx, d in enumerate(active):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        _style_ax(ax)
        vals = [_get_val(d, sk) for sk in STAGE_ORDER]

        ax.fill_between(
            x_pos,
            vals,
            alpha=0.08,
            color=LINE_COLORS[d],
            zorder=2,
        )
        ax.plot(
            x_pos,
            vals,
            marker=LINE_MARKERS[d],
            markersize=16,
            linewidth=4.0,
            color=LINE_COLORS[d],
            zorder=5,
            markeredgecolor="white",
            markeredgewidth=2.0,
        )
        for xi, v in zip(x_pos, vals):
            if not np.isnan(v):
                ax.annotate(
                    f"{v:.3f}",
                    (xi, v),
                    textcoords="offset points",
                    xytext=(0, 14),
                    ha="center",
                    va="bottom",
                    fontsize=36,
                    fontweight="bold",
                    color=LINE_COLORS[d],
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor=LINE_COLORS[d],
                        alpha=0.9,
                        linewidth=0.6,
                    ),
                )

        pass

        ax.set_title(d, fontsize=60, fontweight="bold", color=LINE_COLORS[d], pad=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_labels, fontsize=48, fontweight="normal")
        ax.tick_params(axis="y", labelsize=36)
        ax.set_ylim(0.05, 1.12)
        ax.grid(axis="y", alpha=0.35, zorder=0)
        ax.grid(axis="x", alpha=0.15, zorder=0, linestyle="--")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    title = f"SIRA Pipeline — {metric} Progression"
    if title_extra:
        title += f"\n{title_extra}"
    fig.suptitle(title, fontsize=72, fontweight="bold", y=1.02, color="#1e2a3a")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    metric_slug = metric.lower().replace("@", "")
    out = PLOTS_DIR / f"render:{metric_slug}-stages{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_qe_recall_vs_sota(
    datasets: list[str],
    per_stage: dict[str, dict[str, float]],
    suffix: str = "",
) -> None:
    """QE Recall@50/200 vs best-baseline Recall@10 — shows rerank headroom."""
    from concurrent.futures import ThreadPoolExecutor

    import matplotlib.pyplot as plt

    _apply_chart_style()
    paths = {
        name: os.path.join(DB_ROOT, slug, "eval", "query-enrich", "best.json")
        for name, slug in SLUG_MAP.items()
        if name in datasets
    }
    qe_r50: dict[str, float] = {}
    qe_r200: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        results = pool.map(lambda kv: (kv[0], _load_json(kv[1])), paths.items())
    for name, best in results:
        if best:
            if "Recall@50" in best:
                qe_r50[name] = best["Recall@50"]
            if "Recall@200" in best:
                qe_r200[name] = best["Recall@200"]

    active = [d for d in datasets if d in qe_r50 and d in DATASETS]
    if not active:
        return

    sota_r10 = {}
    for d in active:
        di = DATASETS.index(d)
        sota_r10[d] = max(RECALL_10[m][di] for m in METHODS)

    sira_r10 = {}
    for d in active:
        stages = per_stage.get(d, {})
        if stages:
            sira_r10[d] = list(stages.values())[-1]

    has_r200 = any(d in qe_r200 for d in active)
    n_bars = 3 + (1 if has_r200 else 0)
    x = np.arange(len(active))
    w = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(14, len(active) * 3), 10))
    _style_ax(ax)

    ax.bar(
        x - w * (n_bars - 1) / 2,
        [sota_r10[d] for d in active],
        width=w,
        color="#1976d2",
        alpha=0.88,
        label="Best Baseline R@10",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    ax.bar(
        x - w * (n_bars - 1) / 2 + w,
        [qe_r50[d] for d in active],
        width=w,
        color="#00897b",
        alpha=0.88,
        label="SIRA QE R@50",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    bar_idx = 2
    if has_r200:
        ax.bar(
            x - w * (n_bars - 1) / 2 + w * bar_idx,
            [qe_r200.get(d, 0) for d in active],
            width=w,
            color="#43a047",
            alpha=0.88,
            label="SIRA QE R@200",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        bar_idx += 1
    sira_vals = [sira_r10.get(d, 0) for d in active]
    if any(v > 0 for v in sira_vals):
        ax.bar(
            x - w * (n_bars - 1) / 2 + w * bar_idx,
            sira_vals,
            width=w,
            color="#e65100",
            alpha=0.88,
            label="SIRA Best R@10",
            edgecolor="#bf360c",
            linewidth=1.8,
            zorder=5,
        )

    for i, d in enumerate(active):
        gap50 = qe_r50[d] - sota_r10[d]
        sign = "+" if gap50 >= 0 else ""
        color = "#1b5e20" if gap50 >= 0 else "#b71c1c"
        vals = [sota_r10[d], qe_r50[d], sira_r10.get(d, 0)]
        if d in qe_r200:
            vals.append(qe_r200[d])
        bar_top = max(vals)
        label = f"R@50 {sign}{gap50:.3f}"
        if d in qe_r200:
            gap200 = qe_r200[d] - sota_r10[d]
            s2 = "+" if gap200 >= 0 else ""
            label += f"\nR@200 {s2}{gap200:.3f}"
        ax.annotate(
            label,
            (i, bar_top + 0.012),
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color=color,
            zorder=11,
            linespacing=1.3,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=color,
                alpha=0.88,
                linewidth=0.8,
            ),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(active, fontsize=20, fontweight="bold", rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("Score", fontsize=22, fontweight="bold")
    ax.set_title(
        "Rerank Headroom — SIRA QE Recall@50/200 vs Best Baseline Recall@10",
        fontsize=26,
        fontweight="bold",
        pad=18,
        color="#1e2a3a",
    )
    ax.set_ylim(0, 1.22)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.legend(
        fontsize=19,
        framealpha=0.95,
        edgecolor="#c0c8d4",
        fancybox=True,
        loc="upper center",
        ncol=n_bars,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.text(
        0.5,
        -0.06,
        "Gap = QE Recall − Best Baseline R@10  |  "
        "Positive gap → reranker has enough candidates to beat SOTA",
        fontsize=14,
        color="#5a6a7a",
        style="italic",
        ha="center",
    )

    out = PLOTS_DIR / f"render:qe-recall-vs-sota{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {out}")


def generate_plots(
    sira_best: dict[str, float] | None = None,
    per_stage: dict[str, dict[str, float]] | None = None,
    complete_only: bool = False,
    per_stage_all: dict[str, dict[str, dict[str, float]]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_datasets = list(SLUG_MAP.keys())
    if sira_best is None or per_stage is None:
        sira_best, per_stage, per_stage_all = collect_sira_results()

    plot_rerank_comparison(all_datasets, sira_best, per_stage, metric="Recall@10")
    plot_rerank_comparison(all_datasets, sira_best, per_stage, metric="NDCG@10")
    plot_pipeline_stages(all_datasets, per_stage, metric="Recall@10", per_stage_all=per_stage_all)
    plot_pipeline_stages(all_datasets, per_stage, metric="NDCG@10", per_stage_all=per_stage_all)
    plot_qe_recall_vs_sota(all_datasets, per_stage)

    complete_datasets = [
        d for d in all_datasets if len(per_stage.get(d, {})) == len(STAGE_ORDER)
    ]
    if complete_only and complete_datasets:
        plot_rerank_comparison(
            complete_datasets,
            sira_best,
            per_stage,
            suffix="-complete",
            title_extra="Complete Pipeline Only",
        )
        plot_pipeline_stages(
            complete_datasets,
            per_stage,
            suffix="-complete",
            title_extra="Complete Pipeline Only",
        )
        plot_qe_recall_vs_sota(
            complete_datasets,
            per_stage,
            suffix="-complete",
        )


# ---------------------------------------------------------------------------
# Failure report — Data loading
# ---------------------------------------------------------------------------


def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _stage_meta(ds_dir: str, stage: str) -> dict | None:
    return _load_json(os.path.join(ds_dir, "eval", stage, "best.meta.json"))


def _stage_metrics(ds_dir: str, stage: str) -> dict | None:
    return _load_json(os.path.join(ds_dir, "eval", stage, "best.json"))


def _scan_traces(
    run_dir: str,
    id_key: str,
    ids: set[str],
) -> dict[str, list[dict]]:
    if not run_dir or not ids or not os.path.isdir(run_dir):
        return {}
    import glob

    files = []
    for pat in [
        "trace.kept.jsonl", "trace.kept.shard*.jsonl",
        "trace.failed.jsonl", "trace.failed.shard*.jsonl",
    ]:
        files.extend(glob.glob(os.path.join(run_dir, pat)))
    if not files:
        return {}

    remaining = set(ids)
    out: dict[str, list[dict]] = {}
    for path in files:
        if not remaining:
            break
        try:
            df = pl.read_ndjson(path)
        except Exception:
            continue
        if id_key not in df.columns:
            continue
        matched = df.filter(pl.col(id_key).is_in(list(remaining)))
        for row in matched.iter_rows(named=True):
            out.setdefault(row[id_key], []).append(row)
            remaining.discard(row[id_key])
    return out


def _count_enrich_stats(run_dir: str) -> dict[str, int] | None:
    """Count enrichment outcomes from kept + failed trace files."""
    if not run_dir or not os.path.isdir(run_dir):
        return None
    import glob

    kept_files: list[str] = []
    failed_files: list[str] = []
    for pat in ["trace.kept.jsonl", "trace.kept.shard*.jsonl"]:
        kept_files.extend(glob.glob(os.path.join(run_dir, pat)))
    for pat in ["trace.failed.jsonl", "trace.failed.shard*.jsonl"]:
        failed_files.extend(glob.glob(os.path.join(run_dir, pat)))
    if not kept_files and not failed_files:
        return None

    enriched = 0
    all_filtered = 0
    for path in kept_files:
        try:
            df = pl.read_ndjson(path)
        except Exception:
            continue
        kept_col = next((c for c in ["kept", "kept_phrases"] if c in df.columns), None)
        if kept_col:
            enriched += df.filter(pl.col(kept_col).list.len() > 0).height
            if "proposed" in df.columns:
                has_proposed = df.filter(pl.col("proposed").list.len() > 0)
                all_filtered += has_proposed.filter(
                    pl.col(kept_col).list.len() == 0
                ).height
        else:
            enriched += len(df)

    failed = 0
    for path in failed_files:
        try:
            df = pl.read_ndjson(path)
        except Exception:
            continue
        failed += len(df)

    total = enriched + all_filtered + failed
    return {
        "total": total,
        "enriched": enriched,
        "all_filtered": all_filtered,
        "failed": failed,
    }


def _load_corpus_subset(ds_dir: str, doc_ids: set[str]) -> dict[str, str]:
    if not doc_ids:
        return {}
    path = os.path.join(ds_dir, "raw", "corpus.jsonl")
    if not os.path.exists(path):
        return {}
    remaining = set(doc_ids)
    out = {}
    with open(path) as f:
        for line in f:
            if not remaining:
                break
            row = json.loads(line)
            if row["_id"] in remaining:
                remaining.discard(row["_id"])
                title = row.get("title", "")
                text = row.get("text", "")
                out[row["_id"]] = f"{title}: {text}" if title else text
    return out


def _trace_pipeline_sources(ds_dir: str) -> dict[str, str]:
    """Trace the actual retrieval files each stage consumed.

    Works backward from rerank: if rerank meta has ``retrieval_source``,
    use that for query-enrich.  Then check QE meta for ``doc_enrich_run``
    to resolve doc-enrich.  Falls back to best.jsonl when provenance is
    unavailable.
    """
    sources: dict[str, str] = {}

    # baseline
    p = os.path.join(ds_dir, "retrieval", "baseline.jsonl")
    if os.path.exists(p):
        sources["baseline"] = p

    # rerank
    rerank_meta = _stage_meta(ds_dir, "rerank")
    if rerank_meta:
        run_dir = rerank_meta.get("run_dir", "")
        p = os.path.join(run_dir, "reranked.jsonl")
        if os.path.exists(p):
            sources["rerank"] = p

        # query-enrich: use the retrieval file the reranker actually loaded
        ret_src = rerank_meta.get("retrieval_source", "")
        if ret_src and os.path.exists(ret_src):
            sources["query-enrich"] = ret_src

    # query-enrich fallback
    if "query-enrich" not in sources:
        qe_meta = _stage_meta(ds_dir, "query-enrich")
        if qe_meta:
            stage_dir = os.path.join(ds_dir, "retrieval", "query-enrich")
            run_name = qe_meta.get("run_name", "")
            if run_name:
                p = os.path.join(stage_dir, f"{run_name}.jsonl")
                if os.path.exists(p):
                    sources["query-enrich"] = p
            if "query-enrich" not in sources:
                p = os.path.join(stage_dir, "best.jsonl")
                if os.path.exists(p):
                    sources["query-enrich"] = p

    # doc-enrich: use QE best meta's doc_enrich_run if available, else DE best.
    qe_best_meta = _stage_meta(ds_dir, "query-enrich")
    de_run = (qe_best_meta or {}).get("doc_enrich_run", "")
    de_stage_dir = os.path.join(ds_dir, "retrieval", "doc-enrich")
    if de_run:
        p = os.path.join(de_stage_dir, f"{de_run}.jsonl")
        if os.path.exists(p):
            sources["doc-enrich"] = p
    if "doc-enrich" not in sources:
        de_meta = _stage_meta(ds_dir, "doc-enrich")
        if de_meta:
            run_name = de_meta.get("run_name", "")
            if run_name:
                p = os.path.join(de_stage_dir, f"{run_name}.jsonl")
                if os.path.exists(p):
                    sources["doc-enrich"] = p
            if "doc-enrich" not in sources:
                p = os.path.join(de_stage_dir, "best.jsonl")
                if os.path.exists(p):
                    sources["doc-enrich"] = p

    return sources


def _load_retrieval_flat(path: str) -> pl.DataFrame:
    """Load retrieval JSONL into a flat (query_id, rank, doc_id) DataFrame."""
    qids, ranks, dids = [], [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["query_id"]
            cands = row.get("candidates", [])
            if not cands:
                continue
            if "rank" in cands[0]:
                cands.sort(key=lambda c: c["rank"])
            else:
                cands.sort(key=lambda c: -c.get("score", 0))
            for i, c in enumerate(cands):
                qids.append(qid)
                ranks.append(i + 1)
                dids.append(c["doc_id"])
    return pl.DataFrame({"query_id": qids, "rank": ranks, "doc_id": dids})


# ---------------------------------------------------------------------------
# Analysis (vectorized with polars)
# ---------------------------------------------------------------------------


def analyze_dataset(ds_name: str, slug: str, k: int, db_root: str) -> dict | None:
    ds_dir = os.path.join(db_root, slug)
    if not os.path.isdir(ds_dir):
        return None

    queries_path = os.path.join(ds_dir, "raw", "queries-test.jsonl")
    qrels_path = os.path.join(ds_dir, "raw", "qrels-test.jsonl")
    if not os.path.exists(queries_path) or not os.path.exists(qrels_path):
        return None

    queries_df_raw = pl.read_ndjson(
        queries_path, schema={"_id": pl.Utf8, "text": pl.Utf8}
    )
    queries = dict(
        zip(queries_df_raw["_id"].to_list(), queries_df_raw["text"].to_list())
    )

    qrels_raw = pl.read_ndjson(
        qrels_path,
        schema={"query-id": pl.Utf8, "corpus-id": pl.Utf8, "score": pl.Int32},
    )
    qrels_df = qrels_raw.select(
        pl.col("query-id").alias("query_id"),
        pl.col("corpus-id").alias("doc_id"),
    )
    n_gt = qrels_df.group_by("query_id").len().rename({"len": "n_gt"})

    all_metas: dict[str, dict] = {}
    all_metrics: dict[str, dict] = {}
    for stage in STAGE_ORDER:
        m = _stage_meta(ds_dir, stage)
        if m:
            all_metas[stage] = m
        b = _stage_metrics(ds_dir, stage)
        if b:
            all_metrics[stage] = b

    stage_sources = _trace_pipeline_sources(ds_dir)
    available_stages: list[str] = []
    stage_dfs: dict[str, pl.DataFrame] = {}
    for stage in STAGE_ORDER:
        path = stage_sources.get(stage)
        if path is not None:
            stage_dfs[stage] = _load_retrieval_flat(path)
            available_stages.append(stage)

    if not available_stages:
        return None

    # -- vectorized recall & rank computation per stage --
    stage_recall: dict[str, pl.DataFrame] = {}
    stage_first_rank: dict[str, pl.DataFrame] = {}
    stage_gt_ranks: dict[str, pl.DataFrame] = {}
    for stage in available_stages:
        df = stage_dfs[stage]
        topk = df.filter(pl.col("rank") <= k)
        hits = topk.join(qrels_df, on=["query_id", "doc_id"], how="inner")
        hit_counts = hits.group_by("query_id").len().rename({"len": "hits"})
        recall = hit_counts.join(n_gt, on="query_id", how="right").with_columns(
            (pl.col("hits").fill_null(0) / pl.col("n_gt")).alias("recall")
        )
        stage_recall[stage] = recall.select("query_id", "recall")

        gt_hits = df.join(qrels_df, on=["query_id", "doc_id"], how="inner")
        first = gt_hits.group_by("query_id").agg(
            pl.col("rank").min().alias("first_rank")
        )
        stage_first_rank[stage] = first
        stage_gt_ranks[stage] = gt_hits.select("query_id", "doc_id", "rank")

    # -- assemble wide recall/rank tables --
    final_stage = available_stages[-1]
    final_df = stage_dfs[final_stage]

    all_qids_df = qrels_df.select("query_id").unique()

    def rcol(s: str) -> str:
        return f"r_{s}"

    def kcol(s: str) -> str:
        return f"k_{s}"

    wide = all_qids_df
    for stage in available_stages:
        sr = stage_recall[stage].rename({"recall": rcol(stage)})
        fr = stage_first_rank[stage].rename({"first_rank": kcol(stage)})
        wide = wide.join(sr, on="query_id", how="left").join(
            fr, on="query_id", how="left"
        )
    wide = wide.with_columns(pl.col(rcol(s)).fill_null(0.0) for s in available_stages)

    # is_failure: final stage recall < 1.0
    # is_hard: all stage recalls == 0
    # lost_at: first stage where recall < cummax of previous stages
    wide = wide.with_columns(
        (pl.col(rcol(final_stage)) < 1.0).alias("is_failure"),
        pl.all_horizontal(pl.col(rcol(s)) == 0.0 for s in available_stages).alias(
            "is_hard"
        ),
    )

    lost_at_expr = pl.lit(None, dtype=pl.Utf8)
    for i in range(len(available_stages) - 1, 0, -1):
        stage = available_stages[i]
        prev_max = pl.max_horizontal(pl.col(rcol(s)) for s in available_stages[:i])
        lost_at_expr = (
            pl.when(pl.col(rcol(stage)) < prev_max)
            .then(pl.lit(stage))
            .otherwise(lost_at_expr)
        )
    wide = wide.with_columns(lost_at_expr.alias("lost_at"))

    # top1 doc from final stage
    top1_df = final_df.filter(pl.col("rank") == 1).select("query_id", "doc_id")
    wide = wide.join(top1_df, on="query_id", how="left").rename({"doc_id": "top1_doc"})

    # gt docs per query
    qrels_grouped = qrels_df.group_by("query_id").agg(
        pl.col("doc_id").sort().alias("gt_docs")
    )
    wide = wide.join(qrels_grouped, on="query_id", how="left")

    # query text
    queries_df = pl.DataFrame(
        {"query_id": list(queries.keys()), "query": list(queries.values())}
    )
    wide = wide.join(queries_df, on="query_id", how="left")

    n_total = len(wide)
    n_hard = wide.filter(pl.col("is_hard")).height

    # Per-GT-doc ranks: {(query_id, doc_id) -> {stage: rank}}
    doc_rank_lookup: dict[tuple[str, str], dict[str, int]] = {}
    for stage in available_stages:
        for qid, did, rank in stage_gt_ranks[stage].iter_rows():
            doc_rank_lookup.setdefault((qid, did), {})[stage] = rank

    # Top-10 docs from final stage with per-stage ranks (vectorized)
    top10_final = final_df.filter(pl.col("rank") <= 10)
    top10_wide = top10_final.rename({"rank": f"rank_{final_stage}"})
    for stage in available_stages:
        if stage == final_stage:
            continue
        sdf = stage_dfs[stage].rename({"rank": f"rank_{stage}"})
        top10_wide = top10_wide.join(sdf, on=["query_id", "doc_id"], how="left")
    top10_rank_lookup: dict[str, list[dict]] = {}
    for row in top10_wide.sort("query_id", f"rank_{final_stage}").iter_rows(named=True):
        entry = {
            "doc_id": row["doc_id"],
            "rank": row[f"rank_{final_stage}"],
            "stage_ranks": {
                s: row[f"rank_{s}"]
                for s in available_stages
                if row.get(f"rank_{s}") is not None
            },
        }
        top10_rank_lookup.setdefault(row["query_id"], []).append(entry)

    # Build failure/regression lists from filtered DataFrames
    fail_df = wide.filter(pl.col("is_failure")).sort("query_id")
    regress_df = wide.filter(pl.col("lost_at").is_not_null()).sort("query_id")

    def _row_to_failure(row: dict) -> dict:
        gt_docs = row.get("gt_docs", [])
        qid = row["query_id"]
        return {
            "query_id": qid,
            "query": row.get("query", ""),
            "gt_docs": gt_docs,
            "gt_doc_ranks": {
                did: doc_rank_lookup.get((qid, did), {}) for did in gt_docs
            },
            "top1_doc": row.get("top1_doc"),
            "gt_rank": row.get(kcol(final_stage)),
            "stage_ranks": {s: row.get(kcol(s)) for s in available_stages},
            "lost_at": row.get("lost_at"),
            "stages": {s: row[rcol(s)] for s in available_stages},
            "top10": top10_rank_lookup.get(qid, []),
        }

    def _row_to_regression(row: dict) -> dict:
        return {
            "query_id": row["query_id"],
            "query": row.get("query", ""),
            "regressed_at": row["lost_at"],
            "stages": {s: row[rcol(s)] for s in available_stages},
        }

    failures = [_row_to_failure(r) for r in fail_df.iter_rows(named=True)]
    regressions = [_row_to_regression(r) for r in regress_df.iter_rows(named=True)]

    # Load corpus text only for reported failures
    needed_docs: set[str] = set()
    needed_qids: set[str] = set()
    for f in failures[:100]:
        needed_docs.update(f["gt_docs"])
        needed_qids.add(f["query_id"])
        if f["top1_doc"]:
            needed_docs.add(f["top1_doc"])
        for t10 in f.get("top10", []):
            needed_docs.add(t10["doc_id"])
    corpus_texts = _load_corpus_subset(ds_dir, needed_docs)

    # Load traces from run dirs
    de_meta = all_metas.get("doc-enrich")
    de_run_dir = de_meta.get("run_dir", "") if de_meta else ""
    doc_traces = _scan_traces(de_run_dir, "doc_id", needed_docs)
    if not doc_traces and de_meta:
        note = de_meta.get("note", "")
        if "refiltered from " in note:
            source_dir = os.path.join(
                ds_dir, "runs", "doc-enrich", note.split("refiltered from ")[-1]
            )
            doc_traces = _scan_traces(source_dir, "doc_id", needed_docs)

    qe_meta = all_metas.get("query-enrich")
    qe_run_dir = qe_meta.get("run_dir", "") if qe_meta else ""
    query_traces = _scan_traces(qe_run_dir, "query_id", needed_qids)

    # Dataset-level enrichment stats (with same refilter fallback)
    doc_enrich_stats = _count_enrich_stats(de_run_dir)
    if doc_enrich_stats is None and de_meta:
        note = de_meta.get("note", "")
        if "refiltered from " in note:
            source_dir = os.path.join(
                ds_dir, "runs", "doc-enrich", note.split("refiltered from ")[-1]
            )
            doc_enrich_stats = _count_enrich_stats(source_dir)
    query_enrich_stats = _count_enrich_stats(qe_run_dir)

    rr_meta = all_metas.get("rerank")
    rerank_run_dir = rr_meta.get("run_dir", "") if rr_meta else ""
    rerank_traces_raw: dict[str, dict] = {}
    rerank_scores: dict[str, dict[str, dict]] = {}
    if rerank_run_dir:
        rr_traces = _scan_traces(rerank_run_dir, "query_id", needed_qids)
        for rows in rr_traces.values():
            for row in rows:
                qid = row.get("query_id", "")
                did = row.get("doc_id", "")
                rerank_traces_raw[f"{qid}:{did}"] = row

        reranked_path = os.path.join(rerank_run_dir, "reranked.jsonl")
        if os.path.exists(reranked_path):
            with open(reranked_path) as ef:
                for line in ef:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["query_id"] in needed_qids:
                        qscores: dict[str, dict] = {}
                        for c in row.get("candidates", []):
                            qscores[c["doc_id"]] = {
                                "llm_score": c.get("llm_score"),
                                "bm25_score": c.get("bm25_score"),
                                "rank": c.get("rank"),
                            }
                        rerank_scores[row["query_id"]] = qscores

    for f in failures[:100]:
        f["gt_text"] = [corpus_texts.get(d, "") for d in f["gt_docs"]]
        f["top1_text"] = corpus_texts.get(f["top1_doc"], "") if f["top1_doc"] else ""

        # Doc-enrich: per-doc trace with proposed → kept → filter_stats
        f["doc_traces"] = {}
        for d in f["gt_docs"]:
            traces = doc_traces.get(d, [])
            t = next((x for x in traces if x.get("status") == "ok"), None)
            t = t or (traces[0] if traces else None)
            if t:
                f["doc_traces"][d] = {
                    "proposed": t.get("proposed") or [],
                    "kept": t.get("kept") or t.get("kept_phrases") or [],
                    "filter_stats": t.get("filter_stats") or {},
                    "status": t.get("status"),
                }

        # Query-enrich: trace with proposed → kept
        qt_list = query_traces.get(f["query_id"], [])
        qt = next((x for x in qt_list if x.get("status") == "ok"), None)
        qt = qt or (qt_list[0] if qt_list else None)
        if qt:
            rejected = qt.get("rejected") or []
            f["query_trace"] = {
                "proposed": qt.get("proposed") or [],
                "kept": qt.get("kept_phrases") or qt.get("kept") or [],
                "rejected": rejected,
                "status": qt.get("status"),
            }
        else:
            f["query_trace"] = None

        # Rerank scores + raw LLM response per doc
        qr = rerank_scores.get(f["query_id"], {})
        all_docs = set(f["gt_docs"])
        if f["top1_doc"]:
            all_docs.add(f["top1_doc"])
        for t10 in f.get("top10", []):
            all_docs.add(t10["doc_id"])
        f["rerank_scores"] = {}
        for d in all_docs:
            score = qr.get(d)
            if score:
                entry = dict(score)
                trace = rerank_traces_raw.get(f"{f['query_id']}:{d}")
                if trace:
                    entry["raw_response"] = trace.get("raw_response")
                f["rerank_scores"][d] = entry

    # Load prompts from run dirs
    prompt_files = {
        "doc-enrich": "prompt.txt",
        "query-enrich": "query_prompt.txt",
        "rerank": "prompt.txt",
    }
    prompts: dict[str, str] = {}
    for stage, fname in prompt_files.items():
        meta = all_metas.get(stage)
        if not meta:
            continue
        run_dir = meta.get("run_dir", "")
        p = os.path.join(run_dir, fname)
        if not os.path.exists(p):
            note = meta.get("note", "")
            if "refiltered from " in note:
                source_run = note.split("refiltered from ")[-1]
                p = os.path.join(
                    os.path.dirname(run_dir), source_run, fname
                )
        if os.path.exists(p):
            with open(p) as pf:
                prompts[stage] = pf.read().strip()

    # Build stage summary from pre-loaded metas/metrics
    official_recall: dict[str, float] = {}
    stage_recall_all: dict[str, dict[str, float]] = {}
    stage_meta: dict[str, dict] = {}
    for stage in STAGE_ORDER:
        best = all_metrics.get(stage)
        if best:
            if f"Recall@{k}" in best:
                official_recall[stage] = best[f"Recall@{k}"]
            stage_recall_all[stage] = {
                key: val for key, val in best.items() if key.startswith("Recall@")
            }
        meta = all_metas.get(stage)
        if meta:
            sm: dict = {
                "run_name": meta.get("run_name", meta.get("best_config", "")),
                "model": meta.get("model", ""),
                "score": meta.get("score"),
                "selection_metric": meta.get("selection_metric", f"Recall@{k}"),
                "timestamp": meta.get("timestamp"),
            }
            if stage == "baseline" and "bm25_params" in meta:
                sm["bm25_params"] = meta["bm25_params"]
            if stage == "rerank" and "rerank_params" in meta:
                sm["rerank_params"] = meta["rerank_params"]
            stage_meta[stage] = sm

    baseline_recall_keys = stage_recall_all.get("baseline", {}).keys()
    retrieval_k = max(
        (int(k_str.split("@")[1]) for k_str in baseline_recall_keys if "@" in k_str),
        default=200,
    )
    rerank_top_n = (
        stage_meta.get("rerank", {}).get("rerank_params", {}).get("top_n")
        or retrieval_k
    )
    stage_top_k: dict[str, int] = {}
    for stage in STAGE_ORDER:
        stage_top_k[stage] = rerank_top_n if stage == "rerank" else retrieval_k

    return {
        "dataset": ds_name,
        "slug": slug,
        "k": k,
        "n_queries": n_total,
        "available_stages": available_stages,
        "final_stage": final_stage,
        "n_failures": len(failures),
        "n_regressions": len(regressions),
        "n_hard": n_hard,
        "stage_top_k": stage_top_k,
        "recall_at_k": official_recall,
        "stage_recall_all": stage_recall_all,
        "stage_meta": stage_meta,
        "doc_enrich_stats": doc_enrich_stats,
        "query_enrich_stats": query_enrich_stats,
        "prompts": prompts,
        "failures": failures[:100],
        "regressions": regressions[:50],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _report_is_fresh(json_path: Path, db_root: str) -> bool:
    from concurrent.futures import ThreadPoolExecutor

    try:
        report_mtime = json_path.stat().st_mtime
    except OSError:
        return False

    paths = [
        os.path.join(db_root, slug, "eval", stage, "best.json")
        for slug in SLUG_MAP.values()
        for stage in STAGE_ORDER
    ]

    def _newer(p: str) -> bool:
        try:
            return os.stat(p).st_mtime > report_mtime
        except OSError:
            return False

    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        if any(pool.map(_newer, paths)):
            return False
    return True


def _analyze_one(args: tuple[str, str, int, str]) -> tuple[str, dict | None]:
    ds_name, slug, k, db_root = args
    return ds_name, analyze_dataset(ds_name, slug, k=k, db_root=db_root)


@hydra.main(version_base=None, config_path="configs", config_name="failure_report")
def main(cfg: DictConfig) -> None:
    import time
    from concurrent.futures import ProcessPoolExecutor

    k = cfg.k
    db_root = cfg.db_root
    force = cfg.get("force", False)

    reports_dir = Path.home() / ".cache" / "sira"
    json_path = reports_dir / f"failures-k{k}.json"

    if not force and _report_is_fresh(json_path, db_root):
        print(f"  Report up-to-date: {json_path}")
        return

    t0 = time.perf_counter()
    tasks = [(name, slug, k, db_root) for name, slug in SLUG_MAP.items()]

    all_reports = {}
    from concurrent.futures import as_completed

    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(_analyze_one, t): t[0] for t in tasks}
        done_count = 0
        for future in as_completed(futures):
            ds_name, report = future.result()
            done_count += 1
            print(f"  [{done_count}/{len(tasks)}] {ds_name:<16} {'✓' if report else '—'}")
            if report is not None:
                all_reports[ds_name] = report

    elapsed = time.perf_counter() - t0

    if not all_reports:
        logger.warning("No datasets with per-query data found.")
        return

    # Sort datasets by gap to best baseline (SIRA - best baseline), ascending
    def _gap(ds_name: str) -> float:
        r = all_reports[ds_name]
        sira_recall = r["recall_at_k"].get(r["final_stage"], 0.0)
        di = DATASETS.index(ds_name) if ds_name in DATASETS else -1
        if di < 0:
            return 0.0
        best_baseline = max(RECALL_10[m][di] for m in METHODS)
        return sira_recall - best_baseline

    sorted_names = sorted(all_reports.keys(), key=_gap)

    # Summary
    print(f"\n  Failure Analysis @ k={k}  ({elapsed:.1f}s)\n")
    print(
        f"  {'Dataset':<16} {'Stage':<14} {'Queries':>7} "
        f"{'Failures':>8} {'Regress':>8} {'Hard':>6}  {'Recall':>7}  {'Gap':>7}"
    )
    print("  " + "─" * 90)
    for ds_name in sorted_names:
        r = all_reports[ds_name]
        gap = _gap(ds_name)
        sign = "+" if gap >= 0 else ""
        print(
            f"  {ds_name:<16} {r['final_stage']:<14} "
            f"{r['n_queries']:>7} {r['n_failures']:>8} "
            f"{r['n_regressions']:>8} {r['n_hard']:>6}  "
            f"{r['recall_at_k'][r['final_stage']]:>7.4f}  {sign}{gap:>.4f}"
        )

    # Save (ordered by gap)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ordered_reports = {name: all_reports[name] for name in sorted_names}

    with open(json_path, "w") as f:
        json.dump(ordered_reports, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")


# ---------------------------------------------------------------------------
# TUI browser
# ---------------------------------------------------------------------------


def _browse(k: int) -> None:
    report_path = Path.home() / ".cache" / "sira" / f"failures-k{k}.json"
    needs_generate = not report_path.exists()
    if not needs_generate:
        print("  Checking report freshness … ", end="", flush=True)
        if not _report_is_fresh(report_path, DB_ROOT):
            print("stale, regenerating …")
            needs_generate = True
        else:
            print("ok.", flush=True)

    if needs_generate:
        from concurrent.futures import ProcessPoolExecutor

        print("  Generating report … ", end="", flush=True)
        tasks = [(name, slug, k, DB_ROOT) for name, slug in SLUG_MAP.items()]
        all_reports = {}
        with ProcessPoolExecutor() as pool:
            for ds_name, report in pool.map(_analyze_one, tasks):
                if report is not None:
                    all_reports[ds_name] = report
        if not all_reports:
            print("no data.")
            return
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False)
        print(f"done ({len(all_reports)} datasets).", flush=True)

    print(f"  Loading {report_path} … ", end="", flush=True)
    with open(report_path) as f:
        data_raw = json.load(f)
    n_failures = sum(r.get("n_failures", 0) for r in data_raw.values())
    print(f"{len(data_raw)} datasets, {n_failures} failures.", flush=True)

    def _ds_gap(ds_name: str) -> float:
        r = data_raw[ds_name]
        sira_recall = r.get("recall_at_k", {}).get(r.get("final_stage", ""), 0.0)
        di = DATASETS.index(ds_name) if ds_name in DATASETS else -1
        if di < 0:
            return 0.0
        return sira_recall - max(RECALL_10[m][di] for m in METHODS)

    print("  Sorting datasets by gap … ", end="", flush=True)
    sorted_keys = sorted(data_raw.keys(), key=_ds_gap)
    data = {k: data_raw[k] for k in sorted_keys}
    print("done.", flush=True)

    print("  Importing Textual … ", end="", flush=True)
    from textual import on
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, VerticalScroll
    from textual.widgets import (
        Button,
        Collapsible,
        DataTable,
        Footer,
        Input,
        Label,
        Select,
        Static,
    )

    print("done.", flush=True)

    print("  Launching app …", flush=True)

    def _wrap(text: str, width: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        line: list[str] = []
        length = 0
        for w in words:
            if length + len(w) + 1 > width and line:
                lines.append(" ".join(line))
                line = [w]
                length = len(w)
            else:
                line.append(w)
                length += len(w) + 1
        if line:
            lines.append(" ".join(line))
        return lines or [""]

    class FailureApp(App):
        THEME = "textual-light"

        CSS = """
        Screen { layout: vertical; background: #f0f4f8; }
        #controls { height: 3; padding: 0 1; background: #e1e8f0; }
        #info-bar { height: 3; padding: 0 2; background: #1565c0;
                     color: #ffffff; }
        #controls Select { width: 24; }
        #split-select { width: 14; }
        #controls Input { width: 34; }
        #goto-input { width: 8; }
        #controls Label { width: auto; padding: 0 1; content-align: center middle;
                           color: #37474f; }
        SelectCurrent { background: #e3eaf2; color: #263238; }
        Input { background: #e3eaf2; color: #263238; }
        #main-area { height: 1fr; }
        #table-container { width: 50%; min-width: 20; background: #ffffff;
                            border: round #cfd8dc; scrollbar-size: 0 0; }
        #detail-panel { width: 50%; padding: 0 1; background: #fafcff;
                         border: round #bbdefb; }
        DataTable { background: #ffffff; }
        DataTable > .datatable--header { background: #e3eaf2; color: #263238; }
        DataTable > .datatable--cursor { background: #c8e6c9; color: #1b5e20; }
        DataTable > .datatable--hover { background: #e8f5e9; }
        Footer { background: #37474f; color: #eceff1; }
        Collapsible { background: #fafcff; margin: 0 0; padding: 0;
                       border: none; }
        #detail-content { margin-top: 1; }
        CollapsibleTitle { color: #5a6a7a; padding: 0 1; background: #fafcff; }
        CollapsibleTitle:hover { color: #1565c0; }
        #copy-bar { height: 3; }
        #batch-bar { height: 3; }
        #batch-bar Label { width: auto; padding: 0 1; content-align: center middle;
                           color: #37474f; }
        #batch-from { width: 10; }
        #batch-to { width: 10; }
        .copy-btn { width: 16; min-width: 16; height: 3; margin: 0 1;
                    background: #e3eaf2; color: #37474f; border: round #bbdefb; }
        .copy-btn:hover { background: #bbdefb; }
        .copy-btn.-copied { background: #c8e6c9; color: #1b5e20; }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
        ]

        SPLIT_RATIOS = [30, 40, 50, 60, 70]

        def __init__(self) -> None:
            super().__init__()
            self.current_ds = list(data.keys())[0]
            self.search_text = ""
            self.sort_by_rank = False
            self.split_idx = 2  # start at 50/50
            self._last_row_key: str | None = None
            self._detail_plain: str = ""
            self._stk_cache: dict[str, dict[str, int]] = {}

        def compose(self) -> ComposeResult:
            with Horizontal(id="controls"):
                yield Label("Dataset:")
                yield Select(
                    [(ds, ds) for ds in data],
                    value=self.current_ds,
                    allow_blank=False,
                    id="ds-select",
                )
                yield Label("Search:")
                yield Input(placeholder="query or doc text...", id="search-input")
                yield Label("Go to:")
                yield Input(placeholder="#", id="goto-input")
                yield Label("Split:")
                yield Select(
                    [
                        ("30 / 70", "30"),
                        ("40 / 60", "40"),
                        ("50 / 50", "50"),
                        ("60 / 40", "60"),
                        ("70 / 30", "70"),
                    ],
                    value="50",
                    allow_blank=False,
                    id="split-select",
                )
            yield Static("", id="info-bar")
            with Horizontal(id="main-area"):
                with VerticalScroll(id="table-container"):
                    yield DataTable(id="ftable", cursor_type="row")
                with VerticalScroll(id="detail-panel"):
                    with Horizontal(id="copy-bar"):
                        yield Button("📋 Current", id="copy-result-btn", classes="copy-btn")
                        yield Button("📋 Prompts", id="copy-prompts-btn", classes="copy-btn")
                        yield Button("💾 Report", id="save-md-btn", classes="copy-btn")
                    with Horizontal(id="batch-bar"):
                        yield Label("#")
                        yield Select(
                            [("1", 1)], value=1, allow_blank=False,
                            id="batch-from",
                        )
                        yield Label("→")
                        yield Select(
                            [("1", 1)], value=1, allow_blank=False,
                            id="batch-to",
                        )
                        yield Button("📋 Batch", id="batch-copy-btn", classes="copy-btn")
                    yield Static(
                        "[bold #455a64]━━━ PROMPTS ━━━[/]",
                        id="prompt-header",
                    )
                    yield Collapsible(
                        Static("", id="prompt-doc-enrich"),
                        title="◆ Doc Enrich",
                        id="coll-doc-enrich",
                        collapsed=True,
                    )
                    yield Collapsible(
                        Static("", id="prompt-query-enrich"),
                        title="◈ Query Enrich",
                        id="coll-query-enrich",
                        collapsed=True,
                    )
                    yield Collapsible(
                        Static("", id="prompt-rerank"),
                        title="★ Rerank",
                        id="coll-rerank",
                        collapsed=True,
                    )
                    yield Static(
                        "[bold #1565c0]SIRA Failure Analysis[/]\n\n"
                        "[dim]Navigate rows with ↑↓ to preview.\n"
                        "Use controls above to filter and search.\n"
                        "Press q to quit.[/]",
                        id="detail-content",
                    )
            yield Footer()

        def _rebuild_prompts(self) -> None:
            prompts = data[self.current_ds].get("prompts", {})
            mapping = {
                "doc-enrich": "prompt-doc-enrich",
                "query-enrich": "prompt-query-enrich",
                "rerank": "prompt-rerank",
            }
            for stage, widget_id in mapping.items():
                text = prompts.get(stage, "[dim]No prompt available[/]")
                self.query_one(f"#{widget_id}").update(text)

        def on_mount(self) -> None:
            self._setup_table()
            self._rebuild_prompts()
            self._refresh()

        def on_resize(self) -> None:
            table = self.query_one("#ftable", DataTable)
            table.clear(columns=True)
            self._setup_table()
            self._refresh()

        def _setup_table(self) -> None:
            table = self.query_one("#ftable", DataTable)
            left_pct = self.SPLIT_RATIOS[self.split_idx] / 100
            tw = int(self.size.width * left_pct) - 8
            fixed = 5 * 4 + 4 + 6  # 4 rank cols (5ea) + #GT (4) + column padding (6)
            query_w = max(15, tw - fixed)
            table.add_column("Query", width=query_w)
            table.add_column("Raw", width=5)
            table.add_column("Doc", width=5)
            table.add_column("Qry", width=5)
            table.add_column("Rrk", width=5)
            table.add_column("#GT", width=4)

        def _build_info_bar(self) -> str:
            r = data[self.current_ds]
            sra = r.get("stage_recall_all", {})
            ds_name = r["dataset"]
            ds_idx = DATASETS.index(ds_name) if ds_name in DATASETS else -1

            base_r10 = sra.get("baseline", {}).get("Recall@10")
            doc_r10 = sra.get("doc-enrich", {}).get("Recall@10")
            qe_r10 = sra.get("query-enrich", {}).get("Recall@10")
            rerank_r10 = sra.get("rerank", {}).get("Recall@10")
            qe_r50 = sra.get("query-enrich", {}).get("Recall@50")
            qe_r200 = sra.get("query-enrich", {}).get("Recall@200")

            best_method, best_val = "N/A", 0.0
            if ds_idx >= 0:
                for method, vals in RECALL_10.items():
                    if vals[ds_idx] > best_val:
                        best_val = vals[ds_idx]
                        best_method = method

            def _fmt(v):
                return f"{v:.4f}" if v is not None else "N/A"

            best_str = f"{best_method} {best_val:.4f}" if ds_idx >= 0 else "N/A"

            smeta = r.get("stage_meta", {})
            bm25p = smeta.get("baseline", {}).get("bm25_params", {})
            max_n = bm25p.get("max_n")
            tokenizer = bm25p.get("tokenizer", "")
            bm25_tag = f"{max_n}-gram/{tokenizer}" if max_n else "?"

            line1 = (
                f"BM25 [bold]{bm25_tag}[/]  ·  "
                f"R@10  BM25: [bold]{_fmt(base_r10)}[/]  →  "
                f"DocE: [bold]{_fmt(doc_r10)}[/]  →  "
                f"QE: [bold]{_fmt(qe_r10)}[/]  →  "
                f"Rerank: [bold]{_fmt(rerank_r10)}[/]"
            )
            de_stats = r.get("doc_enrich_stats")
            qe_stats = r.get("query_enrich_stats")

            def _enrich_tag(stats: dict | None, label: str) -> str:
                if not stats:
                    return f"{label}: N/A"
                t, e = stats["total"], stats["enriched"]
                pct = e / t * 100 if t else 0
                return f"{label}: [bold]{e}/{t}[/] ({pct:.0f}%)"

            line2 = (
                f"QE R@50: [bold]{_fmt(qe_r50)}[/]  ·  "
                f"QE R@200: [bold]{_fmt(qe_r200)}[/]  ·  "
                f"Best Baseline R@10: [bold]{best_str}[/]  ·  "
                f"{_enrich_tag(de_stats, 'DocE')}  ·  "
                f"{_enrich_tag(qe_stats, 'QE')}"
            )
            return f"{line1}\n{line2}"

        def _stage_top_k(self) -> dict[str, int]:
            cached = self._stk_cache.get(self.current_ds)
            if cached:
                return cached
            r = data[self.current_ds]
            stk = r.get("stage_top_k")
            if stk:
                self._stk_cache[self.current_ds] = stk
                return stk
            sra = r.get("stage_recall_all", {})
            retrieval_k = max(
                (int(k.split("@")[1]) for k in sra.get("baseline", {}) if "@" in k),
                default=200,
            )
            rerank_top_n = (
                r.get("stage_meta", {})
                .get("rerank", {})
                .get("rerank_params", {})
                .get("top_n")
            )
            if not rerank_top_n:
                slug = r.get("slug", "")
                meta = _load_json(
                    os.path.join(DB_ROOT, slug, "eval", "rerank", "best.meta.json")
                )
                rerank_top_n = (
                    meta.get("rerank_params", {}).get("top_n") if meta else None
                ) or retrieval_k
            result = {
                s: (rerank_top_n if s == "rerank" else retrieval_k) for s in STAGE_ORDER
            }
            self._stk_cache[self.current_ds] = result
            return result

        def _get_filtered(self) -> list[dict]:
            r = data.get(self.current_ds)
            if not r:
                return []
            failures = list(r["failures"])

            if self.search_text:
                q = self.search_text.lower()
                failures = [
                    f
                    for f in failures
                    if q in f["query"].lower()
                    or q in " ".join(f.get("gt_text", [])).lower()
                    or q in f.get("top1_text", "").lower()
                ]

            if self.sort_by_rank:
                failures.sort(key=lambda f: f["gt_rank"] if f["gt_rank"] else 9999)
            return failures

        def _refresh(self) -> None:
            self.query_one("#info-bar").update(self._build_info_bar())
            table = self.query_one("#ftable", DataTable)
            table.clear()
            failures = self._get_filtered()
            stk = self._stage_top_k()
            for qi, f in enumerate(failures, 1):
                gt_doc_ranks = f.get("gt_doc_ranks", {})
                gt_docs = f.get("gt_docs", [])
                n_gt = len(gt_docs)

                # Query header row
                table.add_row(
                    f"{qi:>3}  {f['query']}",
                    "",
                    "",
                    "",
                    "",
                    str(n_gt),
                    key=f"{f['query_id']}:0",
                )
                # Per-doc rows
                for di, doc_id in enumerate(gt_docs, 1):
                    per_doc = gt_doc_ranks.get(doc_id, {})
                    rank_cells = []
                    for s in STAGE_ORDER:
                        if s not in f.get("stages", {}):
                            rank_cells.append("·")
                        elif s in per_doc:
                            rank_cells.append(str(per_doc[s]))
                        else:
                            rank_cells.append(f">{stk.get(s, '?')}")
                    table.add_row(
                        f"  └ {doc_id[:24]}",
                        *rank_cells,
                        "",
                        key=f"{f['query_id']}:{di}",
                    )

            sort_tag = " ↕rank" if self.sort_by_rank else ""
            self.title = (
                f"SIRA — {self.current_ds} · " f"{len(failures)} failures{sort_tag}"
            )

            n = len(failures)
            opts = [(str(i), i) for i in range(1, n + 1)]
            if opts:
                self.query_one("#batch-from", Select).set_options(opts)
                self.query_one("#batch-to", Select).set_options(opts)
                self.query_one("#batch-from", Select).value = 1
                self.query_one("#batch-to", Select).value = min(n, 10)

        def _rank_style(self, rank: int | None, top_k: int = 200) -> str:
            if rank is None:
                return f"[red]>{top_k}[/]"
            if rank <= 10:
                return f"[bold green]{rank}[/]"
            if rank <= 50:
                return f"[#ff9800]{rank}[/]"
            return f"[dim]{rank}[/]"

        def _fmt_keywords(self, proposed: list, kept: list | set) -> str:
            kept_set = set(kept) if not isinstance(kept, set) else kept
            parts = []
            for p in proposed:
                if p in kept_set:
                    parts.append(f"[bold #00897b]{p}[/]")
                else:
                    parts.append(f"[strike dim]{p}[/]")
            return ", ".join(parts)

        def _fmt_rerank(self, rr: dict | None, top1_rr: dict | None = None) -> str:
            if not rr:
                return ""
            llm = rr.get("llm_score")
            rank = rr.get("rank")
            parts = []
            if llm is not None:
                parts.append(f"LLM score: [bold]{llm:.0f}[/]")
            else:
                parts.append("[red]LLM score: N/A (scoring failed)[/]")
            if rank is not None:
                parts.append(f"rerank pos: {rank}")
            if top1_rr and top1_rr.get("llm_score") is not None:
                top1_llm = top1_rr["llm_score"]
                if llm is not None:
                    gap = llm - top1_llm
                    sign = "+" if gap >= 0 else ""
                    color = "green" if gap >= 0 else "red"
                    parts.append(f"vs top-1: [{color}]{sign}{gap:.0f}[/]")
            return " · ".join(parts)

        def _build_detail_lines(
            self, f: dict, wrap_w: int, focused_di: int = 0,
        ) -> list[str]:
            """Build rich-text lines for one failure entry."""
            stk = self._stage_top_k()
            indent = "      "
            gt_doc_ranks = f.get("gt_doc_ranks", {})
            gt_texts = f.get("gt_text", [])
            gt_docs = f["gt_docs"]
            show_docs = [gt_docs[focused_di - 1]] if focused_di > 0 else gt_docs

            L: list[str] = []

            def _delta_str(prev_rank: int | None, cur_rank: int | None) -> str:
                if prev_rank is None or cur_rank is None:
                    return ""
                d = prev_rank - cur_rank
                if d > 0:
                    return f"  [green]↑{d}[/]"
                if d < 0:
                    return f"  [red]↓{-d}[/]"
                return ""

            # ── 1. QUERY ──
            L.append("[bold #1565c0]━━━ QUERY ━━━[/]")
            for chunk in _wrap(f["query"], wrap_w - 2):
                L.append(f"  {chunk}")

            # Query enrichment (query-level, show once)
            qt = f.get("query_trace")
            if qt:
                proposed = qt.get("proposed") or []
                kept = qt.get("kept") or []
                rejected = qt.get("rejected") or []
                if proposed:
                    L.append(
                        f"  [bold #00897b]expansion ({len(kept)}/{len(proposed)}):[/] "
                        f"{self._fmt_keywords(proposed, kept)}"
                    )
                    if rejected:
                        reasons = {}
                        for r in rejected:
                            reason = r.get("reason", "unknown")
                            reasons.setdefault(reason, []).append(r.get("phrase", ""))
                        for reason, phrases in reasons.items():
                            L.append(
                                f"  [dim]filtered ({reason}): "
                                f"{', '.join(phrases)}[/]"
                            )
                    elif len(proposed) > len(kept):
                        L.append(
                            f"  [dim]filtered: "
                            f"{len(proposed) - len(kept)} removed[/]"
                        )
                elif qt.get("status") == "error":
                    L.append("[dim red]query enrichment error[/]")
                else:
                    L.append("[dim]no query expansions proposed[/]")
            else:
                L.append("[dim]no query enrichment trace[/]")
            L.append("")

            # ── 2. PIPELINE: stage-by-stage for each GT doc ──
            top1 = f.get("top1_doc")
            top1_rr = f.get("rerank_scores", {}).get(top1) if top1 else None

            for doc_id in show_docs:
                di = gt_docs.index(doc_id)
                doc_text = gt_texts[di] if di < len(gt_texts) else ""
                per_stage = gt_doc_ranks.get(doc_id, {})
                doc_trace = f.get("doc_traces", {}).get(doc_id)
                rr = f.get("rerank_scores", {}).get(doc_id)

                if len(show_docs) > 1:
                    L.append(
                        f"[bold #2e7d32]━━━ GT DOC {di+1}/{len(gt_docs)}: "
                        f"{doc_id} ━━━[/]"
                    )
                else:
                    L.append(f"[bold #2e7d32]━━━ GT DOC: {doc_id} ━━━[/]")

                if doc_text:
                    for chunk in _wrap(doc_text, wrap_w - 2):
                        L.append(f"  [dim]{chunk}[/]")
                    L.append("")

                # Stage 1: BM25
                base_rank = per_stage.get("baseline")
                L.append(
                    f"  ◇ [bold]BM25[/]       "
                    f"rank {self._rank_style(base_rank, stk.get('baseline', 200))}"
                )

                # Stage 2: DOC ENRICH
                doc_rank = per_stage.get("doc-enrich")
                L.append(
                    f"  ◆ [bold]DOC ENRICH[/] "
                    f"rank {self._rank_style(doc_rank, stk.get('doc-enrich', 200))}"
                    f"{_delta_str(base_rank, doc_rank)}"
                )
                if doc_trace:
                    proposed = doc_trace.get("proposed") or []
                    kept = doc_trace.get("kept") or []
                    fs = doc_trace.get("filter_stats") or {}
                    status = doc_trace.get("status", "")
                    if proposed:
                        L.append(
                            f"{indent}[bold #00897b]expansion "
                            f"({len(kept)}/{len(proposed)}):[/] "
                            f"{self._fmt_keywords(proposed, kept)}"
                        )
                        if fs and any(k != "kept" for k in fs):
                            reasons = " · ".join(
                                f"{k}: {v}" for k, v in fs.items() if k != "kept"
                            )
                            L.append(f"{indent}[dim]filtered ({reasons})[/]")
                        elif len(proposed) > len(kept):
                            L.append(
                                f"{indent}[dim]filtered: "
                                f"{len(proposed) - len(kept)} removed[/]"
                            )
                    elif status == "error":
                        L.append(f"{indent}[dim red]enrichment error[/]")
                    else:
                        L.append(f"{indent}[dim]no keywords proposed[/]")
                else:
                    L.append(f"{indent}[dim]no enrichment trace[/]")

                # Stage 3: QUERY ENRICH
                qe_rank = per_stage.get("query-enrich")
                L.append(
                    f"  ◈ [bold]QRY ENRICH[/] "
                    f"rank {self._rank_style(qe_rank, stk.get('query-enrich', 200))}"
                    f"{_delta_str(doc_rank or base_rank, qe_rank)}"
                )

                # Stage 4: RERANK
                rr_rank = per_stage.get("rerank")
                L.append(
                    f"  ★ [bold]RERANK[/]     "
                    f"rank {self._rank_style(rr_rank, stk.get('rerank', 50))}"
                    f"{_delta_str(qe_rank or doc_rank or base_rank, rr_rank)}"
                )
                if rr:
                    L.append(f"{indent}[#7b1fa2]" f"{self._fmt_rerank(rr, top1_rr)}[/]")

                L.append("")

            # ── 3. RETRIEVED TOP-10 ──
            top10 = f.get("top10", [])
            gt_set = set(gt_docs)
            all_rr = f.get("rerank_scores", {})
            if top10:
                L.append("[bold #c62828]━━━ RETRIEVED TOP-10 ━━━[/]")
                stage_hdrs = "".join(
                    f"{'BM25' if s == 'baseline' else s[:3].upper():>6}"
                    for s in STAGE_ORDER
                )
                L.append(f"  [dim]{'LLM':>5}  {stage_hdrs}  doc_id[/]")
                L.append(f"  [dim]{'─' * (5 + len(STAGE_ORDER) * 6 + 6 + 30)}[/]")
                for entry in top10:
                    did = entry["doc_id"]
                    sr = entry.get("stage_ranks", {})
                    is_gt = did in gt_set
                    doc_rr = all_rr.get(did, {})
                    llm = doc_rr.get("llm_score")
                    llm_str = f"{llm:5.0f}" if llm is not None else "  N/A"
                    rank_parts = []
                    for s in STAGE_ORDER:
                        v = sr.get(s)
                        rank_parts.append(
                            f"{v:6d}"
                            if v is not None
                            else f"{'>' + str(stk.get(s, '?')):>6}"
                        )
                    ranks_str = "".join(rank_parts)
                    tag = " [bold green]◀ GT[/]" if is_gt else ""
                    color = "#1b5e20" if is_gt else ""
                    prefix = f"[{color}]" if color else ""
                    suffix = "[/]" if color else ""
                    L.append(
                        f"  {prefix}{llm_str}  " f"{ranks_str}  {did[:30]}{suffix}{tag}"
                    )
                L.append("")

            return L

        def _show_detail(self, row_key: str) -> None:
            qid, _, di_str = row_key.rpartition(":")
            if not qid:
                qid, di_str = di_str, "0"
            focused_di = int(di_str)

            r = data[self.current_ds]
            f = next((x for x in r["failures"] if x["query_id"] == qid), None)
            if not f:
                return

            right_pct = (100 - self.SPLIT_RATIOS[self.split_idx]) / 100
            wrap_w = max(30, int(self.size.width * right_pct) - 6)
            L = self._build_detail_lines(f, wrap_w, focused_di)

            rich_text = "\n".join(L)
            self.query_one("#detail-content").update(rich_text)
            import re

            self._detail_plain = re.sub(r"\[/?[^\]]*\]", "", rich_text)
            self._last_row_key = row_key

        @on(Select.Changed, "#ds-select")
        def ds_changed(self, event: Select.Changed) -> None:
            if event.value is None or event.value == Select.BLANK:
                return
            self.current_ds = str(event.value)
            self._rebuild_prompts()
            self._refresh()

        @on(Select.Changed, "#split-select")
        def split_changed(self, event: Select.Changed) -> None:
            if event.value is None or event.value == Select.BLANK:
                return
            pct = int(event.value)
            self.split_idx = self.SPLIT_RATIOS.index(pct)
            self._apply_split()

        @on(Input.Submitted, "#goto-input")
        def goto_submitted(self, event: Input.Submitted) -> None:
            try:
                num = int(event.value.strip())
            except ValueError:
                return
            table = self.query_one("#ftable", DataTable)
            failures = self._get_filtered()
            if 1 <= num <= len(failures):
                row_idx = sum(
                    1 + len(failures[i].get("gt_docs", [])) for i in range(num - 1)
                )
                table.move_cursor(row=row_idx)
                event.input.value = ""

        @on(Input.Changed, "#search-input")
        def search_changed(self, event: Input.Changed) -> None:
            self.search_text = event.value
            self._refresh()

        @on(DataTable.RowSelected)
        def row_selected(self, event: DataTable.RowSelected) -> None:
            if event.row_key and event.row_key.value:
                self._show_detail(event.row_key.value)

        @on(DataTable.RowHighlighted)
        def row_highlighted(self, event: DataTable.RowHighlighted) -> None:
            if event.row_key and event.row_key.value:
                self._show_detail(event.row_key.value)

        @on(Button.Pressed, "#copy-result-btn")
        def copy_result(self, event: Button.Pressed) -> None:
            if not self._detail_plain:
                return
            self.copy_to_clipboard(self._detail_plain)
            self._flash_copy_btn(event.button, "📋 Result")

        @on(Button.Pressed, "#copy-prompts-btn")
        def copy_prompts(self, event: Button.Pressed) -> None:
            prompts = data[self.current_ds].get("prompts", {})
            if not prompts:
                return
            parts: list[str] = []
            for stage, label in [
                ("doc-enrich", "DOC ENRICH PROMPT"),
                ("query-enrich", "QUERY ENRICH PROMPT"),
                ("rerank", "RERANK PROMPT"),
            ]:
                if stage in prompts:
                    parts.append(f"━━━ {label} ━━━\n{prompts[stage]}")
            self.copy_to_clipboard("\n\n".join(parts))
            self._flash_copy_btn(event.button, "📋 Prompts")

        def _build_batch_text(self, fr: int, to: int) -> str | None:
            import re

            failures = self._get_filtered()
            fr = max(1, fr)
            to = min(len(failures), to)
            if fr > to:
                return None
            parts: list[str] = []
            for i in range(fr - 1, to):
                f = failures[i]
                lines = self._build_detail_lines(f, wrap_w=120)
                plain = re.sub(r"\[/?[^\]]*\]", "", "\n".join(lines))
                parts.append(f"═══ #{i+1} ═══\n{plain}")
            return "\n\n".join(parts)

        def _get_batch_range(self) -> tuple[int, int] | None:
            fr = self.query_one("#batch-from", Select).value
            to = self.query_one("#batch-to", Select).value
            if fr is None or to is None or fr == Select.BLANK or to == Select.BLANK:
                return None
            return int(fr), int(to)

        @on(Button.Pressed, "#batch-copy-btn")
        def batch_copy(self, event: Button.Pressed) -> None:
            rng = self._get_batch_range()
            if not rng:
                return
            text = self._build_batch_text(rng[0], rng[1])
            if text:
                self.copy_to_clipboard(text)
                self._flash_copy_btn(event.button, "📋 Batch")

        @on(Button.Pressed, "#save-md-btn")
        def save_md(self, event: Button.Pressed) -> None:
            import re

            ds_name = self.current_ds
            failures = self._get_filtered()
            if not failures:
                return

            lines: list[str] = []
            lines.append(f"# SIRA Failure Report — {ds_name}\n")

            # Info summary
            r = data[ds_name]
            lines.append(f"- **Queries**: {r.get('n_queries', '?')}")
            lines.append(f"- **Failures**: {r.get('n_failures', '?')}")
            lines.append(f"- **Hard**: {r.get('n_hard', '?')}")
            lines.append(f"- **Final stage**: {r.get('final_stage', '?')}")
            sra = r.get("stage_recall_all", {})
            for stage in STAGE_ORDER:
                r10 = sra.get(stage, {}).get("Recall@10")
                if r10 is not None:
                    lines.append(f"- **{stage} R@10**: {r10:.4f}")
            de_stats = r.get("doc_enrich_stats")
            qe_stats = r.get("query_enrich_stats")
            if de_stats:
                t, e = de_stats["total"], de_stats["enriched"]
                lines.append(f"- **DocE enriched**: {e}/{t} ({e/t*100:.0f}%)")
            if qe_stats:
                t, e = qe_stats["total"], qe_stats["enriched"]
                lines.append(f"- **QE enriched**: {e}/{t} ({e/t*100:.0f}%)")
            lines.append("")

            # Prompts
            prompts = r.get("prompts", {})
            if prompts:
                lines.append("## Prompts\n")
                for stage, label in [
                    ("doc-enrich", "Doc Enrich"),
                    ("query-enrich", "Query Enrich"),
                    ("rerank", "Rerank"),
                ]:
                    if stage in prompts:
                        lines.append(f"### {label}\n")
                        lines.append("```")
                        lines.append(prompts[stage])
                        lines.append("```\n")

            # All failures
            lines.append("## Failures\n")
            for i, f in enumerate(failures):
                detail = self._build_detail_lines(f, wrap_w=120)
                plain = re.sub(r"\[/?[^\]]*\]", "", "\n".join(detail))
                lines.append(f"### #{i+1}\n")
                lines.append("```")
                lines.append(plain)
                lines.append("```\n")

            slug = r.get("slug", ds_name.lower())
            out_path = PLOTS_DIR.parent / f"analyze:failure-report-{slug}.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(lines), encoding="utf-8")

            btn = event.button
            btn.label = "✓ Saved"
            btn.add_class("-copied")
            self.set_timer(2.0, lambda: self._reset_copy_btn(btn, "💾 Report"))

        def _flash_copy_btn(self, btn: Button, original_label: str) -> None:
            btn.label = "✓ Copied"
            btn.add_class("-copied")
            self.set_timer(1.5, lambda: self._reset_copy_btn(btn, original_label))

        def _reset_copy_btn(self, btn: Button, label: str) -> None:
            btn.label = label
            btn.remove_class("-copied")

        def _apply_split(self) -> None:
            left_pct = self.SPLIT_RATIOS[self.split_idx]
            right_pct = 100 - left_pct
            tc = self.query_one("#table-container")
            dp = self.query_one("#detail-panel")
            tc.styles.width = f"{left_pct}%"
            dp.styles.width = f"{right_pct}%"
            self.on_resize()

    FailureApp().run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    if "--browse" in sys.argv:
        sys.argv.remove("--browse")
        k = 10
        for arg in sys.argv[1:]:
            if arg.startswith("--k"):
                k = int(
                    arg.split("=")[1]
                    if "=" in arg
                    else sys.argv[sys.argv.index(arg) + 1]
                )
        _browse(k)
    else:
        force = "--force" in sys.argv
        if force:
            sys.argv.remove("--force")
            sys.argv.append("+force=true")
        complete_only = "--complete" in sys.argv
        if complete_only:
            sys.argv.remove("--complete")
        no_browse = "--no-browse" in sys.argv
        if no_browse:
            sys.argv.remove("--no-browse")
        quick = "--quick" in sys.argv
        if quick:
            sys.argv.remove("--quick")
        sira_best, per_stage, per_stage_all = collect_sira_results()
        print_summary(sira_best, per_stage)
        save_leaderboard_md(sira_best, per_stage, per_stage_all)
        generate_plots(sira_best, per_stage, complete_only=complete_only, per_stage_all=per_stage_all)
        if not quick:
            main()
        if not no_browse and not quick:
            _browse(10)
