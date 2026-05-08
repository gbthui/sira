# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Build bm25x indices and evaluate BM25 Recall@K on an MTEB dataset.

For each n-gram candidate, builds an index, retrieves top-K, computes
NDCG + Recall, and symlinks the best config as ``index/best`` and
``eval/best.json``.

Usage::

    source sandbox.sh
    python scripts/eval_bm25.py
    python scripts/eval_bm25.py data=fiqa
    python scripts/eval_bm25.py bm25.max_n_candidates='[1]' data.k_values='[10,200]'

Config groups::

    global  — db_root (shared across all scripts)
    data/   — dataset name, k_values
    bm25/   — index parameters (max_n_candidates, tokenizer, method, k1, b, cuda)

Outputs under ``{db_root}/{data.name}/``::

    index/bm25-n{ngrams}-{tokenizer}/    — persisted bm25x index
    index/best -> best config            — symlink
    eval/bm25-n{ngrams}-{tokenizer}.json — evaluation metrics
    eval/best.json -> best eval          — symlink
"""

import json
import logging
import os
import time

import hydra
from bm25x import BM25
from omegaconf import DictConfig
from sira.schema.mteb import (
    COL_ID,
    COL_TEXT,
    DatasetDir,
    load_qrels_dict,
    read_corpus_texts,
    read_queries,
)

logger = logging.getLogger(__name__)


def _ds(cfg: DictConfig) -> DatasetDir:
    return DatasetDir(root=cfg.db_root, name=cfg.data.name)


def _build_and_evaluate_one(
    ds: DatasetDir,
    bm25_cfg: DictConfig,
    max_n: int,
    k_values: list[int],
    doc_ids: list[str],
    texts: list[str],
    query_ids: list[str],
    query_texts: list[str],
    qrels: dict[str, dict[str, int]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Build index + evaluate for a single max_n. Returns (metrics, beir_results)."""
    index_dir = ds.bm25_index(max_n, bm25_cfg.tokenizer)
    eval_path = ds.eval_baseline(max_n, bm25_cfg.tokenizer)

    if os.path.exists(eval_path):
        logger.info("Eval already done at %s — loading.", eval_path)
        with open(eval_path) as f:
            return json.load(f), {}

    use_cuda = bm25_cfg.cuda
    if os.path.exists(os.path.join(index_dir, "header.bin")):
        logger.info("Loading existing index from %s", index_dir)
        bm25 = BM25.load(index_dir, cuda=use_cuda)
    else:
        logger.info("Building bm25x index (max_n=%d) over %d docs …", max_n, len(texts))
        bm25 = BM25(
            index=index_dir,
            cuda=use_cuda,
            max_n=max_n,
            tokenizer=bm25_cfg.tokenizer,
            method=bm25_cfg.method,
            k1=bm25_cfg.k1,
            b=bm25_cfg.b,
        )
        bm25.add(texts)
        logger.info("Index built: %d docs", len(bm25))

    max_k = max(k_values)
    logger.info("Searching %d queries @ k=%d …", len(query_texts), max_k)
    try:
        batch_results = bm25.search(query_texts, k=max_k)
    except (ValueError, RuntimeError) as e:
        if "CUDA_ERROR_OUT_OF_MEMORY" not in str(e) and "out of memory" not in str(e):
            raise
        logger.warning("GPU OOM, falling back to CPU: %s", e)
        bm25 = BM25.load(index_dir, cuda=False)
        batch_results = bm25.search(query_texts, k=max_k)

    beir_results: dict[str, dict[str, float]] = {}
    for qid, hits in zip(query_ids, batch_results):
        beir_results[qid] = {doc_ids[idx]: float(score) for idx, score in hits}

    from beir.retrieval.evaluation import EvaluateRetrieval

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, beir_results, k_values
    )

    metrics = {**ndcg, **recall, **precision}

    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved to %s", eval_path)
    return metrics, beir_results


def build_and_evaluate(cfg: DictConfig) -> None:
    """Build indices for each max_n candidate, evaluate, and pick the best."""
    ds = _ds(cfg)
    bm25_cfg = cfg.bm25
    k_values = list(cfg.data.k_values)
    # max_k = max(k_values)
    selection_metric = cfg.selection_metric

    doc_ids, texts = read_corpus_texts(ds.corpus)
    split = cfg.data.split
    queries_df = read_queries(ds.queries(split))
    query_ids = queries_df.get_column(COL_ID).to_list()
    query_texts = queries_df.get_column(COL_TEXT).to_list()
    qrels = load_qrels_dict(ds.qrels(split))

    results: dict[int, dict[str, float]] = {}
    beir_results_all: dict[int, dict[str, dict[str, float]]] = {}
    for max_n in bm25_cfg.max_n_candidates:
        tag = ds._bm25_tag(max_n, bm25_cfg.tokenizer)
        logger.info("--- %s ---", tag)
        metrics, beir_res = _build_and_evaluate_one(
            ds,
            bm25_cfg,
            max_n,
            k_values,
            doc_ids,
            texts,
            query_ids,
            query_texts,
            qrels,
        )
        results[max_n] = metrics
        if beir_res:
            beir_results_all[max_n] = beir_res

    if not results:
        return

    logger.info("Summary for %s:", ds.name)
    for max_n, metrics in sorted(results.items()):
        tag = ds._bm25_tag(max_n, bm25_cfg.tokenizer)
        parts = [f"  {tag}:"]
        for k in k_values:
            parts.append(
                f"NDCG@{k}={metrics.get(f'NDCG@{k}', 0):.4f} "
                f"Recall@{k}={metrics.get(f'Recall@{k}', 0):.4f}"
            )
        logger.info("  ".join(parts))

    best_n = max(results, key=lambda n: results[n].get(selection_metric, 0))
    best_tag = ds._bm25_tag(best_n, bm25_cfg.tokenizer)
    logger.info(
        "Best: %s (%s=%.4f)",
        best_tag,
        selection_metric,
        results[best_n][selection_metric],
    )

    # Index best symlink (directory, not json)
    best_index = ds.bm25_index(best_n, bm25_cfg.tokenizer)
    if os.path.islink(ds.bm25_index_best):
        os.remove(ds.bm25_index_best)
    os.symlink(
        os.path.relpath(best_index, os.path.dirname(ds.bm25_index_best)),
        ds.bm25_index_best,
    )
    logger.info("Linked %s → %s", ds.bm25_index_best, os.readlink(ds.bm25_index_best))

    meta = {
        "stage": "baseline",
        "dataset": ds.name,
        "best_config": best_tag,
        "metrics": results[best_n],
        "all_configs": {
            ds._bm25_tag(n, bm25_cfg.tokenizer): m for n, m in results.items()
        },
        "bm25_params": {
            "method": bm25_cfg.method,
            "tokenizer": bm25_cfg.tokenizer,
            "k1": bm25_cfg.k1,
            "b": bm25_cfg.b,
            "max_n": best_n,
        },
        "timestamp": int(time.time()),
    }
    # Eval + index best.meta.json
    ds.update_best(
        best_links=[ds.eval_baseline_best],
        target_name=f"{best_tag}.json",
        metrics=results[best_n],
        selection_metric=selection_metric,
        meta=meta,
    )
    # Also write meta next to index best symlink
    with open(
        os.path.join(os.path.dirname(ds.bm25_index_best), "best.meta.json"), "w"
    ) as f:
        json.dump(meta, f, indent=2)

    # Save baseline retrieval JSONL
    baseline_path = os.path.join(ds.retrieval_dir, "baseline.jsonl")
    os.makedirs(ds.retrieval_dir, exist_ok=True)
    best_beir = beir_results_all.get(best_n)
    if not best_beir:
        logger.info("Re-searching best config for baseline.jsonl …")
        bm25 = BM25.load(best_index)
        batch_results = bm25.search(query_texts, k=max(k_values))
        best_beir = {}
        for qid, hits in zip(query_ids, batch_results):
            best_beir[qid] = {doc_ids[idx]: float(score) for idx, score in hits}
    with open(baseline_path, "w") as f:
        for qid in query_ids:
            scored = best_beir.get(qid, {})
            candidates = [
                {"doc_id": did, "score": s, "rank": r}
                for r, (did, s) in enumerate(
                    sorted(scored.items(), key=lambda x: -x[1]), 1
                )
            ]
            f.write(json.dumps({"query_id": qid, "candidates": candidates}) + "\n")
    logger.info("Saved baseline retrieval: %s", baseline_path)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="eval_bm25",
)
def main(cfg: DictConfig) -> None:
    build_and_evaluate(cfg)


if __name__ == "__main__":
    main()
