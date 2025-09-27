import json
import os
import re
import string
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List

from rouge_score import rouge_scorer
from tabulate import tabulate

from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("evaluation")

# Initialize the Rouge scorer for ROUGE metrics
_rouge_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def bool_mapping(s: str) -> str:
    return {"True": "yes", "False": "no"}.get(s, s)


def remove_articles(t: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", t)


def white_space_fix(t: str) -> str:
    return " ".join(t.split())


def remove_punc(t: str) -> str:
    exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
    return "".join(ch if ch not in exclude else " " for ch in t)


def lower(t: str) -> str:
    return t.lower()


def replace_underscore(t: str) -> str:
    return t.replace("_", " ")


def normalize_text(text: str) -> str:
    for func in [
        bool_mapping,
        replace_underscore,
        lower,
        remove_punc,
        remove_articles,
        white_space_fix,
    ]:
        text = func(text)
    return text.strip()


# Accuracy Score
def accuracy_score(gt: List[str], pred: str) -> float:
    pred_norm = normalize_text(pred)
    if not pred_norm:
        return 0.0
    gt_norm_ls = [normalize_text(g) for g in gt]
    return 1.0 if any(pred_norm in g for g in gt_norm_ls) else 0.0


# Exact Match Score
def exact_match_score(gt: List[str], pred: str) -> float:
    pred_norm = normalize_text(pred)
    gt_norm_ls = [normalize_text(g) for g in gt]
    return 1.0 if any(pred_norm == g for g in gt_norm_ls) else 0.0


# Cover Exact Match Score
def cover_exact_match_score(gt: List[str], pred: str) -> float:
    pred_norm = normalize_text(pred)
    gt_norm_ls = [normalize_text(g) for g in gt]

    pred_tokens = pred_norm.split()
    gt_tokens_ls = [g.split() for g in gt_norm_ls]

    for gt_tokens in gt_tokens_ls:
        if all(token in pred_tokens for token in gt_tokens):
            return 1.0
    return 0.0


# String Exact Match Score
def string_em_score(gt: List[str], pred: str) -> float:
    pred_norm = normalize_text(pred)
    gt_norm_ls = [normalize_text(g) for g in gt]

    match_cnt = sum(1 for g in gt_norm_ls if pred_norm == g)
    return match_cnt / len(gt_norm_ls) if gt_norm_ls else 0.0


# F1 Score
def f1_score(gt: List[str], pred: str) -> float:
    def calc_f1(gt_str: str, pred_str: str) -> float:
        pred_norm = normalize_text(pred_str)
        gt_norm = normalize_text(gt_str)

        pred_tokens = pred_norm.split()
        gt_tokens = gt_norm.split()
        if not pred_tokens or not gt_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    scores = [calc_f1(g, pred) for g in gt]
    return max(scores) if scores else 0.0


# ROUGE-1 Score
def rouge1_score(gt: List[str], pred: str) -> float:
    pred_norm = normalize_text(pred)
    gt_norm_ls = [normalize_text(g) for g in gt]
    scores = []
    for g in gt_norm_ls:
        score = _rouge_scorer.score(g, pred_norm)["rouge1"].fmeasure
        scores.append(score)
    return max(scores) if scores else 0.0


# ROUGE-2 Score
def rouge2_score(gt: List[str], pred: str) -> float:
    pred_norm = normalize_text(pred)
    gt_norm_ls = [normalize_text(g) for g in gt]
    scores = []
    for g in gt_norm_ls:
        score = _rouge_scorer.score(g, pred_norm)["rouge2"].fmeasure
        scores.append(score)
    return max(scores) if scores else 0.0


# ROUGE-L Score
def rougel_score(gt: List[str], pred: str) -> float:
    pred_norm = normalize_text(pred)
    gt_norm_ls = [normalize_text(g) for g in gt]
    scores = []
    for g in gt_norm_ls:
        score = _rouge_scorer.score(g, pred_norm)["rougeL"].fmeasure
        scores.append(score)
    return max(scores) if scores else 0.0


def compute_metrics(
    gt_list: List[List[str]],
    pred_list: List[str],
    metrics: List[str] | None = None,
) -> Dict[str, float]:
    METRICS_REGISTRY: Dict[str, Callable[[List[str], str], float]] = {
        "acc": accuracy_score,
        "em": exact_match_score,
        "stringem": string_em_score,
        "coverem": cover_exact_match_score,
        "f1": f1_score,
        "rouge-1": rouge1_score,
        "rouge-2": rouge2_score,
        "rouge-l": rougel_score,
    }
    if not metrics:
        metrics = list(METRICS_REGISTRY.keys())
    metrics = [m.lower() for m in metrics]
    results = {metric: [] for metric in metrics}

    for gt, pred in zip(gt_list, pred_list):
        for metric in metrics:
            if metric in METRICS_REGISTRY:
                score = METRICS_REGISTRY[metric](gt, pred)
                results[metric].append(score)
            else:
                warn_msg = f"Metric '{metric}' is not recognized. Available metrics: {', '.join(METRICS_REGISTRY.keys())}."
                app.logger.warning(warn_msg)

    avg_results = {}
    for metric, scores in results.items():
        if not scores:
            avg_results[f"avg_{metric}"] = 0.0
            continue
        avg_results[f"avg_{metric}"] = sum(scores) / len(scores)
    return {**results, **avg_results}


def save_evaluation_results(
    results: Dict[str, float],
    save_path: str,
) -> Dict[str, Any]:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir, base_file = os.path.split(save_path)
    file_stem = os.path.splitext(base_file)[0]
    output_name = f"{file_stem}_{current_time}.json"
    output_path = os.path.join(base_dir, output_name) if base_dir else output_name

    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        app.logger.error(f"Failed to save evaluation results to {output_path}: {e}")
        raise

    table_data = [["Metric", "Value"]]
    for metric, value in results.items():
        if metric.startswith("avg_"):
            pretty_metric = metric.replace("avg_", "")
            formatted_value = round(value, 4) if isinstance(value, float) else value
            table_data.append([pretty_metric, formatted_value])

    table_md = tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
    app.logger.info(f"Evaluation results saved to {output_path}")
    app.logger.info(f"\n{table_md}")
    return {"eval_res": results}


@app.tool(output="pred_ls,gt_ls,metrics,save_path->eval_res")
def evaluate(
    pred_ls: List[str],
    gt_ls: List[List[str]],
    metrics: List[str] | None,
    save_path: str,
) -> Dict[str, Any]:
    results = compute_metrics(gt_ls, pred_ls, metrics)
    return save_evaluation_results(results, save_path)


if __name__ == "__main__":
    app.run(transport="stdio")
