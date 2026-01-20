import json
import os
import re
import string
import random
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from rouge_score import rouge_scorer
from tabulate import tabulate

from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("evaluation")

# Initialize the Rouge scorer for ROUGE metrics
_rouge_scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def normalize_text(text: str) -> str:
    """Normalize text for evaluation by applying multiple transformations.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text string
    """

    def _bool_mapping(s: str) -> str:
        return {"True": "yes", "False": "no"}.get(s, s)

    def _remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def _white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def _remove_punc(t: str) -> str:
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in t)

    def _lower(t: str) -> str:
        return t.lower()

    def _replace_underscore(t: str) -> str:
        return t.replace("_", " ")

    for func in [
        _bool_mapping,
        _replace_underscore,
        _lower,
        _remove_punc,
        _remove_articles,
        _white_space_fix,
    ]:
        text = func(text)
    return text.strip()


def accuracy_score(gt: List[str], pred: str) -> float:
    """Calculate accuracy score: 1.0 if any ground truth is contained in prediction, else 0.0.

    Args:
        gt: List of ground truth strings
        pred: Prediction string

    Returns:
        Accuracy score (0.0 or 1.0)
    """
    pred_norm = normalize_text(pred)
    if not pred_norm:
        return 0.0
    gt_norm_ls = [normalize_text(g) for g in gt]
    return 1.0 if any(g in pred_norm for g in gt_norm_ls) else 0.0


def exact_match_score(gt: List[str], pred: str) -> float:
    """Calculate exact match score: 1.0 if prediction exactly matches any ground truth, else 0.0.

    Args:
        gt: List of ground truth strings
        pred: Prediction string

    Returns:
        Exact match score (0.0 or 1.0)
    """
    pred_norm = normalize_text(pred)
    gt_norm_ls = [normalize_text(g) for g in gt]
    return 1.0 if any(pred_norm == g for g in gt_norm_ls) else 0.0


def cover_exact_match_score(gt: List[str], pred: str) -> float:
    """Calculate cover exact match score: 1.0 if all tokens of any ground truth are in prediction.

    Args:
        gt: List of ground truth strings
        pred: Prediction string

    Returns:
        Cover exact match score (0.0 or 1.0)
    """
    pred_norm = normalize_text(pred)
    gt_norm_ls = [normalize_text(g) for g in gt]

    pred_tokens = pred_norm.split()
    gt_tokens_ls = [g.split() for g in gt_norm_ls]

    for gt_tokens in gt_tokens_ls:
        if all(token in pred_tokens for token in gt_tokens):
            return 1.0
    return 0.0


def string_em_score(gt: List[str], pred: str) -> float:
    """Calculate string exact match score: fraction of ground truths that exactly match prediction.

    Args:
        gt: List of ground truth strings
        pred: Prediction string

    Returns:
        String exact match score (0.0 to 1.0)
    """
    pred_norm = normalize_text(pred)
    gt_norm_ls = [normalize_text(g) for g in gt]

    match_cnt = sum(1 for g in gt_norm_ls if pred_norm == g)
    return match_cnt / len(gt_norm_ls) if gt_norm_ls else 0.0


def f1_score(gt: List[str], pred: str) -> float:
    """Calculate F1 score: maximum F1 score between prediction and any ground truth.

    Args:
        gt: List of ground truth strings
        pred: Prediction string

    Returns:
        F1 score (0.0 to 1.0)
    """

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


def rouge1_score(gt: List[str], pred: str) -> float:
    """Calculate ROUGE-1 score: maximum ROUGE-1 F-measure between prediction and any ground truth.

    Args:
        gt: List of ground truth strings
        pred: Prediction string

    Returns:
        ROUGE-1 score (0.0 to 1.0)
    """
    pred_norm = normalize_text(pred)
    gt_norm_ls = [normalize_text(g) for g in gt]
    scores = []
    for g in gt_norm_ls:
        score = _rouge_scorer.score(g, pred_norm)["rouge1"].fmeasure
        scores.append(score)
    return max(scores) if scores else 0.0


def rouge2_score(gt: List[str], pred: str) -> float:
    """Calculate ROUGE-2 score: maximum ROUGE-2 F-measure between prediction and any ground truth.

    Args:
        gt: List of ground truth strings
        pred: Prediction string

    Returns:
        ROUGE-2 score (0.0 to 1.0)
    """
    pred_norm = normalize_text(pred)
    gt_norm_ls = [normalize_text(g) for g in gt]
    scores = []
    for g in gt_norm_ls:
        score = _rouge_scorer.score(g, pred_norm)["rouge2"].fmeasure
        scores.append(score)
    return max(scores) if scores else 0.0


def rougel_score(gt: List[str], pred: str) -> float:
    """Calculate ROUGE-L score: maximum ROUGE-L F-measure between prediction and any ground truth.

    Args:
        gt: List of ground truth strings
        pred: Prediction string

    Returns:
        ROUGE-L score (0.0 to 1.0)
    """
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
    """Compute evaluation metrics for predictions against ground truths.

    Args:
        gt_list: List of ground truth lists (one per prediction)
        pred_list: List of prediction strings
        metrics: List of metric names to compute (default: all available metrics)

    Returns:
        Dictionary containing per-sample scores and average scores for each metric
    """
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


def _load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Load TREC qrels file.

    Format: <qid> <iter> <docid> <rel>

    Args:
        qrels_path: Path to qrels file

    Returns:
        Dictionary mapping qid to docid to relevance score
    """
    qrel: Dict[str, Dict[str, int]] = {}  # {qid: {docid: rel_int}}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()  # <qid> <iter> <docid> <rel>
            if len(p) < 4:
                continue
            qid, _, docid, rel = p[0], p[1], p[2], p[3]
            try:
                rel_i = int(rel)
            except ValueError:
                rel_i = 1 if rel != "0" else 0
            qrel.setdefault(qid, {})[docid] = rel_i
    return qrel


def _load_run(run_path: str) -> Dict[str, Dict[str, float]]:
    """Load TREC run file.

    Format: <qid> Q0 <docid> <rank> <score> <tag>

    Args:
        run_path: Path to run file

    Returns:
        Dictionary mapping qid to docid to score
    """
    run: Dict[str, Dict[str, float]] = {}  # {qid: {docid: score_float}}
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()  # <qid> Q0 <docid> <rank> <score> <tag>
            if len(p) < 6:
                continue
            qid, docid, score = p[0], p[2], p[4]
            try:
                s = float(score)
            except ValueError:
                s = 0.0
            run.setdefault(qid, {})[docid] = s
    return run


def _mean(xs) -> float:
    """Calculate mean of a list of numbers.

    Args:
        xs: List of numbers

    Returns:
        Mean value (0.0 if list is empty)
    """
    return sum(xs) / len(xs) if xs else 0.0


def _permutation_test_two_sided(diffs, n_resamples=10000) -> float:
    """Perform two-sided permutation test to calculate p-value.

    Args:
        diffs: List of differences between paired samples
        n_resamples: Number of resampling iterations (default: 10000)

    Returns:
        P-value (0.0 to 1.0)
    """
    if not diffs:
        return 1.0
    obs = abs(_mean(diffs))
    n = len(diffs)
    cnt = 0
    for _ in range(n_resamples):
        s = 0.0
        for d in diffs:
            s += d if random.getrandbits(1) else -d
        if abs(s / n) >= obs:
            cnt += 1
    return cnt / n_resamples


def eval_with_pytrec(
    qrels_path: str,
    run_path: str,
    metrics: List[str] | None,
    ks: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Evaluate retrieval results using pytrec_eval.

    Args:
        qrels_path: Path to TREC qrels file
        run_path: Path to TREC run file
        metrics: List of metrics to compute (mrr, map, recall, ndcg, precision)
        ks: List of k values for recall@k, precision@k, ndcg@k (default: None)

    Returns:
        Dictionary with 'per_query' and 'aggregate' results

    Raises:
        ImportError: If pytrec_eval is not installed
    """
    try:
        import pytrec_eval
    except ImportError:
        err_msg = "pytrec_eval is not installed. Please install it with `pip install pytrec_eval`"
        app.logger.error(err_msg)
        raise ImportError(err_msg)

    qrel = _load_qrels(qrels_path)
    run = _load_run(run_path)

    measures: set[str] = set()
    if metrics:
        if "mrr" in metrics:
            measures.add("recip_rank")
        if "map" in metrics:
            measures.add("map")
        if ks:
            for k in ks:
                if "ndcg" in metrics:
                    measures.add(f"ndcg_cut.{k}")
                if "precision" in metrics:
                    measures.add(f"P.{k}")
                if "recall" in metrics:
                    measures.add(f"recall.{k}")

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, measures)
    per_query = evaluator.evaluate(run)  # {qid: {metric: value}}

    agg: Dict[str, float] = {}
    if per_query:
        pytrec_metrics = sorted(next(iter(per_query.values())).keys())
        n_q = len(per_query)
        for m in pytrec_metrics:
            agg[m] = sum(qres.get(m, 0.0) for qres in per_query.values()) / n_q

    return {"per_query": per_query, "aggregate": agg}


def save_evaluation_results(
    results: Dict[str, float],
    markdown: bool,
    save_path: str,
) -> Dict[str, Any]:
    """Save evaluation results to JSON file and optionally display as Markdown table.

    Args:
        results: Dictionary of metric names to values
        markdown: Whether to display results as Markdown table
        save_path: Path where results will be saved (timestamp will be added)

    Returns:
        Dictionary with 'eval_res' containing the results

    Raises:
        Exception: If file writing fails
    """
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

    if markdown:
        table_data = [["Metric", "Value"]]
        for metric, value in results.items():
            if metric.startswith("avg_"):
                pretty_metric = metric.replace("avg_", "")
                formatted_value = round(value, 4) if isinstance(value, float) else value
                table_data.append([pretty_metric, formatted_value])
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                # Only add non-avg metrics if they are numeric (for other evaluation types)
                table_data.append([metric, round(float(value), 6)])

        table_md = tabulate(
            table_data,
            headers="firstrow",
            tablefmt="fancy_grid",
            colalign=("left", "left"),
        )
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
    """Evaluate predictions against ground truths using specified metrics.

    Args:
        pred_ls: List of prediction strings
        gt_ls: List of ground truth lists (one per prediction)
        metrics: List of metric names (acc, em, stringem, coverem, f1, rouge-1, rouge-2, rouge-l)
        save_path: Path where results will be saved

    Returns:
        Dictionary with 'eval_res' containing evaluation results
    """
    results = compute_metrics(gt_ls, pred_ls, metrics)
    return save_evaluation_results(results, markdown=True, save_path=save_path)


@app.tool(output="run_path,qrels_path,ir_metrics,ks,save_path->eval_res")
def evaluate_trec(
    run_path: str,
    qrels_path: str,
    metrics: List[str] | None,
    ks: List[int] | None,
    save_path: str,
):
    """Evaluate retrieval results in TREC format using IR metrics.

    Args:
        run_path: Path to TREC run file
        qrels_path: Path to TREC qrels file
        metrics: List of IR metrics (mrr, map, recall, ndcg, precision)
        ks: List of k values for recall@k, precision@k, ndcg@k (default: [1, 5, 10, 20, 50, 100])
        save_path: Path where results will be saved

    Returns:
        Dictionary with 'eval_res' containing evaluation results
    """
    if ks is None:
        ks = [1, 5, 10, 20, 50, 100]
    SUPPORT_METRICS = ["mrr", "map", "recall", "ndcg", "precision"]
    if metrics is None:
        metrics = SUPPORT_METRICS
    metrics = [m.lower() for m in metrics]
    for m in metrics:
        if m not in SUPPORT_METRICS:
            warn_msg = f"Metric '{m}' is not recognized. Available metrics: {', '.join(SUPPORT_METRICS)}."
            app.logger.warning(warn_msg)
    res = eval_with_pytrec(
        qrels_path=qrels_path,
        run_path=run_path,
        ks=ks,
        metrics=metrics,
    )
    results = {
        "map": res["aggregate"].get("map", 0.0),
        "mrr": res["aggregate"].get("recip_rank", 0.0),
    }
    for k in ks:
        for key in (f"ndcg_cut_{k}", f"P_{k}", f"recall_{k}"):
            if key in res["aggregate"]:
                if "_cut_" in key:
                    # e.g., ndcg_cut_10
                    results[key.replace("_cut_", "@")] = res["aggregate"][key]
                elif "_" in key:
                    # e.g., recall_10, P_10
                    results[key.replace("_", "@")] = res["aggregate"][key]
                else:
                    results[key] = res["aggregate"][key]

    return save_evaluation_results(results, markdown=True, save_path=save_path)


@app.tool(
    output="run_new_path,run_old_path,qrels_path,ir_metrics,ks,n_resamples,save_path->eval_res"
)
def evaluate_trec_pvalue(
    run_new_path: str,
    run_old_path: str,
    qrels_path: str,
    metrics: List[str] | None,
    ks: List[int] | None,
    n_resamples: int | None,
    save_path: str,
):
    """Compare two TREC run files using permutation test to calculate p-values.

    Args:
        run_new_path: Path to new TREC run file (system A)
        run_old_path: Path to old TREC run file (system B)
        qrels_path: Path to TREC qrels file
        metrics: List of IR metrics to compare
        ks: List of k values for recall@k, precision@k, ndcg@k (default: [1, 5, 10, 20, 50, 100])
        n_resamples: Number of resampling iterations for permutation test (default: 10000)
        save_path: Path where results will be saved

    Returns:
        Dictionary with 'eval_res' containing comparison results with p-values
    """

    def _process_metric_key(base: str, k: int | None) -> str:
        base_l = base.lower()
        if k is None:
            return "map" if base_l == "map" else "recip_rank"
        elif base_l == "ndcg":
            return f"ndcg_cut_{k}"
        elif base_l == "recall":
            return f"recall_{k}"
        elif base_l == "precision":
            return f"P_{k}"
        else:
            return base

    def _get_metric_val(
        per: Dict[str, Dict[str, float]],
        qid: str,
        base: str,
        k: int | None,
    ) -> float:
        row = per.get(qid, {})
        key = _process_metric_key(base, k)
        return row[key] if key in row else 0.0

    if ks is None:
        ks = [1, 5, 10, 20, 50, 100]
    n_resamples = int(n_resamples or 10000)
    res_a = eval_with_pytrec(
        qrels_path=qrels_path,
        run_path=run_new_path,
        metrics=metrics,
        ks=ks,
    )
    res_b = eval_with_pytrec(
        qrels_path=qrels_path,
        run_path=run_old_path,
        metrics=metrics,
        ks=ks,
    )
    per_a, per_b = res_a["per_query"], res_b["per_query"]
    qids = sorted(set(per_a.keys()) | set(per_b.keys()))
    out = {}
    for m in metrics:
        if m in ["recall", "ndcg", "precision"]:
            for k in ks:
                a_vals = [_get_metric_val(per_a, q, m, k) for q in qids]
                b_vals = [_get_metric_val(per_b, q, m, k) for q in qids]
                diffs = [a - b for a, b in zip(a_vals, b_vals)]
                p_val = _permutation_test_two_sided(diffs, n_resamples=n_resamples)

                if m == "ndcg":
                    alias = f"ndcg@{k}"
                elif m == "precision":
                    alias = f"P@{k}"
                elif m == "recall":
                    alias = f"recall@{k}"

                out[alias] = {
                    "A_mean": _mean(a_vals),
                    "B_mean": _mean(b_vals),
                    "diff": _mean(diffs),
                    "p_value": p_val,
                    "significant": bool(p_val < 0.05),
                }
        elif m in ["mrr", "map"]:
            a_vals = [_get_metric_val(per_a, q, m, None) for q in qids]
            b_vals = [_get_metric_val(per_b, q, m, None) for q in qids]
            diffs = [a - b for a, b in zip(a_vals, b_vals)]
            p_val = _permutation_test_two_sided(diffs, n_resamples=n_resamples)

            out[m] = {
                "A_mean": _mean(a_vals),
                "B_mean": _mean(b_vals),
                "diff": _mean(diffs),
                "p_value": p_val,
                "significant": bool(p_val < 0.05),
            }

    table_data = [
        ["metrics", "A_mean", "B_mean", "Diff(A-B)", "p_value", "significant"]
    ]
    for metric, res in out.items():
        table_data.append(
            [
                metric,
                res["A_mean"],
                res["B_mean"],
                res["diff"],
                res["p_value"],
                res["significant"],
            ]
        )
    table_md = tabulate(
        table_data,
        headers="firstrow",
        tablefmt="fancy_grid",
        colalign=("left",) * len(table_data[0]),
    )
    app.logger.info(f"\n{table_md}")
    return save_evaluation_results(out, markdown=False, save_path=save_path)


if __name__ == "__main__":
    app.run(transport="stdio")
