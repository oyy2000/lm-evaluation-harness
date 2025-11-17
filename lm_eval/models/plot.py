# -*- coding: utf-8 -*-
"""
Aggregate lm-evaluation-harness grid results and plot key metrics.
- Recursively scans the eval_grid directory.
- Parses layer (L..) and tag (BASELINE or lamXpY) from folder names like: L24_lam2p0 or L24_BASELINE
- Extracts metrics from results_*.json
- Builds a tidy DataFrame
- Plots (Matplotlib only; one chart per figure; no custom colors):
    * GSM8K CoT: exact_match,flexible-extract (with stderr if present)
    * TriviaQA CoT: exact_match,remove_whitespace
    * TruthfulQA Gen CoT: rougeL_max,none and bleu_acc,none
Figures are saved to /mnt/data and displayed inline.
"""
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Configuration (edit ROOT if needed) ----------
ROOT = Path("/home/youyang7/projects/lm-evaluation-harness/lm_eval/models/eval_grid/triviaqa_cot/")

# Choose which metrics to plot for each task (value_key -> stderr_key or None)
PLOT_SPECS = {
    "gsm8k_cot_zeroshot": {
        "value_key": "exact_match,flexible-extract",
        "stderr_key": "exact_match_stderr,flexible-extract",
        "title": "GSM8K-CoT (zeroshot): EM (flexible-extract) vs λ",
        "ylabel": "Exact Match"
    },
    "triviaqa_cot": {
        "value_key": "exact_match,strict-match",
        "stderr_key": "exact_match_stderr,strict-match",
        "title": "TriviaQA-CoT: EM (strict-match) vs λ",
        "ylabel": "Exact Match"
    },
    "truthfulqa_gen_cot": {
        # We'll produce two separate figures for this task
        # (rougeL_max and bleu_acc); configure below
    }
}

TQ_TASKS_TWOWAY = [
    ("truthfulqa_gen_cot", "rougeL_max,none", "rougeL_max_stderr,none", "TruthfulQA-CoT: ROUGE-L (max) vs λ", "ROUGE-L (max)"),
    ("truthfulqa_gen_cot", "bleu_acc,none",   "bleu_acc_stderr,none",   "TruthfulQA-CoT: BLEU Acc vs λ",     "BLEU Acc"),
]

def _parse_layer_and_lambda_from_dirname(dirname: str) -> Tuple[int, float, str]:
    """
    支持两种命名：
      - L8_lam2p0 / L8_BASELINE
      - Llama-3.1-8B-Instruct_L8_lam2p0 / ..._L8_BASELINE
    """
    # 只看最后一个 `_L..._tag`
    m = re.search(r"_L(\d+)_([A-Za-z0-9\-\+p\.]+)$", dirname)
    if not m:
        # 兼容老格式：L8_lam2p0
        m = re.match(r"^L(\d+)_([A-Za-z0-9\-\+p\.]+)$", dirname)
    if not m:
        raise ValueError(f"Unexpected directory name format: {dirname}")

    layer = int(m.group(1))
    tag = m.group(2)

    if tag.upper() == "BASELINE":
        lam = 0.0
    else:
        # tag like lam2p0, lam-1p5, lam0p25 etc.
        body = tag[3:] if tag.startswith("lam") else tag
        lam_str = body.replace("p", ".")
        try:
            lam = float(lam_str)
        except ValueError:
            lam = float(re.findall(r"[-+]?\d*\.?\d+", lam_str)[0])

    return layer, lam, tag


def _collect_results(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        dirname = dirpath.name

        # We only care about leaf folders of the form L{layer}_{tag}; skip others
         # 不再用 ^L\d+_.+$ 限制，直接尝试 parse
        try:
            layer, lam, tag = _parse_layer_and_lambda_from_dirname(dirname)
        except ValueError:
            continue  # 解析失败就跳过
        # Identify model subfolder(s) under this safe_name
        # e.g., .../L24_lam2p0/meta-llama__Llama-3.1-8B-Instruct/ results_*.json
        json_files = [Path(dirpath, f) for f in filenames if f.startswith("results_") and f.endswith(".json")]
        if not json_files:
            # maybe results are inside the model subdir
            for sub in dirpath.iterdir():
                if sub.is_dir():
                    json_files += list(sub.glob("results_*.json"))

        if not json_files:
            continue

        try:
            layer, lam, tag = _parse_layer_and_lambda_from_dirname(dirname)
        except Exception:
            continue

        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            results = data.get("results", {})
            for task_name, task_dict in results.items():
                if not isinstance(task_dict, dict):
                    continue

                # Collect metrics and stderrs
                # Example keys: "exact_match,flexible-extract" and "exact_match_stderr,flexible-extract"
                # We'll pair value and stderr by shared suffix after the first token (metric name / metric+suffix)
                for k, v in task_dict.items():
                    # Skip alias lines
                    if k == "alias":
                        continue
                    if ",none" in k or "," in k or "strict-match" in k or "remove_whitespace" in k:
                        value = None
                        stderr = None
                        # If this is a stderr key, skip for now (we will pick it when handling the value key)
                        if "_stderr" in k:
                            continue
                        value_key = k
                        value = v
                        stderr_key = None
                        # Find matching stderr key
                        candidate = None
                        # typical: exact_match,flexible-extract -> exact_match_stderr,flexible-extract
                        parts = value_key.split(",")
                        if len(parts) >= 1:
                            candidate = parts[0] + "_stderr," + ",".join(parts[1:])
                        if candidate and candidate in task_dict:
                            stderr_key = candidate
                            stderr = task_dict.get(candidate, None)

                        rows.append({
                            "layer": layer,
                            "lambda": lam,
                            "tag": tag,
                            "task": task_name,
                            "metric": value_key,
                            "value": value,
                            "stderr": stderr,
                            "file": str(jf),
                        })

    if not rows:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["layer", "lambda", "tag", "task", "metric", "value", "stderr", "file"])
    df = pd.DataFrame(rows)
    return df


df = _collect_results(ROOT)

# Show a compact preview and save the full CSV for download
if df.empty:
    print("No results found under:", ROOT)
else:
    # Sort for readability
    df = df.sort_values(by=["task", "layer", "lambda", "metric"]).reset_index(drop=True)

    # Display a small preview to the user
    preview_cols = ["task", "layer", "lambda", "metric", "value", "stderr", "tag"]

    # Save the full table
    out_csv = "/home/youyang7/projects/lm-evaluation-harness/lm_eval/models/eval_grid/triviaqa_cot/eval_grid_aggregated.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved aggregated CSV to: {out_csv}")


def _plot_one_task_metric(df: pd.DataFrame, task: str, metric_key: str, title: str, ylabel: str, fname: str):
    sub = df[(df["task"] == task) & (df["metric"] == metric_key)].copy()
    if sub.empty:
        print(f"[skip] No data for {task} / {metric_key}")
        return None

    # Create a line per layer, x=lambda (sorted), y=value, with error bars if available
    plt.figure()
    for layer in sorted(sub["layer"].unique()):
        d = sub[sub["layer"] == layer].sort_values("lambda")
        x = d["lambda"].values
        y = d["value"].values
        yerr = None
        if "stderr" in d and d["stderr"].notna().any():
            yerr = d["stderr"].values
            # Replace NaNs with zeros for plotting error bars
            if isinstance(yerr, np.ndarray):
                yerr = np.nan_to_num(yerr, nan=0.0)

        if yerr is not None:
            plt.errorbar(x, y, yerr=yerr, label=f"L{layer}")
        else:
            plt.plot(x, y, marker="o", label=f"L{layer}")

    plt.xlabel("λ (steer strength; BASELINE=0.0)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Layer", loc="best")
    plt.grid(True, linestyle="--", linewidth=0.5)

    out_path = f"/home/youyang7/projects/lm-evaluation-harness/lm_eval/models/eval_grid/triviaqa_cot/{fname}"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.show()
    print(f"Saved figure: {out_path}")
    return out_path


saved_files = []

# GSM8K
spec = PLOT_SPECS["gsm8k_cot_zeroshot"]
p = _plot_one_task_metric(
    df, "gsm8k_cot_zeroshot", spec["value_key"], spec["title"], spec["ylabel"], "gsm8k_cot_em_vs_lambda.png"
)
if p:
    saved_files.append(p)

# TriviaQA
spec = PLOT_SPECS["triviaqa_cot"]
p = _plot_one_task_metric(
    df, "triviaqa_cot", spec["value_key"], spec["title"], spec["ylabel"], "triviaqa_cot_em_vs_lambda.png"
)
if p:
    saved_files.append(p)

# TruthfulQA (two figures)
for task, val_key, stderr_key, title, ylabel in TQ_TASKS_TWOWAY:
    p = _plot_one_task_metric(
        df, task, val_key, title, ylabel, f"truthfulqa_{val_key.replace(',','_')}_vs_lambda.png"
    )
    if p:
        saved_files.append(p)

print("\nArtifacts:")
for f in saved_files:
    print(" -", f)
