import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def token_overlap_score(a: str, b: str) -> float:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def build_metrics(df: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "num_user_turns": float(len(df)),
        "avg_generated_chars": float(df["generated_response"].str.len().mean() if len(df) else 0.0),
    }

    if "gold_response" in df.columns and df["gold_response"].notna().any():
        valid = df[df["gold_response"].notna()].copy()
        valid["overlap"] = valid.apply(
            lambda row: token_overlap_score(str(row["generated_response"]), str(row["gold_response"])),
            axis=1,
        )
        metrics["avg_token_overlap_vs_gold"] = float(np.mean(valid["overlap"]))

    return metrics


def write_metrics(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
