import json
from pathlib import Path

import pandas as pd


def overlap(a: str, b: str) -> float:
    a_tokens = set(str(a).lower().split())
    b_tokens = set(str(b).lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    responses = pd.read_csv(reports / "generated_responses.csv")

    responses["user_utterance_chars"] = responses["user_utterance"].astype(str).str.len()
    responses["generated_response_chars"] = responses["generated_response"].astype(str).str.len()
    responses["gold_response_chars"] = responses["gold_response"].fillna("").astype(str).str.len()
    responses["has_gold_response"] = responses["gold_response"].notna()

    responses["token_overlap_vs_gold"] = responses.apply(
        lambda r: overlap(r["generated_response"], r["gold_response"])
        if pd.notna(r["gold_response"])
        else None,
        axis=1,
    )

    responses.to_csv(reports / "full_data_for_report.csv", index=False)

    task_summary = (
        responses.groupby("task_id", dropna=False)
        .agg(
            num_samples=("task_id", "count"),
            avg_generated_chars=("generated_response_chars", "mean"),
            avg_overlap_vs_gold=("token_overlap_vs_gold", "mean"),
        )
        .reset_index()
    )
    task_summary.to_csv(reports / "task_level_summary.csv", index=False)

    conv_summary = (
        responses.groupby(["conversation_id", "session_id"], dropna=False)
        .agg(
            num_user_turns=("conversation_id", "count"),
            avg_generated_chars=("generated_response_chars", "mean"),
            avg_overlap_vs_gold=("token_overlap_vs_gold", "mean"),
        )
        .reset_index()
    )
    conv_summary.to_csv(reports / "conversation_level_summary.csv", index=False)

    pack = {
        "dataset_overview": {
            "num_rows": int(len(responses)),
            "num_conversations": int(responses["conversation_id"].nunique()),
            "num_sessions": int(responses["session_id"].nunique()),
            "num_tasks": int(responses["task_id"].nunique()),
            "avg_user_utterance_chars": float(responses["user_utterance_chars"].mean()) if len(responses) else 0.0,
            "avg_generated_response_chars": float(responses["generated_response_chars"].mean()) if len(responses) else 0.0,
            "avg_overlap_vs_gold": float(responses["token_overlap_vs_gold"].dropna().mean())
            if responses["token_overlap_vs_gold"].notna().any()
            else None,
        },
        "task_level_summary": task_summary.to_dict(orient="records"),
        "conversation_level_summary": conv_summary.to_dict(orient="records"),
    }

    with (reports / "report_data_pack.json").open("w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2)

    lines = []
    lines.append("# Conversational HCI Model - Report")
    lines.append("")
    lines.append("## Dataset Overview")
    for key, value in pack["dataset_overview"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Task-Level Summary (CSV)")
    if len(task_summary):
        lines.append(task_summary.to_csv(index=False))
    else:
        lines.append("No task-level rows available.")
    lines.append("")
    lines.append("## Conversation-Level Summary (CSV)")
    if len(conv_summary):
        lines.append(conv_summary.to_csv(index=False))
    else:
        lines.append("No conversation-level rows available.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- reports/full_data_for_report.csv")
    lines.append("- reports/task_level_summary.csv")
    lines.append("- reports/conversation_level_summary.csv")
    lines.append("- reports/report_data_pack.json")
    lines.append("- reports/final_report.md")

    (reports / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Created report artifacts successfully.")


if __name__ == "__main__":
    main()
