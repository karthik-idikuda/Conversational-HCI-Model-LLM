import argparse

import pandas as pd
from dotenv import load_dotenv

from src.config import CONFIG
from src.data_io import load_jsonl, validate_conversations, write_jsonl
from src.evaluation import build_metrics, write_metrics
from src.model import ConversationalHCIModel


def load_system_prompt() -> str:
    if CONFIG.system_prompt_file.exists():
        return CONFIG.system_prompt_file.read_text(encoding="utf-8").strip()
    return "You are a helpful assistant."


def run(mode: str = "rule_based", max_turns: int = 200) -> None:
    load_dotenv()
    system_prompt = load_system_prompt()

    records = load_jsonl(CONFIG.raw_conversations_file)
    validated = validate_conversations(records)
    write_jsonl(CONFIG.validated_conversations_file, validated)

    model = ConversationalHCIModel(mode=mode, system_prompt=system_prompt)

    rows = []
    for turn in validated:
        if turn.role != "user":
            continue

        rows.append(
            {
                "conversation_id": turn.conversation_id,
                "session_id": turn.session_id,
                "task_id": turn.task_id,
                "turn_index": turn.turn_index,
                "user_utterance": turn.utterance,
                "gold_response": turn.gold_response,
                "generated_response": model.generate(turn.utterance),
            }
        )

        if len(rows) >= max_turns:
            break

    out_df = pd.DataFrame(rows)
    CONFIG.reports_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(CONFIG.generated_responses_file, index=False)

    metrics = build_metrics(out_df)
    write_metrics(metrics, CONFIG.summary_report_file)

    print(f"Validated records saved to: {CONFIG.validated_conversations_file}")
    print(f"Generated responses saved to: {CONFIG.generated_responses_file}")
    print(f"Summary metrics saved to: {CONFIG.summary_report_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conversational HCI model pipeline")
    parser.add_argument(
        "--mode",
        default="rule_based",
        choices=["rule_based", "openai"],
        help="Model mode. Use 'openai' if OPENAI_API_KEY is configured.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=200,
        help="Maximum number of user turns to process.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(mode=args.mode, max_turns=args.max_turns)
