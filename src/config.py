from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    root_dir: Path = Path(__file__).resolve().parent.parent
    raw_data_dir: Path = root_dir / "data" / "raw"
    processed_data_dir: Path = root_dir / "data" / "processed"
    reports_dir: Path = root_dir / "reports"
    prompts_dir: Path = root_dir / "prompts"

    raw_conversations_file: Path = raw_data_dir / "conversations.jsonl"
    validated_conversations_file: Path = processed_data_dir / "validated_conversations.jsonl"
    generated_responses_file: Path = reports_dir / "generated_responses.csv"
    summary_report_file: Path = reports_dir / "summary_metrics.json"
    system_prompt_file: Path = prompts_dir / "system_prompt.txt"


CONFIG = ProjectConfig()
