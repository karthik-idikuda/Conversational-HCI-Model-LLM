import json
from pathlib import Path
from typing import List

from src.schemas import ConversationTurn, SchemaValidationError


class DataValidationError(Exception):
    pass


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return records


def validate_conversations(records: List[dict]) -> List[ConversationTurn]:
    valid_rows: List[ConversationTurn] = []
    errors = []

    for idx, record in enumerate(records, start=1):
        try:
            valid_rows.append(ConversationTurn.from_dict(record))
        except SchemaValidationError as exc:
            errors.append(f"Row {idx}: {exc}")

    if errors:
        joined = "\n".join(errors[:10])
        raise DataValidationError(
            f"Validation failed for {len(errors)} row(s). First errors:\n{joined}"
        )

    return valid_rows


def write_jsonl(path: Path, rows: List[ConversationTurn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_serializable_dict(), ensure_ascii=False))
            f.write("\n")
