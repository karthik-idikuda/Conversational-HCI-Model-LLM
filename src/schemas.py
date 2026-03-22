from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional


class SchemaValidationError(ValueError):
    pass


@dataclass
class ConversationTurn:
    conversation_id: str
    user_id: str
    session_id: str
    task_id: str
    turn_index: int
    role: str
    utterance: str
    timestamp: datetime
    gold_response: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: dict) -> "ConversationTurn":
        required_fields = [
            "conversation_id",
            "user_id",
            "session_id",
            "task_id",
            "turn_index",
            "role",
            "utterance",
            "timestamp",
        ]

        for key in required_fields:
            if key not in raw:
                raise SchemaValidationError(f"Missing required field: {key}")

        for key in ["conversation_id", "user_id", "session_id", "task_id", "utterance"]:
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError(f"Field '{key}' must be a non-empty string")

        turn_index = raw.get("turn_index")
        if not isinstance(turn_index, int) or turn_index < 1:
            raise SchemaValidationError("Field 'turn_index' must be an integer >= 1")

        role = raw.get("role")
        if role not in {"user", "assistant"}:
            raise SchemaValidationError("Field 'role' must be either 'user' or 'assistant'")

        timestamp_raw = raw.get("timestamp")
        if not isinstance(timestamp_raw, str) or not timestamp_raw.strip():
            raise SchemaValidationError("Field 'timestamp' must be a non-empty ISO 8601 string")

        try:
            timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError(f"Invalid timestamp format: {timestamp_raw}") from exc

        gold_response = raw.get("gold_response")
        if gold_response is not None and not isinstance(gold_response, str):
            raise SchemaValidationError("Field 'gold_response' must be a string when provided")

        return cls(
            conversation_id=raw["conversation_id"],
            user_id=raw["user_id"],
            session_id=raw["session_id"],
            task_id=raw["task_id"],
            turn_index=turn_index,
            role=role,
            utterance=raw["utterance"],
            timestamp=timestamp,
            gold_response=gold_response,
        )

    def to_serializable_dict(self) -> dict:
        out = asdict(self)
        out["timestamp"] = self.timestamp.isoformat()
        return out
