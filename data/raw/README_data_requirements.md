Required raw input file:
- conversations.jsonl

Each line in conversations.jsonl must be a JSON object with fields:
- conversation_id (string)
- user_id (string)
- session_id (string)
- task_id (string)
- turn_index (integer, starts from 1)
- role (user or assistant)
- utterance (string)
- timestamp (ISO 8601, example: 2026-03-22T10:00:00Z)
- gold_response (optional string, recommended for evaluation)

Minimum recommendation:
- At least 1,000 user turns across 100+ sessions for baseline experiments.
- Include both successful and unsuccessful interactions.
- Include multilingual examples only if you want multilingual evaluation.

Privacy recommendations:
- Remove names, phone numbers, emails, and IDs.
- Replace sensitive values with placeholders.
