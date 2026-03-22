import os
from typing import Optional

from openai import OpenAI


class ConversationalHCIModel:
    def __init__(self, mode: str = "rule_based", system_prompt: Optional[str] = None) -> None:
        self.mode = mode
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.client = None

        if self.mode == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required when mode='openai'.")
            self.client = OpenAI(api_key=api_key)
            self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, user_utterance: str) -> str:
        if self.mode == "rule_based":
            return self._rule_based_response(user_utterance)

        if self.mode == "openai":
            return self._openai_response(user_utterance)

        raise ValueError(f"Unsupported mode: {self.mode}")

    def _rule_based_response(self, text: str) -> str:
        lowered = text.lower()

        if "password" in lowered:
            return "I can help with password reset. Are you on the app or the website right now?"
        if "appointment" in lowered or "book" in lowered:
            return "I can help schedule that. Which city and preferred time should I use?"
        if "cannot" in lowered or "can't" in lowered:
            return "I understand. Please share what you tried so far, and I will guide the next step."

        return "Thanks for sharing. Could you provide one more detail so I can guide you precisely?"

    def _openai_response(self, user_utterance: str) -> str:
        assert self.client is not None
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_utterance},
            ],
            temperature=0.3,
        )
        content = completion.choices[0].message.content
        return content.strip() if content else ""
