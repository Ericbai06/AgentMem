from __future__ import annotations

import json
import os
from textwrap import dedent
from typing import Iterator, List, Sequence

# 新增：导入 dotenv 以读取 .env 文件
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

class ConversationPreprocessor:
    """Turns raw chat turns into compact, factual memories before storage."""

    def __init__(
        self,
        model: str | None = None,
        max_chunk_chars: int = 1400,
        max_chunk_turns: int = 6,
        temperature: float = 0.2,
        min_fact_words: int = 5,
    ) -> None:
        
        self.client = OpenAI()

        # 修改 2: 优先使用传入的 model，如果没有，则使用环境变量中的 MODEL，最后兜底
        self.model = model or os.getenv("MODEL") or "gpt-4o-mini"
        
        self.max_chunk_chars = max_chunk_chars
        self.max_chunk_turns = max_chunk_turns
        self.temperature = temperature
        self.min_fact_words = min_fact_words
        self.system_prompt = dedent(
            """
            You convert noisy multi-turn transcripts into atomic, factual memories for a single speaker.
            Return JSON with the schema {"facts": ["fact 1", "fact 2"]}.
            Rules:
              - Only capture statements explicitly made by the target speaker.
              - Keep each fact self-contained and reference concrete people, places, and dates.
              - Merge redundant utterances into one canonical statement.
              - Ignore speculation, hedging, or assistant utterances.
              - Prefer chronological phrasing using absolute dates if timestamps are provided.
              - Produce at most four high-quality facts per chunk; skip a chunk if nothing new is learned.
            """
        ).strip()

    def preprocess_messages(
        self,
        messages: Sequence[dict],
        speaker_display_name: str,
        conversation_timestamp: str | None = None,
    ) -> List[dict]:
        if not messages:
            return []

        processed: List[dict] = []
        for chunk in self._chunk_messages(messages):
            chunk_text = self._format_chunk(chunk)
            try:
                facts = self._summarize_chunk(chunk_text, speaker_display_name)
            except Exception as exc:  # noqa: BLE001 - log and fallback gracefully
                print(f"[Preprocessor] Falling back to raw chunk due to error: {exc}")
                return list(messages)

            if not facts:
                continue

            timestamp = chunk[-1].get("chat_time") or conversation_timestamp
            for fact in facts:
                processed.append(
                    {
                        "role": "user",
                        "content": f"{speaker_display_name}: {fact}",
                        "chat_time": timestamp,
                    }
                )

        return processed or list(messages)

    def _chunk_messages(self, messages: Sequence[dict]) -> Iterator[List[dict]]:
        chunk: List[dict] = []
        char_count = 0
        for message in messages:
            content = message.get("content", "")
            if chunk and (
                len(chunk) >= self.max_chunk_turns or char_count + len(content) > self.max_chunk_chars
            ):
                yield chunk
                chunk = []
                char_count = 0
            chunk.append(message)
            char_count += len(content)
        if chunk:
            yield chunk

    def _format_chunk(self, chunk: Sequence[dict]) -> str:
        lines: List[str] = []
        for message in chunk:
            timestamp = message.get("chat_time", "unknown_time")
            role = message.get("role", "user").upper()
            content = message.get("content", "").strip()
            lines.append(f"[{timestamp}] {role}: {content}")
        return "\n".join(lines)

    def _summarize_chunk(self, chunk_text: str, speaker_display_name: str) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": f"{self.system_prompt}\nTarget speaker: {speaker_display_name}.",
            },
            {
                "role": "user",
                "content": chunk_text,
            },
        ]
        
        # 修改 3: 调用时使用 self.model (它现在已经从环境变量中获取了)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content.strip()
        return self._parse_facts(content)

    def _parse_facts(self, content: str) -> List[str]:
        facts: List[str] = []
        try:
            payload = json.loads(content)
            raw_facts = payload.get("facts", []) if isinstance(payload, dict) else []
        except json.JSONDecodeError:
            raw_facts = [line.strip("- ") for line in content.splitlines() if line.strip()]

        for fact in raw_facts:
            fact_clean = fact.strip()
            if len(fact_clean.split()) >= self.min_fact_words:
                facts.append(fact_clean)
        return facts