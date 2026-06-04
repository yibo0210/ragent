"""MemoryExtractor: LLM extracts structured memories from conversation."""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.memory.schemas import MemoryNode, MemoryType, MemoryExtraction


_EXTRACTOR_PROMPT = """Analyze the conversation and extract structured memories about the user.

Output JSON:
{
  "memories": [
    {
      "memory_type": "fact|preference|task|relation",
      "content": "what the memory is about",
      "subject": "who/what this memory is about",
      "object_entity": "related entity name (if any)",
      "predicate": "LIKES|WORKED_ON|PREFERS|KNOWS|MENTIONED|ASKED_ABOUT",
      "importance": 0.0-1.0
    }
  ],
  "summary": "one-line summary of key takeaway"
}

Rules:
- fact: user stated a fact (e.g. "I work at Google")
- preference: user expressed preference (e.g. "I prefer Python over Java")
- task: user asked to complete a task (e.g. "analyze Q2 sales")
- relation: user is connected to an entity/person
- Only extract NEW information about the user, not things the AI said
- importance: 0.8+ for explicit statements, 0.5 for implied, <0.3 for trivial
"""


class MemoryExtractor:
    """Extracts structured memories from conversation using LLM."""

    async def extract(
        self,
        messages: list,
        user_id: int = 0,
        tenant_id: int = 0,
        session_id: str = "",
    ) -> MemoryExtraction:
        if not messages:
            return MemoryExtraction()

        recent = messages[-10:]
        conversation = ""
        for msg in recent:
            role = "User" if hasattr(msg, "type") and msg.type == "human" else "Assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)
            conversation += f"{role}: {content}\n"

        try:
            from backend.agent.model_router import get_model_for_agent
            model = get_model_for_agent("supervisor")
            response = await model.ainvoke([
                SystemMessage(content=_EXTRACTOR_PROMPT),
                HumanMessage(content=f"Conversation:\n{conversation[:4000]}\n\nExtract memories:"),
            ])
            content = response.content if hasattr(response, "content") else str(response)
            json_match = re.search(r"\{[\s\S]*\}", content)
            if not json_match:
                return MemoryExtraction()
            data = json.loads(json_match.group(0))

            memories = []
            for item in data.get("memories", []):
                memories.append(MemoryNode(
                    memory_type=MemoryType(item.get("memory_type", "fact")),
                    content=item.get("content", ""),
                    subject=item.get("subject", ""),
                    object_entity=item.get("object_entity", ""),
                    predicate=item.get("predicate", ""),
                    importance=float(item.get("importance", 0.5)),
                    session_id=session_id, tenant_id=tenant_id, user_id=user_id,
                ))
            return MemoryExtraction(memories=memories, summary=data.get("summary", ""))
        except Exception:
            return MemoryExtraction()


_extractor: MemoryExtractor | None = None


def get_memory_extractor() -> MemoryExtractor:
    global _extractor
    if _extractor is None:
        _extractor = MemoryExtractor()
    return _extractor
