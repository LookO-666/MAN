"""Experience memory: store and retrieve lessons from past experiments."""

import json
import os


class ExperienceMemory:
    def __init__(self, save_path="experience_memory.json"):
        self.save_path = save_path
        self.entries = []
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                self.entries = json.load(f)

    def add_entry(self, entry: dict):
        """
        Add an experience record.

        entry format:
        {
            "round": 3,
            "idea_name": "Width x2",
            "idea_description": "Double channel width in ResNet18",
            "outcome": "rejected",   # "rejected" / "success" / "failure"
            "signals": { ... },
            "lesson": "Increasing width alone without regularization causes overfitting",
            "reusable_components": "Width increase can be combined with dropout/weight decay"
        }
        """
        self.entries.append(entry)
        self.save()

    def get_relevant_lessons(self, current_task_description: str, top_k: int = 5) -> list:
        """Return the most recent top_k experience entries."""
        return self.entries[-top_k:]

    def format_for_prompt(self, lessons: list) -> str:
        """Format lessons into text suitable for LLM prompts."""
        if not lessons:
            return "暂无历史经验。"
        lines = []
        for i, entry in enumerate(lessons, 1):
            outcome_map = {"rejected": "❌ 被淘汰", "success": "✅ 成功", "failure": "💥 执行失败"}
            outcome_str = outcome_map.get(entry.get("outcome", ""), entry.get("outcome", ""))
            lines.append(
                f"{i}. [{outcome_str}] {entry.get('idea_name', '?')} "
                f"(Round {entry.get('round', '?')})\n"
                f"   描述: {entry.get('idea_description', '')}\n"
                f"   教训: {entry.get('lesson', '')}\n"
                f"   可复用: {entry.get('reusable_components', '')}"
            )
        return "\n".join(lines)

    def save(self):
        """Persist to JSON file."""
        with open(self.save_path, "w") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)
