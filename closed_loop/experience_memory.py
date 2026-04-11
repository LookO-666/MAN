"""Experience memory: store, structure, retrieve, and summarize lessons from past experiments."""

import json
import os
import re
from collections import Counter


class ExperienceMemory:
    def __init__(self, save_path="experience_memory.json"):
        self.save_path = save_path
        self.entries = []
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                self.entries = json.load(f)

    def add_entry(self, entry: dict):
        """
        Add an experience record and normalize it to a richer schema.

        Canonical entry format:
        {
            "round": 3,
            "idea_name": "Width x2",
            "idea_description": "Double channel width in ResNet18",
            "outcome": "rejected",   # "rejected" / "success" / "failure"
            "stage": "mini" | "full",
            "signals": { ... },
            "lesson": "Increasing width alone without regularization causes overfitting",
            "reusable_components": "Width increase can be combined with dropout/weight decay",
            "failure_type": "overfitting",
            "confidence": "medium",
            "mechanism_hypothesis": "Capacity grows faster than regularization strength",
            "avoid_pattern": "Do not increase width alone without stronger regularization",
            "decision": "REFINE" | "DISCARD" | "N/A",
            "decision_reason": "Idea shows clear learning signal but weak generalization, worth refinement.",
            "components": ["width", "regularization"],
            "salvageable_parts": ["width increase"],
            "context": {
                "dataset": "CIFAR-10",
                "model": "ResNet-18",
                "stage": "mini"
            }
        }
        """
        normalized = self._normalize_entry(entry)
        self.entries.append(normalized)
        self.save()

    def get_relevant_lessons(self, current_task_description: str, top_k: int = 5) -> list:
        """Return the most relevant experience entries for the current task."""
        if not self.entries:
            return []

        query_text = current_task_description or ""
        query_tokens = self._tokenize(query_text)
        scored_entries = []

        for idx, entry in enumerate(self.entries):
            score = self._score_entry(entry, query_tokens)
            # Prefer recent items slightly when scores tie
            recency_bonus = idx / max(len(self.entries), 1) * 0.05
            scored_entries.append((score + recency_bonus, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for score, entry in scored_entries[:top_k] if score > 0 or not query_tokens]

    def build_generation_brief(self, current_task_description: str, top_k: int = 5) -> dict:
        """Build structured memory brief for idea generation prompts."""
        lessons = self.get_relevant_lessons(current_task_description, top_k=top_k)
        constraints = []
        reusable_components = []
        open_hypotheses = []

        for entry in lessons:
            avoid_pattern = entry.get("avoid_pattern", "")
            if avoid_pattern and avoid_pattern not in constraints:
                constraints.append(avoid_pattern)

            for component in entry.get("salvageable_parts", []):
                if component and component not in reusable_components:
                    reusable_components.append(component)

            hypothesis = entry.get("mechanism_hypothesis", "")
            if hypothesis and hypothesis not in open_hypotheses:
                open_hypotheses.append(hypothesis)

        return {
            "lessons": lessons,
            "constraints": constraints[:5],
            "reusable_components": reusable_components[:5],
            "open_hypotheses": open_hypotheses[:5],
            "failure_patterns": self.summarize_failure_patterns(top_k=3),
        }

    def summarize_failure_patterns(self, top_k: int = 3) -> list:
        """Summarize dominant failure patterns across stored entries."""
        failures = [e for e in self.entries if e.get("outcome") in ("rejected", "failure")]
        if not failures:
            return []

        counter = Counter()
        pattern_examples = {}
        for entry in failures:
            failure_type = entry.get("failure_type", "unknown")
            counter[failure_type] += 1
            pattern_examples.setdefault(failure_type, entry)

        summaries = []
        for failure_type, count in counter.most_common(top_k):
            example = pattern_examples[failure_type]
            mechanism = example.get("mechanism_hypothesis", "")
            avoid_pattern = example.get("avoid_pattern", "")
            parts = [f"{failure_type} ({count}x)"]
            if mechanism:
                parts.append(f"mechanism: {mechanism}")
            if avoid_pattern:
                parts.append(f"avoid: {avoid_pattern}")
            summaries.append(" | ".join(parts))
        return summaries

    def get_ideas_to_refine(self, latest_round_only: bool = True, top_k: int = 3) -> list:
        """Return ideas recently marked as REFINE, ranked by confidence and recency."""
        if not self.entries:
            return []

        refine_entries = [
            entry for entry in self.entries
            if entry.get("decision") == "REFINE"
        ]
        if not refine_entries:
            return []

        if latest_round_only:
            latest_round = max(entry.get("round", 0) for entry in refine_entries)
            refine_entries = [
                entry for entry in refine_entries
                if entry.get("round", 0) == latest_round
            ]

        confidence_order = {"high": 3, "medium": 2, "low": 1}
        stage_order = {"full": 2, "mini": 1}

        refine_entries.sort(
            key=lambda entry: (
                confidence_order.get(entry.get("confidence", "low"), 0),
                stage_order.get(entry.get("stage", "mini"), 0),
                entry.get("round", 0),
            ),
            reverse=True,
        )

        results = []
        seen_names = set()
        for entry in refine_entries:
            idea_name = entry.get("idea_name", "")
            if not idea_name or idea_name in seen_names:
                continue
            results.append({
                "idea_name": idea_name,
                "idea_description": entry.get("idea_description", ""),
                "category": self._infer_category(entry),
                "round": entry.get("round", 0),
                "stage": entry.get("stage", "mini"),
                "decision": entry.get("decision", "REFINE"),
                "decision_reason": entry.get("decision_reason", ""),
                "failure_type": entry.get("failure_type", ""),
                "mechanism_hypothesis": entry.get("mechanism_hypothesis", ""),
                "signals": entry.get("signals", {}),
                "lesson": entry.get("lesson", ""),
                "reusable_components": entry.get("reusable_components", ""),
                "salvageable_parts": entry.get("salvageable_parts", []),
                "confidence": entry.get("confidence", "low"),
            })
            seen_names.add(idea_name)
            if len(results) >= top_k:
                break
        return results

    def format_for_prompt(self, lessons: list) -> str:
        """Format lessons into text suitable for LLM prompts."""
        if not lessons:
            return "暂无历史经验。"
        lines = []
        for i, entry in enumerate(lessons, 1):
            outcome_map = {
                "rejected": "❌ 被淘汰",
                "success": "✅ 成功",
                "failure": "💥 执行失败",
            }
            outcome_str = outcome_map.get(entry.get("outcome", ""), entry.get("outcome", ""))
            stage = entry.get("stage", "unknown")
            failure_type = entry.get("failure_type", "unknown")
            confidence = entry.get("confidence", "medium")
            mechanism = entry.get("mechanism_hypothesis", "")
            avoid_pattern = entry.get("avoid_pattern", "")
            decision = entry.get("decision", "N/A")
            decision_reason = entry.get("decision_reason", "")
            components = ", ".join(entry.get("components", [])) or "无"
            salvageable = ", ".join(entry.get("salvageable_parts", [])) or entry.get("reusable_components", "无") or "无"
            lines.append(
                f"{i}. [{outcome_str}] {entry.get('idea_name', '?')} (Round {entry.get('round', '?')}, stage={stage})\n"
                f"   描述: {entry.get('idea_description', '')}\n"
                f"   教训: {entry.get('lesson', '')}\n"
                f"   失败类型: {failure_type} | 置信度: {confidence}\n"
                f"   机制假设: {mechanism}\n"
                f"   避免模式: {avoid_pattern}\n"
                f"   决策: {decision}\n"
                f"   决策原因: {decision_reason}\n"
                f"   组件: {components}\n"
                f"   可复用: {salvageable}"
            )
        return "\n".join(lines)

    def format_generation_guidance(self, brief: dict) -> str:
        """Format structured memory into constraints for idea generation."""
        if not brief or not any(brief.values()):
            return "暂无额外约束。"

        lines = []
        constraints = brief.get("constraints", [])
        reusable_components = brief.get("reusable_components", [])
        open_hypotheses = brief.get("open_hypotheses", [])
        failure_patterns = brief.get("failure_patterns", [])

        if constraints:
            lines.append("禁止或谨慎重复的模式:")
            for item in constraints:
                lines.append(f"- {item}")

        if reusable_components:
            lines.append("优先考虑复用的组件:")
            for item in reusable_components:
                lines.append(f"- {item}")

        if open_hypotheses:
            lines.append("值得验证的机制假设:")
            for item in open_hypotheses:
                lines.append(f"- {item}")

        if failure_patterns:
            lines.append("近期高频失败模式:")
            for item in failure_patterns:
                lines.append(f"- {item}")

        return "\n".join(lines) if lines else "暂无额外约束。"

    def save(self):
        """Persist to JSON file."""
        with open(self.save_path, "w") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

    def _normalize_entry(self, entry: dict) -> dict:
        normalized = dict(entry)
        normalized.setdefault("stage", normalized.get("context", {}).get("stage", "full"))
        normalized.setdefault("signals", {})
        normalized.setdefault("lesson", "")
        normalized.setdefault("reusable_components", "")
        normalized.setdefault("context", {})
        normalized.setdefault("decision", self._infer_decision(normalized))
        normalized.setdefault("decision_reason", self._infer_decision_reason(normalized))

        context = dict(normalized.get("context", {}))
        context.setdefault("stage", normalized.get("stage", "full"))
        context.setdefault("dataset", "CIFAR-10")
        context.setdefault("model", "ResNet-18")
        normalized["context"] = context

        components = normalized.get("components")
        if not components:
            components = self._extract_components(
                normalized.get("idea_name", "") + " " + normalized.get("idea_description", "")
            )
        normalized["components"] = components

        normalized.setdefault("failure_type", self._infer_failure_type(normalized))
        normalized.setdefault("confidence", self._infer_confidence(normalized))
        normalized.setdefault("mechanism_hypothesis", self._infer_mechanism(normalized))
        normalized.setdefault("avoid_pattern", self._infer_avoid_pattern(normalized))

        salvageable = normalized.get("salvageable_parts")
        if not salvageable:
            salvageable = self._extract_salvageable_parts(normalized)
        normalized["salvageable_parts"] = salvageable

        # Re-run decision defaults after failure type / salvageables are available.
        if not normalized.get("decision"):
            normalized["decision"] = self._infer_decision(normalized)
        if not normalized.get("decision_reason"):
            normalized["decision_reason"] = self._infer_decision_reason(normalized)

        return normalized

    def _score_entry(self, entry: dict, query_tokens: set) -> float:
        if not query_tokens:
            return 1.0

        text_fields = [
            entry.get("idea_name", ""),
            entry.get("idea_description", ""),
            entry.get("lesson", ""),
            entry.get("mechanism_hypothesis", ""),
            entry.get("avoid_pattern", ""),
            entry.get("decision_reason", ""),
            " ".join(entry.get("components", [])),
            " ".join(entry.get("salvageable_parts", [])),
        ]
        entry_tokens = self._tokenize(" ".join(text_fields))
        overlap = len(query_tokens & entry_tokens)

        score = float(overlap)
        if entry.get("outcome") in ("rejected", "failure"):
            score += 0.5
        if entry.get("outcome") == "success":
            score += 0.2
        if entry.get("decision") == "REFINE":
            score += 0.3
        return score

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9_\-\.]+", (text or "").lower()))

    def _extract_components(self, text: str) -> list:
        text_lower = (text or "").lower()
        keywords = {
            "augmentation": ["mixup", "cutmix", "cutout", "augmentation", "randaugment", "autoaugment"],
            "regularization": ["dropout", "weight decay", "label smoothing", "regularization", "stochastic depth"],
            "optimization": ["optimizer", "sgd", "adam", "momentum", "lr", "learning rate", "warmup"],
            "scheduler": ["cosine", "scheduler", "anneal", "decay"],
            "architecture": ["resnet", "width", "depth", "block", "layer", "attention"],
            "loss": ["loss", "focal", "distillation", "contrastive"],
        }
        matched = []
        for name, tokens in keywords.items():
            if any(token in text_lower for token in tokens):
                matched.append(name)
        return matched or ["misc"]

    def _infer_failure_type(self, entry: dict) -> str:
        outcome = entry.get("outcome", "")
        lesson = (entry.get("lesson", "") or "").lower()
        signals = entry.get("signals", {}) or {}

        if outcome == "success":
            return "successful_pattern"
        if "error" in lesson or "crash" in lesson or "exception" in lesson:
            return "implementation_bug"
        if signals.get("gap_trend", 0) > 5 or "overfit" in lesson:
            return "overfitting"
        if signals.get("train_loss_slope", 0) > -0.01 and signals.get("val_acc_delta", 0) <= 0:
            return "no_effect"
        if signals.get("val_loss_var", 0) > 0.05 or "unstable" in lesson:
            return "optimization_instability"
        if outcome == "rejected":
            return "early_reject"
        return "underperformance"

    def _infer_confidence(self, entry: dict) -> str:
        stage = entry.get("stage", "full")
        outcome = entry.get("outcome", "")
        if stage == "full" and outcome in ("success", "failure"):
            return "high"
        if stage == "mini" and outcome == "rejected":
            return "medium"
        return "low"

    def _infer_mechanism(self, entry: dict) -> str:
        if entry.get("mechanism_hypothesis"):
            return entry["mechanism_hypothesis"]

        failure_type = entry.get("failure_type", "")
        components = entry.get("components", [])
        if failure_type == "overfitting":
            return "Model capacity or memorization pressure likely exceeded regularization strength."
        if failure_type == "optimization_instability":
            return "Training dynamics are noisy; optimization or augmentation strength may be too aggressive."
        if failure_type == "no_effect":
            return "The modification likely changed too little in the effective training dynamics."
        if failure_type == "early_reject":
            return "The idea underperformed in early screening and may require better pairing or longer training."
        if "architecture" in components and "regularization" not in components:
            return "Architecture change may need stronger regularization or schedule adaptation."
        return "Outcome likely depends on interaction between components rather than a single factor."

    def _infer_avoid_pattern(self, entry: dict) -> str:
        if entry.get("avoid_pattern"):
            return entry["avoid_pattern"]

        failure_type = entry.get("failure_type", "")
        components = entry.get("components", [])
        comp_text = ", ".join(components)
        if failure_type == "overfitting":
            return f"Avoid repeating {comp_text} changes without adding stronger generalization control."
        if failure_type == "optimization_instability":
            return f"Avoid stacking aggressive {comp_text} changes without stabilizing optimization."
        if failure_type == "no_effect":
            return f"Avoid near-no-op {comp_text} modifications that do not materially change training dynamics."
        if failure_type == "early_reject":
            return f"Treat isolated {comp_text} changes as high risk unless paired with complementary components."
        return f"Do not repeat the same {comp_text} recipe without changing its surrounding assumptions."

    def _infer_decision(self, entry: dict) -> str:
        if entry.get("decision"):
            return entry["decision"]

        outcome = entry.get("outcome", "")
        if outcome == "success":
            return "N/A"

        failure_type = entry.get("failure_type") or self._infer_failure_type(entry)
        signals = entry.get("signals", {}) or {}
        gap_trend = signals.get("gap_trend", 0)
        train_loss_slope = signals.get("train_loss_slope", 0)
        val_loss_var = signals.get("val_loss_var", 0)
        train_val_gap = signals.get("train_val_gap", 0)

        if failure_type == "overfitting":
            return "REFINE"
        if failure_type == "optimization_instability":
            return "REFINE"
        if failure_type == "no_effect":
            return "DISCARD"
        if gap_trend > 2 and train_loss_slope < -0.03:
            return "REFINE"
        if abs(train_val_gap) < 3 and val_loss_var < 0.01 and outcome in ("rejected", "failure"):
            return "REFINE"
        return "DISCARD"

    def _infer_decision_reason(self, entry: dict) -> str:
        if entry.get("decision_reason"):
            return entry["decision_reason"]

        decision = entry.get("decision") or self._infer_decision(entry)
        signals = entry.get("signals", {}) or {}
        gap_trend = signals.get("gap_trend", 0)
        train_loss_slope = signals.get("train_loss_slope", 0)
        val_loss_var = signals.get("val_loss_var", 0)
        val_acc_delta = signals.get("val_acc_delta", 0)
        train_val_gap = signals.get("train_val_gap", 0)

        if decision == "N/A":
            return "This idea already succeeded and does not need triage."
        if decision == "REFINE" and gap_trend > 2 and train_loss_slope < -0.03:
            return (
                f"train_loss_slope={train_loss_slope:.4f} is strong but gap_trend={gap_trend:.2f} is rising, "
                "suggesting overfitting with learnable signal; refine with stronger generalization control."
            )
        if decision == "REFINE" and abs(train_val_gap) < 3 and val_loss_var < 0.01:
            return (
                f"train_val_gap={train_val_gap:.2f} and val_loss_var={val_loss_var:.4f} are both small while "
                f"val_acc_delta={val_acc_delta:.2f}, indicating a stable but slow-learning idea worth refinement."
            )
        if decision == "REFINE" and val_loss_var > 0.05:
            return (
                f"val_loss_var={val_loss_var:.4f} is high, implying unstable but potentially salvageable dynamics; "
                "refine the optimizer or stabilization strategy."
            )
        return (
            f"val_acc_delta={val_acc_delta:.2f}, train_loss_slope={train_loss_slope:.4f}, and "
            f"gap_trend={gap_trend:.2f} do not show enough recoverable signal, so this direction should be discarded."
        )

    def _extract_salvageable_parts(self, entry: dict) -> list:
        reusable = entry.get("reusable_components", "") or ""
        if reusable:
            parts = [p.strip() for p in re.split(r"[,;/]| and ", reusable) if p.strip()]
            if parts:
                return parts[:5]
        if entry.get("outcome") == "success":
            return entry.get("components", [])[:3]
        if entry.get("failure_type") in ("early_reject", "optimization_instability"):
            return entry.get("components", [])[:2]
        return []

    def _infer_category(self, entry: dict) -> str:
        components = entry.get("components", [])
        if "augmentation" in components:
            return "数据增强"
        if "regularization" in components:
            return "正则化"
        if "optimization" in components or "scheduler" in components:
            return "学习率策略"
        if "architecture" in components:
            return "架构修改"
        if "loss" in components:
            return "损失函数"
        return "其他"
