from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class RedactionRule:
    name: str
    pattern: re.Pattern[str]
    replacement: str


def default_redaction_rules() -> List[RedactionRule]:
    # Conservative defaults: remove obvious PII-like strings, keep domain terms.
    return [
        RedactionRule(
            name="email",
            pattern=re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
            replacement="<EMAIL>",
        ),
        RedactionRule(
            name="phone",
            pattern=re.compile(r"(?:(?<=\D)|^)(?:\+?\d[\d\-\s()]{8,}\d)(?:(?=\D)|$)"),
            replacement="<PHONE>",
        ),
        RedactionRule(
            name="url",
            pattern=re.compile(r"(?i)\bhttps?://[^\s)>\"]+"),
            replacement="<URL>",
        ),
        RedactionRule(
            name="obsidian_vault_paths",
            pattern=re.compile(r"(?i)(?:/Users/[^\\s]+/|C:\\\\Users\\\\[^\\s]+\\\\)[^\\s]+"),
            replacement="<PATH>",
        ),
    ]


def redact_text(text: str, rules: Iterable[RedactionRule]) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Returns (redacted_text, counts_per_rule).
    counts_per_rule: list of (rule_name, replacements_count)
    """
    counts: List[Tuple[str, int]] = []
    out = text
    for rule in rules:
        out, n = rule.pattern.subn(rule.replacement, out)
        counts.append((rule.name, n))
    return out, counts

