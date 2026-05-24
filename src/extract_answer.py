"""Parse final answers from model-generated reasoning chains."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionResult:
    value: int | None
    extraction_failure: bool


# Ordered by specificity: GSM8K boxed format first, then natural-language markers.
# No fallback to bare numbers — unmarked digits in running text are not answers.
_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"####\s*([-]?\$?[\d,]+)"),
    re.compile(r"(?i)(?:the\s+)?answer\s+is\s+([-]?\$?[\d,]+)"),
    re.compile(r"(?i)final\s+answer\s*[:\s]\s*([-]?\$?[\d,]+)"),
    re.compile(r"(?i)therefore\s*[:\s]\s*([-]?\$?[\d,]+)"),
]


def _to_int(raw: str) -> int:
    return int(raw.replace("$", "").replace(",", ""))


def extract_answer(text: str) -> ExtractionResult:
    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            return ExtractionResult(value=_to_int(match.group(1)), extraction_failure=False)
    return ExtractionResult(value=None, extraction_failure=True)
