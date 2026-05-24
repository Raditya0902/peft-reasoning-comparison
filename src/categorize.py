"""Assign difficulty or domain category labels to dataset examples."""

import re

# All patterns are matched against lowercased question text.

_ARITHMETIC: list[re.Pattern[str]] = [
    re.compile(r"\bbuys?\b"),
    re.compile(r"\badds?\b"),
    re.compile(r"\bsubtract\w*\b"),
    re.compile(r"\bmultipli\w*\b"),
    re.compile(r"\bdivid\w*\b"),
    re.compile(r"\bgains?\b"),
    re.compile(r"\bloses?\b"),
    re.compile(r"\bgave?\s+away\b"),
    re.compile(r"\btook?\s+away\b"),
    re.compile(r"\bhow\s+many\b"),
    re.compile(r"\btotal\b"),
    re.compile(r"\bsum\b"),
    re.compile(r"\bremaining\b"),
    re.compile(r"\bleft\s+over\b"),
]

_FRACTIONS_PERCENTAGES: list[re.Pattern[str]] = [
    re.compile(r"\d+%"),
    re.compile(r"\bpercent(?:age)?\b"),
    re.compile(r"\bfraction\b"),
    re.compile(r"\bhalf\b"),
    re.compile(r"\bquarter\b"),
    re.compile(r"\bthird\b"),
    re.compile(r"\bratio\b"),
    re.compile(r"\bout\s+of\b"),
]

_UNIT_CONVERSION: list[re.Pattern[str]] = [
    re.compile(r"\bconvert\b"),
    re.compile(
        r"\bto\s+(?:minutes?|hours?|seconds?|days?|weeks?"
        r"|meters?|kilometers?|km|miles?|feet|celsius|fahrenheit)\b"
    ),
]

_MULTI_HOP: list[re.Pattern[str]] = [
    re.compile(r"\bfirst\b.{1,150}\bthen\b", re.DOTALL),
    re.compile(r"\bthen\b.{1,150}\bthen\b", re.DOTALL),
    re.compile(r"\bafter\s+that\b"),
    re.compile(r"\bfinally\b"),
    re.compile(r"\bstep\s+\d\b"),
]

_ALGEBRAIC: list[re.Pattern[str]] = [
    re.compile(r"\bsolve\s+for\b"),
    re.compile(r"\b(?:x|y|z|n|m)\s+if\b"),
    re.compile(r"\b\d+\s*(?:x|y|z|n|m)\s*[+\-\*=]"),
    re.compile(r"\bwhat\s+is\s+(?:x|y|z|n|m)\b"),
    re.compile(r"\bequation\b"),
    re.compile(r"\bvariable\b"),
]

_COMPARISON: list[re.Pattern[str]] = [
    re.compile(r"\bmore\s+than\b"),
    re.compile(r"\bless\s+than\b"),
    re.compile(r"\bgreater\s+than\b"),
    re.compile(r"\bfewer\s+than\b"),
    re.compile(r"\bwho\s+has\s+(?:more|fewer)\b"),
    re.compile(r"\bwhich\s+is\s+(?:more|greater|larger|bigger|smaller|less)\b"),
    re.compile(r"\bcompared?\s+to\b"),
]

# Distractor-heavy: a color/descriptor word appears alongside ≥2 numbers,
# suggesting enumerated context rather than a focused arithmetic operation.
_DISTRACTOR_DESCRIPTOR = re.compile(
    r"\b(?:red|blue|green|yellow|black|white|orange|purple|brown|pink|grey|gray)\b"
)


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(text) for p in patterns)


def _is_distractor_heavy(text: str) -> bool:
    return (
        bool(_DISTRACTOR_DESCRIPTOR.search(text))
        and len(re.findall(r"\d+", text)) >= 2
    )


def categorize(question: str) -> list[str]:
    """Return zero or more category tags for a question via keyword matching."""
    lowered = question.lower()
    tags: list[str] = []
    if _matches_any(lowered, _ARITHMETIC):
        tags.append("arithmetic")
    if _matches_any(lowered, _FRACTIONS_PERCENTAGES):
        tags.append("fractions_percentages")
    if _matches_any(lowered, _UNIT_CONVERSION):
        tags.append("unit_conversion")
    if _matches_any(lowered, _MULTI_HOP):
        tags.append("multi_hop")
    if _matches_any(lowered, _ALGEBRAIC):
        tags.append("algebraic")
    if _matches_any(lowered, _COMPARISON):
        tags.append("comparison")
    if _is_distractor_heavy(lowered):
        tags.append("distractor_heavy")
    return tags
