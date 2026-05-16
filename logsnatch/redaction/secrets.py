import math
import re
from dataclasses import dataclass


@dataclass
class Finding:
    pattern_name: str
    matched_text: str
    redacted_text: str


PATTERNS = [
    (
        "jwt_token",
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ),
    ("anthropic_api_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    (
        "openai_api_key",
        re.compile(r"sk-[A-Za-z0-9]{20,}(?![A-Za-z0-9_-])"),
    ),
    ("google_api_key", re.compile(r"AIzaSy[A-Za-z0-9_-]{33}")),
    ("github_token", re.compile(r"gh[posr]_[A-Za-z0-9]{36}")),
    ("huggingface_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("aws_access_key", re.compile(r"AKIA[A-Z0-9]{16}")),
    ("slack_token", re.compile(r"xox[bpsa]-[A-Za-z0-9-]{10,}")),
    (
        "bearer_token",
        re.compile(r"Bearer\s+[A-Za-z0-9._\-+/]{20,}", re.IGNORECASE),
    ),
    (
        "db_url",
        re.compile(r"(?:postgres|mysql|mongodb|redis)://[^:]+:[^@\s]+@[^\s]+"),
    ),
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----.*?"
            r"-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
    ),
    (
        "env_secret",
        re.compile(
            r"(?:SECRET|KEY|TOKEN|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
            r"(?:_[A-Z_]+)?\s*=\s*[\"']?[A-Za-z0-9+/=_.\-]{8,}[\"']?",
            re.IGNORECASE,
        ),
    ),
]

ALLOWLIST = [
    re.compile(r"noreply@"),
    re.compile(r"@example\.(com|org|net)$"),
    re.compile(r"@github\.com$"),
    re.compile(r"@users\.noreply\.github\.com$"),
    re.compile(r"email@"),
    re.compile(r"test@"),
    re.compile(r"user@"),
    re.compile(r"example@"),
]


def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def _is_allowlisted(text: str) -> bool:
    for pattern in ALLOWLIST:
        if pattern.search(text):
            return True
    return False


def scan_text(text: str) -> list[Finding]:
    """Scan text and return a list of findings."""
    findings = []
    for name, pattern in PATTERNS:
        for match in pattern.finditer(text):
            matched = match.group(0)
            if name == "email" and _is_allowlisted(matched):
                continue
            if name == "env_secret":
                value_match = re.search(
                    r'=\s*["\']?([A-Za-z0-9+/=_.\-]{8,})["\']?', matched
                )
                if value_match:
                    val = value_match.group(1)
                    if _shannon_entropy(val) < 3.5:
                        continue
            findings.append(
                Finding(
                    pattern_name=name,
                    matched_text=matched,
                    redacted_text=f"[REDACTED_{name.upper()}]",
                )
            )
    return findings


def redact_text(text: str) -> tuple[str, int]:
    """Redact secrets from text. Returns (redacted_text, count).

    Findings are applied from right to left by position to avoid offset
    issues caused by earlier replacements changing string length.
    """
    # Collect all match positions with their replacement strings
    replacements: list[tuple[int, int, str]] = []
    for name, pattern in PATTERNS:
        for match in pattern.finditer(text):
            matched = match.group(0)
            if name == "email" and _is_allowlisted(matched):
                continue
            if name == "env_secret":
                value_match = re.search(
                    r'=\s*["\']?([A-Za-z0-9+/=_.\-]{8,})["\']?', matched
                )
                if value_match:
                    val = value_match.group(1)
                    if _shannon_entropy(val) < 3.5:
                        continue
            replacements.append(
                (match.start(), match.end(), f"[REDACTED_{name.upper()}]")
            )

    if not replacements:
        return text, 0

    # Sort by start position descending so later matches don't shift earlier ones
    replacements.sort(key=lambda r: r[0], reverse=True)

    # Deduplicate overlapping matches (keep the rightmost/outermost)
    deduped: list[tuple[int, int, str]] = []
    last_start = len(text)
    for start, end, replacement in replacements:
        if end <= last_start:
            deduped.append((start, end, replacement))
            last_start = start

    result = list(text)
    for start, end, replacement in deduped:
        result[start:end] = list(replacement)

    return "".join(result), len(deduped)
