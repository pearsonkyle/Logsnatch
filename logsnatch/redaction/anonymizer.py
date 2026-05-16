import hashlib
import os
import re
from pathlib import Path


class Anonymizer:
    def __init__(self, extra_usernames: list[str] | None = None):
        self._username = os.environ.get("USER") or os.environ.get("USERNAME") or ""
        self._home = str(Path.home())
        self._extra = extra_usernames or []
        self._cache: dict[str, str] = {}

    def _hash_username(self, username: str) -> str:
        h = hashlib.sha256(username.encode()).hexdigest()[:8]
        return f"user_{h}"

    def text(self, s: str) -> str:
        """Apply all anonymization to a text string."""
        if not s:
            return s
        if self._home and len(self._home) > 3:
            s = s.replace(self._home, "/home/REDACTED_USER")
        if self._username and len(self._username) >= 4:
            anon = self._hash_username(self._username)
            s = re.sub(re.escape(self._username), anon, s)
        for u in self._extra:
            if u and len(u) >= 4:
                anon = self._hash_username(u)
                s = re.sub(re.escape(u), anon, s)
        return s

    def path(self, s: str) -> str:
        """Anonymize a file path."""
        return self.text(s)
