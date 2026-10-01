"""Download the common-password blocklist used by the password policy (spec §7.6).

The list is data, so it stays local (data/security is not in git). Source: SecLists, the
10,000 most common passwords.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/"
       "10k-most-common.txt")
OUT = Path(__file__).resolve().parents[1] / "data" / "security" / "common-passwords.txt"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=30) as r:
        text = r.read().decode("utf-8", errors="ignore")
    words = [w.strip() for w in text.splitlines() if w.strip()]
    if len(words) < 1000:
        print(f"unexpected download ({len(words)} lines); not written", file=sys.stderr)
        return 1
    OUT.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"{len(words)} common passwords -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
