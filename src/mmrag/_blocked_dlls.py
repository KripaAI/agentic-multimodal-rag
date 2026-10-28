"""Stand-ins for compiled modules that Windows Smart App Control blocks on this laptop.

Smart App Control decides per file and changes its mind: on 2026-10-12 it began blocking
uuid_utils (used by langchain-core and langsmith, so every chat question) after weeks of allowing
it. When the real module fails to load for that reason, a pure-Python equivalent of the part the
dependencies use is registered in its place. Anything else that fails to import still fails.
(scikit-learn, blocked the same day, is handled where it is imported: mmrag.index.embed.)
"""

from __future__ import annotations

import os
import sys
import time
import types
import uuid
from importlib.machinery import ModuleSpec


def _blocked(e: ImportError) -> bool:
    return "Application Control policy" in str(e)


def _uuid7() -> uuid.UUID:
    """RFC 9562 version 7: 48-bit Unix milliseconds, then random bits."""
    raw = bytearray(int(time.time() * 1000).to_bytes(6, "big") + os.urandom(10))
    raw[6] = (raw[6] & 0x0F) | 0x70  # version 7
    raw[8] = (raw[8] & 0x3F) | 0x80  # RFC 4122 variant
    return uuid.UUID(bytes=bytes(raw))


def _module(name: str, package: bool = False) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__spec__ = ModuleSpec(name, None, is_package=package)
    if package:
        m.__path__ = []
    return m


def patch_uuid_utils() -> bool:
    """True when the stand-in was installed."""
    try:
        import uuid_utils.compat  # noqa: F401
        return False
    except ImportError as e:
        if not _blocked(e):
            return False
    for name in [n for n in sys.modules if n == "uuid_utils" or n.startswith("uuid_utils.")]:
        del sys.modules[name]
    pkg, compat = _module("uuid_utils", package=True), _module("uuid_utils.compat")
    for m in (pkg, compat):
        m.uuid7 = _uuid7
        m.uuid4, m.uuid1, m.UUID, m.NAMESPACE_DNS = uuid.uuid4, uuid.uuid1, uuid.UUID, uuid.NAMESPACE_DNS
    pkg.compat = compat
    sys.modules["uuid_utils"], sys.modules["uuid_utils.compat"] = pkg, compat
    return True


def install() -> None:
    patch_uuid_utils()
