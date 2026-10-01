"""Per-user long-term memory (Phase 9, spec §7.8, LLD §5.11, constitution P13).

Two kinds, both in LangGraph's Postgres Store with pgvector (D15), one namespace per user:

    memories/{user_id}/semantic   stable facts and preferences, one row per subject
    memories/{user_id}/episodic   a short summary per finished conversation

Memory **personalises; it never informs** (P13). Recalled memories reach the model as context
about the user, labelled as such and given `memory:N` ids, so a model that tries to cite one
is caught by the validator (`agent/validator.py`), which rejects every `memory:` citation.

Nothing here is a hard dependency of answering: every entry point degrades to "no memories"
when the store, the extractor model or the database is unavailable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterator

from mmrag.config import Settings
from mmrag.obs import get_logger, get_tracer

_log = get_logger("mmrag.memory")

ROOT = "memories"
SEMANTIC, EPISODIC = "semantic", "episodic"


def namespace(user_id: str, kind: str) -> tuple[str, str, str]:
    """The user's namespace for one kind of memory. Isolation is the namespace: no query ever
    runs without a user_id in it (NFR-13)."""
    if not user_id:
        raise ValueError("memory needs a user_id")
    if kind not in (SEMANTIC, EPISODIC):
        raise ValueError(f"unknown memory kind: {kind}")
    return (ROOT, str(user_id), kind)


@dataclass(frozen=True)
class Memory:
    """One remembered item, as the CLI, the UI and the agent see it."""

    key: str
    kind: str
    text: str
    subject: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def id(self) -> str:
        """The id the model sees. The `memory:` prefix is what the validator refuses (P13)."""
        return f"memory:{self.kind}:{self.key}"

    @classmethod
    def from_item(cls, kind: str, key: str, value: dict) -> "Memory":
        return cls(key=key, kind=kind, text=value.get("text", ""), subject=value.get("subject"),
                   created_at=value.get("created_at"), updated_at=value.get("updated_at"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- the store

def _lazy_embedder(settings: Settings) -> Callable[[list[str]], list[list[float]]]:
    """The project embedder, built on first use. Listing and deleting memories then need no
    OpenAI key: only storing and searching by meaning do."""
    holder: dict[str, object] = {}

    def embed(texts: list[str]) -> list[list[float]]:
        if "embedder" not in holder:
            from mmrag.index.embed import Embedder

            holder["embedder"] = Embedder(settings)
        return holder["embedder"].embed(texts)

    return embed


def index_config(settings: Settings, embed: Callable[[list[str]], list[list[float]]] | None = None) -> dict:
    """Vector index for the store: the same embedding model as retrieval (D15), over `text`."""
    return {"dims": settings.embed.dims, "embed": embed or _lazy_embedder(settings), "fields": ["text"]}


@contextmanager
def open_store(settings: Settings, embed: Callable[[list[str]], list[list[float]]] | None = None) -> Iterator:
    """The project's LangGraph Store, set up on first use. Yields None when memory is switched
    off in config, so callers can use it unconditionally."""
    if not settings.memory.enabled:
        yield None
        return
    from langgraph.store.postgres import PostgresStore

    with PostgresStore.from_conn_string(settings.secrets.database_url.get_secret_value(),
                                        index=index_config(settings, embed)) as store:
        store.setup()
        yield store


# ---------------------------------------------------------------- per-user switch (FR-25)

def is_enabled(settings: Settings, user_id: str | None) -> bool:
    """Memory is on for this user: the config switch and the user's own switch, both."""
    if not settings.memory.enabled or not user_id:
        return False
    from mmrag import db

    with db.connect(settings) as conn:
        row = conn.execute("SELECT memory_enabled FROM users WHERE user_id = %s", (user_id,)).fetchone()
    return bool(row and row[0])


def set_enabled(settings: Settings, user_id: str, enabled: bool) -> None:
    """Switch this user's memory on or off. Switching off stops storing and recalling; it does
    not delete what is already there (`forget_all` does that)."""
    from mmrag import db

    with db.connect(settings) as conn:
        conn.execute("UPDATE users SET memory_enabled = %s WHERE user_id = %s", (enabled, user_id))
        conn.commit()


# ---------------------------------------------------------------- read

def recall(store, user_id: str, question: str, k: int) -> list[Memory]:
    """The user's most relevant memories for this question, semantic first, then episodic.
    Never raises: a memory failure must not cost the user an answer."""
    if store is None or not user_id or k <= 0:
        return []
    out: list[Memory] = []
    with get_tracer("mmrag.memory").start_as_current_span("memory.recall") as span:
        for kind, limit in ((SEMANTIC, k), (EPISODIC, max(1, k // 2))):
            try:
                for item in store.search(namespace(user_id, kind), query=question, limit=limit):
                    out.append(Memory.from_item(kind, item.key, item.value))
            except Exception as e:  # noqa: BLE001 - memory is optional; answering is not
                _log.warning("memory recall failed (%s): answering without memories", e)
                return []
        span.set_attribute("mmrag.memory.recalled", len(out))
    return out


def list_memories(store, user_id: str, kind: str | None = None) -> list[Memory]:
    """Everything remembered about this user, newest first (FR-25: "see")."""
    if store is None:
        return []
    kinds = (kind,) if kind else (SEMANTIC, EPISODIC)
    out = [Memory.from_item(k, item.key, item.value)
           for k in kinds for item in store.search(namespace(user_id, k), limit=1000)]
    return sorted(out, key=lambda m: m.updated_at or m.created_at or "", reverse=True)


# ---------------------------------------------------------------- write

def remember_statements(store, user_id: str, statements: list[tuple[str, str]]) -> list[Memory]:
    """Store semantic statements as (subject, text). The subject is the key, so a new statement
    about a subject **replaces** the old one instead of piling up (LLD §5.11)."""
    if store is None or not statements:
        return []
    existing = {m.key: m for m in list_memories(store, user_id, SEMANTIC)}
    saved = []
    for subject, text in statements:
        key = subject.strip().lower().replace(" ", "_")[:64]
        if not key or not text.strip():
            continue
        created = existing[key].created_at if key in existing else _now()
        value = {"text": text.strip(), "subject": subject.strip(), "created_at": created, "updated_at": _now()}
        store.put(namespace(user_id, SEMANTIC), key, value)
        saved.append(Memory.from_item(SEMANTIC, key, value))
    return saved


def remember_episode(store, user_id: str, thread_id: str, summary: str) -> Memory | None:
    """Store (or replace) the summary of one conversation; the thread id is the key."""
    if store is None or not summary.strip():
        return None
    value = {"text": summary.strip(), "subject": None, "thread_id": thread_id,
             "created_at": _now(), "updated_at": _now()}
    store.put(namespace(user_id, EPISODIC), thread_id, value)
    return Memory.from_item(EPISODIC, thread_id, value)


# ---------------------------------------------------------------- delete and retention (FR-25, NFR-13)

def delete(store, user_id: str, kind: str, key: str) -> None:
    if store is not None:
        store.delete(namespace(user_id, kind), key)


def forget_all(store, user_id: str) -> int:
    """Delete every memory of one user. Returns how many were deleted."""
    if store is None:
        return 0
    items = list_memories(store, user_id)
    for m in items:
        store.delete(namespace(user_id, m.kind), m.key)
    return len(items)


def delete_user_memories(settings: Settings, user_id: str) -> int:
    """Delete a user's memories directly in SQL, for account deletion: it must work even when
    the store cannot be opened (no API key for the embedder, for instance)."""
    from mmrag import db

    with db.connect(settings) as conn:
        if not conn.execute("SELECT to_regclass('store')").fetchone()[0]:
            return 0
        n = conn.execute("DELETE FROM store WHERE prefix LIKE %s", (f"{ROOT}.{user_id}.%",)).rowcount
        conn.commit()
    return n


def cleanup_expired(settings: Settings) -> int:
    """Delete memories older than `memory.retention_days` (NFR-13). Called by `obs cleanup`."""
    from mmrag import db

    with db.connect(settings) as conn:
        if not conn.execute("SELECT to_regclass('store')").fetchone()[0]:
            return 0
        n = conn.execute("DELETE FROM store WHERE prefix LIKE %s AND updated_at < now() - make_interval(days => %s)",
                         (f"{ROOT}.%", settings.memory.retention_days)).rowcount
        conn.commit()
    return n


# ---------------------------------------------------------------- the agent's view

@dataclass
class MemorySession:
    """What the agent is given for one user: recall before planning, write after answering.

    Both methods swallow their failures. Memory is a convenience; the answer is the product.
    """

    store: object
    user_id: str
    llm: object | None = None  # mmrag.agent.llm.LLM — the low-cost extractor (`memory.model`)
    llm_factory: Callable[[], object] | None = None  # builds it on the first write, not at sign-in
    recall_k: int = 5
    max_statement_chars: int = 200

    def _extractor(self):
        """The extractor model, built the first time something is written. Building it can
        fail (no API key, say); that costs a memory, never an answer."""
        if self.llm is None and self.llm_factory is not None:
            self.llm = self.llm_factory()
        return self.llm

    def recall(self, question: str) -> list[Memory]:
        return recall(self.store, self.user_id, question, self.recall_k)

    def remember(self, question: str, answer_text: str) -> list[Memory]:
        """Extract statements from one finished turn and store them under their subjects."""
        from mmrag.memory.extract import extract_semantic

        try:
            extractor = self._extractor()
            if extractor is None:
                return []
            existing = [m.text for m in list_memories(self.store, self.user_id, SEMANTIC)]
            pairs = extract_semantic(extractor, question, answer_text, existing, self.max_statement_chars)
            return remember_statements(self.store, self.user_id, pairs)
        except Exception as e:  # noqa: BLE001 - never lose an answer over a memory
            _log.warning("storing memories failed (%s)", e)
            return []


def session_for(settings: Settings, store, user_id: str | None, llm=None) -> "MemorySession | None":
    """A session when memory is on for this user and the store is open, otherwise None."""
    if store is None or not is_enabled(settings, user_id):
        return None
    model = settings.memory.model or settings.agent.summary_model or settings.agent.model
    if llm is None and not model:
        return None

    def build():
        from mmrag.agent.llm import OpenAILLM
        from mmrag.llm import get_client

        return OpenAILLM(get_client(settings), model)

    return MemorySession(store=store, user_id=str(user_id), llm=llm, llm_factory=None if llm else build,
                         recall_k=settings.memory.recall_k,
                         max_statement_chars=settings.memory.max_statement_chars)
