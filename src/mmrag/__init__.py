"""Agentic multimodal RAG over PDFs. Design: docs/05-low-level-design.md."""

from mmrag import _blocked_dlls

_blocked_dlls.install()  # before langchain/langgraph load: Smart App Control blocks some DLLs here

__version__ = "0.1.0"
