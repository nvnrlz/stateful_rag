"""Framework wrappers.

LangChain and LlamaIndex are optional integrations. Importing this package must
not hard-fail if they are not installed, so each wrapper is imported lazily and
its absence raises a clear error only when actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["StatefulLangChainRetriever", "StatefulLlamaIndexRetriever"]

if TYPE_CHECKING:  # for type checkers / IDEs only
    from .langchain_wrapper import StatefulLangChainRetriever
    from .llamaindex_wrapper import StatefulLlamaIndexRetriever


def __getattr__(name: str) -> Any:
    if name == "StatefulLangChainRetriever":
        try:
            from .langchain_wrapper import StatefulLangChainRetriever
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "StatefulLangChainRetriever requires 'langchain-core'. "
                "Install it with: pip install langchain-core"
            ) from exc
        return StatefulLangChainRetriever
    if name == "StatefulLlamaIndexRetriever":
        try:
            from .llamaindex_wrapper import StatefulLlamaIndexRetriever
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "StatefulLlamaIndexRetriever requires 'llama-index-core'. "
                "Install it with: pip install llama-index-core"
            ) from exc
        return StatefulLlamaIndexRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
