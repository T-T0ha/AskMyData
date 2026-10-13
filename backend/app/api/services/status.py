"""Backend status — which model, embedder and checkpointer this process is using."""

from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.semantics.embeddings import get_embedder


def backend_status() -> dict[str, Any]:
    from app.cleaning.checkpointing import checkpointer_backend, checkpointer_is_durable
    from app.semantics.claude_client import get_claude_client

    settings = get_settings()
    embedder = get_embedder()
    return {
        "claude": get_claude_client().status(),
        "embeddings": {
            "model": embedder.name,
            "dim": embedder.dim,
            "semantic": embedder.is_semantic,
        },
        "checkpointer": checkpointer_backend(),
        "durable": checkpointer_is_durable(),
        "thresholds": {
            "equivalence": settings.equivalence_threshold,
            "numeric_convert": settings.numeric_convert_threshold,
            "date_parse": settings.date_parse_threshold,
            "date_type": settings.date_type_threshold,
        },
    }
