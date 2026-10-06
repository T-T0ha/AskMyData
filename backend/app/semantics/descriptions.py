"""Column descriptions: the plain template, and telling it from a written one.

A column gets a description in two ways.  The language model writes a sentence
("The average SAT writing score for students who took the test"), or, when it is
unavailable, a deterministic template built from the name and the detected type
("avgscrwrite in table satscores (unknown, text) — data table").  Both are
stored in the same field, and later stages need to know which they have: the
sentence is worth embedding and showing to a model, the template adds nothing
the column name did not already say.
"""

from __future__ import annotations

import re
from typing import Sequence

#: ``<name> in table <table> (<label>, <type>) — ...``; a written sentence does
#: not take this shape.
_TEMPLATE_RE = re.compile(r"^[^()]+ in table \S+ \([^()]*\) — ")


def template_description(
    column_name: str,
    table_name: str,
    label: str,
    column_type: str,
    table_type: str = "data table",
    sample_values: Sequence[object] = (),
) -> str:
    """The deterministic fallback: mechanical, but always correct."""

    samples = ", ".join(str(v) for v in list(sample_values)[:5])
    parts = [
        column_name.replace("_", " "),
        f"in table {table_name}",
        f"({label.replace('_', ' ')}, {column_type})",
        f"— {table_type}",
    ]
    if samples:
        parts.append(f"example values: {samples}")
    return " ".join(parts)


def is_template_description(text: str | None) -> bool:
    """True for an empty description or the deterministic template."""

    if not text or not text.strip():
        return True
    return bool(_TEMPLATE_RE.match(text.strip()))
