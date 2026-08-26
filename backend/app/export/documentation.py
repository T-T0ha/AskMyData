"""The semantic layer as documentation a person can read (FR-18).

For most of the datasets this platform is pointed at, nobody has ever written
down what the columns mean.  By the end of Phase 4 the system knows: it has a
type for every column, a taxonomy label, a null ratio, the keys and references
a human confirmed, and a sentence per table.  Rendering that as Markdown is the
cheapest useful thing the project can do with it — no new analysis, no model
call, just the stored layer in a shape that survives being emailed.

It renders from the **bundle**, the same JSON the user can download, rather
than from the plan or the database.  Three things follow, and each is the
reason:

* the documentation of version 3 can be produced months later from the
  archived version 3, because the bundle is what was archived;
* it cannot drift from the exported JSON, since there is one source for both;
* nothing here can reach the data — the bundle carries schema, statistics and
  at most five sample values per column, which is what a reader needs and the
  most a document left in a shared drive should contain.

Every value that reaches a table cell is escaped: column names, sample values
and descriptions all originate in a user's spreadsheet, and a pipe character in
a product name should not be able to bend the table it is printed in.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

#: Sample values are for recognition, not for reading; a 400-character cell
#: destroys the table it sits in.
MAX_CELL = 60


def _cell(value: Any) -> str:
    """One table cell: escaped, single-line, bounded."""

    text = "" if value is None else str(value)
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    text = " ".join(text.split())
    if len(text) > MAX_CELL:
        text = text[: MAX_CELL - 1].rstrip() + "…"
    return text or "—"


def _anchor(name: str) -> str:
    """GitHub-style heading anchor for the contents list."""

    slug = str(name).lower().replace(" ", "-")
    # Underscores survive: ``order_lines`` anchors as ``#order_lines``, and a
    # contents list whose links do not land is worse than no contents list.
    return "".join(c for c in slug if c.isalnum() or c in "-_")


def _percent(ratio: Any) -> str:
    try:
        return f"{float(ratio):.0%}"
    except (TypeError, ValueError):
        return "—"


def _table_rows(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines.extend("| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows)
    return lines


def _renamed(column: Mapping[str, Any]) -> bool:
    source = str(column.get("source_column") or "")
    return bool(source) and source != str(column.get("column_name") or "")


def _column_role(column: Mapping[str, Any]) -> str:
    roles = []
    if column.get("is_primary_key"):
        roles.append("**key**")
    if column.get("is_foreign_key"):
        roles.append(f"→ `{column.get('references_table')}.{column.get('references_column')}`")
    if column.get("is_additive"):
        roles.append("summable")
    return ", ".join(roles) if roles else "—"


def _table_section(table: Mapping[str, Any]) -> list[str]:
    name = str(table.get("name", ""))
    kind = str(table.get("table_type") or "unknown")
    kind_words = "data" if kind == "unknown" else kind.replace("_", " ")
    columns = list(table.get("columns") or [])

    lines = [f"## `{name}`", ""]
    source_name = str(table.get("source_name") or "")
    if source_name and source_name != name:
        lines.append(f"*Originally “{source_name}”.*")
        lines.append("")
    lines.append(
        f"*{kind_words.capitalize()} table · {int(table.get('row_count') or 0):,} rows "
        f"· {len(columns)} columns*"
    )
    lines.append("")
    if table.get("description"):
        lines.extend([str(table["description"]), ""])

    primary_key = list(table.get("primary_key") or [])
    if primary_key:
        lines.append("**Primary key:** " + ", ".join(f"`{key}`" for key in primary_key))
    else:
        lines.append("**Primary key:** none — no other table can reference this one.")
    lines.append("")

    foreign_keys = list(table.get("foreign_keys") or [])
    if foreign_keys:
        lines.append("**References**")
        lines.append("")
        lines.extend(
            _table_rows(
                [
                    (
                        f"`{fk.get('source_column')}`",
                        f"`{fk.get('target_table')}.{fk.get('target_column')}`",
                        "enforced by the database"
                        if fk.get("enforced")
                        else f"not enforced — {fk.get('reason') or 'see notes'}",
                    )
                    for fk in foreign_keys
                ],
                ("Column", "References", "Status"),
            )
        )
        lines.append("")

    lines.append("**Columns**")
    lines.append("")
    lines.extend(
        _table_rows(
            [
                (
                    f"`{column.get('column_name')}`",
                    f"`{column.get('sql_type')}`",
                    str(column.get("taxonomy_label") or "unknown").replace("_", " "),
                    _column_role(column),
                    _percent(column.get("null_ratio")),
                    ", ".join(str(v) for v in (column.get("sample_values") or [])[:3]),
                )
                for column in columns
            ],
            ("Column", "Type", "Meaning", "Role", "Blank", "Examples"),
        )
    )
    lines.append("")

    described = [c for c in columns if c.get("semantic_description")]
    if described:
        lines.append("<details><summary>What each column holds</summary>")
        lines.append("")
        lines.extend(
            f"- **`{column.get('column_name')}`** — {column['semantic_description']}"
            for column in described
        )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    renamed = [c for c in columns if _renamed(c)]
    if renamed:
        lines.append("**Renamed for SQL:** " + ", ".join(
            f"“{c.get('source_column')}” → `{c.get('column_name')}`"
            for c in renamed
        ))
        lines.append("")

    notes = list(table.get("notes") or [])
    if notes:
        lines.append("**Notes**")
        lines.append("")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    return lines


def render_markdown(
    bundle: Mapping[str, Any],
    *,
    dataset: str = "",
    target: str = "",
    dialect: str = "",
    version: int = 0,
    exported_at: str = "",
) -> str:
    """The whole dataset, documented.  Deterministic: same layer, same bytes."""

    tables = list(bundle.get("tables") or [])
    column_count = sum(len(table.get("columns") or []) for table in tables)
    row_count = sum(int(table.get("row_count") or 0) for table in tables)

    title = dataset.strip() or "Dataset"
    lines = [f"# {title} — schema documentation", ""]

    stamp = [f"{len(tables)} tables", f"{column_count} columns", f"{row_count:,} rows"]
    if version:
        stamp.append(f"semantic layer version {version}")
    if exported_at:
        stamp.append(f"built {exported_at}")
    lines.append(" · ".join(stamp))
    lines.append("")
    if target:
        where = f"schema `{target}`" if dialect == "postgresql" else f"`{target}`"
        lines.append(f"The data itself is queryable in {where}.")
        lines.append("")
    lines.append(
        "> Everything below comes from the dataset's semantic layer: types detected "
        "from the values themselves, and keys and references confirmed by a person. "
        "Nothing was inferred while writing this document."
    )
    lines.append("")

    if tables:
        lines.append("## Contents")
        lines.append("")
        for table in tables:
            name = str(table.get("name", ""))
            kind = str(table.get("table_type") or "unknown")
            kind_words = "data" if kind == "unknown" else kind.replace("_", " ")
            lines.append(
                f"- [`{name}`](#{_anchor(name)}) — {kind_words} table, "
                f"{int(table.get('row_count') or 0):,} rows"
            )
        lines.append("")

    for table in tables:
        lines.extend(_table_section(table))

    references = [
        (table, fk)
        for table in tables
        for fk in (table.get("foreign_keys") or [])
    ]
    if references:
        lines.extend(["## Relationships", ""])
        lines.extend(
            _table_rows(
                [
                    (
                        f"`{fk.get('source_table')}.{fk.get('source_column')}`",
                        f"`{fk.get('target_table')}.{fk.get('target_column')}`",
                        "enforced" if fk.get("enforced") else "join hint only",
                    )
                    for _, fk in references
                ],
                ("From", "To", "Status"),
            )
        )
        lines.append("")

    warnings = list(bundle.get("warnings") or [])
    if warnings:
        lines.extend(["## What could not be asserted", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    skipped = list(bundle.get("skipped") or [])
    if skipped:
        lines.extend(["## Tables that were not exported", ""])
        lines.extend(
            f"- **{entry.get('table')}** — {entry.get('reason')}" for entry in skipped
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
