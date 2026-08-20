"""Phase 0 — multi-row / merged Excel header flattening.

pandas cannot see merged cells: it only receives the top-left value of a merge
and ``NaN`` for every other cell in the span.  Real business spreadsheets lean
on merges constantly ("2024" merged across "Q1 | Q2 | Q3"), so we read the raw
cell grid with openpyxl first, reconstruct what each header cell *means*, and
only then hand a flat frame to pandas.

The output is a single row of names such as ``2024_Q1_Revenue``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any

from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from app.core.config import get_settings

_WS_RE = re.compile(r"\s+")
_UNSAFE_RE = re.compile(r"[^0-9a-zA-Z_]+")
_MULTI_US_RE = re.compile(r"_{2,}")


@dataclass(slots=True)
class HeaderAnalysis:
    """Result of scanning the top of a sheet for its header block."""

    header_rows: list[int]
    data_start_row: int
    columns: list[str]
    original_labels: list[list[str]]
    merged_ranges: int = 0
    banner_rows: list[int] = field(default_factory=list)
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def is_multi_row(self) -> bool:
        return len(self.header_rows) > 1


def build_merged_value_map(sheet: Worksheet) -> dict[tuple[int, int], Any]:
    """Propagate every merged cell's anchor value across the cells it spans.

    openpyxl exposes merges as ranges whose non-anchor cells read ``None``;
    this map lets the header scanner treat a merged block as if every cell in
    it carried the label.
    """

    filled: dict[tuple[int, int], Any] = {}
    for merged in sheet.merged_cells.ranges:
        anchor = sheet.cell(row=merged.min_row, column=merged.min_col).value
        if anchor is None:
            continue
        for row in range(merged.min_row, merged.max_row + 1):
            for col in range(merged.min_col, merged.max_col + 1):
                filled[(row, col)] = anchor
    return filled


def _cell_value(
    sheet: Worksheet, row: int, col: int, merged: dict[tuple[int, int], Any]
) -> Any:
    value = sheet.cell(row=row, column=col).value
    if value is None:
        return merged.get((row, col))
    return value


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _is_texty(value: Any) -> bool:
    """A header-ish cell: non-empty text that is not a bare number."""

    if _is_blank(value):
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return False
    if isinstance(value, (datetime, date, time)):
        return False
    text = str(value).strip()
    if not text:
        return False
    # A string that is really a number ("1200", "3.5", "45%") is data, not a header.
    if re.fullmatch(r"[-+]?[\d,]*\.?\d+%?", text):
        return False
    return True


def _row_text_ratio(values: list[Any]) -> float:
    populated = [v for v in values if not _is_blank(v)]
    if not populated:
        return 0.0
    return sum(1 for v in populated if _is_texty(v)) / len(populated)


def _row_fill_ratio(values: list[Any], width: int) -> float:
    if width == 0:
        return 0.0
    return sum(1 for v in values if not _is_blank(v)) / width


def value_shape(value: Any) -> str:
    """Collapse a cell to a character-class signature.

    ``"ORD-1001"`` -> ``"a-9"``, ``"৳1,200.00"`` -> ``"x9,9.9"``,
    ``"Order ID"`` -> ``"a a"``.  Header labels and the data underneath them
    almost always have different signatures even when both are text, which is
    what makes the header boundary detectable on sheets where *every* value is
    a string.
    """

    if _is_blank(value):
        return ""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (datetime, date, time)):
        return "date"
    if isinstance(value, int):
        return "9"
    if isinstance(value, float):
        return "9" if value.is_integer() else "9.9"

    out: list[str] = []
    previous = ""
    for char in _WS_RE.sub(" ", str(value).strip()):
        if char.isdigit():
            token = "9"
        elif char.isalpha():
            token = "a"
        elif char == " ":
            token = " "
        else:
            token = char
        if token in {"9", "a"} and token == previous:
            continue
        out.append(token)
        previous = token
    shape = "".join(out)
    # Thousands separators and decimal points are formatting noise inside one
    # number: "9,9.9" and "9" describe the same kind of value.
    while True:
        collapsed = shape.replace("9,9", "9").replace("9.9", "9")
        if collapsed == shape:
            break
        shape = collapsed
    return shape[:24]


def _modal_shapes(rows: list[list[Any]], width: int) -> list[str | None]:
    """Most common value-shape per column across a block of body rows."""

    shapes: list[str | None] = []
    for col in range(width):
        counts: dict[str, int] = {}
        for row in rows:
            if col >= len(row):
                continue
            shape = value_shape(row[col])
            if shape:
                counts[shape] = counts.get(shape, 0) + 1
        shapes.append(max(counts.items(), key=lambda i: i[1])[0] if counts else None)
    return shapes


def _header_score(values: list[Any], body_shapes: list[str | None]) -> float:
    """Fraction of columns where this row's shape differs from the body's."""

    comparable = 0
    mismatched = 0
    for col, body_shape in enumerate(body_shapes):
        if body_shape is None or col >= len(values):
            continue
        shape = value_shape(values[col])
        if not shape:
            continue
        comparable += 1
        if shape != body_shape:
            mismatched += 1
    if comparable == 0:
        return 0.0
    return mismatched / comparable


def sanitize_name(raw: str) -> str:
    """Turn an arbitrary header label into a SQL/pandas-friendly identifier."""

    text = _WS_RE.sub(" ", str(raw)).strip()
    text = text.replace("%", " pct ").replace("&", " and ").replace("#", " no ")
    text = text.replace("/", "_").replace("\\", "_").replace("-", "_").replace(".", "_")
    text = _UNSAFE_RE.sub("_", text)
    text = _MULTI_US_RE.sub("_", text).strip("_")
    if not text:
        return ""
    if text[0].isdigit():
        text = f"c_{text}"
    return text.lower()


def _stringify(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y_%m_%d") if (value.hour or value.minute) == 0 else value.isoformat()
    if isinstance(value, date):
        return value.strftime("%Y_%m_%d")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _dedupe(names: list[str]) -> list[str]:
    """Guarantee unique, non-empty column names (``col_3``, ``price_2``, ...)."""

    seen: dict[str, int] = {}
    out: list[str] = []
    for index, name in enumerate(names):
        candidate = name or f"col_{index + 1}"
        if candidate in seen:
            seen[candidate] += 1
            candidate = f"{candidate}_{seen[candidate]}"
        else:
            seen[candidate] = 1
        out.append(candidate)
    return out


def detect_header_block(
    sheet: Worksheet,
    merged: dict[tuple[int, int], Any],
    max_scan: int | None = None,
    text_ratio: float | None = None,
) -> HeaderAnalysis:
    """Find which of the first rows form the header.

    Strategy, in order:

    1. Skip *banner* rows — a merged title or logo caption, recognisable
       because the whole row carries at most one distinct value.
    2. Build a shape profile of the sheet body (rows below the scan window).
    3. Absorb rows from the top while they read as labels: either their value
       shapes disagree with the body profile, or they are fully textual over a
       body that is not.
    4. Stop at the first row that looks like data.
    """

    settings = get_settings()
    max_scan = max_scan or settings.header_scan_rows
    text_ratio = settings.header_text_ratio if text_ratio is None else text_ratio

    width = sheet.max_column or 0
    height = sheet.max_row or 0
    notes: list[str] = []
    if width == 0 or height == 0:
        return HeaderAnalysis([], 1, [], [], 0, [], 0.0, ["sheet is empty"])

    scan_limit = min(max_scan, height)
    rows: list[list[Any]] = [
        [_cell_value(sheet, r, c, merged) for c in range(1, width + 1)]
        for r in range(1, scan_limit + 1)
    ]

    banner_rows: list[int] = []
    cursor = 0
    for index, values in enumerate(rows):
        populated = [v for v in values if not _is_blank(v)]
        distinct = {_stringify(v) for v in populated}
        # A blank row, or a row carrying a single value across a wide sheet
        # (a merged title), cannot be a header.
        if not populated or (width >= 3 and len(distinct) <= 1):
            banner_rows.append(index + 1)
            cursor = index + 1
            continue
        break

    if cursor >= len(rows):
        notes.append("no header row found in scan window; using first row")
        cursor = 0
        banner_rows = []

    # Profile the body from rows below the scan window so header candidates
    # never contaminate the profile they are compared against.
    body_start = max(scan_limit + 1, cursor + 2)
    body_rows: list[list[Any]] = [
        [_cell_value(sheet, r, c, merged) for c in range(1, width + 1)]
        for r in range(body_start, min(height, body_start + 29) + 1)
    ]
    if len(body_rows) < 2 and height > cursor + 1:
        body_rows = [
            [_cell_value(sheet, r, c, merged) for c in range(1, width + 1)]
            for r in range(cursor + 2, min(height, cursor + 31) + 1)
        ]
    body_shapes = _modal_shapes(body_rows, width)
    body_is_all_text = all(
        shape is None or set(shape) <= {"a", " "} for shape in body_shapes
    )

    header_rows: list[int] = []
    scores: list[float] = []
    for index in range(cursor, len(rows)):
        values = rows[index]
        score = _header_score(values, body_shapes)
        fully_textual = _row_text_ratio(values) >= text_ratio
        # A header row disagrees with the body's value shapes in most columns.
        # When the body is itself entirely textual no shape signal exists, so
        # the textual-ratio test is all that remains.
        # Strictly *more* than half the columns must disagree with the body.
        # At exactly half, the evidence is split — and a first data row that
        # merely differs in punctuation from the rows below it ("Central" in a
        # column of "North-East", "South-East") reaches 0.5 on its own, which
        # is enough to swallow it into the header and lose a row of data.
        looks_like_header = score > 0.5 or (body_is_all_text and fully_textual and score > 0)
        if looks_like_header:
            header_rows.append(index + 1)
            scores.append(score)
            continue
        if not header_rows:
            # Nothing distinguishes the first row from the data below it, but a
            # sheet must have a header — take row 1 and flag low confidence.
            header_rows.append(index + 1)
            scores.append(score)
            notes.append(
                f"row {index + 1} looks similar to the data below it; "
                "header detection is uncertain"
            )
        break

    if not header_rows:
        header_rows = [cursor + 1]
        scores = [0.0]

    data_start_row = header_rows[-1] + 1

    original_labels: list[list[str]] = []
    flattened: list[str] = []
    for col in range(1, width + 1):
        parts: list[str] = []
        raw_parts: list[str] = []
        for row in header_rows:
            value = _cell_value(sheet, row, col, merged)
            if _is_blank(value):
                continue
            text = _stringify(value)
            if raw_parts and raw_parts[-1] == text:
                # Same merged label repeated down the header block.
                continue
            raw_parts.append(text)
            piece = sanitize_name(text)
            if piece and piece not in parts:
                parts.append(piece)
        original_labels.append(raw_parts)
        flattened.append("_".join(parts))

    columns = _dedupe(flattened)

    confidence = 0.55 + 0.45 * (max(scores) if scores else 0.0)
    if len(header_rows) > 1:
        confidence -= 0.05
    empty_labels = sum(1 for label in original_labels if not label)
    if empty_labels:
        confidence -= min(0.4, 0.1 * empty_labels)
        notes.append(f"{empty_labels} column(s) had no header text; auto-named")
    if any("uncertain" in n for n in notes):
        confidence -= 0.25
    confidence = min(1.0, confidence)

    return HeaderAnalysis(
        header_rows=header_rows,
        data_start_row=data_start_row,
        columns=columns,
        original_labels=original_labels,
        merged_ranges=len(sheet.merged_cells.ranges),
        banner_rows=banner_rows,
        confidence=max(0.0, round(confidence, 3)),
        notes=notes,
    )


def describe_span(sheet: Worksheet) -> str:
    """Human readable used range, e.g. ``A1:F120``."""

    if not sheet.max_column or not sheet.max_row:
        return "empty"
    return f"A1:{get_column_letter(sheet.max_column)}{sheet.max_row}"
