"""Claude API access.

Two hard rules hold everywhere in this file:

1. **Claude never sees raw data rows.**  Taxonomy classification gets ten
   sample values for one column; plan generation gets statistical summaries
   only.  This is a privacy and token-budget decision, and it is also why the
   platform scales to a ten-million-row database.
2. **Every call degrades gracefully.**  With no API key, no network, or a
   malformed response, the caller receives ``None`` and falls back to
   deterministic behaviour (``unknown`` labels, heuristic cleaning plan).  The
   platform is never blocked on the LLM.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.config import get_settings
from app.core.schemas import TAXONOMY_LABELS, TAXONOMY_UNKNOWN
from app.semantics.taxonomy import TaxonomyDecision

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

#: Phrases that mean "I am guessing" — the paper assigns ``unknown`` rather
#: than record a low-confidence label, and so do we.
HEDGE_MARKERS = (
    "could be",
    "might be",
    "not sure",
    "unclear",
    "hard to say",
    "ambiguous",
    "cannot determine",
    "insufficient",
)

TAXONOMY_SYSTEM_PROMPT = """You label columns of business spreadsheets with a semantic taxonomy.

You will be given one column: its name, its table name, its detected storage \
type, and at most 10 sample values. Choose the single label that best \
describes what the column MEANS (not how it is stored).

You must answer with JSON only, in exactly this shape:
{"label": "<one label from the allowed list>", "confidence": <0.0-1.0>, "reasoning": "<one short sentence>"}

Allowed labels: %s

Rules:
- Pick "unknown" if the column's meaning is genuinely unclear from the evidence.
- confidence must reflect real certainty; do not inflate it.
- Never invent a label outside the allowed list.
- Answer with the JSON object and nothing else."""

RELATIONSHIP_SYSTEM_PROMPT = """You help a non-technical person decide whether a \
detected relationship between two spreadsheet columns is real.

You are given the two columns, their taxonomy labels, a few sample values, and \
the value-overlap statistics that a detection algorithm computed. You are NOT \
deciding anything: the user decides, and the algorithm has already scored it.

Write 2-3 plain sentences that answer: given what these columns appear to mean, \
is it plausible that the first one refers to the second? Say plainly when it is \
not plausible, and say what would make you doubt it.

Rules:
- Write for someone who does not know what a foreign key is.
- Never claim certainty the statistics do not support.
- If the two columns look unrelated despite the overlap, say so directly.
- No preamble, no headings, no bullet points, no markdown. Prose only."""

PLAN_SYSTEM_PROMPT = """You are a data cleaning planner for a human-in-the-loop platform.

You receive STATISTICAL SUMMARIES of spreadsheet tables — never the raw rows. \
Propose a concrete, ordered cleaning plan that a human will review, edit and \
approve step by step before anything executes.

Answer with JSON only: an array of steps, each shaped exactly like:
{"type": "<step type>", "table": "<table name>", "description": "<plain English explanation of WHY this step is needed>", "params": {...}}

Allowed step types and their params:
- rename_column      {"column": str, "new_name": str}
- drop_column        {"column": str}
- standardize_format {"column": str, "format": "upper"|"lower"|"title"|"strip"|"date", "date_format": str (date only)}
- standardize_casing {"column": str, "casing": "upper"|"lower"|"title"}
- strip_whitespace   {"column": str, "collapse_internal": bool (optional)}
- merge_sheets       {"left_table": str, "right_table": str, "left_key": str, "right_key": str, "how": "inner"|"left", "result_table": str}
- split_column       {"column": str, "delimiter": str, "into": [str, str]}
- deduplicate        {"subset": [str] | null}
- type_cast          {"column": str, "to": "numeric"|"datetime"|"string"|"boolean"}
- add_synthetic_key  {"column": "row_id"}

Rules:
- Only reference tables and columns that appear in the summaries.
- Order matters: structural fixes (rename, type_cast, split) before row-level \
fixes (deduplicate), and merge_sheets last.
- Only propose merge_sheets on a column pair the summary lists as a CONFIRMED equivalence.
- Do not propose a step that the summary shows is unnecessary.
- NEVER propose filling, imputing, interpolating or otherwise replacing a \
missing value, with any strategy, for any column, no matter how high its \
null_ratio is. There is no step type for it and any such step will be \
rejected. A blank cell means "not known" and must still mean that in the \
database: SQL excludes NULL from AVG and SUM, so an invented value silently \
corrupts every total computed later, and nothing downstream can tell it apart \
from a real measurement. A high null_ratio is a fact to report, not a defect \
to patch. The only step that may respond to emptiness is drop_column, and only \
for a column that is almost entirely empty and therefore carries no information.
- Prefer standardize_casing over standardize_format for capitalisation \
mismatches, and strip_whitespace for stray spaces.
- Ground every description in the evidence you were given: quote the actual \
sample values or counts from the summary. Never invent an illustrative example.
- Keep descriptions non-technical: the reader is a business owner, not an engineer.
- Answer with the JSON array and nothing else."""

DESCRIBE_SYSTEM_PROMPT = """You write one-sentence semantic descriptions of spreadsheet columns.

You receive ONE table: its name, and for each column the detected storage type, \
the semantic taxonomy label, and at most five sample values. Never a full data row.

For every column, write the single sentence a business analyst would use to \
explain what that column holds and what it is for — what it measures or \
identifies, whether it can be summed, and how it relates to the table it \
belongs to. These descriptions are embedded for semantic search, so include the \
words someone would naturally use when asking a question about this column.

Answer with JSON only, in exactly this shape:
{"columns": [{"name": "<column name exactly as given>", "description": "<one sentence>"}]}

Rules:
- One entry per column you were given, using the exact column name.
- One sentence each, no bullet points, no markdown.
- Describe meaning, not storage ("total value of the sale in BDT", not "a float").
- Do not invent business context the evidence does not support.
- Answer with the JSON object and nothing else."""


def _extract_json(text: str) -> Any:
    """Pull a JSON value out of a response that may be fenced or prefixed."""

    text = text.strip()
    block = _JSON_BLOCK_RE.search(text)
    if block:
        text = block.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("response did not contain valid JSON")


class ClaudeClient:
    """Thin wrapper over the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.resolved_api_key()
        self._model = model or settings.claude_model
        self._max_tokens = settings.claude_max_tokens
        self._timeout = settings.claude_timeout_seconds
        self._effort = settings.claude_effort
        self._client: Any = None
        self.last_error: str | None = None
        if not self._api_key:
            self.last_error = "ANTHROPIC_API_KEY is not set"
            return
        try:
            from anthropic import Anthropic  # noqa: PLC0415

            self._client = Anthropic(api_key=self._api_key, timeout=self._timeout)
        except Exception as exc:  # pragma: no cover - import/config failure
            self.last_error = f"anthropic client unavailable: {exc}"
            logger.warning(self.last_error)

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def model(self) -> str:
        return self._model

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "model": self._model if self.available else None,
            "reason": self.last_error,
        }

    # -- low level -------------------------------------------------------
    def _complete(self, system: str, user: str, max_tokens: int | None = None) -> str | None:
        if not self.available:
            return None
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens or self._max_tokens,
                output_config={"effort": self._effort},
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Claude call failed: %s", exc)
            return None

        # A refusal is a successful HTTP call with no usable content; treat it
        # like any other unavailable answer so the caller falls back.
        if getattr(response, "stop_reason", None) == "refusal":
            self.last_error = "the model declined this request"
            logger.warning("Claude declined the request")
            return None
        if getattr(response, "stop_reason", None) == "max_tokens":
            self.last_error = "the model's reply was cut off (raise CLAUDE_MAX_TOKENS)"
            logger.warning("Claude reply hit max_tokens — JSON is likely truncated")

        parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "\n".join(parts).strip() or None

    # -- Phase 1 ---------------------------------------------------------
    def classify_taxonomy(
        self,
        column_name: str,
        table_name: str,
        column_type: str,
        samples: list[str],
    ) -> TaxonomyDecision | None:
        """Label one column the rule engine could not claim."""

        payload = {
            "table": table_name,
            "column": column_name,
            "detected_type": column_type,
            "sample_values": samples[:10],
        }
        raw = self._complete(
            TAXONOMY_SYSTEM_PROMPT % ", ".join(TAXONOMY_LABELS),
            json.dumps(payload, ensure_ascii=False, default=str),
            # The JSON answer is tiny, but max_tokens also has to cover the
            # model's thinking; 300 truncates before the reply is ever written.
            max_tokens=min(self._max_tokens, 4096),
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("Claude taxonomy response was not JSON: %r", raw[:200])
            return None
        if not isinstance(parsed, dict):
            return None

        label = str(parsed.get("label", "")).strip().lower()
        reasoning = str(parsed.get("reasoning", "") or "")
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        hedged = any(marker in reasoning.lower() for marker in HEDGE_MARKERS)
        if label not in TAXONOMY_LABELS or hedged or confidence < 0.5:
            return TaxonomyDecision(
                label=TAXONOMY_UNKNOWN,
                confidence=0.0,
                source="claude",
                reasoning=(
                    reasoning
                    or f"model proposed {label!r} with low confidence — needs manual review"
                ),
            )
        return TaxonomyDecision(
            label=label,
            confidence=min(confidence, 0.95),  # never outrank a deterministic rule
            source="claude",
            reasoning=reasoning or None,
        )

    def describe_columns(
        self,
        table_name: str,
        columns: list[dict[str, Any]],
    ) -> dict[str, str] | None:
        """Rich semantic descriptions for one table's columns.

        One call per *table*, not per column: the model needs to see the columns
        together to describe how they relate, and it keeps the token cost linear
        in tables rather than in columns.  The returned descriptions are what
        gets embedded into pgvector, so their quality is what Phase 4/5
        retrieval accuracy ultimately rests on.

        Returns ``{column_name: description}``, or ``None`` when unavailable —
        the caller falls back to a deterministic template.
        """

        if not columns:
            return {}
        payload = {"table": table_name, "columns": columns}
        raw = self._complete(
            DESCRIBE_SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False, default=str),
            max_tokens=min(self._max_tokens, 200 * len(columns) + 500),
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("Claude description response was not JSON: %r", raw[:200])
            return None
        if not isinstance(parsed, dict):
            return None

        entries = parsed.get("columns")
        if not isinstance(entries, list):
            return None
        known = {str(c.get("name", "")) for c in columns}
        descriptions: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            description = str(entry.get("description", "") or "").strip()
            # A description for a column that does not exist is a hallucination;
            # drop it rather than store it against the wrong column.
            if name in known and description:
                descriptions[name] = description
        return descriptions

    # -- Phase 2 ---------------------------------------------------------
    # -- Phase 3 ---------------------------------------------------------
    def explain_relationship(self, candidate: dict[str, Any]) -> str | None:
        """Plain English for a borderline relationship candidate.

        The model's role in Phase 3 is **explanation only**.  It does not
        detect relationships, does not score them, and cannot promote or
        demote one: the value-overlap arithmetic decides what is proposed, and
        the user decides what is confirmed.  What a non-technical user cannot
        do is read ``ratio_overlap 0.82, ratio_distinct 0.34`` and know whether
        that is a real reference or a coincidence, and that gap is what this
        fills.

        Returns ``None`` with no API key, no network, or an unusable reply —
        the evidence panel then shows the arithmetic alone, which was always
        the substance of the claim.
        """

        raw = self._complete(
            RELATIONSHIP_SYSTEM_PROMPT,
            json.dumps(candidate, ensure_ascii=False, default=str),
            max_tokens=min(self._max_tokens, 4096),
        )
        if raw is None:
            return None
        text = raw.strip()
        if not text:
            return None
        # A paragraph, not an essay: this sits inside an evidence panel next to
        # the rows themselves.
        return text[:600]

    def propose_cleaning_plan(self, summary: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Ask for an ordered cleaning plan from statistical summaries only."""

        raw = self._complete(
            PLAN_SYSTEM_PROMPT,
            json.dumps(summary, ensure_ascii=False, default=str),
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("Claude plan response was not JSON: %r", raw[:200])
            return None
        if isinstance(parsed, dict):
            parsed = parsed.get("steps", [])
        if not isinstance(parsed, list):
            return None
        return [step for step in parsed if isinstance(step, dict)]


_client: ClaudeClient | None = None


def get_claude_client() -> ClaudeClient:
    global _client
    if _client is None:
        _client = ClaudeClient()
    return _client


def reset_claude_client() -> None:
    global _client
    _client = None
