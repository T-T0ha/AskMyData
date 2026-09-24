"""LLM access, backed by OpenRouter.

The module, the class, and the ``source="claude"`` values stored in the
database all still say "claude" — renaming them would ripple into routes,
services, the frontend's status labels, and taxonomy rows already on disk —
but every network call this file makes goes through OpenRouter's
OpenAI-compatible endpoint (https://openrouter.ai), which itself proxies to
whichever underlying model ``OPENROUTER_MODEL`` names.

Two hard rules hold everywhere in this file:

1. **The model never sees raw data rows.**  Taxonomy classification gets ten
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

TAXONOMY_BATCH_SYSTEM_PROMPT = """You label columns of business spreadsheets with a semantic taxonomy.

You will be given one table and a list of its columns that a rule engine could \
not confidently label: each with its name, its detected storage type, and at \
most 10 sample values. For EACH column, choose the single label that best \
describes what the column MEANS (not how it is stored).

You must answer with JSON only, in exactly this shape:
{"labels": [{"column": "<column name exactly as given>", "label": "<one label from the allowed list>", "confidence": <0.0-1.0>, "reasoning": "<one short sentence>"}]}

Allowed labels: %s

Rules:
- Exactly one entry per column you were given, using the exact column name.
- Judge each column independently on its own evidence — do not let one column's \
label influence another's.
- Pick "unknown" if a column's meaning is genuinely unclear from the evidence.
- confidence must reflect real certainty; do not inflate it.
- Never invent a label outside the allowed list.
- Answer with the JSON object and nothing else."""

TAXONOMY_DATASET_SYSTEM_PROMPT = """You label columns of business spreadsheets with a semantic taxonomy.

You will be given every table of ONE dataset, each with the columns a rule engine \
could not confidently label: each column's name, its detected storage type, and at \
most 10 sample values. For EACH column of EACH table, choose the single label that \
best describes what the column MEANS (not how it is stored).

You must answer with JSON only, in exactly this shape:
{"labels": [{"table": "<table name exactly as given>", "column": "<column name exactly as given>", "label": "<one label from the allowed list>", "confidence": <0.0-1.0>, "reasoning": "<one short sentence>"}]}

Allowed labels: %s

Rules:
- Exactly one entry per column you were given, using the exact table and column names.
- Judge each column independently on its own evidence — do not let one column's \
label influence another's, even within the same table.
- Pick "unknown" if a column's meaning is genuinely unclear from the evidence.
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
- Only propose merge_sheets when two tables hold the same columns and are really \
one table split across sheets — never to join two different entities (an orders \
table and a customers table, say) just because they share a key column; that \
relationship is what a foreign key is for, and joining on it would denormalise \
the very structure this platform exists to recover. A CONFIRMED equivalence on \
the shared key is what tells you the split-sheet case is real, not a license to \
merge on it in every case.
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

DESCRIBE_DATASET_SYSTEM_PROMPT = """You write one-sentence semantic descriptions of spreadsheet columns.

You receive every table of ONE dataset: its name, and for each column the detected \
storage type, the semantic taxonomy label, and at most five sample values. Never a \
full data row.

For every column of every table, write the single sentence a business analyst would \
use to explain what that column holds and what it is for — what it measures or \
identifies, whether it can be summed, and how it relates to the table it belongs to. \
These descriptions are embedded for semantic search, so include the words someone \
would naturally use when asking a question about this column.

Answer with JSON only, in exactly this shape:
{"columns": [{"table": "<table name exactly as given>", "name": "<column name exactly as given>", "description": "<one sentence>"}]}

Rules:
- Exactly one entry per column you were given, using the exact table and column names.
- One sentence each, no bullet points, no markdown.
- Describe meaning, not storage ("total value of the sale in BDT", not "a float").
- Do not invent business context the evidence does not support.
- Answer with the JSON object and nothing else."""

DESCRIBE_TABLES_SYSTEM_PROMPT = """You write one-sentence descriptions of the tables \
in a business database.

You receive every table of ONE dataset: its name, the kind of table the analysis \
decided it is, its primary key, the tables it references and is referenced by, and \
its column names with their semantic labels. You never receive a data row or a \
sample value.

For every table, write the single sentence a business analyst would use to say what \
the table holds and what it is for — what one row of it represents, and what \
questions it is the right table to answer. These sentences are how the system picks \
which tables a question is even read against, so include the words someone would \
naturally use when asking about this table.

Answer with JSON only, in exactly this shape:
{"tables": [{"name": "<table name exactly as given>", "description": "<one sentence>"}]}

Rules:
- One entry per table you were given, using the exact table name.
- One sentence each, no bullet points, no markdown.
- Say what a row means ("one line of one customer order"), not how it is stored.
- Do not invent business context the evidence does not support.
- Answer with the JSON object and nothing else."""


SQL_SYSTEM_PROMPT = """You translate a business question into one read-only SQL query.

You are given the question, the SQL dialect ("postgresql" or "sqlite"), and the schema of \
the tables that matter for it — grouped by table, with each column's name, SQL type, \
semantic label, whether it is a primary or foreign key (and what it references), whether \
it is additive (safe to SUM or AVG), its null ratio, and up to five example values. You may \
also be given a previous attempt and the error it produced; fix that specific error rather \
than starting over from a different query.

You may also be given "business_rules" — plain-English domain rules the user has recorded \
about this data (e.g. "only orders with status 'shipped' count as revenue") — and \
"examples" — verified question/SQL pairs already confirmed correct for this same dataset. \
Follow a business rule even where it overrides what the schema alone would suggest, and \
when a question closely resembles one of the examples, follow its pattern rather than \
inventing a different one.

Answer with JSON only, in exactly this shape:
{"sql": "<one SELECT statement>", "explanation": "<one short plain-English sentence saying what it computes>", "confidence": <0.0-1.0>}

Rules:
- Exactly one statement, and it must be a SELECT (a WITH ... SELECT is allowed; a bare \
UNION/INTERSECT/EXCEPT of two SELECTs is allowed). Never a semicolon followed by anything else.
- Never INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, GRANT, or any other statement that is \
not a read.
- Reference only the tables and columns you were given, by the exact names given. Never \
invent one, and never qualify a name with a schema or database ("public.orders" is wrong — \
write "orders").
- Only SUM or AVG a column marked additive. A non-additive numeric column (an id, a year, a \
rating, a percentage) may be selected, grouped or filtered on, never summed or averaged.
- Join only through the primary/foreign key pairs you were given.
- Always return your best-effort SQL, even when the schema only partially fits the \
question — but "confidence" must honestly reflect how likely this query is to be exactly \
right, not just syntactically valid. Lower it for a guessed join, a wording resolved \
arbitrarily, an aggregate over a column you are unsure is additive, or a schema that only \
partially covers what was asked. Do not inflate it: a caller downstream decides whether to \
show this answer to the user based on this number alone.
- No comments, no markdown fencing, no prose outside the JSON object."""

AMBIGUITY_SYSTEM_PROMPT = """You decide whether a business question is too ambiguous to \
safely turn into one SQL query, before any SQL is written.

You are given the question and the tables that were pre-selected as relevant to it — each \
with its name and one-sentence description. You are NOT deciding what the answer is; you \
are deciding whether "what the user is actually asking" is clear enough that a specific SQL \
query would not just be one arbitrary guess among several equally plausible ones.

Answer with JSON only, in exactly this shape:
{"ambiguous": <true|false>, "reason": "<one short sentence, empty string if not ambiguous>", "options": ["<clarification 1>", "<clarification 2>"]}

Rules:
- Mark ambiguous only when the question genuinely admits multiple, materially different \
readings given these tables (e.g. "top customers" with no metric named, when both order \
count and order value are plausible; a time period named nowhere in the tables shown). A \
question that is merely broad but has one natural reading is NOT ambiguous.
- "options" has at most 3 entries, each a complete, specific rephrasing of the original \
question a user could click to ask instead of typing again. Empty when not ambiguous.
- No markdown, no prose outside the JSON object."""

SUGGEST_QUESTIONS_SYSTEM_PROMPT = """You propose business questions worth asking about a \
dataset, before anyone has asked anything.

You are given every table in ONE dataset: its name, its one-sentence description, and its \
columns with their semantic labels. You never receive a data row.

Answer with JSON only, in exactly this shape:
{"questions": ["<question 1>", "<question 2>", ...]}

Rules:
- Between 10 and 15 questions.
- Each must be answerable from the tables and columns you were given, and specific to this \
dataset's actual subject matter — never generic ("show me some data").
- Phrased the way a business person would type them, under fifteen words each, no question \
mark required.
- Cover different tables and different kinds of question (totals, trends, comparisons, \
top-N) rather than fifteen variations on one theme.
- No markdown, no numbering, no prose outside the JSON object."""


FOLLOWUP_SYSTEM_PROMPT = """You suggest follow-up questions after a business question was \
answered against a database.

You are given the original question, the SQL that answered it, the columns of \
the result, and the tables it read — never any row of the result.

Answer with JSON only, in exactly this shape:
{"questions": ["<question 1>", "<question 2>", "<question 3>"]}

Rules:
- Exactly three questions.
- Each must be a natural next thing to ask — drill into a category, compare a \
period, or ask for a trend — answerable from the same tables.
- Short: under twelve words, phrased the way a person would type them, no \
question mark required.
- Never repeat the original question or trivially reword it.
- No markdown, no numbering, no prose outside the JSON object."""


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
    """Thin wrapper over OpenRouter's OpenAI-compatible chat completions endpoint."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.resolved_api_key()
        self._model = model or settings.openrouter_model
        self._max_tokens = settings.openrouter_max_tokens
        self._timeout = settings.openrouter_timeout_seconds
        self._client: Any = None
        self.last_error: str | None = None
        if not self._api_key:
            self.last_error = "OPENROUTER_API_KEY is not set"
            return
        try:
            from openai import OpenAI  # noqa: PLC0415

            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=self._timeout,
            )
        except Exception as exc:  # pragma: no cover - import/config failure
            self.last_error = f"openrouter client unavailable: {exc}"
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
    def _complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ) -> str | None:
        if not self.available:
            return None
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": 0.0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if json_mode:
                # Not every free OpenRouter model honours structured JSON
                # output; retry once in plain text and let _extract_json
                # pull the object out of whatever prose comes back.
                kwargs.pop("response_format", None)
                try:
                    response = self._client.chat.completions.create(**kwargs)
                except Exception as exc2:
                    self.last_error = str(exc2)
                    logger.warning("OpenRouter call failed: %s", exc2)
                    return None
            else:
                self.last_error = str(exc)
                logger.warning("OpenRouter call failed: %s", exc)
                return None

        choices = getattr(response, "choices", None) or []
        if not choices:
            self.last_error = "the model returned no choices"
            logger.warning("OpenRouter call produced no choices")
            return None

        # A refusal/content-filter block is a successful HTTP call with no
        # usable content; treat it like any other unavailable answer so the
        # caller falls back.
        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason == "length":
            self.last_error = "the model's reply was cut off (raise OPENROUTER_MAX_TOKENS)"
            logger.warning("OpenRouter reply hit the token limit — JSON is likely truncated")
        elif finish_reason == "content_filter":
            self.last_error = "the model declined this request (content filter)"
            logger.warning("OpenRouter declined the request: content filter")
            return None

        text = (getattr(choices[0].message, "content", None) or "").strip()
        return text or None

    # -- Phase 1 ---------------------------------------------------------
    @staticmethod
    def _taxonomy_decision(label_raw: Any, confidence_raw: Any, reasoning_raw: Any) -> TaxonomyDecision:
        """Shared validation between :meth:`classify_taxonomy` and its batch sibling."""

        label = str(label_raw or "").strip().lower()
        reasoning = str(reasoning_raw or "")
        try:
            confidence = float(confidence_raw)
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

    def classify_taxonomy(
        self,
        column_name: str,
        table_name: str,
        column_type: str,
        samples: list[str],
    ) -> TaxonomyDecision | None:
        """Label one column the rule engine could not claim.

        Prefer :meth:`classify_taxonomy_batch` when labelling more than one
        column of the same table — it does the same job in one request
        instead of one per column.
        """

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
            logger.warning("OpenRouter taxonomy response was not JSON: %r", raw[:200])
            return None
        if not isinstance(parsed, dict):
            return None

        return self._taxonomy_decision(
            parsed.get("label"), parsed.get("confidence", 0.0), parsed.get("reasoning")
        )

    def classify_taxonomy_batch(
        self,
        table_name: str,
        columns: list[dict[str, Any]],
    ) -> dict[str, TaxonomyDecision] | None:
        """Label every rule-unresolved column of one table in a single call.

        Same evidence boundary as :meth:`classify_taxonomy` — name, detected
        type, at most ten sample values — but one request per *table* rather
        than one per column.  A wide table can leave a couple dozen columns
        unclaimed by the rule engine, and paying network latency (plus a
        rate-limited free-tier quota) once per column instead of once per
        table made Phase 1 unusably slow.

        ``columns`` is a list of ``{"column", "detected_type",
        "sample_values"}`` dicts (mirroring :meth:`classify_taxonomy`'s
        arguments). Returns ``{column_name: TaxonomyDecision}`` — missing
        entries mean the model skipped that column, and the caller falls back
        to ``unknown`` for those same as it would for a ``None`` result.
        """

        if not columns:
            return {}
        payload = {"table": table_name, "columns": columns}
        raw = self._complete(
            TAXONOMY_BATCH_SYSTEM_PROMPT % ", ".join(TAXONOMY_LABELS),
            json.dumps(payload, ensure_ascii=False, default=str),
            max_tokens=min(self._max_tokens, 300 * len(columns) + 1000),
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("OpenRouter batch taxonomy response was not JSON: %r", raw[:200])
            return None
        if not isinstance(parsed, dict):
            return None

        entries = parsed.get("labels")
        if not isinstance(entries, list):
            return None
        known = {str(c.get("column", "")) for c in columns}
        decisions: dict[str, TaxonomyDecision] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("column", "")).strip()
            # A label for a column that does not exist is a hallucination;
            # drop it rather than store it against the wrong column.
            if name not in known:
                continue
            decisions[name] = self._taxonomy_decision(
                entry.get("label"), entry.get("confidence", 0.0), entry.get("reasoning")
            )
        return decisions

    def classify_taxonomy_dataset(
        self,
        tables: list[dict[str, Any]],
    ) -> dict[str, dict[str, TaxonomyDecision]] | None:
        """Label every rule-unresolved column of an entire dataset in one call.

        Same evidence boundary as :meth:`classify_taxonomy` — name, detected
        type, at most ten sample values per column — but one request for the
        *whole dataset* rather than one per table or one per column. This is
        what :func:`app.semantics.pipeline.analyze_tables` calls so that
        Phase 1 costs exactly one LLM request no matter how many sheets or
        columns a dataset has.

        ``tables`` is ``[{"table": name, "columns": [{"column",
        "detected_type", "sample_values"}, ...]}, ...]`` — one entry per
        table that has at least one rule-unresolved column. Returns
        ``{table_name: {column_name: TaxonomyDecision}}``; a table or column
        missing from the result means the model skipped it, and the caller
        falls back to ``unknown`` for those same as it would for a ``None``
        result.
        """

        if not tables:
            return {}
        raw = self._complete(
            TAXONOMY_DATASET_SYSTEM_PROMPT % ", ".join(TAXONOMY_LABELS),
            json.dumps({"tables": tables}, ensure_ascii=False, default=str),
            max_tokens=min(
                self._max_tokens,
                300 * sum(len(t.get("columns", [])) for t in tables) + 1000,
            ),
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("OpenRouter dataset taxonomy response was not JSON: %r", raw[:200])
            return None
        if not isinstance(parsed, dict):
            return None

        entries = parsed.get("labels")
        if not isinstance(entries, list):
            return None
        known = {
            (str(t.get("table", "")), str(c.get("column", "")))
            for t in tables
            for c in t.get("columns", [])
        }
        decisions: dict[str, dict[str, TaxonomyDecision]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            table_name = str(entry.get("table", "")).strip()
            column_name = str(entry.get("column", "")).strip()
            # A label for a table/column pair that was not sent is a
            # hallucination; drop it rather than store it against the wrong
            # column.
            if (table_name, column_name) not in known:
                continue
            decisions.setdefault(table_name, {})[column_name] = self._taxonomy_decision(
                entry.get("label"), entry.get("confidence", 0.0), entry.get("reasoning")
            )
        return decisions

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
            logger.warning("OpenRouter description response was not JSON: %r", raw[:200])
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

    def describe_columns_dataset(
        self,
        tables: list[dict[str, Any]],
    ) -> dict[str, dict[str, str]] | None:
        """Rich semantic descriptions for every table's columns, in one call.

        Same evidence boundary as :meth:`describe_columns` — detected type,
        taxonomy label, five sample values, never a data row — but one
        request for the whole dataset instead of one per table, so Phase 1
        costs one taxonomy call (:meth:`classify_taxonomy_dataset`) plus one
        description call, no matter how many sheets the dataset has.

        ``tables`` is ``[{"table": name, "columns": [...]}, ...]`` (the same
        per-column payload shape :meth:`describe_columns` takes). Returns
        ``{table_name: {column_name: description}}``; the caller falls back
        to its deterministic template for anything missing, same as it would
        for a ``None`` result.
        """

        if not tables:
            return {}
        total_columns = sum(len(t.get("columns", [])) for t in tables)
        if total_columns == 0:
            return {}
        raw = self._complete(
            DESCRIBE_DATASET_SYSTEM_PROMPT,
            json.dumps({"tables": tables}, ensure_ascii=False, default=str),
            max_tokens=min(self._max_tokens, 200 * total_columns + 500),
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("OpenRouter dataset description response was not JSON: %r", raw[:200])
            return None
        if not isinstance(parsed, dict):
            return None

        entries = parsed.get("columns")
        if not isinstance(entries, list):
            return None
        known = {
            (str(t.get("table", "")), str(c.get("name", "")))
            for t in tables
            for c in t.get("columns", [])
        }
        descriptions: dict[str, dict[str, str]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            table_name = str(entry.get("table", "")).strip()
            column_name = str(entry.get("name", "")).strip()
            description = str(entry.get("description", "") or "").strip()
            # A description for a table/column pair that was not sent is a
            # hallucination; drop it rather than store it against the wrong
            # column.
            if (table_name, column_name) not in known or not description:
                continue
            descriptions.setdefault(table_name, {})[column_name] = description
        return descriptions

    def describe_tables(self, tables: list[dict[str, Any]]) -> dict[str, str] | None:
        """One sentence per table, for the whole dataset in one call.

        Per *dataset* rather than per table, because the sentences are used to
        choose between tables: the model writes better ones when it can see
        that ``orders`` is the one with the money in it and ``order_status`` is
        a lookup.  It also keeps the cost at one call per enrichment run.

        Returns ``{table_name: description}``, or ``None`` when unavailable —
        the caller keeps the composed sentence, which is never absent.
        """

        if not tables:
            return {}
        raw = self._complete(
            DESCRIBE_TABLES_SYSTEM_PROMPT,
            json.dumps({"tables": tables}, ensure_ascii=False, default=str),
            max_tokens=min(self._max_tokens, 200 * len(tables) + 500),
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("OpenRouter table description response was not JSON: %r", raw[:200])
            return None
        if not isinstance(parsed, dict):
            return None

        entries = parsed.get("tables")
        if not isinstance(entries, list):
            return None
        known = {str(table.get("name", "")) for table in tables}
        descriptions: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            description = str(entry.get("description", "") or "").strip()
            # A description for a table that was not sent is a hallucination.
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
            json_mode=False,
        )
        if raw is None:
            return None
        text = raw.strip()
        if not text:
            return None
        # A paragraph, not an essay: this sits inside an evidence panel next to
        # the rows themselves.
        return text[:600]

    # -- Phase 5 -----------------------------------------------------------
    def generate_sql(
        self,
        question: str,
        dialect: str,
        schema: list[dict[str, Any]],
        prior_sql: str | None = None,
        prior_error: str | None = None,
        business_rules: list[str] | None = None,
        examples: list[dict[str, str]] | None = None,
    ) -> dict[str, Any] | None:
        """One SQL attempt for ``question`` against ``schema`` — never the data itself.

        ``schema`` is the retrieval step's output
        (:func:`app.query.retrieval.schema_context`): table and column names,
        types and roles, at most five example values per column.  No row of
        the actual table is ever part of this call, the same boundary every
        other OpenRouter call in this platform holds.

        ``business_rules``/``examples`` are the third retrieval index
        (:func:`app.query.retrieval.retrieve_business_rules` /
        :func:`~app.query.retrieval.retrieve_examples`) — the user's own
        recorded rules and previously verified question/SQL pairs for this
        dataset, ranked by relevance to ``question``.

        Passing ``prior_sql``/``prior_error`` is what makes this a *retry*
        rather than a second independent guess — the model sees exactly what
        it wrote and exactly what was wrong with it.

        The returned ``confidence`` (0.0-1.0, defaulting to 0.0 for a missing
        or unparsable value) is not acted on here — the caller applies
        :attr:`~app.core.config.Settings.query_confidence_threshold` and
        decides whether to run the query at all (§Phase 5 abstention gate).
        """

        payload: dict[str, Any] = {
            "question": question,
            "dialect": dialect,
            "tables": schema,
            "business_rules": business_rules or [],
            "examples": examples or [],
        }
        if prior_sql is not None:
            payload["previous_attempt"] = {"sql": prior_sql, "error": prior_error}
        raw = self._complete(
            SQL_SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False, default=str),
            max_tokens=min(self._max_tokens, 4096),
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("OpenRouter SQL response was not JSON: %r", raw[:200])
            return None
        if not isinstance(parsed, dict):
            return None
        sql = str(parsed.get("sql", "") or "").strip()
        if not sql:
            return None
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))
        return {
            "sql": sql,
            "explanation": str(parsed.get("explanation", "") or "").strip(),
            "confidence": confidence,
        }

    def check_ambiguity(
        self, question: str, tables: list[dict[str, str]]
    ) -> dict[str, Any] | None:
        """Whether ``question`` admits multiple, materially different SQL readings.

        Runs after table pre-selection (so the model has real table
        descriptions to reason about) but before column ranking or SQL
        generation — "before any SQL is generated" (§3.1.6). ``None`` (no
        client, or an unusable reply) degrades to "not ambiguous": the
        generate → guard → EXPLAIN loop still runs, the same way every other
        optional OpenRouter call in this file leaves the deterministic path intact
        when the model is unavailable.
        """

        raw = self._complete(
            AMBIGUITY_SYSTEM_PROMPT,
            json.dumps({"question": question, "tables": tables}, ensure_ascii=False, default=str),
            max_tokens=min(self._max_tokens, 1024),
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("OpenRouter ambiguity response was not JSON: %r", raw[:200])
            return None
        if not isinstance(parsed, dict):
            return None
        options = parsed.get("options")
        return {
            "ambiguous": bool(parsed.get("ambiguous")),
            "reason": str(parsed.get("reason", "") or ""),
            "options": (
                [str(item).strip() for item in options if str(item).strip()][:3]
                if isinstance(options, list)
                else []
            ),
        }

    def suggest_questions(self, tables: list[dict[str, Any]]) -> list[str] | None:
        """10-15 dataset-specific questions, generated before anyone asks anything (§5.4).

        Same privacy boundary as :meth:`describe_tables`: names, descriptions
        and column labels only, never a row.  ``None`` when unavailable — the
        caller shows no proactive suggestions rather than blocking on this.
        """

        if not tables:
            return []
        raw = self._complete(
            SUGGEST_QUESTIONS_SYSTEM_PROMPT,
            json.dumps({"tables": tables}, ensure_ascii=False, default=str),
            max_tokens=min(self._max_tokens, 2048),
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("OpenRouter suggested-questions response was not JSON: %r", raw[:200])
            return None
        if not isinstance(parsed, dict):
            return None
        entries = parsed.get("questions")
        if not isinstance(entries, list):
            return None
        return [str(item).strip() for item in entries if str(item).strip()][:15]

    # -- Phase 6 -----------------------------------------------------------
    def suggest_followups(
        self,
        question: str,
        sql: str,
        columns: list[str],
        tables: list[str],
    ) -> list[str] | None:
        """Three short next questions, or ``None`` when unavailable.

        A dashboard nicety, not a load-bearing part of the answer — a caller
        that gets ``None`` back shows no suggestions rather than failing the
        question that already succeeded.
        """

        payload = {"question": question, "sql": sql, "columns": columns, "tables": tables}
        raw = self._complete(
            FOLLOWUP_SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False, default=str),
            max_tokens=min(self._max_tokens, 1024),
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("OpenRouter follow-up response was not JSON: %r", raw[:200])
            return None
        if not isinstance(parsed, dict):
            return None
        entries = parsed.get("questions")
        if not isinstance(entries, list):
            return None
        return [str(item).strip() for item in entries if str(item).strip()][:3]

    def propose_cleaning_plan(self, summary: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Ask for an ordered cleaning plan from statistical summaries only."""

        raw = self._complete(
            PLAN_SYSTEM_PROMPT,
            json.dumps(summary, ensure_ascii=False, default=str),
            # This prompt's contract is a top-level JSON *array*, which
            # conflicts with response_format: json_object (that mode forces
            # an object) — some models then wrap a single step directly
            # instead of {"steps": [...]}. Leave JSON mode off here and rely
            # on _extract_json's bracket matching instead.
            json_mode=False,
        )
        if raw is None:
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            logger.warning("OpenRouter plan response was not JSON: %r", raw[:200])
            return None
        if isinstance(parsed, dict):
            # A model that ignores the "array" instruction may hand back
            # either {"steps": [...]} or a single step object directly.
            parsed = parsed.get("steps", [parsed] if "type" in parsed else [])
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
