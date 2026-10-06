"""The one model interface both arms of the comparison go through.

Three jobs, kept together because they all sit between the harness and
``app.semantics.claude_client.ClaudeClient``:

* :class:`UsageMeter` counts every token (and, when OpenRouter reports it, every
  dollar) the run spends, per question and per phase, by wrapping the OpenAI
  client's one network call.  Nothing in ``app`` is modified for this.
* :class:`CreditsExhausted` stops the run the moment the key is rejected or the
  account is out of credit.  ``ClaudeClient._complete`` swallows every ordinary
  exception and reports "no usable answer", which a benchmark would then score
  as a *wrong answer* — silently turning an empty wallet into a bad accuracy
  number.  A ``BaseException`` is not caught by ``except Exception``, so it
  climbs out of the product code untouched and the runner can checkpoint and stop.
* :class:`EvalModel` is the object ``answer_question`` (semantic arm) and
  :func:`evaluation.bird.predict.generate_with_retry` (raw arm) are handed in
  place of the raw client.  It sends the product's own ``SQL_SYSTEM_PROMPT``
  unchanged and the product's own payload shape, with exactly three differences,
  all identical for both arms: BIRD's evidence rides in ``business_rules`` (the
  slot the prompt already defines for domain rules) — added here for the raw arm,
  handed to ``answer_question(extra_rules=...)`` for the semantic arm, which puts
  it in the same place first — the JSON is sent compactly (token saving, no
  information removed), and ``max_tokens`` is the harness's setting rather than
  the product's fixed 4096 ceiling.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from app.semantics.claude_client import SQL_SYSTEM_PROMPT, _extract_json

#: HTTP statuses that mean "stop, do not keep spending" rather than "try again".
#: 401 — the key is wrong or revoked; 402 — OpenRouter's "insufficient credits".
FATAL_STATUSES = frozenset({401, 402})

#: How often a reply the *provider* failed to produce is asked for again, and the
#: seconds waited before each repeat.  OpenRouter reports an upstream failure
#: (``finish_reason: "error"``, no tokens) or an empty message as a *successful*
#: HTTP call, so the OpenAI client does not retry it and the product reads "no
#: usable answer".  Left alone, that scores a transient provider hiccup as a wrong
#: SQL answer — and which arm it lands on is luck.
PROVIDER_FAULT_RETRIES = 2
PROVIDER_FAULT_BACKOFF = (2.0, 6.0)


class CreditsExhausted(BaseException):
    """Raised from inside the model call when continuing cannot succeed.

    Deliberately *not* an ``Exception`` subclass — see the module docstring.
    """


# ---------------------------------------------------------------------------
# usage metering
# ---------------------------------------------------------------------------


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class Usage:
    """Counters for one scope (a question, a build stage)."""

    __slots__ = (
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "cost_usd",
        "api_errors",
        "last_api_error",
        "call_seconds",
        "finish_reasons",
    )

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.cost_usd = 0.0
        self.api_errors = 0
        self.last_api_error: str | None = None
        self.call_seconds = 0.0
        self.finish_reasons: Counter[str] = Counter()

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_tokens += other.cached_tokens
        self.cost_usd += other.cost_usd
        self.api_errors += other.api_errors
        self.call_seconds += other.call_seconds
        self.finish_reasons.update(other.finish_reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "api_errors": self.api_errors,
            "call_seconds": round(self.call_seconds, 2),
            "finish_reasons": dict(self.finish_reasons),
        }


class UsageMeter:
    """Thread-safe token accounting with nested, per-thread scopes.

    A call is added to the innermost scope open *on the calling thread*, and to a
    run-wide total under that scope's phase label, so concurrent lanes never
    mix their numbers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self.by_phase: dict[str, Usage] = {}

    @contextmanager
    def scope(self, phase: str) -> Iterator[Usage]:
        usage = Usage()
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        stack.append((phase, usage))
        try:
            yield usage
        finally:
            stack.pop()
            with self._lock:
                self.by_phase.setdefault(phase, Usage()).add(usage)

    def _current(self) -> Usage | None:
        stack = getattr(self._local, "stack", None)
        return stack[-1][1] if stack else None

    def record_response(self, response: Any, seconds: float) -> None:
        usage = self._current()
        if usage is None:
            return
        # ``ClaudeClient._complete`` answers a rejected JSON-mode request by
        # immediately asking again in plain text.  An error that the very next
        # call on this thread recovered from is not a failed question, so it is
        # taken back out of the count; two errors in a row, or an error nobody
        # recovered, stay in it.
        if getattr(self._local, "errored", None) is usage and usage.api_errors > 0:
            usage.api_errors -= 1
        self._local.errored = None
        raw = getattr(response, "usage", None)
        usage.calls += 1
        usage.call_seconds += seconds
        usage.prompt_tokens += int(_number(getattr(raw, "prompt_tokens", 0)))
        usage.completion_tokens += int(_number(getattr(raw, "completion_tokens", 0)))
        details = getattr(raw, "completion_tokens_details", None)
        usage.reasoning_tokens += int(_number(getattr(details, "reasoning_tokens", 0)))
        # Prompt tokens the provider served from its cache (billed at a discount, where it caches at all).
        prompt_details = getattr(raw, "prompt_tokens_details", None)
        usage.cached_tokens += int(_number(getattr(prompt_details, "cached_tokens", 0)))
        usage.cost_usd += _number(getattr(raw, "cost", 0))
        choices = getattr(response, "choices", None) or []
        if choices:
            usage.finish_reasons[str(getattr(choices[0], "finish_reason", None))] += 1

    def record_error(self, exc: BaseException, seconds: float) -> None:
        usage = self._current()
        if usage is None:
            return
        usage.calls += 1
        usage.call_seconds += seconds
        usage.api_errors += 1
        usage.last_api_error = f"{type(exc).__name__}: {exc}"[:300]
        self._local.errored = usage

    def record_fault(self, detail: str) -> None:
        """The provider answered, every repeat included, with a failure instead of a reply.

        Counted like an exception — the question is then an *API failure*, not a
        wrong answer — but never marked as recoverable by "the next call": the
        product moves on to its next call after a failed one (the next batch of
        columns, say) and that success must not erase this.  The calls themselves
        were already counted by :meth:`record_response`.
        """

        usage = self._current()
        if usage is None:
            return
        usage.api_errors += 1
        usage.last_api_error = detail[:300]
        self._local.errored = None

    def totals(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {phase: usage.to_dict() for phase, usage in self.by_phase.items()}


def provider_fault(response: Any) -> str | None:
    """Why ``response`` is the provider's failure rather than the model's reply, or ``None``.

    Only what no amount of reading could turn into an answer: no choices, an
    upstream ``error`` finish, or an empty message.  A reply cut off by the token
    limit (``length``) or withheld by a content filter would come back the same
    way every time, so those are the model's result and are left to the caller.
    """

    choices = getattr(response, "choices", None) or []
    if not choices:
        return "the provider returned no choices"
    choice = choices[0]
    finish = getattr(choice, "finish_reason", None)
    if finish == "error":
        detail = getattr(choice, "error", None) or getattr(choice, "native_finish_reason", None)
        return "the provider ended the reply with an error" + (f" ({detail})" if detail else "")
    message = getattr(choice, "message", None)
    if message is not None and finish not in ("length", "content_filter"):
        if not (getattr(message, "content", None) or "").strip():
            return f"the provider returned an empty reply (finish_reason={finish})"
    return None


def install_meter(claude: Any, meter: UsageMeter, reasoning_effort: str | None = None) -> None:
    """Wrap the OpenAI client's ``chat.completions.create`` with metering.

    ``reasoning_effort`` ("low"/"medium"/"high") is added to calls that do not
    already say anything about reasoning — i.e. SQL generation, the call whose
    hidden thinking dominates the bill.  Calls that explicitly disable reasoning
    (labelling, descriptions, the ambiguity check) are left exactly as the
    product makes them.  Every call also asks OpenRouter to include its cost in
    ``usage``.

    A reply the provider itself failed to produce (:func:`provider_fault`) is
    asked for again, up to ``PROVIDER_FAULT_RETRIES`` times; a call that fails
    every time counts as an API error, so the question is skipped and asked again
    on the next run instead of being scored as a wrong answer.
    """

    client = getattr(claude, "_client", None)
    if client is None:
        return
    completions = client.chat.completions
    real_create = completions.create

    def create(*args: Any, **kwargs: Any) -> Any:
        extra = dict(kwargs.get("extra_body") or {})
        if reasoning_effort and "reasoning" not in extra:
            extra["reasoning"] = {"effort": reasoning_effort}
        # OpenRouter reports the dollar cost of a call in ``usage`` when asked.
        extra.setdefault("usage", {"include": True})
        kwargs["extra_body"] = extra
        for attempt in range(PROVIDER_FAULT_RETRIES + 1):
            started = time.monotonic()
            try:
                response = real_create(*args, **kwargs)
            except Exception as exc:
                elapsed = time.monotonic() - started
                meter.record_error(exc, elapsed)
                status = getattr(exc, "status_code", None)
                if status in FATAL_STATUSES:
                    raise CreditsExhausted(
                        f"OpenRouter answered {status} ({str(exc)[:200]}) — out of credits or the key "
                        "was rejected; progress so far is saved, add credit and re-run the same command"
                    ) from exc
                raise
            meter.record_response(response, time.monotonic() - started)
            fault = provider_fault(response)
            if fault is None:
                return response
            if attempt < PROVIDER_FAULT_RETRIES:
                time.sleep(PROVIDER_FAULT_BACKOFF[min(attempt, len(PROVIDER_FAULT_BACKOFF) - 1)])
                continue
            meter.record_fault(fault)
        return response

    completions.create = create


# ---------------------------------------------------------------------------
# the model both arms are given
# ---------------------------------------------------------------------------

#: Printed once in the payload when a schema has had its default-valued keys
#: dropped, so the omission is read as "false / none / unknown" and not as
#: "not provided".  The product's SQL prompt describes the full set of fields.
SCHEMA_NOTE = (
    "a column key that is absent has its default: is_primary_key/is_foreign_key/"
    "is_additive false, references none, taxonomy unknown, null_ratio 0, no sample_values"
)

#: Longest sample value sent.  Five long free-text samples are most of the token
#: cost of a wide text column and say nothing a 40-character prefix does not.
SAMPLE_CHARS = 40


def _compact_column(column: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"name": column.get("name")}
    if column.get("type"):
        out["type"] = column["type"]
    taxonomy = column.get("taxonomy")
    if taxonomy and taxonomy != "unknown":
        out["taxonomy"] = taxonomy
    if column.get("is_primary_key"):
        out["is_primary_key"] = True
    if column.get("is_foreign_key"):
        out["is_foreign_key"] = True
    if column.get("references"):
        out["references"] = column["references"]
    if column.get("is_additive"):
        out["is_additive"] = True
    null_ratio = column.get("null_ratio") or 0.0
    if null_ratio > 0:
        out["null_ratio"] = round(float(null_ratio), 3)
    samples = [
        value[:SAMPLE_CHARS] if isinstance(value, str) else value
        for value in (column.get("sample_values") or [])
    ]
    if samples:
        out["sample_values"] = samples
    return out


def compact_schema(tables: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """The same schema with default-valued keys dropped.  ``(tables, was_compacted)``.

    A schema that carries nothing but names (the raw arm) comes back untouched.
    """

    if not any("type" in column for table in tables for column in table.get("columns", [])):
        return list(tables), False
    out = []
    for table in tables:
        entry: dict[str, Any] = {"table": table.get("table")}
        if table.get("table_type") and table["table_type"] != "unknown":
            entry["table_type"] = table["table_type"]
        entry["columns"] = [_compact_column(column) for column in table.get("columns", [])]
        out.append(entry)
    return out, True


class EvalModel:
    """What ``answer_question`` / the raw loop call instead of the raw client.

    Only the methods the harness wants the product to use are defined, on
    purpose.  ``answer_question`` probes for ``check_ambiguity``,
    ``rewrite_question`` and ``suggest_followups`` with ``getattr(..., None)``
    and degrades gracefully when they are missing, so leaving them out is how
    the follow-up suggestions (a model call per answered question, useless to a
    benchmark) and — unless ``product_gates`` — the ambiguity check are turned
    off without touching the product.
    """

    def __init__(
        self,
        base: Any,
        *,
        evidence: str = "",
        max_tokens: int = 8192,
        compact: bool = True,
        product_gates: bool = False,
    ) -> None:
        self._base = base
        self._evidence = (evidence or "").strip()
        self._max_tokens = max_tokens
        self._compact = compact
        self.last_error: str | None = None
        #: What the first generation call actually sent, for the result record.
        self.sent: dict[str, Any] = {}
        if product_gates:
            self.check_ambiguity = self._check_ambiguity  # type: ignore[assignment]

    @property
    def available(self) -> bool:
        return bool(self._base.available)

    def _rules(self, business_rules: Sequence[str] | None) -> list[str]:
        rules = [self._evidence] if self._evidence else []
        rules.extend(str(rule) for rule in (business_rules or []))
        return rules

    def generate_sql(
        self,
        question: str,
        dialect: str,
        schema: list[dict[str, Any]],
        prior_sql: str | None = None,
        prior_error: str | None = None,
        business_rules: list[str] | None = None,
        examples: list[dict[str, str]] | None = None,
        conversation: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        tables, compacted = compact_schema(schema) if self._compact else (list(schema), False)
        rules = self._rules(business_rules)
        payload: dict[str, Any] = {
            "question": question,
            "dialect": dialect,
            "tables": tables,
            "business_rules": rules,
            "examples": examples or [],
        }
        if compacted:
            payload["schema_note"] = SCHEMA_NOTE
        if conversation:
            payload["conversation"] = conversation
        if prior_sql is not None:
            payload["previous_attempt"] = {"sql": prior_sql, "error": prior_error}

        if not self.sent:
            self.sent = {
                "tables": [table.get("table") for table in schema],
                "columns": sum(len(table.get("columns", [])) for table in schema),
                "column_names": [
                    str(column.get("name")).lower()
                    for table in schema
                    for column in table.get("columns", [])
                ],
                "examples": [str(example.get("question")) for example in (examples or [])],
                "rules": len(rules),
            }

        text = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
        raw = self._base._complete(SQL_SYSTEM_PROMPT, text, max_tokens=self._max_tokens)
        if raw is None:
            self.last_error = self._base.last_error or "the model returned no usable answer"
            return None
        try:
            parsed = _extract_json(raw)
        except ValueError:
            self.last_error = "the model's reply was not valid JSON"
            return None
        if not isinstance(parsed, dict):
            self.last_error = "the model's reply was not a JSON object"
            return None
        sql = str(parsed.get("sql", "") or "").strip()
        if not sql:
            self.last_error = "the model's reply held no SQL"
            return None
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "sql": sql,
            "explanation": str(parsed.get("explanation", "") or "").strip(),
            "confidence": max(0.0, min(confidence, 1.0)),
        }

    def _check_ambiguity(
        self, question: str, tables: list[dict[str, Any]], business_rules: list[str] | None = None
    ) -> dict[str, Any] | None:
        return self._base.check_ambiguity(question, tables, business_rules=self._rules(business_rules))
