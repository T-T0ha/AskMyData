"""The BIRD evaluation harness's own logic — scoring, sampling, the model proxy,
name translation, accounting and reporting.

No model call is made and no pipeline is built: every test works on plain values.
(An end-to-end run needs an OpenRouter key and the downloaded dataset; see
``evaluation/bird/README.md``.)
"""

from __future__ import annotations

import json

import pytest

from evaluation.bird import dataset, llm, metric, report


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _compare(pred, gold, max_extra=3):
    return metric.compare(pred, gold, len(pred[0]) if pred else 1, len(gold[0]) if gold else 1, max_extra)


def test_identical_results_pass_every_verdict():
    verdict = _compare([("a", 1), ("b", 2)], [("b", 2), ("a", 1)])
    assert verdict.strict and verdict.normalized and verdict.relaxed
    assert verdict.extra_columns == 0


def test_a_number_and_the_same_number_as_text_differ_only_strictly():
    """The export stores some numeric columns as text; that is not a wrong answer."""

    verdict = _compare([("103",), ("452",)], [(103.0,), (452.0,)])
    assert not verdict.strict
    assert verdict.normalized and verdict.relaxed


def test_a_midnight_timestamp_equals_its_date():
    verdict = _compare([("2014-02-20 00:00:00.000000",)], [("2014-02-20",)])
    assert not verdict.strict and verdict.normalized


def test_an_extra_column_passes_relaxed_but_not_normalized():
    """"Which schools…?" answered with the name *and* the id has found the answer."""

    pred = [("Acme", 1), ("Bolt", 2)]
    gold = [("Acme",), ("Bolt",)]
    verdict = _compare(pred, gold)
    assert not verdict.strict and not verdict.normalized
    assert verdict.relaxed and verdict.extra_columns == 1


def test_extra_columns_may_come_first_and_in_any_order():
    verdict = _compare([(1, "Acme", "x"), (2, "Bolt", "y")], [("Acme",), ("Bolt",)])
    assert verdict.relaxed and verdict.extra_columns == 2


def test_gold_columns_are_matched_in_any_order():
    verdict = _compare([("x", 1), ("y", 2)], [(1, "x"), (2, "y")])
    assert verdict.relaxed and not verdict.normalized and verdict.extra_columns == 0


def test_extra_rows_are_never_forgiven():
    """The row condition is what stops ``SELECT *`` over everything from passing."""

    pred = [("Acme", 1), ("Bolt", 2), ("Cobb", 3)]
    gold = [("Acme",), ("Bolt",)]
    assert not _compare(pred, gold).relaxed


def test_the_right_columns_with_the_wrong_pairing_fail():
    """Two columns whose values match the gold's as sets, but not row by row."""

    pred = [("a", 2), ("b", 1)]
    gold = [("a", 1), ("b", 2)]
    assert not _compare(pred, gold).relaxed


def test_too_many_surplus_columns_is_not_a_small_extra():
    pred = [("Acme", 1, 2, 3, 4)]
    gold = [("Acme",)]
    verdict = _compare(pred, gold, max_extra=3)
    assert verdict.relaxed_match and verdict.extra_columns == 4
    assert not verdict.relaxed
    assert _compare(pred, gold, max_extra=4).relaxed


def test_a_missing_column_fails():
    assert not _compare([("Acme",)], [("Acme", 1)]).relaxed


def test_empty_results_agree_with_each_other():
    verdict = metric.compare([], [], 1, 1)
    assert verdict.strict and verdict.normalized and verdict.relaxed
    assert metric.compare([], [], 3, 1).relaxed  # an empty answer with extra columns is still empty
    assert not metric.compare([("x",)], [], 1, 1).relaxed


def test_the_three_verdicts_are_nested():
    cases = [
        ([("1",)], [(1,)]),
        ([("a", 1)], [("a",)]),
        ([("a",)], [("b",)]),
        ([(1.0000001,)], [(1.0,)]),
    ]
    for pred, gold in cases:
        verdict = _compare(pred, gold)
        assert (not verdict.strict) or verdict.normalized
        assert (not verdict.normalized) or verdict.relaxed


def test_score_runs_both_queries(tmp_path):
    import sqlite3

    path = tmp_path / "t.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("create table t (id integer, name text)")
        conn.executemany("insert into t values (?, ?)", [(1, "a"), (2, "b")])

    ok = metric.score("select id, name from t", "select name from t", path)
    assert ok.relaxed and not ok.normalized and ok.extra_columns == 1
    assert ok.predicted_rows == 2 and ok.gold_columns == 1

    broken = metric.score("select nope from t", "select name from t", path)
    assert broken.predicted_error and not broken.relaxed
    assert metric.score(None, "select name from t", path).predicted_error == "no SQL produced"
    assert metric.score("select 1", "select nope from t", path).gold_error


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

needs_data = pytest.mark.skipif(not dataset.DEV_JSON.exists(), reason="BIRD dev set not downloaded")


@needs_data
def test_the_default_sample_is_100_100_100_and_reproducible():
    examples = dataset.load_all()
    bands = {"simple": 100, "moderate": 100, "challenging": 100}
    first = dataset.sample_by_difficulty(examples, bands, dataset.DEFAULT_DATABASES, seed=0)
    again = dataset.sample_by_difficulty(examples, bands, dataset.DEFAULT_DATABASES, seed=0)
    other = dataset.sample_by_difficulty(examples, bands, dataset.DEFAULT_DATABASES, seed=1)

    assert len(first) == 300
    for difficulty, wanted in bands.items():
        assert sum(1 for e in first if e.difficulty == difficulty) == wanted
    assert {e.db_id for e in first} <= set(dataset.DEFAULT_DATABASES)
    assert [e.question_id for e in first] == [e.question_id for e in again]
    assert [e.question_id for e in first] != [e.question_id for e in other]
    assert len({(e.db_id, e.question_id) for e in first}) == 300


@needs_data
def test_a_band_the_databases_cannot_fill_is_an_error_not_a_short_sample():
    with pytest.raises(ValueError, match="challenging"):
        dataset.sample_by_difficulty(
            dataset.load_all(), {"challenging": 100}, ["california_schools"], seed=0
        )


# ---------------------------------------------------------------------------
# the model proxy
# ---------------------------------------------------------------------------


class _FakeBase:
    available = True
    last_error = None

    def __init__(self, reply='{"sql": "SELECT 1", "explanation": "one", "confidence": 0.9}'):
        self.calls = []
        self.reply = reply

    def _complete(self, system, user, max_tokens=None, **kwargs):
        self.calls.append({"system": system, "user": json.loads(user), "max_tokens": max_tokens})
        return self.reply

    def check_ambiguity(self, question, tables, business_rules=None):
        self.calls.append({"ambiguity_rules": business_rules})
        return {"ambiguous": False}


SEMANTIC_TABLES = [
    {
        "table": "schools",
        "table_type": "unknown",
        "columns": [
            {
                "name": "cdscode",
                "type": "VARCHAR(100)",
                "taxonomy": "identifier",
                "is_primary_key": True,
                "is_foreign_key": False,
                "references": None,
                "is_additive": False,
                "null_ratio": 0.0,
                "sample_values": ["01100170109835", "x" * 200],
            },
            {
                "name": "county",
                "type": "TEXT",
                "taxonomy": "unknown",
                "is_primary_key": False,
                "is_foreign_key": False,
                "references": None,
                "is_additive": False,
                "null_ratio": 0.0123456,
                "sample_values": [],
            },
        ],
    }
]


def test_the_semantic_schema_loses_only_default_values():
    tables, compacted = llm.compact_schema(SEMANTIC_TABLES)
    assert compacted
    code, county = tables[0]["columns"]
    assert "table_type" not in tables[0]
    assert code == {
        "name": "cdscode",
        "type": "VARCHAR(100)",
        "taxonomy": "identifier",
        "is_primary_key": True,
        "sample_values": ["01100170109835", "x" * llm.SAMPLE_CHARS],
    }
    assert county == {"name": "county", "type": "TEXT", "null_ratio": 0.012}


def test_a_names_only_schema_is_left_alone():
    bare = [{"table": "t", "columns": [{"name": "a"}, {"name": "b"}]}]
    tables, compacted = llm.compact_schema(bare)
    assert tables == bare and not compacted


def test_the_evidence_travels_as_a_business_rule_and_the_prompt_is_the_products():
    from app.semantics.claude_client import SQL_SYSTEM_PROMPT

    base = _FakeBase()
    model = llm.EvalModel(base, evidence="rate = a / b", max_tokens=1234)
    answer = model.generate_sql(
        "q?", "sqlite", SEMANTIC_TABLES, business_rules=["recorded rule"], examples=[{"question": "e", "sql": "s"}]
    )
    call = base.calls[0]
    assert call["system"] == SQL_SYSTEM_PROMPT
    assert call["max_tokens"] == 1234
    assert call["user"]["business_rules"] == ["rate = a / b", "recorded rule"]
    assert call["user"]["examples"] == [{"question": "e", "sql": "s"}]
    assert call["user"]["dialect"] == "sqlite" and "schema_note" in call["user"]
    assert answer == {"sql": "SELECT 1", "explanation": "one", "confidence": 0.9}
    assert model.sent["examples"] == ["e"] and model.sent["rules"] == 2
    assert model.sent["column_names"] == ["cdscode", "county"]


def test_no_evidence_means_no_business_rule():
    base = _FakeBase()
    llm.EvalModel(base).generate_sql("q?", "sqlite", [{"table": "t", "columns": [{"name": "a"}]}])
    assert base.calls[0]["user"]["business_rules"] == []
    assert "schema_note" not in base.calls[0]["user"]


def test_a_retry_carries_the_previous_attempt():
    base = _FakeBase()
    llm.EvalModel(base).generate_sql("q?", "sqlite", [], prior_sql="SELECT x", prior_error="no such column")
    assert base.calls[0]["user"]["previous_attempt"] == {"sql": "SELECT x", "error": "no such column"}


def test_unusable_replies_are_none_with_a_reason():
    for reply, reason in [(None, "no usable"), ("not json", "valid JSON"), ('{"sql": ""}', "no SQL")]:
        model = llm.EvalModel(_FakeBase(reply))
        assert model.generate_sql("q?", "sqlite", []) is None
        assert reason in model.last_error


def test_the_optional_phase_5_calls_are_absent_unless_asked_for():
    """``answer_question`` probes for these with ``getattr`` and skips what is missing."""

    plain = llm.EvalModel(_FakeBase())
    for name in ("check_ambiguity", "rewrite_question", "suggest_followups", "suggest_questions"):
        assert getattr(plain, name, None) is None

    base = _FakeBase()
    gated = llm.EvalModel(base, evidence="e", product_gates=True)
    gated.check_ambiguity("q?", [], business_rules=["r"])
    assert base.calls[-1]["ambiguity_rules"] == ["e", "r"]


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------


class _Resp:
    class usage:
        prompt_tokens = 100
        completion_tokens = 40
        cost = 0.002

        class completion_tokens_details:
            reasoning_tokens = 25

        class prompt_tokens_details:
            cached_tokens = 60

    class _Choice:
        finish_reason = "stop"

    choices = [_Choice()]


class _Boom(Exception):
    def __init__(self, status):
        super().__init__(f"http {status}")
        self.status_code = status


def _client(*outcomes):
    """A stand-in client whose ``create`` plays back responses / exceptions."""

    queue = list(outcomes)

    class Completions:
        def create(self, **kwargs):
            outcome = queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    class Claude:
        _client = Client()

    return Claude()


def test_tokens_are_counted_per_scope_and_per_phase():
    meter = llm.UsageMeter()
    claude = _client(_Resp(), _Resp())
    llm.install_meter(claude, meter)
    with meter.scope("raw") as first:
        claude._client.chat.completions.create(model="m")
    with meter.scope("raw") as second:
        claude._client.chat.completions.create(model="m")

    assert first.to_dict()["prompt_tokens"] == 100 and first.to_dict()["reasoning_tokens"] == 25
    assert first.to_dict()["cached_tokens"] == 60
    assert first.to_dict()["finish_reasons"] == {"stop": 1}
    assert meter.totals()["raw"]["total_tokens"] == 280
    assert meter.totals()["raw"]["cost_usd"] == pytest.approx(0.004)
    assert second.calls == 1


def test_an_error_the_next_call_recovered_from_is_not_a_failure():
    """``_complete`` retries a rejected JSON-mode request as plain text."""

    meter = llm.UsageMeter()
    claude = _client(_Boom(400), _Resp())
    llm.install_meter(claude, meter)
    create = claude._client.chat.completions.create
    with meter.scope("build") as usage:
        with pytest.raises(_Boom):
            create(model="m")
        create(model="m")
    assert usage.api_errors == 0 and usage.calls == 2


def test_an_error_nothing_recovered_from_is_counted():
    meter = llm.UsageMeter()
    claude = _client(_Boom(500), _Boom(500))
    llm.install_meter(claude, meter)
    create = claude._client.chat.completions.create
    with meter.scope("build") as usage:
        for _ in range(2):
            with pytest.raises(_Boom):
                create(model="m")
    assert usage.api_errors == 2 and "500" in usage.last_api_error


def _reply(finish="stop", content='{"sql": "SELECT 1"}', error=None, tokens=100):
    """A response as the OpenAI client hands it back, with the fields the harness reads."""

    from types import SimpleNamespace

    choice = SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))
    if error is not None:
        choice.error = error
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=tokens, completion_tokens=tokens, cost=0.0),
    )


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER_FAULT_BACKOFF", (0.0, 0.0))


def test_a_reply_the_provider_failed_to_produce_is_asked_for_again(no_backoff):
    """OpenRouter reports an upstream failure as a *successful* HTTP call whose
    ``finish_reason`` is ``error``.  The product reads that as "no usable answer"
    and the question was scored as a wrong SQL answer — the same question the
    other arm, called a moment later, answered."""

    meter = llm.UsageMeter()
    good = _reply()
    claude = _client(_reply(finish="error", content="", tokens=0), good)
    llm.install_meter(claude, meter)
    with meter.scope("semantic") as usage:
        got = claude._client.chat.completions.create(model="m")

    assert got is good
    assert usage.calls == 2 and usage.api_errors == 0
    assert usage.to_dict()["finish_reasons"] == {"error": 1, "stop": 1}


def test_a_provider_that_keeps_failing_is_an_api_failure_not_a_wrong_answer(no_backoff):
    """Counted, so the runner skips the question and the next run asks it again."""

    meter = llm.UsageMeter()
    outcomes = [
        _reply(finish="error", content="", tokens=0, error={"code": 502, "message": "upstream"})
        for _ in range(llm.PROVIDER_FAULT_RETRIES + 1)
    ]
    claude = _client(*outcomes, _reply())
    llm.install_meter(claude, meter)
    create = claude._client.chat.completions.create
    with meter.scope("semantic") as usage:
        failed = create(model="m")
        assert failed.choices[0].finish_reason == "error", "the product still sees the failure"
        assert usage.calls == llm.PROVIDER_FAULT_RETRIES + 1
        assert usage.api_errors == 1
        assert "502" in usage.last_api_error
        create(model="m")  # the product goes on to its next call: that must not erase the failure
        assert usage.api_errors == 1


def test_an_empty_reply_is_asked_for_again_but_a_cut_off_one_is_not(no_backoff):
    meter = llm.UsageMeter()
    good = _reply()
    claude = _client(_reply(finish="stop", content="  "), good)
    llm.install_meter(claude, meter)
    with meter.scope("raw") as usage:
        assert claude._client.chat.completions.create(model="m") is good
    assert usage.calls == 2 and usage.api_errors == 0

    # the reasoning used up the token budget: the same call would do it again, so
    # it is the model's result (the product reports it) rather than something to retry
    cut = _reply(finish="length", content="")
    claude = _client(cut)  # a second call would pop from an empty queue and fail the test
    llm.install_meter(claude, meter)
    with meter.scope("raw") as usage:
        assert claude._client.chat.completions.create(model="m") is cut
    assert usage.calls == 1 and usage.api_errors == 0


def test_a_provider_fault_does_not_swallow_a_rejected_key(no_backoff):
    meter = llm.UsageMeter()
    claude = _client(_reply(finish="error", content="", tokens=0), _Boom(402))
    llm.install_meter(claude, meter)
    with meter.scope("raw"):
        with pytest.raises(llm.CreditsExhausted, match="402"):
            claude._client.chat.completions.create(model="m")


@pytest.mark.parametrize("status", [401, 402])
def test_a_rejected_key_or_empty_wallet_stops_the_run_outright(status):
    """A ``BaseException``, so ``ClaudeClient._complete``'s ``except Exception`` cannot swallow it."""

    meter = llm.UsageMeter()
    claude = _client(_Boom(status))
    llm.install_meter(claude, meter)
    assert not issubclass(llm.CreditsExhausted, Exception)
    with meter.scope("raw"):
        with pytest.raises(llm.CreditsExhausted, match=str(status)):
            claude._client.chat.completions.create(model="m")


def test_reasoning_effort_is_added_only_where_the_product_did_not_choose():
    seen = []

    class Completions:
        def create(self, **kwargs):
            seen.append(kwargs)
            return _Resp()

    class Claude:
        class _client:
            class chat:
                completions = Completions()

    claude = Claude()
    llm.install_meter(claude, llm.UsageMeter(), reasoning_effort="low")
    claude._client.chat.completions.create(model="m")
    claude._client.chat.completions.create(model="m", extra_body={"reasoning": {"enabled": False}})
    assert seen[0]["extra_body"]["reasoning"] == {"effort": "low"}
    assert seen[1]["extra_body"]["reasoning"] == {"enabled": False}
    assert all(call["extra_body"]["usage"] == {"include": True} for call in seen)


# ---------------------------------------------------------------------------
# BIRD names -> exported names
# ---------------------------------------------------------------------------


def _names():
    from evaluation.bird.session_build import NameMap

    names = NameMap(
        table_origin={"frpm": "frpm", "schools": "schools"},
        labels={
            "free meal count (k-12)": {"free_meal_count_k_12"},
            "enrollment (k-12)": {"enrollment_k_12"},
            "county": {"county"},
            "cdscode": {"cdscode"},
            "id": {"id", "id_2"},
        },
    )
    names._build_rewrite()
    return names


def test_evidence_is_rewritten_into_the_names_the_semantic_arm_sees():
    names = _names()
    evidence = "Eligible free rate = `Free Meal Count (K-12)` / `Enrollment (K-12)` where County = 'Alameda'"
    assert names.translate(evidence) == (
        "Eligible free rate = `free_meal_count_k_12` / `enrollment_k_12` where County = 'Alameda'"
    )


def test_a_name_that_only_changed_case_is_left_alone():
    assert _names().translate("frpm.CDSCode and County") == "frpm.CDSCode and County"


def test_an_ambiguous_label_is_not_guessed_at():
    assert _names().translate("the ID column") == "the ID column"


def test_the_gold_columns_are_looked_up_under_their_exported_names():
    names = _names()
    assert names.exported_column_names("Free Meal Count (K-12)") == {"free_meal_count_k_12"}
    assert names.exported_column_names("Whatever") == {"whatever"}


def test_the_drift_report_tells_a_changed_value_from_a_retyped_column_from_a_respelling(tmp_path):
    import sqlite3

    from evaluation.bird.session_build import NameMap, column_drift

    def make(path, rows, score_type):
        with sqlite3.connect(path) as con:
            con.execute(f"CREATE TABLE t (element TEXT, score {score_type}, opened TEXT, label TEXT)")
            con.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", rows)

    original, exported = tmp_path / "bird.sqlite", tmp_path / "export.sqlite"
    make(original, [("na", 512, "2002-09-08", "x"), ("cl", 480, "2002-05-21", "y")], "INTEGER")
    # 'na' became missing; score became text; the dates gained a time; label is untouched.
    make(exported, [(None, "512", "2002-09-08 00:00:00.000000", "x"), ("cl", "480", "2002-05-21 00:00:00.000000", "y")], "TEXT")

    names = NameMap(columns={("t", c): ("t", c) for c in ("element", "score", "opened", "label")})
    drift = column_drift(original, exported, names)

    kinds = {(table, column): found for table, column, found in drift["drifted_columns"]}
    assert kinds == {
        ("t", "element"): ["values"],
        ("t", "score"): ["retyped"],
        ("t", "opened"): ["reformatted"],
    }
    assert drift["by_kind"] == {"values": 1, "retyped": 1, "reformatted": 1}
    assert drift["columns_compared"] == 4


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _record(arm, qid, *, strict, normalized=None, relaxed=None, extra=None, difficulty="simple", tokens=100, **more):
    normalized = strict if normalized is None else normalized
    relaxed = normalized if relaxed is None else relaxed
    return {
        "arm": arm,
        "db_id": "db",
        "question_id": qid,
        "difficulty": difficulty,
        "outcome": "answered",
        "strict": strict,
        "normalized": normalized,
        "relaxed": relaxed,
        "relaxed_match": relaxed,
        "extra_columns": extra if extra is not None else (0 if relaxed else None),
        "correct": relaxed,
        "n_attempts": 1,
        "first_attempt_ok": True,
        "predicted_sql": "select 1",
        "elapsed_seconds": 2.0,
        "usage": {"total_tokens": tokens, "prompt_tokens": tokens - 10, "completion_tokens": 10, "cost_usd": 0.01},
        **more,
    }


def test_the_summary_reads_a_run_under_any_metric():
    records = [
        _record("raw", 1, strict=True),
        _record("raw", 2, strict=False, normalized=False, relaxed=True, extra=2),
        _record("raw", 3, strict=False, normalized=False, relaxed=False),
        _record("semantic", 1, strict=True),
        _record("semantic", 2, strict=False, normalized=True),
        _record("semantic", 3, strict=True),
    ]
    summary = report.summarize(records, primary="relaxed")
    raw, sem = summary["arms"]["raw"], summary["arms"]["semantic"]

    assert raw["by_metric"] == {"strict": 0.3333, "normalized": 0.3333, "relaxed": 0.6667}
    assert raw["rescued_by_extra_columns"] == 1
    assert sem["by_metric"] == {"strict": 0.6667, "normalized": 1.0, "relaxed": 1.0}
    assert sem["rescued_by_value_normalisation"] == 1
    assert sem["accuracy"] == 1.0 and raw["accuracy"] == pytest.approx(0.6667)

    strict = report.summarize(records, primary="strict")
    assert strict["arms"]["semantic"]["accuracy"] == pytest.approx(0.6667)

    capped = report.summarize(records, primary="relaxed", max_extra=1)
    assert capped["arms"]["raw"]["accuracy"] == pytest.approx(0.3333)  # 2 surplus columns no longer pass


def test_the_paired_table_counts_who_won_each_question():
    records = [
        _record("raw", 1, strict=True),
        _record("raw", 2, strict=False),
        _record("raw", 3, strict=False),
        _record("raw", 4, strict=True),
        _record("semantic", 1, strict=True),
        _record("semantic", 2, strict=True),
        _record("semantic", 3, strict=False),
        _record("semantic", 4, strict=False),
    ]
    paired = report.summarize(records)["paired"]["by_metric"]["strict"]
    assert (paired["both_correct"], paired["semantic_only"], paired["raw_only"], paired["neither"]) == (1, 1, 1, 1)
    assert paired["semantic_minus_raw"] == 0.0 and paired["mcnemar_p"] == 1.0


def test_the_paired_table_splits_on_whether_the_gold_answer_survives_the_export():
    records = [
        _record("raw", 1, strict=True),
        _record("raw", 2, strict=True),
        _record("raw", 3, strict=True),
        _record("semantic", 1, strict=True, gold_on_export="same", drift_kinds=[], gold_touches_drift=False),
        _record("semantic", 2, strict=False, gold_on_export="different", drift_kinds=["values"], gold_touches_drift=True),
        _record("semantic", 3, strict=False, gold_on_export="not_runnable", drift_kinds=["retyped"], gold_touches_drift=True),
    ]
    summary = report.summarize(records)
    by_export = summary["paired"]["by_gold_on_export"]
    assert by_export["answer_preserved"]["both_correct"] == 1
    assert by_export["answer_changed"]["raw_only"] == 1
    assert by_export["not_runnable_unchanged"]["n"] == 1
    assert summary["arms"]["semantic"]["pipeline"]["gold_on_export"] == {"same": 1, "different": 1, "not_runnable": 1}
    by_kind = summary["paired"]["by_data_drift"]
    assert by_kind["gold_reads_a_values_column"]["n"] == 1 and by_kind["gold_reads_a_retyped_column"]["n"] == 1


def test_the_semantic_arm_reports_retrieval_and_selective_accuracy():
    records = [
        _record("semantic", 1, strict=True, confidence=0.95, table_recall=1.0, column_recall=0.5, examples_in_store=0, examples_sent=[]),
        _record("semantic", 2, strict=False, confidence=0.3, table_recall=0.5, column_recall=1.0, examples_in_store=3, examples_sent=["a"], example_saved=False),
    ]
    pipeline = report.summarize(records)["arms"]["semantic"]["pipeline"]
    assert pipeline["table_recall_avg"] == 0.75 and pipeline["column_recall_avg"] == 0.75
    assert pipeline["withheld_by_product_gate"] == 1
    at_floor = next(c for c in pipeline["selective_accuracy"] if c["min_confidence"] == report.CONFIDENCE_FLOOR)
    assert at_floor == {"min_confidence": 0.45, "coverage": 0.5, "accuracy_of_kept": 1.0}
    assert pipeline["accuracy_by_examples_in_store"]["0"]["n"] == 1


def test_a_half_written_last_line_is_skipped(tmp_path):
    path = tmp_path / "records.jsonl"
    good = _record("raw", 1, strict=True)
    path.write_text(json.dumps(good) + "\n" + '{"arm": "raw", "db_id": "d')
    assert [r["question_id"] for r in report.load_records(path)] == [1]


def test_the_flat_tables_are_written(tmp_path):
    records = [
        _record("raw", 1, strict=True, question="q", evidence="", gold_sql="g"),
        _record("semantic", 1, strict=False, question="q", evidence="", gold_sql="g"),
    ]
    report.write_tables(records, tmp_path)
    paired = (tmp_path / "paired.csv").read_text().splitlines()
    assert "winner" in paired[0] and paired[1].endswith("raw")
    assert len((tmp_path / "records.csv").read_text().splitlines()) == 3


def test_a_wilson_interval_brackets_the_estimate():
    low, high = report.wilson(60, 100)
    assert low < 0.6 < high and 0.0 <= low and high <= 1.0
    assert report.wilson(0, 0) is None


def test_a_session_built_before_the_data_revision_is_refused_rather_than_reused():
    """Ingestion writes the tables, so an old session cannot be fixed by re-exporting.

    Reusing one would put the old pipeline's numbers under the new pipeline's name.
    """

    import pytest

    from evaluation.bird import session_build

    with pytest.raises(session_build.StaleSession, match="--rebuild-sessions"):
        session_build.check_revision("superhero", {"session_id": "abc", "stage": "exported"})
    with pytest.raises(session_build.StaleSession):
        session_build.check_revision(
            "superhero", {"session_id": "abc", "data_revision": session_build.DATA_REVISION - 1}
        )

    session_build.check_revision(
        "superhero", {"session_id": "abc", "data_revision": session_build.DATA_REVISION}
    )
    session_build.check_revision("superhero", {})  # nothing built yet: nothing to refuse
    assert issubclass(session_build.StaleSession, session_build.BuildDegraded), (
        "the runner already reports a degraded build and carries on with the other databases"
    )


def test_a_lane_waits_for_a_write_lock_another_lane_holds(tmp_path):
    """The first real run lost toxicology's semantic arm to "database is locked".

    Each build stage is one transaction, so formula_1's ingestion held SQLite's one
    write lock for longer than the other lanes were willing to wait for it.
    """

    import threading
    import time

    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError

    from evaluation.bird import session_build

    def contend(busy_timeout_ms):
        engine = create_engine(
            f"sqlite:///{tmp_path / f'lock_{busy_timeout_ms}.db'}",
            connect_args={"check_same_thread": False},
        )
        session_build.tune_sqlite(engine, busy_timeout_ms)
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE t (x INTEGER)"))
        holding = threading.Event()

        def hold():
            with engine.begin() as connection:
                connection.execute(text("INSERT INTO t VALUES (1)"))
                holding.set()
                time.sleep(1.0)

        holder = threading.Thread(target=hold)
        holder.start()
        assert holding.wait(5)
        try:
            with engine.begin() as connection:
                connection.execute(text("INSERT INTO t VALUES (2)"))
        finally:
            holder.join()
        with engine.connect() as connection:
            return connection.execute(text("SELECT COUNT(*) FROM t")).scalar()

    assert contend(10_000) == 2, "waits for the holder to commit, then writes"
    with pytest.raises(OperationalError, match="locked"):
        contend(50)  # the same contention with a short wait is the failure that was seen
    assert session_build.BUSY_TIMEOUT_MS >= 5 * 60 * 1000, "a stage is minutes, not seconds"


def test_a_run_begun_on_an_older_export_is_not_resumed(tmp_path, monkeypatch):
    """Its semantic answers were measured on data the pipeline no longer produces;
    finishing it with new ones would put two different exports in one accuracy figure."""

    from dataclasses import asdict

    from evaluation.bird import runner, session_build

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    cfg = runner.RunConfig(run_name="older", model="m")
    assert cfg.data_revision == session_build.DATA_REVISION
    started_under = {**asdict(cfg), "data_revision": session_build.DATA_REVISION - 1}
    (tmp_path / "older").mkdir()
    (tmp_path / "older" / "config.json").write_text(json.dumps({"config": started_under}))
    (tmp_path / "older" / "questions.json").write_text("[]")

    with pytest.raises(SystemExit) as refused:
        runner.Run(cfg, object(), tmp_path).prepare()

    message = str(refused.value)
    assert "data_revision" in message and "new --run-name" in message


def _started_under(tmp_path, monkeypatch, **changed):
    """A run directory whose stored config differs from the default one in ``changed``."""

    from dataclasses import asdict

    from evaluation.bird import runner

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    cfg = runner.RunConfig(run_name="earlier", model="m")
    (tmp_path / "earlier").mkdir()
    (tmp_path / "earlier" / "config.json").write_text(json.dumps({"config": {**asdict(cfg), **changed}}))
    (tmp_path / "earlier" / "questions.json").write_text("[]")
    return cfg


def test_a_run_begun_with_the_evidence_on_is_not_resumed_with_it_off(tmp_path, monkeypatch):
    from evaluation.bird import runner

    assert runner.RunConfig().use_evidence is True, "BIRD's own protocol is the default"
    cfg = _started_under(tmp_path, monkeypatch, use_evidence=False)

    with pytest.raises(SystemExit) as refused:
        runner.Run(cfg, object(), tmp_path).prepare()
    assert "use_evidence" in str(refused.value)


def test_a_run_begun_under_other_retrieval_settings_is_not_resumed(tmp_path, monkeypatch):
    """The column cap, the similarity an example needs and the embedding model decide what
    the semantic arm's model is shown; a resumed run would be half one pipeline, half another."""

    from evaluation.bird import runner

    settings = runner.retrieval_settings()
    assert {"max_ranked_columns", "example_min_similarity", "embedding_model"} <= set(settings)
    cfg = _started_under(tmp_path, monkeypatch, retrieval={**settings, "max_ranked_columns": 40})

    with pytest.raises(SystemExit) as refused:
        runner.Run(cfg, object(), tmp_path).prepare()
    message = str(refused.value)
    assert "retrieval" in message and "new --run-name" in message


def test_a_run_resumed_under_the_same_settings_is_accepted(tmp_path, monkeypatch):
    from evaluation.bird import runner

    cfg = _started_under(tmp_path, monkeypatch)
    (tmp_path / "earlier" / "questions.json").write_text("[]")
    runner.Run(cfg, object(), tmp_path).prepare()  # does not raise

