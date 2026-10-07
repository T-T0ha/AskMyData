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



# ---------------------------------------------------------------------------
# asking order, batches, checkpoints, and refusing a layer the model barely wrote
# ---------------------------------------------------------------------------


def _q(db, question_id, difficulty):
    return dataset.BirdExample(question_id, db, f"question {question_id}", "", "select 1", difficulty)


def test_questions_are_asked_database_by_database_simple_first():
    mixed = [
        _q("b", 1, "challenging"),
        _q("a", 2, "moderate"),
        _q("c", 3, "simple"),
        _q("a", 4, "simple"),
        _q("b", 5, "simple"),
        _q("a", 6, "simple"),
        _q("z", 7, "simple"),
        _q("a", 8, "unknown"),
    ]
    ordered = dataset.run_order(mixed, ["b", "a"])

    assert [(q.db_id, q.difficulty, q.question_id) for q in ordered] == [
        ("b", "simple", 5),
        ("b", "challenging", 1),
        ("a", "simple", 4),  # 4 before 6: inside a band the incoming (seeded) order is kept
        ("a", "simple", 6),
        ("a", "moderate", 2),
        ("a", "unknown", 8),
        ("c", "simple", 3),  # databases that are not listed follow, by name
        ("z", "simple", 7),
    ]
    assert sorted(q.question_id for q in ordered) == sorted(q.question_id for q in mixed), "nothing added or dropped"


@needs_data
def test_the_sample_does_not_depend_on_the_order_databases_are_listed():
    """Listing the databases in another order only changes when each is asked."""

    bands = {"simple": 100, "moderate": 100, "challenging": 100}
    examples = dataset.load_all()
    listed = list(dataset.DEFAULT_DATABASES)
    one = dataset.sample_by_difficulty(examples, bands, listed, seed=0)
    other = dataset.sample_by_difficulty(examples, bands, list(reversed(listed)), seed=0)
    assert [e.question_id for e in one] == [e.question_id for e in other]


@needs_data
def test_the_default_order_lines_three_batches_of_100_up_with_the_databases():
    """The numbers the run's commands promise: what each invocation of 100 covers."""

    bands = {"simple": 100, "moderate": 100, "challenging": 100}
    sample = dataset.run_order(
        dataset.sample_by_difficulty(dataset.load_all(), bands, dataset.DEFAULT_DATABASES, seed=0),
        dataset.DEFAULT_DATABASES,
    )
    first, second, third = sample[:100], sample[100:200], sample[200:]

    per_database = {db: sum(1 for q in sample if q.db_id == db) for db in dataset.DEFAULT_DATABASES}
    assert per_database == {
        "formula_1": 53,
        "student_club": 47,
        "thrombosis_prediction": 69,
        "california_schools": 27,
        "toxicology": 60,
        "superhero": 44,
    }
    assert {q.db_id for q in first} == {"formula_1", "student_club"}
    assert {q.db_id for q in second} == {"thrombosis_prediction", "california_schools", "toxicology"}
    assert {q.db_id for q in third} == {"toxicology", "superhero"}
    assert sum(1 for q in second if q.db_id == "toxicology") == 4, "the only database split between batches"


def test_an_invocation_asks_the_next_n_unfinished_questions_in_asking_order(tmp_path, monkeypatch):
    from evaluation.bird import runner

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    questions = dataset.run_order(
        [_q("b", 4, "challenging"), _q("a", 1, "moderate"), _q("b", 3, "simple"), _q("a", 2, "simple"), _q("a", 5, "challenging")],
        ["a", "b"],
    )
    run = runner.Run(runner.RunConfig(run_name="r", model="m", batch_size=2), object(), tmp_path)
    run.questions = questions
    assert [q.question_id for q in questions] == [2, 1, 5, 3, 4]
    assert run.database_order == ["a", "b"]
    assert [q.question_id for q in run.batch()] == [2, 1]

    # question 2 has both arms' answers; question 1 only the raw arm's, so it is still owed one
    run.done = {("raw", "a", 2): {}, ("semantic", "a", 2): {}, ("raw", "a", 1): {}}
    assert [q.question_id for q in run.batch()] == [1, 5]

    run.cfg.batch_size = None
    assert [q.question_id for q in run.batch()] == [1, 5, 3, 4], "no batch size: everything that is left"
    run.cfg.batch_size = 0
    assert len(run.batch()) == 4

    run.cfg.arms = ["raw"]
    assert [q.question_id for q in run.outstanding()] == [5, 3, 4], "judged by the arms of this run"


@needs_data
def test_the_order_databases_are_listed_in_is_not_part_of_the_experiment(tmp_path, monkeypatch):
    """Each database has its own session, so no answer depends on which one went first."""

    from evaluation.bird import runner

    listed = list(runner.RunConfig().databases)
    reordered, narrower = tmp_path / "reordered", tmp_path / "narrower"
    reordered.mkdir()
    narrower.mkdir()

    cfg = _started_under(reordered, monkeypatch, databases=list(reversed(listed)))
    runner.Run(cfg, object(), reordered).prepare()  # accepted

    cfg = _started_under(narrower, monkeypatch, databases=listed[:-1])
    with pytest.raises(SystemExit) as refused:
        runner.Run(cfg, object(), narrower).prepare()
    assert "databases" in str(refused.value), "but drawing from other databases is a different experiment"


def test_the_other_questions_are_every_question_the_earlier_run_did_not_ask():
    pool = (
        [_q("a", i, "simple") for i in range(1, 7)]
        + [_q("b", i, "moderate") for i in range(7, 11)]
        + [_q("c", 11, "simple")]
    )
    asked = {("a", 1), ("a", 3), ("b", 8)}

    rest = dataset.rest_of(pool, asked, ["a", "b"], seed=0)

    assert sorted((q.db_id, q.question_id) for q in rest) == [
        ("a", 2), ("a", 4), ("a", 5), ("a", 6), ("b", 7), ("b", 9), ("b", 10),
    ], "all that is left of the chosen databases, nothing of database c, nothing that was asked"
    assert [q.question_id for q in dataset.rest_of(pool, asked, ["a", "b"], seed=0)] == [q.question_id for q in rest]
    assert {q.question_id for q in dataset.rest_of(pool, asked, ["a", "b"], seed=1)} == {q.question_id for q in rest}
    assert len(dataset.rest_of(pool, asked)) == len(pool) - len(asked), "no database list: every database"


def _earlier_run(tmp_path, questions, records=(), **config):
    """A run directory ``tmp_path/first`` that a later run can continue."""

    from dataclasses import asdict

    from evaluation.bird import runner

    cfg = runner.RunConfig(run_name="first", model="m", **config)
    first = tmp_path / "first"
    first.mkdir()
    (first / "questions.json").write_text(json.dumps([{"db_id": d, "question_id": q} for d, q in questions]))
    (first / "config.json").write_text(
        json.dumps(
            {"config": asdict(cfg), "system_prompt_sha": runner._prompt_sha(), "question_count": len(questions)}
        )
    )
    (first / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return cfg


def test_a_continuation_is_the_earlier_questions_followed_by_every_other_one(tmp_path, monkeypatch):
    from evaluation.bird import runner

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    _earlier_run(tmp_path, [("a", 3), ("a", 1)])
    pool = [_q("a", i, "simple") for i in range(1, 6)] + [_q("z", 9, "simple")]

    cfg = runner.RunConfig(run_name="more", model="m", databases=["a"], continue_from="first")
    drawn = runner.draw_questions(cfg, pool)
    assert [q.question_id for q in drawn[:2]] == [3, 1], "the earlier run's questions, in the order it stored them"
    assert sorted(q.question_id for q in drawn[2:]) == [2, 4, 5], "then every other question of the database, none of z"

    cfg.limit = 1
    assert len(runner.draw_questions(cfg, pool)) == 3, "limit keeps only the first N added questions"


@needs_data
def test_continuing_the_default_sample_gives_the_whole_dev_set_of_the_six_databases(tmp_path, monkeypatch):
    from evaluation.bird import runner

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    examples = dataset.load_all()
    bands = {"simple": 100, "moderate": 100, "challenging": 100}
    sample = dataset.sample_by_difficulty(examples, bands, dataset.DEFAULT_DATABASES, seed=0)
    _earlier_run(tmp_path, [(q.db_id, q.question_id) for q in sample])

    drawn = runner.draw_questions(runner.RunConfig(run_name="more", model="m", continue_from="first"), examples)

    assert len(drawn) == len({(q.db_id, q.question_id) for q in drawn}) == 858
    assert [(q.db_id, q.question_id) for q in drawn[:300]] == [(q.db_id, q.question_id) for q in sample]


@needs_data
def test_a_continuation_needs_the_earlier_runs_settings_and_a_refused_one_leaves_nothing_behind(tmp_path, monkeypatch):
    from evaluation.bird import runner

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    _earlier_run(tmp_path, [], use_evidence=False)

    cfg = runner.RunConfig(run_name="more", model="m", continue_from="first")  # evidence on: not that run's
    with pytest.raises(SystemExit) as refused:
        runner.Run(cfg, object(), tmp_path).prepare()
    assert "use_evidence" in str(refused.value) and "--no-evidence" in str(refused.value)
    assert not (tmp_path / "more").exists(), "a refused continuation creates nothing"

    cfg = runner.RunConfig(run_name="more", model="m", continue_from="never_started")
    with pytest.raises(SystemExit) as refused:
        runner.Run(cfg, object(), tmp_path).prepare()
    assert "never_started" in str(refused.value) and not (tmp_path / "more").exists()


@needs_data
def test_a_continuation_and_a_resume_both_refuse_another_sql_prompt(tmp_path, monkeypatch):
    """The two parts of a run are one experiment only if both arms were given the same prompt."""

    from evaluation.bird import runner

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    _earlier_run(tmp_path, [])
    stored = json.loads((tmp_path / "first" / "config.json").read_text())
    stored["system_prompt_sha"] = "000000000000"
    (tmp_path / "first" / "config.json").write_text(json.dumps(stored))

    cfg = runner.RunConfig(run_name="more", model="m", continue_from="first")
    with pytest.raises(SystemExit) as refused:
        runner.Run(cfg, object(), tmp_path).prepare()
    assert "SQL prompt" in str(refused.value) and not (tmp_path / "more").exists()

    with pytest.raises(SystemExit) as refused:  # resuming "first" itself, begun under that other prompt
        runner.Run(runner.RunConfig(run_name="first", model="m"), object(), tmp_path).prepare()
    assert "another SQL prompt" in str(refused.value)


@needs_data
def test_a_continuation_carries_the_answers_in_and_leaves_the_earlier_run_untouched(tmp_path, monkeypatch):
    import hashlib
    import types

    from evaluation.bird import runner

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(runner, "get_embedder", lambda: types.SimpleNamespace(name="stub"))
    db = "superhero"
    two = [e for e in dataset.load_all() if e.db_id == db][:2]
    records = [
        {"arm": arm, "db_id": db, "question_id": e.question_id, "correct": True, "example_saved": False}
        for e in two
        for arm in ("raw", "semantic")
    ]
    _earlier_run(tmp_path, [(db, e.question_id) for e in two], records, databases=[db])

    def fingerprint():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((tmp_path / "first").iterdir())}

    before = fingerprint()

    run = runner.Run(
        runner.RunConfig(run_name="more", model="m", databases=[db], continue_from="first", limit=3),
        object(),
        tmp_path,
    )
    run.prepare()

    assert fingerprint() == before, "the earlier run's directory is not touched"
    assert len(run.questions) == 5 and len(run.done) == 4, "its questions and its answers are both carried in"
    owed = {q.question_id for q in run.outstanding()}
    assert len(owed) == 3 and not owed & {e.question_id for e in two}, "only the added questions are still owed"
    stored = json.loads((tmp_path / "more" / "config.json").read_text())
    assert (stored["continued_from"], stored["carried_records"], stored["question_count"]) == ("first", 4, 5)

    # resumed later without repeating --limit: the continuation's own quota flags are not part of its contract
    again = runner.Run(runner.RunConfig(run_name="more", model="m", databases=[db]), object(), tmp_path)
    again.prepare()
    assert len(again.questions) == 5 and len(again.done) == 4


@needs_data
def test_the_default_sample_and_the_other_questions_are_the_whole_dev_set_of_the_six_databases():
    from collections import Counter

    examples = dataset.load_all()
    databases = list(dataset.DEFAULT_DATABASES)
    bands = {"simple": 100, "moderate": 100, "challenging": 100}
    sample = dataset.sample_by_difficulty(examples, bands, databases, seed=0)
    taken = {(q.db_id, q.question_id) for q in sample}
    rest = dataset.rest_of(examples, taken, databases, seed=0)

    assert len(sample) + len(rest) == len([e for e in examples if e.db_id in databases]) == 858
    assert not taken & {(q.db_id, q.question_id) for q in rest}
    assert dict(Counter(q.difficulty for q in rest)) == {"simple": 379, "moderate": 177, "challenging": 2}, (
        "the 100 challenging questions of the sample are nearly all that these databases hold"
    )
    assert dict(Counter(q.db_id for q in rest)) == {
        "formula_1": 121,
        "student_club": 111,
        "thrombosis_prediction": 94,
        "california_schools": 62,
        "toxicology": 85,
        "superhero": 85,
    }


def test_a_layer_the_model_barely_wrote_is_refused_and_a_written_one_is_not():
    from evaluation.bird import session_build

    superhero = {"columns": 31, "described_by_model": 2, "labelled_by_model": 0}
    with pytest.raises(session_build.BuildDegraded, match=r"only 2 of 31") as refused:
        session_build.check_model_written("superhero", superhero, 0.5)
    assert "--databases superhero --prepare-only --rebuild-sessions" in str(refused.value)
    assert not isinstance(refused.value, session_build.StaleSession)

    session_build.check_model_written("formula_1", {"columns": 94, "described_by_model": 94}, 0.5)
    session_build.check_model_written("x", {"columns": 31, "described_by_model": 16}, 0.5)  # just over half
    with pytest.raises(session_build.BuildDegraded):
        session_build.check_model_written("x", {"columns": 31, "described_by_model": 15}, 0.5)
    session_build.check_model_written("superhero", superhero, 0.0)  # switched off
    session_build.check_model_written("empty", {"columns": 0, "described_by_model": 0}, 0.5)


def test_a_degraded_or_stale_layer_stops_the_run_before_anything_is_asked(tmp_path, monkeypatch):
    from evaluation.bird import runner, session_build

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")
    work = tmp_path / "work"
    work.mkdir()
    good = {
        "session_id": "s1",
        "stage": "exported",
        "data_revision": session_build.DATA_REVISION,
        "model_written": {"columns": 94, "described_by_model": 94, "labelled_by_model": 70},
    }
    degraded = {**good, "session_id": "s2", "model_written": {"columns": 31, "described_by_model": 2, "labelled_by_model": 0}}
    stale = {**good, "session_id": "s3", "data_revision": session_build.DATA_REVISION - 1}
    unfinished = {"session_id": "s4", "stage": "ingested", "data_revision": session_build.DATA_REVISION}
    (work / "sessions.json").write_text(
        json.dumps({"formula_1": good, "superhero": degraded, "toxicology": stale, "thrombosis_prediction": unfinished})
    )

    run = runner.Run(runner.RunConfig(run_name="r", model="m"), object(), work)
    run.questions = [
        _q("formula_1", 1, "simple"),
        _q("superhero", 2, "simple"),
        _q("toxicology", 3, "simple"),
        _q("thrombosis_prediction", 5, "simple"),
        _q("student_club", 4, "simple"),
    ]
    problems = run.preflight()  # student_club has no layer yet: it is built, and judged, when its turn comes

    assert len(problems) == 3
    assert any("superhero" in p and "only 2 of 31" in p and "--rebuild-sessions" in p for p in problems)
    assert any("toxicology" in p and "data revision" in p for p in problems)
    assert any("thrombosis_prediction" in p and "never finished" in p and "'ingested'" in p for p in problems)

    run.cfg.min_model_written = 0.0
    assert len(run.preflight()) == 2, "the share check can be switched off; a stale or half-built layer cannot"
    run.cfg.arms = ["raw"]
    assert run.preflight() == [], "the raw arm uses no layer"


def test_a_copy_of_the_records_is_kept_beside_them(tmp_path, monkeypatch):
    from evaluation.bird import runner

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    run = runner.Run(runner.RunConfig(run_name="r", model="m"), object(), tmp_path)
    run.dir.mkdir(parents=True)
    assert run._snapshot("formula_1") is None, "nothing to copy before the first answer"

    whole = [{"arm": "raw", "db_id": "a", "question_id": 1}, {"arm": "semantic", "db_id": "a", "question_id": 1}]
    run.records_path.write_text("".join(json.dumps(r) + "\n" for r in whole) + '{"arm": "ra')  # last line cut short
    run.done = {("raw", "a", 1): {}, ("semantic", "a", 1): {}}
    copy = run._snapshot("formula_1")

    assert copy.parent == run.dir / "checkpoints"
    assert copy.read_text() == run.records_path.read_text()
    assert "0002answers" in copy.name and copy.name.endswith("-formula_1.jsonl")
    assert not list((run.dir / "checkpoints").glob("*.tmp")), "written whole or not at all"
    assert len(report.load_records(copy)) == 2, "and a copy with a cut-short line still loads"


def test_a_scope_can_keep_the_replies_and_errors_it_saw_in_full():
    """The product logs the first 200 characters of a reply it cannot use; a build scope keeps all of it."""

    meter = llm.UsageMeter()
    system = [{"role": "system", "content": "You write one-sentence semantic descriptions of spreadsheet columns."}]
    claude = _client(_Boom(400), _reply(content='{"columns": [{"table": "t", "name": "id", "description": "The "Good" one"}]}'), _reply())
    llm.install_meter(claude, meter)

    with meter.scope("build", capture=True) as kept:
        with pytest.raises(_Boom):
            claude._client.chat.completions.create(model="m", messages=system)
        claude._client.chat.completions.create(model="m", messages=system)
    with meter.scope("raw") as plain:
        claude._client.chat.completions.create(model="m", messages=system)

    assert [("error" in r) for r in kept.replies] == [True, False]
    assert kept.replies[0]["error"].startswith("_Boom") and kept.replies[0]["call"].startswith("You write one-sentence")
    assert kept.replies[1]["text"].endswith('"The "Good" one"}]}') and kept.replies[1]["finish_reason"] == "stop"
    assert plain.replies == [], "nothing is kept unless the scope asks for it"
    assert "replies" not in kept.to_dict(), "and the totals never carry them"


def test_a_rejected_layer_leaves_the_models_replies_to_read(tmp_path):
    from evaluation.bird import session_build

    usage = llm.Usage()
    usage.calls = 2
    usage.replies = [{"call": "x", "error": "boom"}, {"call": "x", "finish_reason": "stop", "text": '{"columns": ['}]
    path = session_build.save_replies(tmp_path, "superhero", "semantics", usage)

    saved = json.loads(path.read_text())
    assert path.parent == tmp_path / "build_failures" and path.name.startswith("superhero-semantics-")
    assert saved["database"] == "superhero" and saved["calls"] == 2
    assert saved["replies"][1]["text"] == '{"columns": ['


def test_a_new_session_does_not_inherit_what_the_old_one_measured(tmp_path):
    from evaluation.bird import session_build

    session_build._update_manifest(
        tmp_path, "superhero", session_id="old", stage="exported",
        model_written={"columns": 31, "described_by_model": 2}, drift={"columns_drifted": 0}, build_usage={"total_tokens": 5},
    )
    entry = session_build._update_manifest(tmp_path, "superhero", replace=True, session_id="new", stage="created")

    assert entry == {"session_id": "new", "stage": "created"}
    assert session_build.read_manifest(tmp_path)["superhero"] == entry
    merged = session_build._update_manifest(tmp_path, "superhero", stage="ingested")
    assert merged == {"session_id": "new", "stage": "ingested"}, "later stages still merge"
