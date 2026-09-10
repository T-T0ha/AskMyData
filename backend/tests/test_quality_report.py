"""The automated data-quality report (§5.4, C-3) — pure functions over plain dicts,
standing in for what :class:`~app.db.models.ColumnSemantics` rows and an archived
:class:`~app.db.models.SemanticLayerVersion` bundle would hand them."""

from __future__ import annotations

from app.export.quality import build_quality_report, score_table, score_table_basic


def test_a_fully_clean_table_scores_perfectly():
    columns = [
        {"null_ratio": 0.0, "unique_ratio": 1.0, "type_confidence": 1.0, "taxonomy_confidence": 1.0},
        {"null_ratio": 0.0, "unique_ratio": 1.0, "type_confidence": 1.0, "taxonomy_confidence": 1.0},
    ]
    result = score_table(columns)
    assert result["completeness"] == 100.0
    assert result["uniqueness"] == 100.0
    assert result["consistency"] == 100.0
    assert result["validity"] == 100.0
    assert result["overall"] == 100.0


def test_missing_values_lower_completeness_only():
    columns = [
        {"null_ratio": 0.5, "unique_ratio": 1.0, "type_confidence": 1.0, "taxonomy_confidence": 1.0},
    ]
    result = score_table(columns)
    assert result["completeness"] == 50.0
    assert result["uniqueness"] == 100.0
    assert result["consistency"] == 100.0
    assert result["validity"] == 100.0


def test_an_empty_table_scores_zero_rather_than_dividing_by_zero():
    result = score_table([])
    assert result["overall"] == 0.0
    assert result["summary"]


def test_the_summary_names_the_weakest_dimension():
    columns = [
        {"null_ratio": 0.0, "unique_ratio": 1.0, "type_confidence": 1.0, "taxonomy_confidence": 0.1},
    ]
    result = score_table(columns)
    assert "validity" in result["summary"]


def test_basic_score_only_reports_what_an_archived_export_can_support():
    """The exported ``sem_metadata`` shape has no confidence fields — an
    archived version's score must say so honestly rather than guess."""

    columns = [
        {"null_ratio": 0.0, "taxonomy_label": "email"},
        {"null_ratio": 0.0, "taxonomy_label": "unknown"},
    ]
    result = score_table_basic(columns)
    assert result["completeness"] == 100.0
    assert result["validity"] == 50.0
    assert "uniqueness" not in result
    assert "consistency" not in result


def test_basic_score_of_no_columns_is_none_not_zero():
    assert score_table_basic([]) == {"completeness": None, "validity": None}


def test_build_quality_report_scores_every_current_table():
    current = [
        {
            "name": "orders",
            "columns": [
                {"null_ratio": 0.0, "unique_ratio": 1.0, "type_confidence": 1.0, "taxonomy_confidence": 1.0}
            ],
        },
        {
            "name": "customers",
            "columns": [
                {"null_ratio": 0.5, "unique_ratio": 0.5, "type_confidence": 0.5, "taxonomy_confidence": 0.5}
            ],
        },
    ]
    report = build_quality_report(current)
    names = {t["name"] for t in report["tables"]}
    assert names == {"orders", "customers"}
    assert report["overall"] == 75.0
    assert "previous" not in report["tables"][0]


def test_build_quality_report_attaches_a_before_after_comparison_by_table_name():
    current = [
        {
            "name": "orders",
            "columns": [
                {"null_ratio": 0.0, "unique_ratio": 1.0, "type_confidence": 1.0, "taxonomy_confidence": 1.0}
            ],
        }
    ]
    previous = [{"name": "orders", "columns": [{"null_ratio": 0.2, "taxonomy_label": "unknown"}]}]

    report = build_quality_report(current, previous)

    orders = report["tables"][0]
    assert orders["previous"]["completeness"] == 80.0
    assert orders["previous"]["validity"] == 0.0


def test_a_table_with_no_prior_build_has_no_previous_key():
    current = [
        {
            "name": "new_table",
            "columns": [
                {"null_ratio": 0.0, "unique_ratio": 1.0, "type_confidence": 1.0, "taxonomy_confidence": 1.0}
            ],
        }
    ]
    previous = [{"name": "some_other_table", "columns": []}]

    report = build_quality_report(current, previous)

    assert "previous" not in report["tables"][0]
