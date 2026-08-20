"""Phase 1 — column taxonomy classification.

SemTabla trains a Random Forest on 600–800 hand-labelled columns.  Labelling
that corpus is a project in itself, so this implementation replaces the
classifier with a two-layer hybrid that needs no training data:

* **Layer 1** — a rule engine of 40 rules matching on column name keywords and
  on value regexes.  High precision, instant, free, and every hit is
  explainable ("matched rule ``email``").
* **Layer 2** — for columns no rule claims, Claude is asked to choose a label
  from the closed vocabulary, given only the column name, table name, detected
  type and ten sample values.

Anything Claude hedges on becomes ``unknown`` and is highlighted in the UI for
manual labelling — the paper's own fallback behaviour for low-confidence
predictions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import pandas as pd

from app.core.schemas import ADDITIVE_LABELS, TAXONOMY_UNKNOWN, ColumnType

_NAME_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize(name: str) -> str:
    return _NAME_NORM_RE.sub("_", str(name).lower()).strip("_")


@dataclass(slots=True)
class TaxonomyRule:
    """One high-precision rule.

    A rule fires when its name keywords match *and* — if it defines one — its
    value pattern holds for most sampled values.  ``value_pattern`` alone is
    enough for formats that are unmistakable (email, URL, UUID).
    """

    label: str
    name_keywords: tuple[str, ...] = ()
    value_pattern: re.Pattern[str] | None = None
    value_ratio: float = 0.8
    allowed_types: tuple[ColumnType, ...] = ()
    confidence: float = 0.9
    predicate: Callable[[str, pd.Series], bool] | None = None
    name_exact: tuple[str, ...] = ()

    def name_matches(self, normalized_name: str) -> bool:
        if normalized_name in self.name_exact:
            return True
        tokens = set(normalized_name.split("_"))
        for keyword in self.name_keywords:
            if "_" in keyword:
                if keyword in normalized_name:
                    return True
            elif keyword in tokens:
                return True
        return False

    def values_match(self, values: Sequence[str]) -> bool:
        if self.value_pattern is None:
            return True
        if not values:
            return False
        hits = sum(1 for v in values if self.value_pattern.match(v))
        return hits / len(values) >= self.value_ratio


_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_URL = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)
_PHONE = re.compile(r"^\+?[\d][\d\s\-().]{6,}\d$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_POSTAL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\s-]{2,9}$")
_CURRENCY_CODE = re.compile(r"^[A-Z]{3}$")
_LATLON = re.compile(r"^[-+]?\d{1,3}\.\d+$")
_YEAR = re.compile(r"^(19|20)\d{2}$")
_TIME = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?\s?([AaPp][Mm])?$")
_SKU = re.compile(r"^[A-Za-z0-9]{1,6}[-_/]?[A-Za-z0-9]{1,12}([-_/][A-Za-z0-9]{1,12})?$")

NUMERIC_TYPES = (
    ColumnType.INTEGER_CONTINUOUS,
    ColumnType.INTEGER_ORDINAL,
    ColumnType.INTEGER_NOMINAL,
    ColumnType.FLOAT,
    ColumnType.CURRENCY,
)

#: The 40-rule engine.  Ordered most-specific first; the first hit wins.
RULES: tuple[TaxonomyRule, ...] = (
    # --- unmistakable value formats ---------------------------------------
    TaxonomyRule("email", ("email", "e_mail", "mail"), _EMAIL, 0.7, confidence=0.98),
    TaxonomyRule("url", ("url", "website", "link", "site", "homepage"), _URL, 0.7, confidence=0.97),
    TaxonomyRule("phone", ("phone", "mobile", "tel", "telephone", "contact", "cell", "msisdn"), _PHONE, 0.6, confidence=0.94),
    TaxonomyRule("identifier", (), _UUID, 0.9, confidence=0.97),
    TaxonomyRule("geo_coordinate", ("lat", "latitude", "lon", "lng", "longitude", "coord"), _LATLON, 0.7, confidence=0.93),
    TaxonomyRule("currency_code", ("currency", "curr", "ccy"), _CURRENCY_CODE, 0.8, confidence=0.93),
    TaxonomyRule("time", ("time", "hour", "clock"), _TIME, 0.7, confidence=0.9),
    # --- identifiers -------------------------------------------------------
    TaxonomyRule("sku", ("sku", "item_code", "product_code", "barcode", "upc", "ean"), _SKU, 0.6, confidence=0.93),
    TaxonomyRule("invoice_number", ("invoice", "bill_no", "bill_number", "receipt"), confidence=0.9),
    TaxonomyRule("order_number", ("order_id", "order_no", "order_number", "orderid", "po_number", "tracking"), confidence=0.92),
    # Whether an identifier is a primary or a foreign key is decided by Phase 3
    # from the data, never guessed from the column name here.
    TaxonomyRule(
        "identifier",
        ("id", "uuid", "guid", "code", "key", "ref", "reference", "customer_id", "client_id", "user_id", "supplier_id", "employee_id", "product_id", "account_id"),
        allowed_types=(ColumnType.IDENTIFIER, ColumnType.TEXT, ColumnType.INTEGER_CONTINUOUS, ColumnType.INTEGER_NOMINAL),
        confidence=0.85,
    ),
    # --- ratios (checked before money: "discount_pct" is a fraction) --------
    TaxonomyRule("percentage", ("percent", "pct", "percentage", "ratio", "margin", "share"), allowed_types=NUMERIC_TYPES, confidence=0.9),
    # --- money -------------------------------------------------------------
    TaxonomyRule("price", ("price", "rate", "unit_price", "mrp", "tariff"), allowed_types=NUMERIC_TYPES, confidence=0.93),
    TaxonomyRule("cost", ("cost", "expense", "expenditure", "spend"), allowed_types=NUMERIC_TYPES, confidence=0.92),
    TaxonomyRule("revenue", ("revenue", "sales", "income", "turnover", "earning", "gross", "net_sales", "amount", "total", "subtotal", "grand_total"), allowed_types=NUMERIC_TYPES, confidence=0.9),
    TaxonomyRule("discount", ("discount", "rebate", "markdown", "off"), allowed_types=NUMERIC_TYPES, confidence=0.9),
    TaxonomyRule("tax", ("tax", "vat", "gst", "duty", "levy"), allowed_types=NUMERIC_TYPES, confidence=0.91),
    TaxonomyRule("salary", ("salary", "wage", "payroll", "stipend", "compensation"), allowed_types=NUMERIC_TYPES, confidence=0.93),
    TaxonomyRule("balance", ("balance", "outstanding", "due", "credit", "debit"), allowed_types=NUMERIC_TYPES, confidence=0.88),
    TaxonomyRule("payment_method", ("payment_method", "pay_method", "payment_type", "paymode", "payment_mode"), confidence=0.9),
    # --- measures ----------------------------------------------------------
    TaxonomyRule("quantity", ("qty", "quantity", "count", "units", "stock", "pieces", "volume", "in_stock"), allowed_types=NUMERIC_TYPES, confidence=0.92),
    TaxonomyRule("weight", ("weight", "kg", "gram", "lbs", "mass", "tonnage"), allowed_types=NUMERIC_TYPES, confidence=0.9),
    TaxonomyRule("dimension", ("length", "width", "height", "depth", "size", "diameter"), allowed_types=NUMERIC_TYPES, confidence=0.85),
    TaxonomyRule("age", ("age",), allowed_types=NUMERIC_TYPES, confidence=0.9),
    TaxonomyRule("rating", ("rating", "score", "stars", "rank", "grade"), allowed_types=NUMERIC_TYPES, confidence=0.87),
    TaxonomyRule("duration", ("duration", "elapsed", "days", "hours", "minutes", "lead_time", "tenure"), allowed_types=NUMERIC_TYPES, confidence=0.86),
    # --- time --------------------------------------------------------------
    TaxonomyRule("year", ("year", "yr", "fy"), _YEAR, 0.8, confidence=0.92),
    TaxonomyRule("month", ("month", "mon", "mm"), confidence=0.85),
    TaxonomyRule("weekday", ("weekday", "day_of_week", "dow"), confidence=0.88),
    TaxonomyRule("datetime", ("timestamp", "datetime", "created_at", "updated_at", "logged_at"), allowed_types=(ColumnType.DATE,), confidence=0.93),
    TaxonomyRule("date", ("date", "day", "dob", "birthday", "expiry", "deadline", "due_date"), allowed_types=(ColumnType.DATE, ColumnType.TEXT), confidence=0.92),
    # --- people / places ---------------------------------------------------
    TaxonomyRule("person_name", ("first_name", "last_name", "full_name", "customer_name", "client_name", "employee_name", "contact_name", "person"), confidence=0.9),
    TaxonomyRule("organization_name", ("company", "organization", "org", "vendor", "supplier", "firm", "business", "branch", "store"), confidence=0.87),
    TaxonomyRule("product_name", ("product", "item", "product_name", "item_name", "goods", "model"), confidence=0.85),
    TaxonomyRule("job_title", ("designation", "job_title", "position", "role", "title"), confidence=0.85),
    TaxonomyRule("department", ("department", "dept", "division", "team", "unit"), confidence=0.88),
    TaxonomyRule("postal_address", ("address", "street", "location", "addr"), confidence=0.9),
    TaxonomyRule("city", ("city", "town", "district", "thana", "upazila"), confidence=0.9),
    TaxonomyRule("country", ("country", "nation"), confidence=0.92),
    TaxonomyRule("region", ("region", "state", "province", "zone", "area", "territory", "division_name"), confidence=0.86),
    TaxonomyRule("postal_code", ("zip", "zipcode", "postal", "postcode", "pin_code"), _POSTAL, 0.7, confidence=0.9),
    TaxonomyRule("gender", ("gender", "sex"), confidence=0.93),
    TaxonomyRule("language", ("language", "locale", "lang"), confidence=0.88),
    # --- categorical / descriptive ----------------------------------------
    # A two-valued column is a flag first and a "status" second.
    TaxonomyRule("boolean_flag", (), allowed_types=(ColumnType.BOOLEAN,), confidence=0.92),
    TaxonomyRule("status", ("status", "state", "stage", "condition", "active", "is_active"), confidence=0.88),
    TaxonomyRule("priority", ("priority", "urgency", "severity"), confidence=0.9),
    TaxonomyRule("category", ("category", "type", "class", "group", "segment", "kind", "genre", "tag"), confidence=0.85),
    TaxonomyRule("description", ("description", "details", "summary", "desc"), confidence=0.88),
    TaxonomyRule("note", ("note", "notes", "comment", "remark", "memo", "feedback"), confidence=0.88),
)


@dataclass(slots=True)
class TaxonomyDecision:
    label: str
    confidence: float
    source: str  # "rule" | "claude" | "fallback"
    rule: str | None = None
    reasoning: str | None = None
    alternatives: list[str] = field(default_factory=list)

    @property
    def is_additive(self) -> bool:
        return self.label in ADDITIVE_LABELS

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "rule": self.rule,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
            "is_additive": self.is_additive,
        }


def sample_strings(series: pd.Series, limit: int = 20) -> list[str]:
    values = series.dropna()
    if values.empty:
        return []
    return [str(v).strip() for v in values.head(limit)]


def classify_by_rules(
    name: str,
    column_type: ColumnType,
    series: pd.Series,
) -> TaxonomyDecision | None:
    """Layer 1.  Returns ``None`` when no rule claims the column."""

    normalized = normalize(name)
    values = sample_strings(series, 20)

    for rule in RULES:
        if rule.allowed_types and column_type not in rule.allowed_types:
            continue
        has_name_rule = bool(rule.name_keywords or rule.name_exact)
        name_hit = rule.name_matches(normalized) if has_name_rule else False

        if has_name_rule and not name_hit:
            continue
        if not has_name_rule and not rule.allowed_types and rule.value_pattern is None:
            continue
        if not rule.values_match(values):
            continue
        if rule.predicate and not rule.predicate(normalized, series):
            continue

        confidence = rule.confidence
        if rule.value_pattern is not None and name_hit:
            confidence = min(0.99, confidence + 0.03)
        return TaxonomyDecision(
            label=rule.label,
            confidence=confidence,
            source="rule",
            rule=f"{rule.label}:{'|'.join(rule.name_keywords) or 'value-pattern'}",
        )
    return None


def classify_column(
    name: str,
    table: str,
    column_type: ColumnType,
    series: pd.Series,
    claude=None,
    sample_limit: int = 10,
) -> TaxonomyDecision:
    """Full two-layer classification for one column."""

    decision = classify_by_rules(name, column_type, series)
    if decision is not None:
        return decision

    if claude is not None and claude.available:
        samples = sample_strings(series, sample_limit)
        result = claude.classify_taxonomy(
            column_name=name,
            table_name=table,
            column_type=column_type.value,
            samples=samples,
        )
        if result is not None:
            return result

    return TaxonomyDecision(
        label=TAXONOMY_UNKNOWN,
        confidence=0.0,
        source="fallback",
        reasoning="No rule matched and no LLM label was available — label manually.",
    )


def classify_table(
    table: str,
    df: pd.DataFrame,
    column_types: dict[str, ColumnType],
    claude=None,
) -> dict[str, TaxonomyDecision]:
    return {
        str(column): classify_column(
            str(column), table, column_types[str(column)], df[column], claude=claude
        )
        for column in df.columns
    }
