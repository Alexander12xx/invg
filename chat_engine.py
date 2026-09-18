"""
ALTECH SOFTWARE DEVELOPERS
INTELLIGENT COMMAND ENGINE v28
------------------------------
Document-aware command execution with robust price parsing,
read-cell queries, self-naming column creation, provenance
passthrough, and flexible rename/replace/use-as handling.

NEW IN v28 (vs v27):
  • _rename_column now handles these phrasings:
      "rename X to Y"
      "rename X as Y"
      "rename the column X to Y"
      "change X to Y"
      "change the name of X to Y"
      "replace X with Y"
      "replace the column name X with Y"
      "use X as Y"
      "use X for Y"
      "in X use Y"
      "call X Y"
      "label X as Y"
  • _looks_like_total_row is stricter: any cell whose text is
    total / grand total / subtotal / sum / balance due marks the row
    as a summary row, even when other cells in the row are numbers.
  • price parser accepts "use" as a soft verb.
  • All other v26 behavior preserved.

CARRIED FORWARD FROM v26:
  • _parse_price_command re-anchored on the price verb.
  • Multi-line input joined with "; ".
  • Read-cell queries: "what colour is Hydrangea scarlet"
  • Smart column-naming for "ADD LINE TOTAL COLUMN".
"""

from __future__ import annotations

import os
import re
import json
import math
import logging
import urllib.request
import urllib.error
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

log = logging.getLogger("altech-chat-engine")

MAX_ITEMS = 20000
MAX_COLUMNS = 400
MAX_MESSAGE = 6000
MAX_COMMANDS = 40


# ===========================================================================
#  ROLE MODEL
# ===========================================================================
ROLE_ALIASES = {
    "product_name": [
        "product", "product name", "item", "item name", "flower",
        "flower name", "flower names", "description", "item description",
        "particulars", "goods", "article", "articles",
        "flower variety", "variety", "cultivar",
    ],
    "variety": ["variety", "cultivar", "cultivar name"],
    "quantity": [
        "quantity", "qty", "qnty", "stems", "total stems", "pieces",
        "pcs", "units", "count", "total quantity",
    ],
    "boxes": ["boxes", "box", "bx", "cartons", "carton", "ctn", "cases"],
    "pack_rate": [
        "packrate", "pack rate", "pack_rate", "per box", "per carton",
        "stems per box", "stems/box", "qty per box", "quantity per box",
    ],
    "unit_price": [
        "price", "unit price", "unit_price", "rate", "cost",
        "unit cost", "price per stem", "price/stem", "unit price (usd)",
        "prices totals", "prices", "price total",
    ],
    "total": [
        "total", "amount", "line total", "line amount",
        "total amount", "extended price", "revenue", "line total (usd)",
        "prices totals",
    ],
    "length_cm": [
        "length", "length cm", "length (cm)", "length(cm)",
        "stem length", "size", "cm",
    ],
    "head_size_cm": ["head size", "head size cm", "head size (cm)"],
    "color": ["color", "colour", "shade"],
    "farm_code": ["farm code", "farm", "farmcode"],
    "invoice_number": ["invoice number", "invoice no", "invoice #",
                       "reference"],
    "date": ["date", "invoice date", "shipment date"],
    "due_date": ["due date", "payment due", "valid until"],
    "currency": ["currency"],
    "vat_rate": ["vat", "vat rate", "tax", "tax rate"],
    "discount": ["discount"],
    "n": ["n", "no", "no.", "#", "s/n", "sr", "index"],
}

NUMERIC_ROLES = {
    "quantity", "boxes", "pack_rate", "length_cm", "head_size_cm",
    "unit_price", "total", "vat_rate", "discount", "n",
}


# ===========================================================================
#  LOW-LEVEL HELPERS
# ===========================================================================
def _norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.lower().replace("–", "-").replace("—", "-").replace("’", "'")
    s = s.replace("\t", " ")
    return re.sub(r"\s+", " ", s.strip())


STOPWORDS = {
    "a", "an", "the", "of", "to", "for", "all", "and", "or", "is", "are",
    "on", "in", "at", "by", "with", "as", "each", "then", "every", "row",
    "rows", "please", "me", "get", "give", "show", "can", "you", "i",
    "want", "need", "make", "create", "add", "new", "column", "field",
    "called", "named", "that", "which", "using", "from", "into", "set",
    "change", "update", "apply", "fill", "put", "use", "containing",
    "contains", "hold", "holding", "with",
}


def _tokens(s: str) -> List[str]:
    return [
        x for x in re.findall(r"[a-z0-9]+", _norm(s))
        if x not in STOPWORDS and len(x) > 1
    ]


def _num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    s = _norm(v)
    if not s:
        return None
    s = re.sub(r"\b(?:usd|us\$|kes|ksh|kshs|eur|gbp|aed|sar|qar)\b", "", s)
    s = s.replace("$", "").replace("€", "").replace("£", "")
    if "," in s and "." in s:
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(cur[j] + 1, prev[j + 1] + 1,
                           prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]


def _clean_items(items):
    """
    Preserve the "_meta" provenance block emitted by document_engine.
    Also DROP rows that look like totals (defense in depth).
    """
    cleaned = []
    for r in (items or [])[:MAX_ITEMS]:
        if not isinstance(r, dict):
            continue
        if _looks_like_total_row(r):
            continue
        x = dict(r)
        if isinstance(r.get("_meta"), dict):
            x["_meta"] = r["_meta"]
        cleaned.append(x)
    return cleaned


def _clean_columns(columns):
    out = []
    for c in (columns or [])[:MAX_COLUMNS]:
        if isinstance(c, dict) and c.get("key"):
            x = dict(c)
            x["label"] = str(x.get("label") or x["key"])[:200]
            out.append(x)
    return out


TOTAL_ROW_PATTERN = re.compile(
    r"\b(?:sub\s*total|grand\s*total|invoice\s*total|total\s+amount|"
    r"balance\s+due|amount\s+due|total\s+price|total\s+stems|"
    r"total\s+qty|sum|totals?)\b",
    re.I,
)


def _looks_like_total_row(row: Dict[str, Any]) -> bool:
    """
    Stronger than the app.py version: if ANY cell in the row is
    exactly a total-word (total/totals/subtotal/sum), treat the row
    as a summary row. Also treat rows whose only meaningful cell is
    a bare total keyword + number as summaries.
    """
    if not isinstance(row, dict):
        return False
    # Skip _meta / _sheet keys
    values = []
    for k, v in row.items():
        if k.startswith("_"):
            continue
        values.append(v)
    if not values:
        return False

    # Rule 1: any cell contains a total keyword
    for v in values:
        if v is None:
            continue
        s = _norm(v)
        if not s:
            continue
        if TOTAL_ROW_PATTERN.search(s):
            # If the cell is basically just "total" / "totals" / "sum",
            # or "total <number>", treat as summary.
            stripped = re.sub(r"[^a-z0-9]+", " ", s).strip()
            if stripped in {"total", "totals", "subtotal", "sub total",
                            "grand total", "sum", "balance due",
                            "amount due"}:
                return True
            if re.fullmatch(
                r"(?:total|totals|subtotal|sub total|grand total|sum)"
                r"[\s:]*[-+]?\d[\d,.\s]*", stripped):
                return True

    # Rule 2: cell that combines "TOTAL" and a number in any order
    joined = " ".join(_norm(v) for v in values if v not in (None, ""))
    if re.search(
        r"\b(?:total|totals|subtotal|grand\s+total)\b"
        r"[^\n]{0,30}?[-+]?\d",
        joined,
    ):
        # Only if the row has <= 4 non-empty cells (typical summary row)
        non_empty = sum(1 for v in values if v not in (None, ""))
        if non_empty <= 4:
            return True

    # Rule 3: single-cell row that is just a number and the sheet has a
    # summary keyword somewhere else — handled by Rule 1 already.
    return False


# ===========================================================================
#  COLUMN RESOLUTION
# ===========================================================================
def _role_candidates(columns, query):
    q = _norm(query)
    if not q:
        return []
    scored = []
    for c in columns:
        key = str(c.get("key", ""))
        label = str(c.get("label", key))
        role = c.get("role")
        aliases = ROLE_ALIASES.get(role, [])
        variants = [key.replace("_", " "), label, *aliases]
        best = 0
        for v in variants:
            if not v:
                continue
            nv = _norm(v)
            if nv == q:
                best = 100
                break
            s = max(fuzz.WRatio(q, nv), fuzz.token_set_ratio(q, nv))
            if s > best:
                best = s
        scored.append((best, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def resolve_column(columns, query, numeric=False):
    candidates = _role_candidates(columns, query)
    if numeric:
        candidates = [
            x for x in candidates if x[1].get("role") in NUMERIC_ROLES
        ]
    if not candidates:
        return None, []
    best_score, best = candidates[0]
    second = candidates[1][0] if len(candidates) > 1 else 0
    if best_score >= 92 or (best_score >= 80 and best_score - second >= 8):
        return best, [c for s, c in candidates[:5]]
    return None, [c for s, c in candidates[:5]]


def _product_keys(columns):
    keys = [
        c["key"] for c in columns
        if c.get("role") in ("product_name", "variety", "description")
    ]
    if not keys:
        keys = [
            c["key"] for c in columns
            if c.get("role") not in NUMERIC_ROLES
        ]
    return keys


def _extract_columns_in_order(text, columns, numeric_only=False):
    if not text or not columns:
        return []
    text_low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    hits = []
    for c in columns:
        if numeric_only and c.get("role") not in NUMERIC_ROLES:
            continue
        variants = set()
        if c.get("key"):
            variants.add(c["key"].replace("_", " ").lower())
            variants.add(c["key"].replace("_", "").lower())
        if c.get("label"):
            variants.add(str(c["label"]).lower())
        for alias in ROLE_ALIASES.get(c.get("role"), []):
            variants.add(alias.lower())

        ordered = sorted(variants, key=len, reverse=True)
        for v in ordered:
            v_clean = re.sub(r"\s+", " ", v).strip()
            if len(v_clean) < 2:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(v_clean)}(?![a-z0-9])"
            m = re.search(pattern, text_low)
            if m:
                hits.append((m.start(), c))
                break

    hits.sort(key=lambda x: x[0])
    seen = set()
    out = []
    for _, col in hits:
        if col["key"] in seen:
            continue
        seen.add(col["key"])
        out.append(col)
    return out


def _resolve_single_column(columns, text_fragment):
    if not text_fragment:
        return None
    frag = re.sub(r"[^\w\s]", " ", text_fragment).strip()
    frag = re.sub(r"^(?:the|of|by|as|from|to)\s+", "", frag, flags=re.I)
    frag = re.sub(r"\s+(?:the|of|by|as|from|to)$", "", frag, flags=re.I)
    if not frag:
        return None
    frag_norm = _norm(frag)
    for c in columns:
        key_n = _norm(c.get("key", "").replace("_", " "))
        label_n = _norm(c.get("label", ""))
        if frag_norm == key_n or frag_norm == label_n:
            return c
    best_col, best_score = None, 0
    for c in columns:
        candidates = [
            _norm(c.get("key", "").replace("_", " ")),
            _norm(c.get("label", "")),
        ] + [_norm(a) for a in ROLE_ALIASES.get(c.get("role"), [])]
        score = max(
            (max(fuzz.WRatio(frag_norm, cd), fuzz.partial_ratio(frag_norm, cd))
             for cd in candidates if cd),
            default=0)
        if score > best_score:
            best_score, best_col = score, c
    return best_col if best_score >= 82 else None


# ===========================================================================
#  PRODUCT MATCHING
# ===========================================================================
def resolve_product(items, columns, query):
    q = _norm(query)
    if not q:
        return [], []
    product_keys = _product_keys(columns)
    ranked: List[Tuple[float, int]] = []
    qt = _tokens(q)

    for i, row in enumerate(items):
        text = " ".join(_norm(row.get(k, "")) for k in product_keys)
        if not text.strip():
            continue
        if q in text:
            score = 100.0
        else:
            score = max(
                fuzz.WRatio(q, text),
                fuzz.partial_ratio(q, text),
                fuzz.token_set_ratio(q, text),
            )
            row_tokens = set(text.split())
            if qt and all(t in row_tokens for t in qt):
                score = max(score, 94.0)
            for t in qt:
                if len(t) < 4:
                    continue
                for rt in row_tokens:
                    if abs(len(t) - len(rt)) <= 2 and _lev(t, rt) <= 1:
                        score = max(score, 88.0)

        if score >= 55:
            ranked.append((score, i))

    ranked.sort(reverse=True)
    if not ranked:
        return [], []
    best = ranked[0][0]
    if best >= 88:
        threshold = max(82.0, best - 8.0)
        return [i for s, i in ranked if s >= threshold], ranked[:8]
    second = ranked[1][0] if len(ranked) > 1 else 0
    if best >= 72 and (best - second) >= 8:
        return [ranked[0][1]], ranked[:8]
    return [], ranked[:8]


# ===========================================================================
#  RESULT HELPERS
# ===========================================================================
def _ok(items, columns, explanation, via="deterministic", **extra):
    return {
        "status": "ok", "success": True, "items": items,
        "columns": columns, "explanation": explanation,
        "applied_via": via, "needs_clarification": False, **extra,
    }


def _clarify(items, columns, question, options, **extra):
    return {
        "status": "clarify", "success": True, "items": items,
        "columns": columns, "needs_clarification": True,
        "question": question, "options": options[:8], **extra,
    }


# ===========================================================================
#  WILDCARD HELPERS
# ===========================================================================
WILDCARDS = {
    "all", "every", "each", "everything",
    "all rows", "all items", "all flowers", "all products",
}


def _is_all(target):
    t = _norm(target)
    if t in WILDCARDS:
        return True
    stripped = re.sub(r"^(?:all|every|each)\s+", "", t).strip()
    if not stripped:
        return True
    generic = {
        "rows", "row", "items", "item", "entries", "lines", "line",
        "flowers", "flower", "products", "product", "records",
        "everything", "here", "only",
    }
    return stripped in generic


def _target_rows(items, columns, target):
    if _is_all(target):
        return list(range(len(items))), []
    t = _norm(target)
    t = re.sub(r"^(?:all|every|each)\s+", "", t).strip()
    t = re.sub(r"\s+only$", "", t).strip()
    if not t:
        return list(range(len(items))), []
    hits, ranked = resolve_product(items, columns, t)
    if hits:
        return hits, []
    return [], ranked


# ===========================================================================
#  COLUMN ADD / LOOKUP
# ===========================================================================
def _add_column(columns, label, role=None):
    label = str(label or "Computed").strip()[:80]
    base = re.sub(r"[^a-z0-9]+", "_", _norm(label)).strip("_") or "computed"
    keys = {c["key"] for c in columns}
    key = base
    i = 2
    while key in keys:
        key = f"{base}_{i}"
        i += 1
    return list(columns) + [{
        "key": key, "label": label, "role": role, "source": "added",
    }], key


def _find_role(columns, role):
    for c in columns:
        if c.get("role") == role:
            return c
    return None


def _find_column_by_label(columns, label):
    if not label:
        return None
    n = _norm(label)
    for c in columns:
        if _norm(c.get("label", "")) == n:
            return c
        if _norm(c.get("key", "").replace("_", " ")) == n:
            return c
    return None


# ===========================================================================
#  READ-CELL QUERIES
# ===========================================================================
QUESTION_WORDS = {
    "colour", "color", "colours", "colors", "shade",
    "price", "prices", "cost", "costs", "rate", "rates",
    "variety", "varieties", "cultivar", "cultivars",
    "length", "lengths", "size", "sizes",
    "quantity", "quantities", "qty", "stems",
    "total", "totals", "amount", "amounts",
    "box", "boxes", "cartons", "carton",
    "pack", "rate", "head", "size",
    "flower", "flowers", "item", "items",
    "description", "descriptions",
}


def _handle_read_cell(items, columns, msg):
    low = _norm(msg)
    if not low:
        return None

    m = re.match(
        r"^(?:what(?:'s|\s+is|\s+are)?|show(?:\s+me)?|tell(?:\s+me)?|"
        r"give(?:\s+me)?|list|find)\s+"
        r"(?:the\s+)?(.+?)\s*$", low)
    if not m:
        return None
    rest = m.group(1).strip()

    m2 = re.match(
        r"^(?P<field>[a-z][a-z0-9 _\-]*?)\s+"
        r"(?:is|of|for|in)\s+"
        r"(?P<target>.+)$", rest)
    if m2:
        field_word = m2.group("field").strip()
        target = m2.group("target").strip()
    else:
        m3 = re.match(
            r"^(?P<target>.+?)[\s'’]s?\s+"
            r"(?P<field>[a-z][a-z0-9 _\-]*?)\s*$", rest)
        if m3:
            target = m3.group("target").strip()
            field_word = m3.group("field").strip()
        else:
            return None

    field_word = re.sub(r"\s+(?:is|are|of|for|in)$", "",
                        field_word).strip()
    if field_word not in QUESTION_WORDS:
        return None

    col, _ = resolve_column(columns, field_word)
    if not col:
        return None

    rows, ranked = _target_rows(items, columns, target)
    if not rows:
        return _clarify(
            items, columns,
            f"I couldn't find a row matching “{target}”.", [])

    idx = rows[0]
    row = items[idx]
    value = row.get(col["key"])
    if value in (None, ""):
        return _ok(items, columns,
                   f"“{col['label']}” is empty for that row.")
    name = None
    for pk in _product_keys(columns):
        v = row.get(pk)
        if v not in (None, ""):
            name = v
            break
    label = f"“{name}”" if name else f"row {idx + 1}"
    return _ok(items, columns,
               f"{label} — {col['label']}: {value}")


# ===========================================================================
#  PRICE COMMAND PARSER
# ===========================================================================
PRICE_VERBS = r"(?:add|set|change|apply|make|assign|put|update|give|fill|use)"
PRICE_WORDS = r"(?:unit\s*price|unit_price|price|rate|cost|unit\s*cost)"


def _parse_price_command(msg: str):
    m_text = msg.strip()
    m_text = re.sub(r"\s+", " ", m_text)

    # --- Pattern A: VERB-LEADING with target in a preposition tail ---
    m = re.match(
        rf"^\s*{PRICE_VERBS}\s+(?:a\s+|the\s+)?"
        rf"(?P<field>[a-z][a-z0-9 _\-]*?)\s+"
        rf"\$?\s*(?P<value>[\d,]+(?:\.\d+)?)\s+"
        rf"(?:to|for|on|in)\s+(?:all\s+|every\s+|the\s+)?"
        rf"(?P<target>.+?)\s*$",
        m_text, re.I)
    if m:
        field = m.group("field").strip()
        if re.match(r"^[a-z][a-z0-9 _\-]{1,30}$", field, re.I):
            return (m.group("target").strip(),
                    _num(m.group("value")), field)

    # --- Pattern B: VERB-LEADING with target at front ---
    m = re.match(
        rf"^\s*{PRICE_VERBS}\s+(?P<target>.+?)\s+"
        rf"(?P<field>[a-z][a-z0-9 _\-]*?)\s+"
        rf"(?:to|=|:)\s*\$?\s*(?P<value>[\d,]+(?:\.\d+)?)\s*$",
        m_text, re.I)
    if m:
        return (m.group("target").strip(),
                _num(m.group("value")), m.group("field").strip())

    # --- Pattern C: VERB-LEADING with target in a preposition tail ---
    m = re.match(
        rf"^\s*{PRICE_VERBS}\s+(?P<field>[a-z][a-z0-9 _\-]*?)\s+"
        rf"\$?\s*(?P<value>[\d,]+(?:\.\d+)?)\s+"
        rf"(?:to|for|on|in)\s+(?:all\s+|every\s+|the\s+)?"
        rf"(?P<target>.+?)\s*$",
        m_text, re.I)
    if m:
        field = m.group("field").strip()
        if re.match(r"^[a-z][a-z0-9 _\-]{1,30}$", field, re.I):
            return (m.group("target").strip(),
                    _num(m.group("value")), field)

    # --- Pattern D: TARGET-LEADING, verb-less ---
    m = re.match(
        rf"^\s*(?:to\s+)?(?P<target>.+?)\s+"
        rf"{PRICE_VERBS}\s+(?:a\s+|the\s+)?"
        rf"(?P<field>[a-z][a-z0-9 _\-]*?)\s+"
        rf"\$?\s*(?P<value>[\d,]+(?:\.\d+)?)\s*$",
        m_text, re.I)
    if m:
        target = m.group("target").strip()
        target = re.sub(r"^to\s+", "", target, flags=re.I).strip()
        field = m.group("field").strip()
        if re.match(r"^[a-z][a-z0-9 _\-]{1,30}$", field, re.I):
            return target, _num(m.group("value")), field

    # --- Pattern E: "<target> <field> <value>" (verb-less, no "to") ---
    m = re.match(
        rf"^\s*(?:to\s+)?(?P<target>.+?)\s+"
        rf"(?P<field>[a-z][a-z0-9 _\-]*?)\s+"
        rf"\$?\s*(?P<value>[\d,]+(?:\.\d+)?)\s*$",
        m_text, re.I)
    if m:
        target = m.group("target").strip()
        target = re.sub(r"^to\s+", "", target, flags=re.I).strip()
        field = m.group("field").strip()
        if field.lower() in ("to", "for", "on", "in"):
            return None, None, None
        if re.match(r"^[a-z][a-z0-9 _\-]{1,30}$", field, re.I):
            return target, _num(m.group("value")), field

    # --- Pattern F: "<target> @ <value>" ---
    m = re.match(
        r"^\s*(?:to\s+)?(?P<target>.+?)\s*@\s*\$?\s*"
        r"(?P<value>[\d,]+(?:\.\d+)?)\s*$",
        m_text, re.I)
    if m:
        target = re.sub(r"^to\s+", "", m.group("target"), flags=re.I).strip()
        if target:
            return target, _num(m.group("value")), "unit price"

    return None, None, None


# ===========================================================================
#  COMMAND IMPLEMENTATIONS
# ===========================================================================
def _column_mentioned_in_message(msg, columns):
    low = _norm(msg)
    candidates = []
    for c in columns:
        if c.get("role") in ("product_name", "variety", "description"):
            continue
        variants = set()
        if c.get("key"):
            variants.add(c["key"].replace("_", " ").lower())
        if c.get("label"):
            variants.add(str(c["label"]).lower())
        for v in variants:
            v_clean = re.sub(r"\s+", " ", v).strip()
            if len(v_clean) < 2:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(v_clean)}(?![a-z0-9])",
                         low):
                candidates.append((len(v_clean), c))
                break
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def _set_value(items, columns, msg):
    target, value, field = _parse_price_command(msg)
    if target is not None:
        col = None
        if field:
            col, _ = resolve_column(columns, field)
        if col is None:
            col = _column_mentioned_in_message(msg, columns)
        if col is None:
            col = _find_role(columns, "unit_price")
        if col is None:
            columns, key = _add_column(columns, "Unit Price", "unit_price")
            col = next(c for c in columns if c["key"] == key)
        return _apply_column_value(items, columns, col, value, target)

    m = re.match(
        rf"^\s*{PRICE_VERBS}\s+(?P<field>.+?)\s+(?:to|=|as)\s+(?P<value>.+)$",
        msg, re.I)
    if m:
        field = m.group("field").strip()
        right = m.group("value").strip()
        col, _ = resolve_column(columns, field)
        if col:
            val = _num(right)
            if val is None:
                val = right.strip().strip('"\'')
            return _apply_column_value(items, columns, col, val, "all")
    return None


def _apply_column_value(items, columns, col, value, target):
    rows, ranked = _target_rows(items, columns, target)
    if not rows:
        t_norm = _norm(re.sub(r"^(?:all|every|each)\s+", "", target))
        t_norm = re.sub(r"\s+only$", "", t_norm).strip()
        if t_norm:
            product_keys = _product_keys(columns)
            for i, row in enumerate(items):
                text = " ".join(_norm(row.get(k, "")) for k in product_keys)
                if t_norm in text:
                    rows.append(i)
    if not rows:
        qt = [t for t in _tokens(target) if len(t) >= 4]
        if qt:
            product_keys = _product_keys(columns)
            for i, row in enumerate(items):
                row_tokens = set()
                for k in product_keys:
                    row_tokens.update(
                        re.findall(r"[a-z0-9]+", _norm(row.get(k, ""))))
                if not row_tokens:
                    continue
                if all(any(t in rt or rt in t or
                            (abs(len(t) - len(rt)) <= 2 and _lev(t, rt) <= 1)
                            for rt in row_tokens) for t in qt):
                    rows.append(i)
    if not rows:
        options = []
        for score, i in ranked[:6]:
            name = (items[i].get("product_name")
                    or items[i].get("item")
                    or items[i].get("description")
                    or f"row {i+1}")
            options.append(str(name))
        return _clarify(
            items, columns,
            f"I couldn't identify which rows “{target}” refers to.",
            options)

    out = deepcopy(items)
    for i in rows:
        out[i][col["key"]] = value
    display = f"{value:g}" if isinstance(value, (int, float)) else str(value)
    return _ok(out, columns,
               f"Set {col.get('label', col['key'])} to {display} on "
               f"{len(rows)} row(s) matching “{target}”.")


def _calculate_totals(items, columns):
    q = _find_role(columns, "quantity")
    p = _find_role(columns, "unit_price")
    t = _find_role(columns, "total")
    if not q or not p:
        return _clarify(
            items, columns,
            "I need a Quantity and a Unit Price column before I can "
            "calculate line totals.",
            [c.get("label", c["key"]) for c in columns])
    if not t:
        columns, key = _add_column(columns, "Line Total", "total")
        t = next(c for c in columns if c["key"] == key)
    out = deepcopy(items)
    calculated = 0
    for row in out:
        qv = _num(row.get(q["key"]))
        pv = _num(row.get(p["key"]))
        if qv is not None and pv is not None:
            row[t["key"]] = round(qv * pv, 4)
            row["total"] = row[t["key"]]
            calculated += 1
    return _ok(out, columns,
               f"Calculated {t.get('label', t['key'])} for "
               f"{calculated} row(s).")


def _grand_total(items, columns):
    t = _find_role(columns, "total")
    if not t:
        calc = _calculate_totals(items, columns)
        if calc.get("status") != "ok":
            return calc
        items, columns = calc["items"], calc["columns"]
        t = _find_role(columns, "total")
    total = sum(_num(r.get(t["key"])) or 0 for r in items)
    return _ok(items, columns, f"Total amount: {total:.2f}.",
               grand_total=round(total, 2), query_result=round(total, 2))


def _compute(items, columns, msg):
    low = _norm(msg)

    op = None
    if re.search(r"\b(multiply|times|product|multiplied\s+by|×)\b", low):
        op = "multiply"
    elif re.search(r"\b(divide|divided|over|per|÷)\b", low):
        op = "divide"
    elif re.search(r"\b(add|plus|sum|\+)\b", low):
        op = "add"
    elif re.search(r"\b(subtract|minus|less|−)\b", low):
        op = "subtract"
    if not op:
        return None

    target_label = None
    rest_of_msg = low

    m = re.search(
        rf"\b(?:add|create|make|new)\s+(?:a\s+|the\s+)?(?:column|field)\s+"
        rf"(?:called\s+|named\s+|for\s+|with\s+)?([a-z0-9 _\-]+?)\s+"
        rf"(?:that\s+is\s+|which\s+is\s+|as\s+|to\s+|=|equals?|"
        rf"containing\s+)\s*(.+)$", low)
    if m:
        target_label = m.group(1).strip()
        rest_of_msg = m.group(2).strip()
    else:
        m = re.search(
            rf"\b(?:calculate|compute|get|give|show|derive|work\s+out|"
            rf"figure\s+out|fill|make)\s+(?:me\s+)?(?:the\s+)?"
            rf"([a-z0-9 _\-]+?)\s+(?:as|by|from|=|equals?|using)\s+(.+)$",
            low)
        if m:
            target_label = m.group(1).strip()
            rest_of_msg = m.group(2).strip()
        else:
            m = re.match(r"^\s*([a-z0-9 _\-]+?)\s*[:=]\s*(.+)$", low)
            if m:
                target_label = m.group(1).strip()
                rest_of_msg = m.group(2).strip()

    operand_columns = _extract_columns_in_order(rest_of_msg, columns,
                                                 numeric_only=True)

    if not operand_columns:
        q_col = _find_role(columns, "quantity")
        p_col = _find_role(columns, "unit_price")
        if q_col and p_col and op == "multiply":
            operand_columns = [q_col, p_col]
        else:
            return None

    if len(operand_columns) == 1:
        for verb in ("multiply", "multiplying", "times", "product of",
                     "divide", "dividing", "divided by", "over",
                     "add", "adding", "plus", "sum of",
                     "subtract", "subtracting", "minus"):
            parts = re.split(rf"\b{re.escape(verb)}\b", rest_of_msg,
                             maxsplit=1, flags=re.I)
            if len(parts) == 2:
                left_col = _resolve_single_column(columns, parts[0])
                right_col = _resolve_single_column(columns, parts[1])
                if left_col and right_col:
                    operand_columns = [left_col, right_col]
                    break

    if len(operand_columns) < 2:
        return None

    constants = []
    for cm in re.finditer(
            r"(?:times|multiplied\s+by|×|\*)\s+(\d+(?:\.\d+)?)\b",
            rest_of_msg, re.I):
        try:
            constants.append(float(cm.group(1)))
        except ValueError:
            pass

    if target_label:
        target_label = re.sub(r"^(?:the\s+)", "", target_label, flags=re.I)
        existing = _find_column_by_label(columns, target_label)
        if existing:
            target_key = existing["key"]
        else:
            columns, target_key = _add_column(columns, target_label, None)
    else:
        if op == "multiply":
            existing_total = _find_role(columns, "total")
            if existing_total:
                target_key = existing_total["key"]
            else:
                columns, target_key = _add_column(columns, "Line Total",
                                                   "total")
        else:
            label = {"divide": "Ratio", "add": "Sum",
                     "subtract": "Difference"}.get(op, "Computed")
            columns, target_key = _add_column(columns, label, None)

    def combine(vals):
        try:
            if op == "multiply":
                r = vals[0]
                for v in vals[1:]:
                    r *= v
                return r
            if op == "divide":
                r = vals[0]
                for v in vals[1:]:
                    if v == 0:
                        return None
                    r /= v
                return r
            if op == "add":
                return sum(vals)
            if op == "subtract":
                r = vals[0]
                for v in vals[1:]:
                    r -= v
                return r
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return None

    updated = []
    zero_count = 0
    for row in items:
        vals = []
        for c in operand_columns:
            v = _num(row.get(c["key"]))
            if v is None:
                v = 0
            vals.append(v)
        for c in constants:
            vals.append(c)
        result = combine(vals)
        r = dict(row)
        r[target_key] = round(result, 4) if result is not None else None
        if result == 0:
            zero_count += 1
        updated.append(r)

    label_of = {c["key"]: (c.get("label") or c["key"]) for c in columns}
    target_lbl = label_of.get(target_key, "Computed")
    operand_labels = [label_of.get(c["key"], c["key"]) for c in operand_columns]
    summary = (f"Computed “{target_lbl}” = "
               f"{f' {op} '.join(operand_labels)}"
               + (" × " + " × ".join(f"{c:g}" for c in constants)
                  if constants else "")
               + f" — {len(updated)} row(s).")
    if zero_count == len(updated) and len(updated) > 0:
        summary += "  ⚠ All values came out 0."

    return _ok(updated, columns, summary)


def _aggregate(items, columns, msg):
    low = _norm(msg)
    m = re.search(
        r"\b(average|avg|mean|sum|total|count|min(?:imum)?|max(?:imum)?|"
        r"highest|lowest)\b\s+(?:of\s+|the\s+)?([a-z0-9 _\-]+?)"
        r"(?:\s+(?:per|by|for\s+each|group\s+by|grouped\s+by)\s+"
        r"([a-z0-9 _\-]+?))?(?:\s*$|[,.;!?])", low)
    if not m:
        return None
    func = m.group(1).lower()
    target = m.group(2).strip()
    group_by = (m.group(3) or "").strip()
    if func == "highest":
        func = "max"
    if func == "lowest":
        func = "min"

    tcol, _ = resolve_column(columns, target, numeric=True)
    if not tcol:
        return None

    if group_by:
        gcol, _ = resolve_column(columns, group_by)
        if not gcol:
            return None
        groups: Dict[str, List[float]] = {}
        for row in items:
            g = str(row.get(gcol["key"]) or "").strip()
            v = _num(row.get(tcol["key"]))
            if v is None:
                continue
            groups.setdefault(g, []).append(v)
        lines = []
        for g in sorted(groups.keys()):
            vals = groups[g]
            if func in ("average", "avg", "mean"):
                r = sum(vals) / len(vals)
            elif func in ("sum", "total"):
                r = sum(vals)
            elif func == "count":
                r = len(vals)
            elif func.startswith("min"):
                r = min(vals)
            elif func.startswith("max"):
                r = max(vals)
            else:
                continue
            r = round(r, 4) if isinstance(r, float) and not r.is_integer() \
                else int(r)
            lines.append(f"  • {g}: {r}")
        return _ok(items, columns,
                   f"{func.capitalize()} of {tcol['label']} by "
                   f"{gcol['label']}:\n" + "\n".join(lines))

    vals = [_num(r.get(tcol["key"])) for r in items]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if func in ("average", "avg", "mean"):
        r = sum(vals) / len(vals)
    elif func in ("sum", "total"):
        r = sum(vals)
    elif func == "count":
        r = len(vals)
    elif func.startswith("min"):
        r = min(vals)
    elif func.startswith("max"):
        r = max(vals)
    else:
        return None
    r = round(r, 4) if isinstance(r, float) and not r.is_integer() else int(r)
    return _ok(items, columns,
               f"{func.capitalize()} of {tcol['label']}: {r}")


def _remove_duplicates(items, columns, msg):
    seen = {}
    out = []
    removed = 0
    for row in items:
        parts = []
        for c in columns:
            if c.get("role") in NUMERIC_ROLES:
                continue
            v = row.get(c["key"])
            parts.append("" if v in (None, "") else _norm(v))
        fp = "|".join(parts)
        if fp in seen:
            removed += 1
            continue
        seen[fp] = True
        out.append(row)
    return _ok(out, columns, f"Removed {removed} duplicate row(s).")


def _add_row(items, columns, msg):
    m = re.search(
        r"\badd\s+(?:a\s+|one\s+)?row\s+(?:for\s+|with\s+|called\s+)?(.+)$",
        msg, re.I)
    if not m:
        return None
    rest = m.group(1).strip()

    parts = re.split(r"\s+", rest)
    name_tokens = []
    i = 0
    while i < len(parts) and not re.match(r"^-?\d", parts[i]):
        name_tokens.append(parts[i])
        i += 1
    name = " ".join(name_tokens).strip(" ,;")
    if not name:
        return _clarify(items, columns,
                        "What should the new row's product name be?", [])

    numeric_cols = [c for c in columns if c.get("role") in NUMERIC_ROLES]
    new_row = {c["key"]: None for c in columns}
    for c in columns:
        if c.get("role") == "product_name":
            new_row[c["key"]] = name
    for tok, col in zip(parts[i:], numeric_cols):
        v = _num(tok)
        if v is not None:
            new_row[col["key"]] = v

    out = list(items) + [new_row]
    return _ok(out, columns, f"Added a new row for “{name}”.")


# ===========================================================================
#  COLUMN CREATION
# ===========================================================================
COLUMN_NOUN_ALIASES = {
    "line total": "Line Total",
    "total": "Total",
    "amount": "Amount",
    "revenue": "Revenue",
    "price": "Price",
    "unit price": "Unit Price",
    "cost": "Cost",
    "rate": "Rate",
    "quantity": "Quantity",
    "qty": "Qty",
    "stems": "Stems",
    "boxes": "Boxes",
    "pack rate": "Pack Rate",
    "length": "Length",
    "size": "Size",
    "cm": "cm",
    "colour": "Color",
    "color": "Color",
    "variety": "Variety",
    "flower name": "Flower",
    "flower names": "Flower",
    "flower": "Flower",
    "item": "Item",
    "description": "Description",
    "notes": "Notes",
    "comments": "Comments",
}


def _smart_column_label(raw):
    n = _norm(raw)
    if n in COLUMN_NOUN_ALIASES:
        return COLUMN_NOUN_ALIASES[n]
    return " ".join(w.capitalize() for w in raw.split())[:80]


def _handle_column_creation(items, columns, msg):
    low = _norm(msg)

    m = re.match(
        r"^(?:add|create|make|new|insert|need|want|build|generate)\s+"
        r"(?:a\s+|the\s+|one\s+)?"
        r"(?P<label>[a-z][a-z0-9 _\-]*?)\s+"
        r"(?:column|field)\s*$",
        low)
    if m:
        label = m.group("label").strip()
        return _create_or_compute_column(items, columns, label)

    m = re.match(
        r"^(?:add|create|make|new|insert|need|want|build|generate)\s+"
        r"(?:a\s+|the\s+|one\s+)?(?:column|field)\s+"
        r"(?:called\s+|named\s+)?"
        r"(?:(?P<label>[a-z][a-z0-9 _\-]*?)\s+)?"
        r"(?:(?:for|with|of|holding|containing|that\s+holds?|that\s+contains)"
        r"\s+(?P<source>.+?))?\s*$",
        low)
    if m:
        label = (m.group("label") or "").strip()
        source = (m.group("source") or "").strip()
        if source:
            return _create_or_compute_column(items, columns,
                                             label or source, source)
        if label:
            return _create_or_compute_column(items, columns, label)

    m = re.match(
        r"^(?:add|create|make|new|insert)\s+"
        r"(?:a\s+|the\s+|one\s+)?(?:column|field)\s+"
        r"(?:called\s+|named\s+)(?P<label>.+?)\s*$",
        low)
    if m:
        return _create_or_compute_column(items, columns,
                                         m.group("label").strip())

    return None


def _create_or_compute_column(items, columns, label, source_noun=None):
    label = _smart_column_label(label)
    label_norm = _norm(label)

    if source_noun:
        col, _ = resolve_column(columns, source_noun)
        if col:
            columns, key = _add_column(columns, label, col.get("role"))
            out = [dict(r, **{key: r.get(col["key"])}) for r in items]
            return _ok(out, columns,
                       f"Added column “{label}” with values from "
                       f"“{col['label']}”.")
        options = [c.get("label") or c["key"] for c in columns][:8]
        return _clarify(
            items, columns,
            f"Which existing column should I copy into “{label}”?",
            options)

    if label_norm in ("line total", "total", "amount", "revenue"):
        q = _find_role(columns, "quantity")
        p = _find_role(columns, "unit_price")
        if q and p:
            columns, key = _add_column(columns, label, "total")
            out = []
            for r in items:
                v = (_num(r.get(q["key"])) or 0) * \
                    (_num(r.get(p["key"])) or 0)
                out.append(dict(r, **{key: round(v, 4)}))
            return _ok(out, columns,
                       f"Added column “{label}” = "
                       f"{q['label']} × {p['label']} for {len(out)} row(s).")
        columns, key = _add_column(columns, label, "total")
        out = [dict(r, **{key: None}) for r in items]
        return _ok(out, columns,
                   f"Added empty column “{label}” "
                   f"(needed Quantity and Unit Price to compute).")

    if label_norm in ("unit price", "price", "rate", "cost"):
        columns, key = _add_column(columns, label, "unit_price")
        out = [dict(r, **{key: None}) for r in items]
        return _ok(out, columns, f"Added empty column “{label}”.")

    col, _ = resolve_column(columns, label)
    if col:
        columns, key = _add_column(columns, label, col.get("role"))
        out = [dict(r, **{key: r.get(col["key"])}) for r in items]
        return _ok(out, columns,
                   f"Added column “{label}” copying from "
                   f"“{col['label']}”.")

    columns, key = _add_column(columns, label, None)
    out = [dict(r, **{key: None}) for r in items]
    return _ok(out, columns, f"Added a new empty column “{label}”.")


# ===========================================================================
#  RENAME / REPLACE / USE-AS  (v28 — full coverage)
# ===========================================================================
def _rename_column(items, columns, msg):
    """
    Handles all the phrasings:
      rename X to Y
      rename X as Y
      rename the column X to Y
      change X to Y
      change the name of X to Y
      replace X with Y
      replace the column name X with Y
      use X as Y
      use X for Y
      in X use Y
      call X Y
      label X as Y
      set X as Y      (rename form, not price form)
    """
    low = _norm(msg)

    # Strip leading "please" / "can you" noise
    low = re.sub(r"^(?:please|can you|could you|kindly)\s+", "", low)

    # Try each pattern in order; return on the first match that resolves.
    patterns = [
        # rename X to Y  |  rename X as Y  |  rename the column X to Y
        r"^(?:re)?name\s+(?:the\s+)?(?:column\s+|field\s+)?"
        r"(?P<old>.+?)\s+(?:to|as)\s+(?P<new>.+?)\s*$",

        # replace X with Y  |  replace the column name X with Y
        r"^replace\s+(?:the\s+)?(?:column\s+)?(?:name\s+)?"
        r"(?P<old>.+?)\s+(?:with|by|as)\s+(?P<new>.+?)\s*$",

        # change X to Y  |  change the name of X to Y
        r"^change\s+(?:the\s+)?(?:name\s+of\s+)?(?:column\s+)?"
        r"(?P<old>.+?)\s+(?:to|as|into)\s+(?P<new>.+?)\s*$",

        # use X as Y  |  use X for Y
        r"^use\s+(?:the\s+)?(?:column\s+)?"
        r"(?P<old>.+?)\s+(?:as|for)\s+(?P<new>.+?)\s*$",

        # in X use Y
        r"^in\s+(?P<old>.+?)\s+use\s+(?P<new>.+?)\s*$",

        # call X Y  |  call column X Y
        r"^call\s+(?:the\s+)?(?:column\s+)?"
        r"(?P<old>.+?)\s+(?P<new>[a-z0-9 _\-]+)\s*$",

        # label X as Y
        r"^label\s+(?:the\s+)?(?:column\s+)?"
        r"(?P<old>.+?)\s+(?:as|to)\s+(?P<new>.+?)\s*$",
    ]

    for pat in patterns:
        m = re.match(pat, low)
        if not m:
            continue
        old_needle = (m.group("old") or "").strip()
        new_label = (m.group("new") or "").strip()
        if not old_needle or not new_label:
            continue

        col, _ = resolve_column(columns, old_needle)
        if not col:
            # If it's the "use X as Y" form where X wasn't a known column,
            # try to match a role alias for X to produce a helpful message.
            continue

        new_label = _smart_column_label(new_label)
        new_cols = []
        for c in columns:
            if c["key"] == col["key"]:
                # Also update the role if the new label maps cleanly
                new_role = col.get("role")
                if _norm(new_label) in ("unit price", "price", "rate", "cost"):
                    new_role = "unit_price"
                elif _norm(new_label) in ("line total", "total", "amount",
                                          "revenue"):
                    new_role = "total"
                elif _norm(new_label) in ("qty", "quantity", "stems"):
                    new_role = "quantity"
                new_cols.append({**c, "label": new_label, "role": new_role})
            else:
                new_cols.append(c)
        return _ok(items, new_cols,
                   f"Renamed “{col['label']}” to “{new_label}”.")

    return None


# ===========================================================================
#  REMOVE / SORT / ROUND / DISCOUNT
# ===========================================================================
def _remove_row(items, columns, msg):
    m = re.search(r"\b(?:remove|delete|drop)\s+row\s+(\d+)", msg, re.I)
    if m:
        idx = int(m.group(1)) - 1
        if idx < 0 or idx >= len(items):
            return _clarify(items, columns,
                            f"There's no row {m.group(1)}.", [])
        out = items[:idx] + items[idx+1:]
        return _ok(out, columns, f"Removed row {idx+1}.")

    m = re.search(
        r"\b(?:remove|delete|drop)\s+row\s+(?:for\s+|with\s+)?(.+)$",
        msg, re.I)
    if m:
        target = m.group(1).strip()
        rows, _ = _target_rows(items, columns, target)
        if not rows:
            return _clarify(items, columns,
                            f"I couldn't find a row for “{target}”.", [])
        rm = set(rows[:1])
        out = [r for i, r in enumerate(items) if i not in rm]
        return _ok(out, columns, f"Removed row for “{target}”.")
    return None


def _remove_column(items, columns, msg):
    m = re.search(
        r"\b(?:remove|delete|drop)\s+(?:the\s+)?column\s+(.+)$", msg, re.I)
    if not m:
        return None
    col, _ = resolve_column(columns, m.group(1).strip())
    if not col:
        return _clarify(items, columns,
                        f"I couldn't find a column called "
                        f"“{m.group(1).strip()}”.", [])
    new_cols = [c for c in columns if c["key"] != col["key"]]
    out = [{k: v for k, v in r.items() if k != col["key"]}
           for r in items]
    return _ok(out, new_cols, f"Removed column “{col['label']}”.")


def _remove_rows(items, columns, msg):
    low = _norm(msg)

    if re.search(r"\bempty\b|\bblank\b", low):
        out = [r for r in items
               if not all(v in (None, "") for v in r.values())]
        return _ok(out, columns,
                   f"Removed {len(items) - len(out)} empty row(s).")

    m = re.search(r"\b(?:zero|0)\b.*\b(quantity|qty|stems?)\b", low)
    if m:
        col = _find_role(columns, "quantity") or \
              _column_mentioned_in_message(msg, columns)
        if not col:
            return _clarify(items, columns,
                            "Which column should I check for zeros?", [])
        out, removed = [], 0
        for r in items:
            if (_num(r.get(col["key"])) or 0) == 0:
                removed += 1
            else:
                out.append(r)
        return _ok(out, columns,
                   f"Removed {removed} row(s) with zero in "
                   f"{col['label']}.")

    m = re.search(
        r"\bwhere\s+([a-z0-9 _\-]+?)\s+(?:is|=|equals?|contains?)\s+"
        r"([a-z0-9 _\-\.]+)", low)
    if m:
        col, _ = resolve_column(columns, m.group(1).strip())
        if not col:
            return _clarify(items, columns,
                            f"I couldn't find column “{m.group(1)}”.", [])
        val = m.group(2).strip()
        v_num = _num(val)
        out, removed = [], 0
        for r in items:
            rv = r.get(col["key"])
            match = (v_num is not None and _num(rv) == v_num) or \
                    (_norm(rv) == val)
            if match:
                removed += 1
            else:
                out.append(r)
        return _ok(out, columns,
                   f"Removed {removed} row(s) where {col['label']} is "
                   f"“{val}”.")
    return None


def _sort(items, columns, msg):
    m = re.search(
        r"(?:sort|order)\s+(?:by\s+)?(.+?)"
        r"(?:\s+(ascending|descending|asc|desc))?$", _norm(msg))
    if not m:
        return None
    col, _ = resolve_column(columns, m.group(1).strip())
    if not col:
        return None
    desc = (m.group(2) or "").startswith("desc")

    def key_fn(r):
        v = r.get(col["key"])
        if col.get("role") in NUMERIC_ROLES:
            return (_num(v) is None, _num(v) or 0)
        return (_norm(v) == "", _norm(v))

    out = sorted(items, key=key_fn, reverse=desc)
    return _ok(out, columns,
               f"Sorted by {col['label']} "
               f"({'descending' if desc else 'ascending'}).")


def _round(items, columns, msg):
    m = re.search(
        r"round\s+(?:all\s+|the\s+)?([a-z0-9 _\-]+?)\s+to\s+"
        r"(\d+)\s*decimals?", _norm(msg))
    if not m:
        return None
    col, _ = resolve_column(columns, m.group(1).strip(), numeric=True)
    if not col:
        return None
    decimals = max(0, min(int(m.group(2)), 8))
    out = []
    for r in items:
        v = _num(r.get(col["key"]))
        x = dict(r)
        if v is not None:
            x[col["key"]] = round(v, decimals)
        out.append(x)
    return _ok(out, columns,
               f"Rounded {col['label']} to {decimals} decimal(s).")


def _discount_or_increase(items, columns, msg):
    low = _norm(msg)
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*(discount|off|increase|more|less)", low)
    if not m:
        return None
    pct = float(m.group(1))
    kind = m.group(2)
    factor = (1.0 - (pct / 100.0) if kind in ("discount", "off", "less")
              else 1.0 + (pct / 100.0))

    col = _find_role(columns, "unit_price") or _find_role(columns, "total")
    if not col:
        return None

    out = []
    applied = 0
    for r in items:
        v = _num(r.get(col["key"]))
        x = dict(r)
        if v is not None:
            x[col["key"]] = round(v * factor, 4)
            applied += 1
        out.append(x)
    verb = "Applied" if factor < 1 else "Increased"
    return _ok(out, columns,
               f"{verb} {pct}% on {col['label']} for {applied} row(s).")


def _answer_question(items, columns, msg):
    low = _norm(msg)
    if re.search(r"\b(how\s+many|count|number\s+of)\b", low):
        target = re.sub(r"^.*?(?:how\s+many|count|number\s+of)\s+", "", low)
        target = re.sub(r"\s+(?:do\s+we\s+have|are\s+there|rows?|\?)\s*$",
                        "", target).strip()
        if target:
            rows, _ = _target_rows(items, columns, target)
            return _ok(items, columns,
                       f"I found {len(rows)} row(s) matching “{target}”.")
    if re.search(r"\bwhat\s+is\s+the\s+total\b", low):
        return _grand_total(items, columns)
    return None


# ===========================================================================
#  DISPATCHER
# ===========================================================================
def _single_command(items, columns, msg):
    low = _norm(msg)
    if not low:
        return None

    # Read-cell queries MUST come before any other match
    r = _handle_read_cell(items, columns, msg)
    if r:
        return r

    # Totals shorthand
    if re.search(r"\b(?:calculate|recalculate|compute)\s+(?:all\s+)?"
                 r"(?:line\s+)?totals?\b", low) \
       or low in {"calculate totals", "recalculate totals",
                  "calculate total"}:
        return _calculate_totals(items, columns)

    if low in {"total amount", "grand total", "invoice total",
               "calculate total amount", "calculate invoice total",
               "what is the total", "how much is the total"}:
        return _grand_total(items, columns)

    # Duplicates
    if re.search(r"\b(?:remove|delete|drop)\s+"
                 r"(?:all\s+)?duplicates?\b", low):
        return _remove_duplicates(items, columns, msg)

    # Rename / replace / use-as  (v28 — before column creation so
    # "use cm as unit price" doesn't get misinterpreted as add-column)
    r = _rename_column(items, columns, msg)
    if r:
        return r

    # Column creation
    r = _handle_column_creation(items, columns, msg)
    if r:
        return r

    # Row / column removal
    r = _remove_row(items, columns, msg)
    if r: return r
    r = _remove_column(items, columns, msg)
    if r: return r
    r = _remove_rows(items, columns, msg)
    if r: return r

    # Add row
    r = _add_row(items, columns, msg)
    if r: return r

    # Set value (price)
    r = _set_value(items, columns, msg)
    if r: return r

    # Aggregate
    r = _aggregate(items, columns, msg)
    if r: return r

    # Compute
    r = _compute(items, columns, msg)
    if r: return r

    # Discount / increase
    r = _discount_or_increase(items, columns, msg)
    if r: return r

    # Sort
    r = _sort(items, columns, msg)
    if r: return r

    # Round
    r = _round(items, columns, msg)
    if r: return r

    # Questions
    r = _answer_question(items, columns, msg)
    if r: return r

    if re.search(r"\b(?:total|sum)\b", low):
        return _grand_total(items, columns)

    return None


def _split_commands(msg: str) -> List[str]:
    raw_lines = [x.strip() for x in re.split(r"\r?\n+", msg) if x.strip()]
    merged: List[str] = []
    for line in raw_lines:
        is_standalone = bool(re.search(
            r"\b(?:add|set|change|remove|delete|drop|create|make|"
            r"calculate|compute|sort|rename|replace|apply|update|"
            r"use|call|label|reset|clear)\b",
            line, re.I))
        if merged and not is_standalone:
            merged[-1] = merged[-1] + " " + line
        else:
            merged.append(line)

    final = []
    for piece in merged:
        for sub in re.split(r"\s*;\s*|\s+\bthen\b\s+", piece, flags=re.I):
            sub = sub.strip()
            if sub:
                final.append(sub)
    return final[:MAX_COMMANDS]


# ===========================================================================
#  AI PLANNER (fallback only)
# ===========================================================================
def _ai_plan(message, items, columns, history=None):
    provider = os.getenv("AI_PROVIDER", "auto").lower()
    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    system = """Semantic planner for a document editor. Return ONLY JSON:
{
  "action": "set_value|calculate_totals|grand_total|compute|add_row|
             add_column|create_column|remove_row|remove_column|
             remove_rows|remove_duplicates|rename_column|sort|
             aggregate|answer_question|clarify|none",
  "target":     "<row filter>",
  "field":      "<column label>",
  "source":     "<source column label, for create_column>",
  "new_column": "<label for new column>",
  "value":      <number or null>,
  "operation":  "multiply|divide|add|subtract",
  "explanation":"<one short sentence>"
}

Rules:
- The user's literal column names win.
- Never invent values.
- If unclear, use action "clarify"."""

    payload = json.dumps({
        "message": message,
        "columns": columns,
        "sample_rows": items[:15],
        "recent_history": (history or [])[-3:],
    }, ensure_ascii=False)

    if provider == "groq" or (provider == "auto" and not key):
        gkey = os.getenv("GROQ_API_KEY", "").strip()
        if not gkey:
            return None
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        body = {
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "temperature": 0,
            "max_tokens": 700,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": payload},
            ],
        }
        try:
            req = urllib.request.Request(
                endpoint, data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {gkey}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw = json.loads(resp.read().decode())
            txt = raw["choices"][0]["message"]["content"]
            return json.loads(re.sub(r"^```json|```$", "", txt.strip()))
        except Exception as e:
            log.warning(f"Groq planner failed: {e}")
            return None

    if key:
        endpoint = (f"https://generativelanguage.googleapis.com/v1beta/"
                    f"models/{model}:generateContent?key={key}")
        prompt = (system + "\nUSER: " + message +
                  "\nCOLUMNS: " + json.dumps(columns) +
                  "\nROWS: " + json.dumps(items[:15], ensure_ascii=False))
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0,
                                 "responseMimeType": "application/json"},
        }
        try:
            req = urllib.request.Request(
                endpoint, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw = json.loads(resp.read().decode())
            txt = raw["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(txt)
        except Exception as e:
            log.warning(f"Gemini planner failed: {e}")
    return None


def _execute_ai_plan(items, columns, plan):
    if not isinstance(plan, dict):
        return None
    action = str(plan.get("action", "none"))
    if action == "set_value":
        field = str(plan.get("field") or "unit_price")
        target = str(plan.get("target") or "all")
        value = plan.get("value")
        if value is None:
            return _clarify(items, columns,
                            "What value should I apply?", [])
        return _set_value(items, columns,
                          f"set {target} {field} to {value}")
    if action == "calculate_totals":
        return _calculate_totals(items, columns)
    if action == "grand_total":
        return _grand_total(items, columns)
    if action == "compute":
        field = str(plan.get("new_column") or "Computed")
        op = str(plan.get("operation") or "multiply")
        return _compute(items, columns, f"calculate {field} by {op}")
    if action == "create_column":
        new_label = str(plan.get("new_column") or "New Column")
        source = str(plan.get("source") or plan.get("field") or "")
        return _create_or_compute_column(items, columns, new_label, source)
    if action == "add_column":
        return _create_or_compute_column(
            items, columns, str(plan.get("new_column") or "New Column"))
    if action == "add_row":
        return _add_row(items, columns,
                        f"add a row for {plan.get('target')}")
    if action == "remove_row":
        return _remove_row(items, columns,
                           f"remove row for {plan.get('target')}")
    if action == "remove_column":
        return _remove_column(items, columns,
                              f"remove column {plan.get('field')}")
    if action == "remove_rows":
        return _remove_rows(items, columns,
                            f"remove rows where {plan.get('field')} "
                            f"is {plan.get('value')}")
    if action == "remove_duplicates":
        return _remove_duplicates(items, columns, "remove duplicates")
    if action == "rename_column":
        return _rename_column(items, columns,
                              f"rename {plan.get('field')} to "
                              f"{plan.get('new_column')}")
    if action == "sort":
        return _sort(items, columns, f"sort by {plan.get('field')}")
    if action == "aggregate":
        func = str(plan.get("operation") or "average")
        col = str(plan.get("field") or "quantity")
        return _aggregate(items, columns, f"{func} {col}")
    if action == "answer_question":
        text = str(plan.get("explanation") or "").strip()
        if text:
            return _ok(items, columns, text, via="ai")
    if action == "clarify":
        return _clarify(items, columns,
                        str(plan.get("explanation")
                            or "Please clarify that instruction."), [])
    return None


# ===========================================================================
#  PUBLIC ENTRY POINT
# ===========================================================================
def process_message(items, columns, message, history=None):
    items = _clean_items(items)
    columns = _clean_columns(columns)
    message = str(message or "").strip()[:MAX_MESSAGE]

    if not message:
        return {"status": "unrecognized", "success": False,
                "items": items, "columns": columns,
                "explanation": "No instruction was provided.",
                "needs_clarification": False}

    current_items, current_columns = items, columns
    explanations: List[str] = []
    used: List[str] = []
    pending: List[str] = []

    for cmd in _split_commands(message):
        r = _single_command(current_items, current_columns, cmd)
        via = "deterministic"
        if not r:
            plan = _ai_plan(cmd, current_items, current_columns, history)
            r = _execute_ai_plan(current_items, current_columns, plan)
            via = "ai"
        if not r:
            pending.append(cmd)
            continue
        if r.get("status") == "clarify":
            return {**r, "pending_items": current_items,
                    "pending_columns": current_columns,
                    "explanations": explanations}
        current_items = r["items"]
        current_columns = r["columns"]
        explanations.append(r.get("explanation", "Done."))
        used.append(via)

    if pending and not explanations:
        return {"status": "clarify", "success": True, "items": items,
                "columns": columns, "needs_clarification": True,
                "question": f"I could not interpret: “{pending[0]}”.",
                "options": []}

    if pending:
        for p in pending:
            explanations.append(f"Not applied (ambiguous): “{p}”.")

    return {
        "status": "ok", "success": True,
        "items": current_items, "columns": current_columns,
        "explanation": "\n".join(
            f"{i+1}. {x}" for i, x in enumerate(explanations)),
        "applied_via": "+".join(sorted(set(used))) if used else "none",
        "needs_clarification": bool(pending),
    }
