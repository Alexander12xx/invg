"""
ALTECH SOFTWARE DEVELOPERS
INTELLIGENT COMMAND ENGINE v21
------------------------------
Universal, document-aware command execution engine.

Design:
  1. Commands operate on semantic roles, not hard-coded column names.
  2. Natural language, typos, and ambiguity are handled deterministically
     where possible; the AI tier is only used when necessary.
  3. Compound commands run as a transaction-like pipeline.
  4. Ambiguous instructions return a clarification instead of guessing.
  5. Never silently drops data; every no-op returns an explanation.
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
#  ROLE MODEL  (mirrors app.py so both engines agree on classification)
# ===========================================================================
ROLE_ALIASES = {
    "product_name": [
        "product", "product name", "item", "item name", "flower",
        "flower name", "description", "item description", "particulars",
        "goods", "article", "articles", "flower variety", "variety",
        "cultivar", "product/service", "service", "commodity",
    ],
    "variety": ["variety", "cultivar", "cultivar name", "flower variety name"],
    "quantity": [
        "quantity", "qty", "qnty", "stems", "total stems", "pieces", "pcs",
        "units", "count", "total quantity", "qty trial only - stems",
        "qty trial only-stems", "qty trial only stems", "trial qty",
    ],
    "boxes": ["boxes", "box", "bx", "cartons", "carton", "ctn",
              "cases", "case", "bundles", "bundle", "packages", "pkg"],
    "pack_rate": [
        "packrate", "pack rate", "pack_rate", "per box", "per carton",
        "stems per box", "stems/box", "qty per box",
        "quantity per box", "stems per carton", "qty/carton",
    ],
    "unit_price": [
        "price", "unit price", "unit_price", "rate", "cost",
        "unit cost", "price per stem", "price/stem", "cost per stem",
        "selling price", "unit selling price", "price per unit",
    ],
    "total": [
        "total", "amount", "line total", "line amount", "total amount",
        "extended price", "revenue", "line value",
    ],
    "length_cm": ["length", "length cm", "length (cm)", "length(cm)",
                  "stem length", "size", "size cm", "height"],
    "head_size_cm": ["head size", "head size cm", "head size (cm)"],
    "color": ["color", "colour", "shade"],
    "farm_code": ["farm code", "farm", "farmcode", "farm ref",
                  "grower code", "supplier code"],
    "invoice_number": ["invoice number", "invoice no", "invoice #",
                       "invoice no.", "inv no", "inv #", "reference",
                       "ref no", "document number"],
    "date": ["date", "invoice date", "shipment date", "date of shipment",
             "issue date", "document date"],
    "due_date": ["due date", "payment due", "valid until", "expiry"],
    "currency": ["currency", "currency code", "ccy"],
    "vat_rate": ["vat", "vat rate", "vat %", "tax", "tax rate", "tax %"],
    "discount": ["discount", "disc.", "rebate"],
    "consignee": ["consignee", "bill to", "ship to", "buyer", "customer"],
    "seller": ["seller", "vendor", "supplier", "exporter"],
    "awb": ["awb", "air waybill", "waybill", "tracking number"],
    "net_weight": ["net weight", "net kg", "net weight (kgs)"],
    "gross_weight": ["gross weight", "gross kg"],
}

NUMERIC_ROLES = {
    "quantity", "boxes", "pack_rate", "length_cm", "head_size_cm",
    "unit_price", "total", "vat_rate", "discount", "net_weight",
    "gross_weight",
}


def _norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.lower().replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", s.strip())


STOPWORDS = {
    "a", "an", "the", "of", "to", "for", "all", "and", "or", "is", "are",
    "on", "in", "at", "by", "with", "as", "each", "then", "every", "row",
    "rows", "please", "me", "get", "give", "show", "can", "you", "i",
    "want", "need", "make", "create", "add", "new", "column", "field",
    "called", "named", "that", "which", "using", "from", "into", "set",
    "change", "update", "apply", "fill", "put", "use",
}


def _tokens(s: str) -> List[str]:
    return [x for x in re.findall(r"[a-z0-9]+", _norm(s))
            if x not in STOPWORDS and len(x) > 1]


def _num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
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


def _clean_items(items):
    out = []
    for row in (items or [])[:MAX_ITEMS]:
        if isinstance(row, dict):
            out.append(dict(row))
    return out


def _clean_columns(columns):
    out = []
    for c in (columns or [])[:MAX_COLUMNS]:
        if isinstance(c, dict) and c.get("key"):
            x = dict(c)
            x["label"] = str(x.get("label") or x["key"])[:200]
            out.append(x)
    return out


# ===========================================================================
#  ROLE / COLUMN RESOLUTION
# ===========================================================================
def _role_candidates(columns: List[Dict[str, Any]], query: str):
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
        candidates = [x for x in candidates if x[1].get("role") in NUMERIC_ROLES]
    if not candidates:
        return None, []
    best_score, best = candidates[0]
    second = candidates[1][0] if len(candidates) > 1 else 0
    # Accept if clearly the best match
    if best_score >= 92 or (best_score >= 80 and best_score - second >= 8):
        return best, [c for s, c in candidates[:5]]
    return None, [c for s, c in candidates[:5]]


def _product_keys(columns):
    keys = [c["key"] for c in columns
            if c.get("role") in ("product_name", "variety", "description")]
    if not keys:
        keys = [c["key"] for c in columns
                if c.get("role") not in NUMERIC_ROLES]
    return keys


def resolve_product(items, columns, query):
    """
    Return (matching_indices, ranked_preview).
    Tolerant of typos and word order.
    """
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

        # Exact substring is best
        if q in text:
            score = 100.0
        else:
            # Combined fuzzy signals
            wr = fuzz.WRatio(q, text)
            pr = fuzz.partial_ratio(q, text)
            ts = fuzz.token_set_ratio(q, text)
            score = max(wr, pr, ts)

            # Token containment bonus (e.g. "garden roses" ⊂ "garden roses alina")
            row_tokens = set(text.split())
            if qt and all(t in row_tokens for t in qt):
                score = max(score, 94.0)

            # Typo tolerance: any token within edit distance 1 of a query token
            for t in qt:
                for rt in row_tokens:
                    if abs(len(t) - len(rt)) <= 2 and _lev(t, rt) <= 1:
                        score = max(score, 88.0)

        if score >= 55:
            ranked.append((score, i))

    ranked.sort(reverse=True)
    if not ranked:
        return [], []

    best = ranked[0][0]
    second = ranked[1][0] if len(ranked) > 1 else 0

    # Strict acceptance: high absolute or strong margin
    if best >= 88 and (best >= 96 or best - second >= 6):
        threshold = max(82, best - 8)
        return [i for s, i in ranked if s >= threshold], ranked[:8]

    return [], ranked[:8]


def _lev(a: str, b: str) -> int:
    """Levenshtein distance — for typo tolerance."""
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
            cur.append(min(
                cur[j] + 1,          # insert
                prev[j + 1] + 1,     # delete
                prev[j] + (ca != cb) # substitute
            ))
        prev = cur
    return prev[-1]


# ===========================================================================
#  RESULT HELPERS
# ===========================================================================
def _ok(items, columns, explanation, via="deterministic", **extra):
    return {
        "status": "ok",
        "success": True,
        "items": items,
        "columns": columns,
        "explanation": explanation,
        "applied_via": via,
        "needs_clarification": False,
        **extra,
    }


def _clarify(items, columns, question, options, **extra):
    return {
        "status": "clarify",
        "success": True,
        "items": items,
        "columns": columns,
        "needs_clarification": True,
        "question": question,
        "options": options[:8],
        **extra,
    }


# ===========================================================================
#  TARGET RESOLUTION
# ===========================================================================
WILDCARDS = {"all", "every", "each", "everything", "all rows", "all items",
             "all flowers", "all products"}


def _is_all(target: str) -> bool:
    t = _norm(target)
    if t in WILDCARDS:
        return True
    if t.startswith(("all ", "every ", "each ")):
        return True
    return False


def _target_rows(items, columns, target):
    if _is_all(target):
        return list(range(len(items))), []
    hits, ranked = resolve_product(items, columns, target)
    if hits:
        return hits, []
    return [], ranked


# ===========================================================================
#  COLUMN ADD / LOOKUP
# ===========================================================================
def _add_column(columns: List[Dict[str, Any]], label: str, role=None):
    label = str(label or "Computed").strip()[:80]
    base = re.sub(r"[^a-z0-9]+", "_", _norm(label)).strip("_") or "computed"
    keys = {c["key"] for c in columns}
    key = base
    i = 2
    while key in keys:
        key = f"{base}_{i}"
        i += 1
    new_cols = list(columns) + [{
        "key": key,
        "label": label,
        "role": role,
        "source": "added",
    }]
    return new_cols, key


def _find_role(columns, role):
    for c in columns:
        if c.get("role") == role:
            return c
    return None


# ===========================================================================
#  PRICE COMMAND PARSER
#  Handles both verbed and verb-less forms:
#     "To all Hydrangea unit price 1.3"
#     "set price 1.3 for Hydrangea"
#     "change unit price of Garden roses to 0.7"
#     "All Carnations @ 0.21"
# ===========================================================================
VERBS = r"(?:set|change|update|make|apply|assign|give|put|add|fill|edit)"

PRICE_WORDS = r"(?:unit\s*price|unit_price|price|rate|cost|unit\s*cost)"


def _parse_price_command(msg: str):
    """
    Extract (target, value). Returns (None, None) if not a price command.
    """
    m_text = msg.strip()

    # Pattern 1 — verb-less: "To all X unit price V" or "all X price V"
    m = re.match(
        rf"^\s*(?:to\s+)?(?P<target>.+?)\s+{PRICE_WORDS}\s*[:=]?\s*"
        rf"\$?\s*(?P<value>[\d,]+(?:\.\d+)?)\s*$",
        m_text, re.I)
    if m:
        target = m.group("target").strip()
        # Clean target: strip a leading "to"
        target = re.sub(r"^to\s+", "", target, flags=re.I).strip()
        if target:
            return target, _num(m.group("value"))

    # Pattern 2 — verbed: "set price 1.3 for X" / "change unit price of X to 1.3"
    m = re.match(
        rf"^\s*{VERBS}\s+{PRICE_WORDS}\s+(?:of\s+|for\s+|to\s+|on\s+)?"
        rf"(?P<target>.+?)\s+(?:to|=|:)\s*\$?\s*"
        rf"(?P<value>[\d,]+(?:\.\d+)?)\s*$",
        m_text, re.I)
    if m:
        return m.group("target").strip(), _num(m.group("value"))

    # Pattern 3 — verbed, target-first: "set X price to 1.3"
    m = re.match(
        rf"^\s*{VERBS}\s+(?P<target>.+?)\s+{PRICE_WORDS}\s+"
        rf"(?:to|=|:)\s*\$?\s*(?P<value>[\d,]+(?:\.\d+)?)\s*$",
        m_text, re.I)
    if m:
        return m.group("target").strip(), _num(m.group("value"))

    # Pattern 4 — shorthand "X @ 1.3"
    m = re.match(
        r"^\s*(?:to\s+)?(?P<target>.+?)\s*@\s*\$?\s*"
        r"(?P<value>[\d,]+(?:\.\d+)?)\s*$",
        m_text, re.I)
    if m:
        target = re.sub(r"^to\s+", "", m.group("target"), flags=re.I).strip()
        if target:
            return target, _num(m.group("value"))

    # Pattern 5 — "X: 1.3"
    m = re.match(
        r"^\s*(?:to\s+)?(?P<target>.+?)\s*[:=]\s*\$?\s*"
        r"(?P<value>[\d,]+(?:\.\d+)?)\s*$",
        m_text, re.I)
    if m:
        target = re.sub(r"^to\s+", "", m.group("target"), flags=re.I).strip()
        if target and not _norm(target).startswith(("all", "every")):
            return target, _num(m.group("value"))

    return None, None


# ===========================================================================
#  COMMAND IMPLEMENTATIONS
# ===========================================================================
def _set_value(items, columns, msg):
    """Handle any price/value assignment."""
    target, value = _parse_price_command(msg)
    if target is None:
        # Try generic "set <col> to <value>"
        m = re.match(
            rf"^\s*{VERBS}\s+(?P<field>.+?)\s+(?:to|=|as)\s+(?P<value>.+)$",
            msg, re.I)
        if m:
            field, right = m.group("field").strip(), m.group("value").strip()
            col, _ = resolve_column(columns, field)
            if col:
                val = _num(right)
                if val is None:
                    val = right.strip().strip('"\'')
                return _apply_column_value(items, columns, col, val, "all")
        return None

    # Determine destination column: prefer existing unit_price role
    price_col = _find_role(columns, "unit_price")
    if price_col is None:
        # Fall back to a column literally called "Price" / "Rate" / "Cost"
        price_col, _ = resolve_column(columns, "unit price")
    if price_col is None:
        columns, key = _add_column(columns, "Unit Price", "unit_price")
        price_col = next(c for c in columns if c["key"] == key)

    return _apply_column_value(items, columns, price_col, value, target)


def _apply_column_value(items, columns, col, value, target):
    rows, ranked = _target_rows(items, columns, target)
    if not rows:
        # Try a broader fallback: any row containing every keyword
        words = _tokens(target)
        if words:
            rows = []
            product_keys = _product_keys(columns)
            for i, row in enumerate(items):
                text = " ".join(_norm(row.get(k, "")) for k in product_keys)
                if all(w in text for w in words):
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
    display_value = (f"{value:g}" if isinstance(value, (int, float))
                     else str(value))
    return _ok(
        out, columns,
        f"Set {col.get('label', col['key'])} to {display_value} on "
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
        qv, pv = _num(row.get(q["key"])), _num(row.get(p["key"]))
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
    if re.search(r"\b(multiply|times|product)\b", low):
        op = "multiply"
    elif re.search(r"\b(divide|divided|over|per)\b", low):
        op = "divide"
    elif re.search(r"\b(add|plus|sum)\b", low):
        op = "add"
    elif re.search(r"\b(subtract|minus)\b", low):
        op = "subtract"
    if not op:
        return None

    # Collect numeric column mentions
    mentions = []
    for c in columns:
        variants = [c["key"].replace("_", " "), c.get("label", "")]
        for v in variants:
            if v and re.search(rf"(?<!\w){re.escape(_norm(v))}(?!\w)", low):
                if c.get("role") in NUMERIC_ROLES:
                    mentions.append(c)
                    break
    if len(mentions) < 2 and op == "multiply":
        q, p = _find_role(columns, "quantity"), _find_role(columns, "unit_price")
        if q and p:
            mentions = [q, p]
    if len(mentions) < 2:
        return None

    # Name the new column
    m = re.search(
        rf"(?:calculate|get|create|make|add|new)\s+(?:the\s+)?"
        rf"(?:column\s+)?([a-z0-9 _\-]+?)\s+(?:by|as|=)\s+",
        low)
    label = (m.group(1).strip().title() if m else
             ("Line Total" if op == "multiply" else "Computed"))

    columns, key = _add_column(
        columns, label, "total" if op == "multiply" else None)

    out = deepcopy(items)
    for row in out:
        vals = [_num(row.get(c["key"])) for c in mentions]
        if any(v is None for v in vals):
            row[key] = None
            continue
        try:
            if op == "multiply":
                r = math.prod(vals)
            elif op == "divide":
                r = vals[0]
                for v in vals[1:]:
                    r /= v
            elif op == "add":
                r = sum(vals)
            else:
                r = vals[0]
                for v in vals[1:]:
                    r -= v
            row[key] = round(r, 4)
        except Exception:
            row[key] = None
    return _ok(out, columns,
               f"Computed {label} using {op} across {len(out)} row(s).")


def _aggregate(items, columns, msg):
    low = _norm(msg)
    m = re.search(
        r"\b(average|avg|mean|sum|total|count|min(?:imum)?|max(?:imum)?|"
        r"highest|lowest)\b\s+(?:of\s+|the\s+)?([a-z0-9 _\-]+?)"
        r"(?:\s+(?:per|by|for\s+each|group\s+by|grouped\s+by)\s+"
        r"([a-z0-9 _\-]+?))?(?:\s*$|[,.;!?])",
        low)
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
            r = round(r, 4) if isinstance(r, float) and not r.is_integer() else int(r)
            lines.append(f"  • {g}: {r}")
        return _ok(items, columns,
                   f"{func.capitalize()} of {tcol['label']} by {gcol['label']}:\n"
                   + "\n".join(lines))

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
    return _ok(items, columns, f"{func.capitalize()} of {tcol['label']}: {r}")


def _modify(items, columns, msg):
    low = _norm(msg)

    # Remove rows
    m = re.match(r"(?:remove|delete|exclude|drop)\s+(?:all\s+)?(.+?)(?:\s+rows?)?$",
                 low)
    if m:
        target = m.group(1).strip()
        if "empty" in target or "blank" in target:
            out = [r for r in items if not all(v in (None, "") for v in r.values())]
            return _ok(out, columns,
                       f"Removed {len(items) - len(out)} empty row(s).")
        rows, _ = _target_rows(items, columns, target)
        if rows:
            rm = set(rows)
            out = [r for i, r in enumerate(items) if i not in rm]
            return _ok(out, columns,
                       f"Removed {len(rows)} row(s) matching “{target}”.")
        # Fallback: remove rows with zero in a role
        if "zero" in target or target == "0":
            q = _find_role(columns, "quantity")
            if q:
                out = []
                removed = 0
                for r in items:
                    v = _num(r.get(q["key"]))
                    if v is None or v == 0:
                        removed += 1
                        continue
                    out.append(r)
                return _ok(out, columns, f"Removed {removed} zero-quantity row(s).")

    # Remove column
    m = re.match(r"(?:remove|delete|drop)\s+(?:the\s+)?column\s+(.+)$", low)
    if m:
        col, _ = resolve_column(columns, m.group(1).strip())
        if not col:
            return None
        new_cols = [c for c in columns if c["key"] != col["key"]]
        out = [{k: v for k, v in r.items() if k != col["key"]} for r in items]
        return _ok(out, new_cols, f"Removed column “{col['label']}”.")

    # Clear column
    m = re.match(r"clear\s+(?:all\s+)?(.+?)(?:\s+(?:column|values|prices))?$",
                 low)
    if m:
        col, _ = resolve_column(columns, m.group(1).strip(), numeric=True)
        if col:
            out = [{**r, col["key"]: None} for r in items]
            return _ok(out, columns, f"Cleared {col['label']} on all rows.")
    return None


def _sort(items, columns, msg):
    m = re.search(
        r"(?:sort|order)\s+(?:by\s+)?(.+?)(?:\s+(ascending|descending|asc|desc))?$",
        _norm(msg))
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
               f"Sorted by {col['label']} ({'descending' if desc else 'ascending'}).")


def _round(items, columns, msg):
    m = re.search(
        r"round\s+(?:all\s+|the\s+)?([a-z0-9 _\-]+?)\s+to\s+(\d+)\s*decimals?",
        _norm(msg))
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
    factor = 1.0 - (pct / 100.0) if kind in ("discount", "off", "less") \
        else 1.0 + (pct / 100.0)

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


# ===========================================================================
#  SINGLE COMMAND DISPATCHER
# ===========================================================================
def _single_command(items, columns, msg):
    low = _norm(msg)
    if not low:
        return None

    # Totals
    if re.search(r"\b(?:calculate|recalculate|compute)\s+(?:all\s+)?"
                 r"(?:line\s+)?totals?\b", low) \
       or low in {"calculate totals", "recalculate totals", "calculate total"}:
        return _calculate_totals(items, columns)

    if low in {"total amount", "grand total", "invoice total",
               "calculate total amount", "calculate invoice total",
               "what is the total", "how much is the total"}:
        return _grand_total(items, columns)

    # Price assignment
    r = _set_value(items, columns, msg)
    if r:
        return r

    # Aggregation
    r = _aggregate(items, columns, msg)
    if r:
        return r

    # Compute
    r = _compute(items, columns, msg)
    if r:
        return r

    # Discount / increase
    r = _discount_or_increase(items, columns, msg)
    if r:
        return r

    # Modify (remove)
    r = _modify(items, columns, msg)
    if r:
        return r

    # Sort
    r = _sort(items, columns, msg)
    if r:
        return r

    # Round
    r = _round(items, columns, msg)
    if r:
        return r

    # Fallback: any mention of "total" → grand total
    if re.search(r"\b(?:total|sum)\b", low):
        return _grand_total(items, columns)

    return None


# ===========================================================================
#  COMPOUND SPLITTER
# ===========================================================================
def _split_commands(msg: str) -> List[str]:
    """Split on ';' or ' then ' — authoritative boundaries."""
    parts = [x.strip() for x in re.split(r"\s*;\s*|\s+\bthen\b\s+",
                                          msg, flags=re.I) if x.strip()]
    return parts[:MAX_COMMANDS]


# ===========================================================================
#  AI PLANNER (optional fallback)
# ===========================================================================
def _ai_plan(message: str, items, columns) -> Optional[dict]:
    provider = os.getenv("AI_PROVIDER", "auto").lower()
    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    system = """You are a semantic planner for document commands.

The user typed an instruction in natural language. Return ONE JSON object
matching this schema:

{"action": "set_value|calculate_totals|grand_total|compute|remove_rows|
            remove_column|sort|clarify|none",
 "target": "<row filter, e.g. 'Hydrangea'>",
 "field": "<column label or role, e.g. 'unit_price'>",
 "value": <number or null>,
 "operation": "multiply|divide|add|subtract",
 "new_column": "<label if creating a new column>",
 "explanation": "<one short sentence>"}

Never invent prices. If the intent is unclear, use action "clarify"."""

    payload = json.dumps({
        "message": message,
        "columns": columns,
        "sample_rows": items[:15],
    }, ensure_ascii=False)

    # --- Groq ---
    if provider == "groq" or (provider == "auto" and not key):
        gkey = os.getenv("GROQ_API_KEY", "").strip()
        if not gkey:
            return None
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        payload_data = {
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
                endpoint,
                data=json.dumps(payload_data).encode(),
                headers={"Authorization": f"Bearer {gkey}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw = json.loads(resp.read().decode())
            txt = raw["choices"][0]["message"]["content"]
            return json.loads(re.sub(r"^```json|```$", "", txt.strip()))
        except Exception as e:
            log.warning(f"Groq planner failed: {e}")
            return None

    # --- Gemini ---
    if key:
        endpoint = (f"https://generativelanguage.googleapis.com/v1beta/"
                    f"models/{model}:generateContent?key={key}")
        prompt = system + "\n\nUSER: " + message + \
            "\nCOLUMNS: " + json.dumps(columns) + \
            "\nROWS: " + json.dumps(items[:15], ensure_ascii=False)
        payload_data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload_data).encode(),
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
        return _set_value(items, columns, f"set {target} {field} to {value}")

    if action == "calculate_totals":
        return _calculate_totals(items, columns)
    if action == "grand_total":
        return _grand_total(items, columns)

    if action == "compute":
        field = str(plan.get("new_column") or "Computed")
        op = str(plan.get("operation") or "multiply")
        return _compute(items, columns,
                        f"calculate {field} by {op}")

    if action == "remove_rows":
        return _modify(items, columns,
                       "remove " + str(plan.get("target") or ""))
    if action == "remove_column":
        return _modify(items, columns,
                       "remove column " + str(plan.get("field") or ""))
    if action == "sort":
        return _sort(items, columns,
                     "sort by " + str(plan.get("field") or ""))
    if action == "clarify":
        return _clarify(
            items, columns,
            str(plan.get("explanation") or "Please clarify that instruction."),
            [])
    return None


# ===========================================================================
#  PUBLIC ENTRY POINT
# ===========================================================================
def process_message(items, columns, message):
    items = _clean_items(items)
    columns = _clean_columns(columns)
    message = str(message or "").strip()[:MAX_MESSAGE]

    if not message:
        return {
            "status": "unrecognized",
            "success": False,
            "items": items,
            "columns": columns,
            "explanation": "No instruction was provided.",
            "needs_clarification": False,
        }

    current_items, current_columns = items, columns
    explanations: List[str] = []
    used: List[str] = []
    pending: List[str] = []

    for cmd in _split_commands(message):
        r = _single_command(current_items, current_columns, cmd)
        via = "deterministic"
        if not r:
            plan = _ai_plan(cmd, current_items, current_columns)
            r = _execute_ai_plan(current_items, current_columns, plan)
            via = "ai"
        if not r:
            pending.append(cmd)
            continue
        if r.get("status") == "clarify":
            return {
                **r,
                "pending_items": current_items,
                "pending_columns": current_columns,
                "explanations": explanations,
            }
        current_items = r["items"]
        current_columns = r["columns"]
        explanations.append(r.get("explanation", "Done."))
        used.append(via)

    if pending and not explanations:
        return {
            "status": "clarify",
            "success": True,
            "items": items,
            "columns": columns,
            "needs_clarification": True,
            "question": f"I could not safely interpret: {pending[0]}",
            "options": [],
        }

    if pending:
        for p in pending:
            explanations.append(f"Not applied because it was ambiguous: {p}")

    return {
        "status": "ok",
        "success": True,
        "items": current_items,
        "columns": current_columns,
        "explanation": "\n".join(
            f"{i+1}. {x}" for i, x in enumerate(explanations)),
        "applied_via": "+".join(sorted(set(used))) if used else "none",
        "needs_clarification": bool(pending),
    }
