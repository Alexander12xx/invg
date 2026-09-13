"""
chat_engine.py — Smart Docs conversational engine v15.

THE USER NEVER SEES A FORMULA. They type natural language:
  "get the total by multiplying quantity and price"
  "add a column for what we would earn if the price were 20% higher"
  "average quantity for each flower"
  "give me the sum of quantity and boxes"
  "total revenue by variety"

The engine:
  1. Runs an NL intent layer that maps English → internal operation.
  2. Executes the operation deterministically.
  3. Only falls back to the AI if the NL layer doesn't recognise the text.

The AI returns the same structured operations, never raw formulas or values.
"""

import os
import re
import json
import math
import ast
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("chat-engine")

AI_ENABLED = os.getenv("AI_ENABLED", "0") == "1"
AI_PROVIDER = os.getenv("AI_PROVIDER", "auto").lower()
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "10"))

MAX_ITEMS = 5000
MAX_COLUMNS = 200
MAX_MESSAGE_LEN = 2000
MAX_FIELD_LEN = 500
MAX_COMPOUND_COMMANDS = 10
MAX_CLARIFY_OPTIONS = 8


# ===========================================================================
#  INPUT SANITIZATION
# ===========================================================================
_INJECTION_PATTERNS = re.compile(
    r"(?:<\s*script|javascript:|data:text/html|"
    r"\b(?:DROP|TRUNCATE|ALTER)\s+TABLE\b|"
    r"\b(?:UNION\s+SELECT|INSERT\s+INTO|DELETE\s+FROM)\b|"
    r"\.\./|\bchmod\b|\bwget\b|\bcurl\b\s+http|\brm\s+-rf\b|"
    r"\beval\s*\(|\bexec\s*\(|\bsystem\s*\(|"
    r"\$\(|\`|\|\s*(?:sh|bash|cmd|powershell)\b)",
    re.IGNORECASE)

_SAFE_KEY_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,100}$")


def sanitize_message(msg: Any) -> str:
    if not isinstance(msg, str):
        return ""
    msg = msg[:MAX_MESSAGE_LEN].replace("\x00", "")
    if _INJECTION_PATTERNS.search(msg):
        return ""
    return msg


def safe_column_key(key: Any) -> Optional[str]:
    if not isinstance(key, str) or not _SAFE_KEY_RE.match(key):
        return None
    return key


def sanitize_label(label: Any) -> str:
    if not isinstance(label, str):
        return ""
    return re.sub(r"[<>]", "", label.strip()[:MAX_FIELD_LEN]).replace("\x00", "")


def sanitize_value(v: Any) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
        except Exception:
            return None
        return v
    if isinstance(v, str):
        return v[:MAX_FIELD_LEN]
    return None


def sanitize_items(items) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out = []
    for row in items[:MAX_ITEMS]:
        if not isinstance(row, dict):
            continue
        clean = {}
        for k, v in row.items():
            kk = safe_column_key(k) or re.sub(r"[^a-zA-Z0-9_\-]", "_", str(k))[:100]
            clean[kk] = sanitize_value(v)
        out.append(clean)
    return out


def sanitize_columns(columns) -> List[Dict[str, Any]]:
    if not isinstance(columns, list):
        return []
    out, seen = [], set()
    for col in columns[:MAX_COLUMNS]:
        if not isinstance(col, dict):
            continue
        key = safe_column_key(col.get("key"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({
            "key": key,
            "label": sanitize_label(col.get("label", "")) or key,
            "role": col.get("role") if isinstance(col.get("role"), str) else None,
            "source": col.get("source") if col.get("source") in ("original", "added") else "original",
        })
    return out


# ===========================================================================
#  STRUCTURE HELPERS
# ===========================================================================
NUMERIC_ROLES = {"quantity", "boxes", "pack_rate", "length_cm",
                 "head_size_cm", "unit_price", "total", "n"}


def is_numeric_column(col) -> bool:
    return col.get("role") in NUMERIC_ROLES


def numeric_columns(columns):
    return [c for c in columns if is_numeric_column(c)]


def text_columns(columns):
    return [c for c in columns if not is_numeric_column(c)]


def find_role_column(columns, role):
    for c in columns:
        if c.get("role") == role:
            return c["key"]
    return None


def find_column_by_label(columns, needle):
    if not needle:
        return None
    n = str(needle).lower().strip()
    for c in columns:
        if n == c.get("key", "").lower():
            return c["key"]
    for c in columns:
        if n == (c.get("label") or "").lower():
            return c["key"]
    for c in columns:
        if n and n in (c.get("label") or "").lower():
            return c["key"]
    n_words = [w for w in re.split(r"\s+", n) if w]
    best, best_score = None, 0
    for c in columns:
        label = (c.get("label") or "").lower()
        score = sum(1 for w in n_words if w in label)
        if score > best_score:
            best_score, best = score, c["key"]
    return best if best_score > 0 else None


def ensure_column(columns, role, label):
    key = find_role_column(columns, role)
    if key:
        return key, list(columns)
    taken = {c["key"] for c in columns}
    base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or role
    key = base
    i = 2
    while key in taken:
        key = f"{base}_{i}"
        i += 1
    new_cols = list(columns) + [{
        "key": key, "label": sanitize_label(label),
        "role": role, "source": "added",
    }]
    return key, new_cols


def to_num(v):
    if v is None or v == "" or isinstance(v, bool):
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        try:
            f = float(str(v).replace(",", "").replace("$", "").strip())
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except (TypeError, ValueError):
            return None


STOPWORDS = {"a", "an", "the", "of", "to", "for", "all", "and", "or",
             "is", "are", "on", "in", "at", "by", "with", "as",
             "each", "then", "every", "row", "rows", "please", "me",
             "get", "give", "show", "can", "you", "i", "want", "need"}


def normalize_keywords(text):
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 2]


WILDCARD_PHRASES = {
    "all", "every", "everything", "each", "any", "all rows", "every row",
    "all flowers", "all items", "all entries", "all lines",
    "the whole list", "the entire list", "every flower", "every item",
    "everything here", "the whole thing", "the entire table",
    "all products", "all records",
}


def is_wildcard(target):
    if not target:
        return False
    t = re.sub(r"\s+", " ", str(target).lower().strip())
    if t in WILDCARD_PHRASES:
        return True
    if t.startswith(("all ", "every ", "each ")):
        return True
    return False


def match_rows_by_target(items, columns, target):
    if is_wildcard(target):
        return list(range(len(items)))
    words = normalize_keywords(target)
    if not words:
        return []
    text_cols = [c["key"] for c in text_columns(columns)]
    hits = []
    for idx, row in enumerate(items):
        row_text = " ".join(str(row.get(ck) or "").lower() for ck in text_cols)
        if len(words) == 1:
            if words[0] in row_text:
                hits.append(idx)
        else:
            if all(w in row_text for w in words):
                hits.append(idx)
    return hits


# ===========================================================================
#  COMPOUND SPLITTER
# ===========================================================================
def split_compound(msg):
    if not msg or len(msg) < 4:
        return [msg.strip()] if msg and msg.strip() else []
    parts = re.split(r"\s*(?:;\s*|\s+then\s+|\s+&\s+)\s*", msg, flags=re.I)
    parts = [p.strip() for p in parts if p.strip()]
    verbs = (r"\b(?:add|set|change|apply|remove|delete|drop|multiply|divide|"
             r"calculate|compute|recalc|update|clear|round|sort|increase|"
             r"raise|discount|rename|make|give|assign|show|how|count|sum|"
             r"total|average|avg|mean|min|max|halve|double|triple|empty|fill)\b")
    final = []
    for p in parts:
        pieces = re.split(r"\s+and\s+", p, flags=re.I)
        if len(pieces) <= 1:
            final.append(p)
            continue
        merged = [pieces[0]]
        for piece in pieces[1:]:
            if re.search(verbs, piece, re.I):
                merged.append(piece)
            else:
                merged[-1] = merged[-1] + " and " + piece
        final.extend(merged)
    return [f.strip() for f in final if f.strip()][:MAX_COMPOUND_COMMANDS]


# ===========================================================================
#  RESULT HELPERS
# ===========================================================================
def Ok(items, columns, explanation, tier="deterministic", **extra):
    out = {"status": "ok", "items": items, "columns": columns,
           "explanation": explanation, "applied_via": tier,
           "needs_clarification": False}
    out.update(extra)
    return out


def Clarify(items, columns, question, options, **extra):
    out = {"status": "clarify", "items": items, "columns": columns,
           "needs_clarification": True, "question": question,
           "options": options}
    out.update(extra)
    return out


def NotUnderstood(items, columns):
    return {
        "status": "unrecognized",
        "items": items, "columns": columns,
        "needs_clarification": False,
        "explanation": (
            "I didn't quite understand that. Try things like: "
            "“get the total by multiplying quantity and unit price”, "
            "“average quantity per flower”, "
            "“set price for Garden roses to 2.50”, or "
            "“add a column for total cost”."
        ),
    }


# ===========================================================================
#  NUMERIC TOKEN / PRICE EXTRACTION
# ===========================================================================
PRICE_TOKEN_RE = re.compile(
    r"(?:(?:\$|usd|kes|eur|gbp|aed)\s*)?(\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?)"
    r"(?:\s*(?:usd|kes|eur|gbp|aed|dollars?|shillings?|percent|%))?",
    re.IGNORECASE)

PRICE_VERBS = {"add", "set", "change", "apply", "make", "assign",
               "put", "update", "give", "charge"}


def extract_price_token(text):
    m = PRICE_TOKEN_RE.search(text)
    if not m:
        return None, text
    raw = m.group(1).replace(",", "").replace(" ", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, text
    return value, text[:m.start()] + " " + text[m.end():]


def extract_target_phrase(text):
    text = text.strip()
    if not text:
        return None
    m = re.search(
        r"\b(?:to|for|on)\s+(.+?)\s+(?:" + "|".join(PRICE_VERBS) + r")\b",
        text, re.I)
    if m and m.group(1).strip().lower() not in ("a", "an", "the"):
        return m.group(1).strip()
    m = re.match(r"^(.+?)\s*[:=]", text)
    if m and m.group(1).strip().lower() not in ("a", "an", "the"):
        return m.group(1).strip()
    m = re.search(
        r"\b(?:" + "|".join(PRICE_VERBS) + r")\b\s+(?:\w+\s+){0,3}?(?:to|for|on)\s+(.+?)$",
        text, re.I)
    if m:
        c = re.sub(r"\b(?:please|now|thanks|thank you)\b\.?$", "",
                   m.group(1), flags=re.I).strip(" .,;:!?")
        if c and c.lower() not in ("a", "an", "the"):
            return c
    m = re.search(r"\ball\s+([a-z][a-z0-9 \-_']+?)(?:\s|$|[,.;!?])", text, re.I)
    if m:
        return "all " + m.group(1).strip()
    rest = text
    for verb in PRICE_VERBS:
        rest = re.sub(rf"\b{verb}\b", " ", rest, flags=re.I)
    rest = re.sub(r"\b(?:the|a|an|please|to|for|on|of|in|at|with)\b",
                  " ", rest, flags=re.I)
    rest = re.sub(r"\s+", " ", rest).strip(" .,;:!?")
    return rest if rest and rest.lower() not in ("a", "an", "the") else None


# ===========================================================================
#  NATURAL-LANGUAGE INTENT LAYER
#  Maps plain English → internal action names.
# ===========================================================================
NL_AGGS = {
    "average": ["average", "avg", "mean", "typical"],
    "sum": ["sum", "total", "add up", "add together"],
    "count": ["count", "how many", "number of"],
    "min": ["minimum", "smallest", "lowest"],
    "max": ["maximum", "largest", "highest"],
}

NL_VERBS_MULTIPLY = ["multiply", "times", "product of", "x", "multiplied by"]
NL_VERBS_DIVIDE = ["divide", "divided by", "over", "split"]
NL_VERBS_ADD = ["add", "sum", "plus", "combined with"]
NL_VERBS_SUBTRACT = ["subtract", "minus", "less", "minus out"]


def _detect_multiply_divider(text: str) -> Optional[str]:
    """Return 'multiply' or 'divide' if the text contains the intent."""
    t = text.lower()
    for kw in NL_VERBS_MULTIPLY:
        if re.search(rf"\b{re.escape(kw)}\b", t):
            return "multiply"
    for kw in NL_VERBS_DIVIDE:
        if re.search(rf"\b{re.escape(kw)}\b", t):
            return "divide"
    return None


def _extract_column_mentions(text: str, columns) -> List[str]:
    """
    Find every column label or key mentioned in the text.
    Returns keys ordered by where they appear in the text.
    """
    found: List[Tuple[int, str]] = []
    t = text.lower()
    for c in columns:
        candidates = set()
        if c.get("key"):
            candidates.add(c["key"].lower())
        if c.get("label"):
            candidates.add(c["label"].lower())
        for cand in candidates:
            if not cand or len(cand) < 2:
                continue
            # word boundary match
            for m in re.finditer(
                    rf"(?<![a-z0-9]){re.escape(cand)}(?![a-z0-9])", t):
                found.append((m.start(), c["key"]))
                break
    # De-dup, preserve order
    seen = set()
    out = []
    for _, key in sorted(found, key=lambda x: x[0]):
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def handle_natural_compute(items, columns, msg):
    """
    Handles the following natural language patterns:

    "get the total by multiplying quantity and unit price"
    "calculate line total as quantity times price"
    "multiply quantity and boxes for a new column called total items"
    "total = quantity * unit price"
    "add a column for total that is quantity * price"
    "what is the average quantity per flower"
    "sum of quantity by variety"
    "total revenue by flower"
    """
    low = msg.lower()

    # -------- aggregation without grouping column: "average quantity"
    for func, kws in NL_AGGS.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", low):
                m = re.search(
                    rf"\b{re.escape(kw)}\b\s+(?:of\s+|the\s+)?(.+?)"
                    rf"(?:\s+(?:per|by|for\s+each|group\s+by)\s+(.+?))?"
                    rf"(?:\s*$|[,.;!?])",
                    low)
                if m:
                    target_text = m.group(1).strip() if m.group(1) else ""
                    group_text = m.group(2).strip() if m.group(2) else ""
                    target_key = find_column_by_label(columns, target_text)
                    if target_key:
                        synth = f"{func} {target_text}"
                        if group_text:
                            synth += f" per {group_text}"
                        return handle_aggregation(items, columns, synth)

    # -------- direct compute: "X = A * B" or "add a column X as A times B"
    m = re.search(
        r"\b(?:add|create|make|new)\s+(?:a\s+|the\s+)?column\s+"
        r"(?:called\s+|named\s+|for\s+)?([a-z0-9 _\-]+?)\s+"
        r"(?:as|to|=|equals?|that\s+is|which\s+is|that\s+equals?|"
        r"by\s+(?:multiplying|dividing|adding|subtracting))\s+(.+?)"
        r"(?:\s*$|[,.;!?])",
        msg, re.I)
    if m:
        new_label = m.group(1).strip()
        rest = m.group(2).strip()
        return _compute_new_column(items, columns, new_label, rest)

    # also allow "add column X = A * B"
    m = re.search(
        r"\b(?:add|create|make|new)\s+(?:a\s+|the\s+)?column\s+"
        r"([a-z0-9 _\-]+?)\s*[:=]\s*(.+?)(?:\s*$|[,.;!?])",
        msg, re.I)
    if m:
        return _compute_new_column(items, columns, m.group(1).strip(), m.group(2).strip())

    # -------- "get the total by multiplying X and Y"
    #           "calculate total as X times Y"
    #           "total of X and Y"
    m = re.search(
        r"\b(?:get|give|show|calculate|compute|make|find|derive|"
        r"work\s+out|figure\s+out)\s+(?:the\s+|me\s+the\s+)?"
        r"([a-z0-9 _\-]+?)\s+"
        r"(?:by\s+)?(?:multiplying|times|product\s+of|"
        r"dividing|over|"
        r"adding|sum\s+of|plus)\s+(.+?)(?:\s*$|[,.;!?])",
        msg, re.I)
    if m:
        new_label = m.group(1).strip()
        rest = m.group(2).strip()
        return _compute_new_column(items, columns, new_label, rest)

    # -------- "multiply X and Y" (no explicit target name)
    m = re.search(
        r"\b(?:multiply|times|product\s+of)\s+(.+?)\s+(?:and|by|\*)\s+(.+?)"
        r"(?:\s*$|[,.;!?])", msg, re.I)
    if m and not re.search(r"\b(?:column|as|=)\b", msg, re.I):
        # No column name given; ask for one
        left = m.group(1).strip()
        right = m.group(2).strip()
        left_key = find_column_by_label(columns, left)
        right_key = find_column_by_label(columns, right)
        if left_key and right_key:
            suggested = f"{_safe_name(left)}_{_safe_name(right)}"
            return Clarify(
                items, columns,
                f"Multiply {left} by {right} — what should I call the new column?",
                [{"label": f"Call it “{suggested}”", "key": "", "action": "compute_column",
                  "value": {"label": suggested, "formula_keys": [left_key, right_key],
                            "op": "multiply"}}],
            )

    return None


def _safe_name(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_") or "col"


def _compute_new_column(items, columns, new_label, rest):
    """
    Given a target label and a natural-language expression like
    "quantity and unit price" or "quantity times unit price",
    build the new column.
    """
    new_label = sanitize_label(new_label)[:60] or "Computed"
    op = None

    # Which operation?
    if re.search(r"\b(?:multiply|times|product|multiplied)\b", rest, re.I):
        op = "multiply"
    elif re.search(r"\b(?:divide|divided|over|per)\b", rest, re.I):
        op = "divide"
    elif re.search(r"\b(?:add|sum|plus|combined)\b", rest, re.I):
        op = "add"
    elif re.search(r"\b(?:subtract|minus|less)\b", rest, re.I):
        op = "subtract"
    else:
        # Default to multiply if two columns are listed and no verb
        op = "multiply"

    # Column references
    col_keys = _extract_column_mentions(rest, columns)
    if len(col_keys) < 2:
        # Try to find any two mentioned column labels manually
        parts = re.split(r"\s+(?:and|times|by|\*|,|/|plus|minus|and\s+then)\s+",
                         rest, flags=re.I)
        col_keys = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            key = find_column_by_label(columns, p)
            if key and key not in col_keys:
                col_keys.append(key)

    if len(col_keys) < 2:
        return None

    # Constants
    constants = re.findall(r"\b(\d+(?:\.\d+)?)\b", rest)

    # Create new column
    taken = {c["key"] for c in columns}
    base = _safe_name(new_label)
    key = base
    i = 2
    while key in taken:
        key = f"{base}_{i}"
        i += 1

    columns = list(columns) + [{
        "key": key, "label": new_label,
        "role": None, "source": "added",
    }]

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
    for row in items:
        vals = []
        for ck in col_keys:
            v = to_num(row.get(ck))
            if v is None:
                v = 0
            vals.append(v)
        # Multiply by any explicit constants at the end (e.g. "times 1.1")
        for c in constants:
            try:
                vals.append(float(c))
            except (TypeError, ValueError):
                pass
        result = combine(vals)
        r = dict(row)
        r[key] = round(result, 4) if result is not None else None
        updated.append(r)

    return Ok(updated, columns,
              f"Added column “{new_label}” using "
              f"{op} of {len(col_keys)} column(s).")


# ===========================================================================
#  EXISTING HANDLERS (unchanged behaviour, new Ok/Clarify return shape)
# ===========================================================================
def handle_calculate_totals(items, columns, msg):
    low = msg.lower()
    if not re.search(r"\b(total|totals|line\s*total|amount|amounts)\b", low):
        return None
    qty_key = find_role_column(columns, "quantity")
    price_key = find_role_column(columns, "unit_price")
    if not qty_key or not price_key:
        return None
    total_key, columns = ensure_column(columns, "total", "Line Total")
    updated = []
    for row in items:
        q = to_num(row.get(qty_key)) or 0
        u = to_num(row.get(price_key)) or 0
        r = dict(row)
        r[total_key] = round(q * u, 2)
        updated.append(r)
    return Ok(updated, columns, f"Calculated line total for {len(updated)} row(s).")


def handle_set_price(items, columns, msg):
    text = re.sub(r"\b(usd|kes|eur|gbp|aed)\b", "$", msg, flags=re.I)
    if not re.search(r"\b(?:" + "|".join(PRICE_VERBS) + r")\b", text, re.I) \
       and "$" not in text:
        return None
    price, text_no_price = extract_price_token(text)
    if price is None or price < 0 or price > 1_000_000_000:
        return None
    target = extract_target_phrase(text_no_price)
    if not target or re.search(r"\d", target):
        return None
    target = target[:200]
    price_key, columns = ensure_column(columns, "unit_price", "Unit Price")
    wildcard = is_wildcard(target)
    if not wildcard and re.search(rf"\ball\s+{re.escape(target)}\b", msg, re.I):
        if not match_rows_by_target(items, columns, target):
            wildcard = True
    if wildcard:
        hits = list(range(len(items)))
    else:
        hits = match_rows_by_target(items, columns, target)
        if not hits:
            words = [w for w in normalize_keywords(target) if len(w) >= 4]
            if words:
                hits = []
                for idx, row in enumerate(items):
                    row_text = " ".join(
                        str(row.get(ck) or "").lower()
                        for ck in [c["key"] for c in text_columns(columns)])
                    if any(w in row_text for w in words):
                        hits.append(idx)
    if not hits:
        return None
    hit_set = set(hits)
    updated = []
    for idx, row in enumerate(items):
        r = dict(row)
        if idx in hit_set:
            r[price_key] = price
        updated.append(r)
    label = "all rows" if wildcard else f"“{target}”"
    return Ok(updated, columns,
              f"Set unit price to {price} on {len(hits)} row(s) matching {label}.")


def handle_set_column_value(items, columns, msg):
    m = re.search(
        r"\bset\s+(?:the\s+)?([a-z0-9 _\-]+?)\s+(?:to|as|=)\s+"
        r"([a-z0-9.$_\-]+)"
        r"(?:\s+on\s+(?:all\s+|the\s+)?([a-z0-9 _\-']+?))?"
        r"(?:\s*$|[,.;!?])", msg, re.I)
    if not m:
        return None
    col_needle = m.group(1).strip().lower()
    if "price" in col_needle or col_needle in ("cost", "rate"):
        return None
    value_raw = m.group(2).strip()
    filter_target = (m.group(3) or "").strip()
    key = find_column_by_label(columns, col_needle)
    if not key:
        return None
    v_num = to_num(value_raw)
    value = v_num if v_num is not None else value_raw
    updated = []
    applied = 0
    hit_set = None
    if filter_target and not is_wildcard(filter_target):
        hits = match_rows_by_target(items, columns, filter_target)
        if not hits:
            return None
        hit_set = set(hits)
    for idx, row in enumerate(items):
        r = dict(row)
        if hit_set is None or idx in hit_set:
            r[key] = value
            applied += 1
        updated.append(r)
    if applied == 0:
        return None
    label = next((c["label"] for c in columns if c["key"] == key), col_needle)
    suffix = ""
    if filter_target:
        suffix = " on all rows" if is_wildcard(filter_target) else f" on “{filter_target}”"
    return Ok(updated, columns,
              f"Set {label} to {value}{suffix} on {applied} row(s).")


def handle_bare_number(items, columns, msg):
    stripped = msg.strip(" .,;:!?")
    m = re.fullmatch(r"\$?\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?)\s*", stripped)
    if m:
        value = to_num(m.group(1))
        if value is None:
            return None
        numeric_cols = numeric_columns(columns)
        if not numeric_cols:
            return None
        options = [{
            "label": f"Put {value} in “{c.get('label') or c['key']}”",
            "key": c["key"], "action": "set_value", "value": value,
        } for c in numeric_cols[:MAX_CLARIFY_OPTIONS]]
        return Clarify(items, columns,
                       f"You typed {value} but didn't say where it should go. "
                       f"Which column should I put it in?",
                       options, detected_value=value)
    return None


def handle_aggregation(items, columns, msg):
    m = re.search(
        r"\b(average|avg|mean|sum|total|count|min(?:imum)?|max(?:imum)?)\b"
        r"\s+(?:of\s+|the\s+)?"
        r"(?:the\s+)?([a-z0-9 _\-]+?)"
        r"(?:\s+(?:per|by|for\s+each|group\s+by|grouped\s+by)\s+"
        r"([a-z0-9 _\-]+?))?"
        r"(?:\s*$|[,.;!?])", msg, re.I)
    if not m:
        return None
    func = m.group(1).lower()
    target = (m.group(2) or "").strip()
    group_by = (m.group(3) or "").strip()
    target_key = find_column_by_label(columns, target)
    if not target_key:
        return None
    if group_by:
        group_key = find_column_by_label(columns, group_by)
        if not group_key:
            return None
        groups: Dict[str, List[float]] = {}
        for row in items:
            g = str(row.get(group_key) or "").strip()
            v = to_num(row.get(target_key))
            if v is None:
                continue
            groups.setdefault(g, []).append(v)
        lines = []
        for g in sorted(groups.keys()):
            values = groups[g]
            if func in ("average", "avg", "mean"):
                r = sum(values) / len(values)
            elif func in ("sum", "total"):
                r = sum(values)
            elif func == "count":
                r = len(values)
            elif func.startswith("min"):
                r = min(values)
            elif func.startswith("max"):
                r = max(values)
            else:
                continue
            if isinstance(r, float) and r.is_integer():
                r = int(r)
            else:
                r = round(r, 4) if isinstance(r, float) else r
            lines.append(f"  • {g}: {r}")
        header = {"average": "Average", "avg": "Average", "mean": "Average",
                  "sum": "Sum", "total": "Total", "count": "Count",
                  "min": "Minimum", "minimum": "Minimum",
                  "max": "Maximum", "maximum": "Maximum"}.get(func, func.capitalize())
        tlabel = next((c["label"] for c in columns if c["key"] == target_key), target)
        glabel = next((c["label"] for c in columns if c["key"] == group_key), group_by)
        return Ok(items, columns,
                  f"{header} of “{tlabel}” by “{glabel}”:\n" + "\n".join(lines))
    values = [to_num(r.get(target_key)) for r in items]
    values = [v for v in values if v is not None]
    if not values:
        return None
    if func in ("average", "avg", "mean"):
        r = sum(values) / len(values)
    elif func in ("sum", "total"):
        r = sum(values)
    elif func == "count":
        r = len(values)
    elif func.startswith("min"):
        r = min(values)
    elif func.startswith("max"):
        r = max(values)
    else:
        return None
    if isinstance(r, float) and r.is_integer():
        r = int(r)
    else:
        r = round(r, 4) if isinstance(r, float) else r
    label = next((c["label"] for c in columns if c["key"] == target_key), target)
    return Ok(items, columns, f"{func.capitalize()} of “{label}”: {r}")


def handle_remove_column(items, columns, msg):
    m = re.search(
        r"\b(?:remove|delete|drop|hide|get\s+rid\s+of|take\s+out)\s+"
        r"(?:the\s+|a\s+|an\s+)?(?:column\s+|field\s+)?"
        r"([a-z0-9][a-z0-9 _\-]+?)"
        r"(?:\s+column|\s+field)?(?:\s*$|[,.;!?])", msg, re.I)
    if not m:
        return None
    needle = m.group(1).strip()
    key = find_column_by_label(columns, needle)
    if not key:
        return None
    new_cols = [c for c in columns if c["key"] != key]
    updated = [{k: v for k, v in row.items() if k != key} for row in items]
    label = next((c["label"] for c in columns if c["key"] == key), needle)
    return Ok(updated, new_cols, f"Removed column “{label}”.")


def handle_remove_rows(items, columns, msg):
    low = msg.lower()
    if not re.search(r"\b(remove|delete|drop|filter\s+out|exclude|"
                     r"get\s+rid\s+of|hide|skip)\b.*\brows?\b", low) \
       and not re.search(r"\b(remove|delete|drop|filter|exclude)\b.*"
                         r"\b(items?|entries|lines?)\b", low):
        return None
    qty_key = find_role_column(columns, "quantity")
    price_key = find_role_column(columns, "unit_price")
    if re.search(r"\bempty\b|\bblank\b|\bno\s+content\b", low):
        kept, removed = [], 0
        for row in items:
            if all(v in (None, "") for v in row.values()):
                removed += 1
                continue
            kept.append(row)
        return Ok(kept, columns, f"Removed {removed} empty row(s).")
    if re.search(r"\b(?:zero|0|no)\b.*\b(?:quantity|qty|stems?)\b", low) or \
       re.search(r"\b(?:quantity|qty|stems?)\b.*\b(?:is|equals?|=)\s*"
                 r"(?:zero|0)\b", low):
        if not qty_key:
            return None
        kept, removed = [], 0
        for row in items:
            q = to_num(row.get(qty_key))
            if q is None or q == 0:
                removed += 1
                continue
            kept.append(row)
        return Ok(kept, columns, f"Removed {removed} row(s) with zero quantity.")
    if re.search(r"\b(no|missing|empty|without)\s+price\b", low):
        if not price_key:
            return None
        kept, removed = [], 0
        for row in items:
            if to_num(row.get(price_key)) is None:
                removed += 1
                continue
            kept.append(row)
        return Ok(kept, columns, f"Removed {removed} row(s) without a price.")
    return None


def _resolve_column_for(columns, what):
    w = what.lower()
    if w.startswith(("quantit", "qty", "stem")):
        return find_role_column(columns, "quantity")
    if "price" in w or w.startswith("cost"):
        return find_role_column(columns, "unit_price")
    if w.startswith(("amount", "total")):
        return find_role_column(columns, "total")
    if w.startswith("box"):
        return find_role_column(columns, "boxes")
    if "pack" in w:
        return find_role_column(columns, "pack_rate")
    return find_column_by_label(columns, w)


def handle_multiply(items, columns, msg):
    m = re.search(
        r"\bmultiply\s+(?:all\s+|the\s+)?"
        r"(quantit(?:y|ies)|qty|stems?|prices?|unit\s*price|"
        r"amounts?|totals?|boxes?|pack\s*rate)"
        r"\s+by\s+\$?\s*(\d+(?:[.,]\d+)?)", msg, re.I)
    if not m:
        m2 = re.search(
            r"\b(double|triple|halve|half)\s+"
            r"(quantit(?:y|ies)|qty|stems?|prices?|unit\s*price|amounts?|totals?)",
            msg, re.I)
        if not m2:
            return None
        word = m2.group(1).lower()
        what = m2.group(2).lower()
        factor = {"double": 2.0, "triple": 3.0, "halve": 0.5, "half": 0.5}[word]
        key = _resolve_column_for(columns, what)
        if not key:
            return None
        updated = []
        for row in items:
            r = dict(row)
            v = to_num(r.get(key))
            if v is not None:
                r[key] = round(v * factor, 4)
            updated.append(r)
        return Ok(updated, columns, f"{word.capitalize()}d {what}.")
    what = m.group(1).lower()
    factor = float(m.group(2).replace(",", ""))
    key = _resolve_column_for(columns, what)
    if not key:
        return None
    updated = []
    for row in items:
        r = dict(row)
        v = to_num(r.get(key))
        if v is not None:
            r[key] = round(v * factor, 4)
        updated.append(r)
    return Ok(updated, columns, f"Multiplied {what} by {factor}.")


def handle_discount(items, columns, msg):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:discount|off|less|reduction)",
                  msg, re.I)
    if not m:
        m = re.search(r"\bdiscount\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", msg, re.I)
    if not m:
        return None
    pct = float(m.group(1))
    factor = 1.0 - (pct / 100.0)
    m_filter = re.search(r"\b(?:on|for|to)\s+(?:all\s+|the\s+)?"
                         r"([a-z0-9 _\-']+?)(?:\s*$|[,.;!?])", msg, re.I)
    target_rows = None
    if m_filter:
        t = m_filter.group(1).strip()
        if not is_wildcard(t):
            hits = match_rows_by_target(items, columns, t)
            if hits:
                target_rows = set(hits)
    key = find_role_column(columns, "unit_price") or find_role_column(columns, "total")
    if not key:
        return None
    updated = []
    applied = 0
    for idx, row in enumerate(items):
        r = dict(row)
        if target_rows is None or idx in target_rows:
            v = to_num(r.get(key))
            if v is not None:
                r[key] = round(v * factor, 4)
                applied += 1
        updated.append(r)
    if applied == 0:
        return None
    suffix = ""
    if m_filter:
        t = m_filter.group(1).strip()
        suffix = " on all rows" if is_wildcard(t) else f" on “{t}”"
    return Ok(updated, columns, f"Applied a {pct}% discount{suffix}.")


def handle_increase(items, columns, msg):
    m = re.search(
        r"\b(?:increase|raise|bump|add)\s+(?:all\s+|the\s+)?"
        r"(?:prices?|unit\s*price|amounts?|totals?)\s+by\s+"
        r"(\d+(?:\.\d+)?)\s*%", msg, re.I)
    if not m:
        return None
    pct = float(m.group(1))
    factor = 1.0 + (pct / 100.0)
    key = find_role_column(columns, "unit_price") or find_role_column(columns, "total")
    if not key:
        return None
    updated = []
    for row in items:
        r = dict(row)
        v = to_num(r.get(key))
        if v is not None:
            r[key] = round(v * factor, 4)
        updated.append(r)
    return Ok(updated, columns, f"Increased prices by {pct}%.")


def handle_add_fixed(items, columns, msg):
    m = re.search(
        r"\b(?:add|subtract)\s+\$?(\d+(?:\.\d+)?)\s+"
        r"(?:to|from)\s+(?:all\s+|the\s+)?"
        r"(?:prices?|unit\s*price|amounts?|totals?)", msg, re.I)
    if not m:
        return None
    delta = float(m.group(1))
    if msg.lower().startswith("subtract"):
        delta = -delta
    key = find_role_column(columns, "unit_price") or find_role_column(columns, "total")
    if not key:
        return None
    updated = []
    for row in items:
        r = dict(row)
        v = to_num(r.get(key))
        if v is not None:
            r[key] = round(v + delta, 4)
        updated.append(r)
    verb = "Added" if delta > 0 else "Subtracted"
    return Ok(updated, columns, f"{verb} {abs(delta)} on prices.")


def handle_sort(items, columns, msg):
    m = re.search(
        r"\bsort\s+(?:rows?\s+)?(?:by\s+)?([a-z0-9 _\-]+?)"
        r"(?:\s+(asc(?:ending)?|desc(?:ending)?|asc|desc))?"
        r"(?:\s+(?:order|direction))?(?:\s*$|[,.;!?])",
        msg, re.I)
    if not m:
        m = re.search(r"\border\s+by\s+([a-z0-9 _\-]+?)"
                      r"(?:\s+(asc|desc))?(?:\s*$|[,.;!?])", msg, re.I)
        if not m:
            return None
    needle = m.group(1).strip()
    direction = (m.group(2) or "").lower()
    desc = direction.startswith("desc")
    key = find_column_by_label(columns, needle)
    if not key:
        return None

    def key_fn(row):
        v = row.get(key)
        if v is None:
            return (1, "")
        num = to_num(v)
        if num is not None:
            return (0, num)
        return (1, str(v).lower())

    updated = sorted(items, key=key_fn, reverse=desc)
    label = next((c["label"] for c in columns if c["key"] == key), needle)
    return Ok(updated, columns,
              f"Sorted by “{label}” ({'descending' if desc else 'ascending'}).")


def handle_clear(items, columns, msg):
    m = re.search(r"\bclear\s+(?:all\s+|the\s+)?"
                  r"(prices?|unit\s*price|quantit(?:y|ies)|qty|stems?|totals?|amounts?)",
                  msg, re.I)
    if not m:
        return None
    what = m.group(1).lower()
    key = _resolve_column_for(columns, what)
    if not key:
        return None
    updated = [{**row, key: None} for row in items]
    return Ok(updated, columns, f"Cleared all {what}.")


def handle_round(items, columns, msg):
    m = re.search(
        r"\bround\s+(?:all\s+|the\s+)?"
        r"(prices?|unit\s*price|quantit(?:y|ies)|qty|totals?|amounts?)"
        r"\s+to\s+(\d+)\s*decimals?", msg, re.I)
    if not m:
        return None
    target, decimals = m.group(1).lower(), int(m.group(2))
    decimals = max(0, min(decimals, 8))
    key = _resolve_column_for(columns, target)
    if not key:
        return None
    updated = []
    for row in items:
        r = dict(row)
        v = to_num(r.get(key))
        if v is not None:
            r[key] = round(v, decimals)
        updated.append(r)
    return Ok(updated, columns, f"Rounded {target} to {decimals} decimal(s).")


def handle_rename_column(items, columns, msg):
    m = re.search(
        r"\brename\s+(?:the\s+)?([a-z0-9 _\-]+?)\s+(?:column\s+)?"
        r"(?:to|as)\s+([a-z0-9 _\-]+)", msg, re.I)
    if not m:
        return None
    old_needle, new_label = m.group(1).strip(), m.group(2).strip()
    key = find_column_by_label(columns, old_needle)
    if not key:
        return None
    new_label = sanitize_label(new_label) or new_label
    new_cols = []
    for c in columns:
        if c["key"] == key:
            new_cols.append({**c, "label": new_label})
        else:
            new_cols.append(c)
    return Ok(items, new_cols, f"Renamed “{old_needle}” to “{new_label}”.")


def handle_question(items, columns, msg):
    low = msg.lower().strip()
    qty_key = find_role_column(columns, "quantity")
    price_key = find_role_column(columns, "unit_price")
    total_key = find_role_column(columns, "total")
    if re.search(r"\b(how\s+many\s+rows?|row\s+count|count\s+(?:of\s+)?rows?|"
                 r"number\s+of\s+rows?|how\s+many\s+"
                 r"(?:items?|entries|lines?))\b", low):
        return Ok(items, columns, f"There are {len(items)} rows.")
    if (re.search(r"\b(total|sum|overall|how\s+many)\b", low)
        and re.search(r"\b(quantity|qty|stems?|units?|pieces?)\b", low)
        and not re.search(r"\b(add|set|change|calc|remove|delete|discount|multiply|"
                          r"calculate|compute|update)\b", low)):
        if qty_key:
            total = sum(to_num(r.get(qty_key)) or 0 for r in items)
            return Ok(items, columns,
                      f"Total quantity is "
                      f"{int(total) if float(total).is_integer() else round(total, 2)}.")
    if (re.search(r"\b(total|sum|grand|overall)\s+(?:of\s+)?"
                  r"(amount|price|value|cost|invoice)\b", low)
        or re.search(r"\bhow\s+much\b.*\btotal\b", low)):
        if total_key:
            total = sum(to_num(r.get(total_key)) or 0 for r in items)
        elif qty_key and price_key:
            total = sum((to_num(r.get(qty_key)) or 0) *
                        (to_num(r.get(price_key)) or 0) for r in items)
        else:
            return None
        return Ok(items, columns, f"Total amount is {round(total, 2)}.")
    m = re.search(r"\bhow\s+many\s+([a-z0-9 _\-']+?)(?:\s*\?|\s*$|,|\.|!)", low)
    if m:
        target = m.group(1).strip()
        hits = match_rows_by_target(items, columns, target)
        return Ok(items, columns, f"I found {len(hits)} row(s) matching “{target}”.")
    m = re.search(r"\bcount\s+(?:of\s+)?([a-z0-9 _\-']+?)(?:\s*$|[,.;!?])", low)
    if m:
        target = m.group(1).strip()
        hits = match_rows_by_target(items, columns, target)
        return Ok(items, columns, f"{len(hits)} row(s) match “{target}”.")
    return None


# ===========================================================================
#  DISPATCHER
# ===========================================================================
DETERMINISTIC_HANDLERS = [
    # natural-language front-end goes first
    handle_natural_compute,
    # structured operations
    handle_aggregation,
    handle_calculate_totals,
    handle_question,
    handle_set_price,
    handle_set_column_value,
    handle_multiply,
    handle_discount,
    handle_increase,
    handle_add_fixed,
    handle_remove_column,
    handle_remove_rows,
    handle_sort,
    handle_clear,
    handle_round,
    handle_rename_column,
    handle_bare_number,   # last: single number → clarify
]


def run_deterministic(items, columns, msg):
    for handler in DETERMINISTIC_HANDLERS:
        try:
            result = handler(items, columns, msg)
        except Exception as e:
            logger.warning(f"{handler.__name__} raised: {e}")
            continue
        if result:
            return result
    return None


# ===========================================================================
#  AI FALLBACK — updated to accept NL compute
# ===========================================================================
AI_SYSTEM = """You are the AI assistant for a document transformation tool.

The user has a table of line items and types a plain-English instruction.
Pick the ONE action that best matches their intent and return it as JSON.

Actions (whitelist):
  set_price | set_value | calculate_totals | multiply | add_fixed |
  discount | increase_pct | remove_column | remove_rows | clear_column |
  round_column | sort | rename_column | compute_column | aggregate |
  answer_question | clarify | none

Return schema:
{
  "action": "<name>",
  "args": { ... },
  "explanation": "<one short sentence>"
}

For compute_column (used whenever the user wants a new column derived from
two or more existing columns), return:
  args = {
    "label": "<new column name, e.g. 'Total'>",
    "left":  "<label or key of first column>",
    "right": "<label or key of second column>",
    "op":    "multiply|divide|add|subtract",
    "extra_constants": [<number>, ...]   // optional multipliers like 1.1
  }

For aggregate (used for "average X per Y", "sum of X", etc.):
  args = {"func": "average|sum|count|min|max",
          "column": "<label>",
          "group_by": "<label or null>"}

Rules:
- Never invent prices or quantities.
- Numbers must be numbers, not strings.
- "all"/"every" as filter = null (all rows).
"""

_VALID_ACTIONS = {
    "set_price", "set_value", "calculate_totals", "multiply", "add_fixed",
    "discount", "increase_pct", "remove_column", "remove_rows",
    "clear_column", "round_column", "sort", "rename_column",
    "compute_column", "aggregate", "answer_question", "clarify", "none",
}


def _ai_clean_json(text):
    if not text or not isinstance(text, str):
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _ai_call(prompt):
    logger.info(f"AI call: provider={AI_PROVIDER}")
    if AI_PROVIDER in ("gemini", "auto"):
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if api_key:
                genai.configure(api_key=api_key)
                candidates = [
                    os.getenv("GEMINI_MODEL", "").strip(),
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-2.5-flash",
                    "gemini-flash-latest",
                ]
                candidates = [m for m in candidates if m]
                for model_name in candidates:
                    try:
                        model = genai.GenerativeModel(model_name)
                        resp = model.generate_content(
                            prompt,
                            generation_config={
                                "temperature": 0.0,
                                "response_mime_type": "application/json",
                                "max_output_tokens": 1024,
                            },
                            request_options={"timeout": AI_TIMEOUT_SECONDS},
                        )
                        if resp and resp.text:
                            logger.info(f"Gemini succeeded with model: {model_name}")
                            return resp.text
                    except Exception as inner:
                        if "404" in str(inner) or "not found" in str(inner).lower():
                            continue
                        logger.warning(f"Gemini ({model_name}) failed: {inner}")
                        break
        except Exception as e:
            logger.warning(f"Gemini chat call failed: {e}")
    if AI_PROVIDER in ("groq", "auto"):
        try:
            import groq as groq_mod
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if api_key:
                try:
                    client = groq_mod.Groq(api_key=api_key)
                except TypeError:
                    client = groq_mod.Groq(api_key=api_key, timeout=AI_TIMEOUT_SECONDS)
                resp = client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    messages=[{"role": "system", "content": AI_SYSTEM},
                              {"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                    timeout=AI_TIMEOUT_SECONDS,
                )
                if resp.choices:
                    logger.info("Groq call succeeded")
                    return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq chat call failed: {e}")
    return None


def run_ai(items, columns, msg):
    if not AI_ENABLED:
        return None
    payload = json.dumps({
        "columns": columns,
        "items": items[:80],
        "message": msg[:800],
    }, ensure_ascii=False)[:30000]
    raw = _ai_call(AI_SYSTEM + "\n\nINPUT:\n" + payload)
    logger.info(f"AI raw response: {(raw[:400] if raw else None)}")
    data = _ai_clean_json(raw) if raw else None
    if not data or not isinstance(data, dict):
        return None
    action = data.get("action")
    if action not in _VALID_ACTIONS:
        return None
    logger.info(f"AI parsed action: {action}")
    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    explanation = str(data.get("explanation", ""))[:300]

    try:
        if action == "clarify":
            q = str(args.get("question", ""))[:300]
            opts = args.get("options") or []
            if q and isinstance(opts, list):
                clean_opts = []
                for o in opts[:MAX_CLARIFY_OPTIONS]:
                    if not isinstance(o, dict):
                        continue
                    clean_opts.append({
                        "label": str(o.get("label", ""))[:200],
                        "key": safe_column_key(o.get("key")) or "",
                        "action": str(o.get("action", "set_value"))[:40],
                        "value": o.get("value"),
                        "filter": o.get("filter"),
                    })
                if clean_opts:
                    return Clarify(items, columns, q, clean_opts)

        if action == "answer_question":
            text = str(args.get("text", ""))[:500]
            if text:
                return Ok(items, columns, text, tier="ai")

        if action == "compute_column":
            label = str(args.get("label", "Computed"))[:60]
            left = str(args.get("left", ""))
            right = str(args.get("right", ""))
            op = str(args.get("op", "multiply")).lower()
            extras = args.get("extra_constants") or []
            rest = f"{left} {op} {right}"
            for c in extras:
                try:
                    rest += f" times {float(c)}"
                except (TypeError, ValueError):
                    pass
            r = _compute_new_column(items, columns, label, rest)
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "aggregate":
            func = str(args.get("func", "average")).lower()
            col = str(args.get("column", ""))
            group = args.get("group_by")
            synth = f"{func} {col}" + (f" per {group}" if group else "")
            r = handle_aggregation(items, columns, synth)
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "set_price":
            filter_val = args.get("filter")
            price = args.get("price")
            if price is None:
                return None
            price = float(price)
            filter_str = "" if filter_val in (None, "", "all", "everything") \
                else str(filter_val)[:200]
            synth = f"set price for {filter_str or 'all'} to {price}"
            r = handle_set_price(items, columns, synth)
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "set_value":
            column = str(args.get("column", ""))[:200]
            value = args.get("value")
            filter_val = args.get("filter")
            if not column or value is None:
                return None
            if "price" in column.lower() or "cost" in column.lower():
                return None
            key = find_column_by_label(columns, column)
            if not key:
                return None
            hit_set = None
            if filter_val not in (None, "", "all"):
                hits = match_rows_by_target(items, columns, str(filter_val))
                if hits:
                    hit_set = set(hits)
            updated = []
            for idx, row in enumerate(items):
                r = dict(row)
                if hit_set is None or idx in hit_set:
                    r[key] = sanitize_value(value)
                updated.append(r)
            return Ok(updated, columns,
                      explanation or f"Set {column} on {len(updated)} row(s).",
                      tier="ai")

        if action == "calculate_totals":
            r = handle_calculate_totals(items, columns, "calculate totals")
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "multiply":
            target = str(args.get("target", "")).lower()
            factor = float(args.get("factor"))
            r = handle_multiply(items, columns, f"multiply {target} by {factor}")
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "add_fixed":
            target = str(args.get("target", "")).lower()
            delta = float(args.get("delta"))
            r = handle_add_fixed(items, columns, f"add {delta} to {target}")
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "discount":
            pct = float(args.get("percent"))
            filter_val = args.get("filter")
            synth = f"apply {pct}% discount"
            if filter_val:
                synth += f" on {filter_val}"
            r = handle_discount(items, columns, synth)
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "increase_pct":
            pct = float(args.get("percent"))
            r = handle_increase(items, columns, f"increase prices by {pct}%")
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "remove_column":
            r = handle_remove_column(items, columns,
                                     f"remove column {args.get('label','')}")
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "remove_rows":
            where = str(args.get("where", "")).lower()
            if "quantity_is_zero" in where:
                r = handle_remove_rows(items, columns,
                                       "remove rows with zero quantity")
            elif "price_missing" in where:
                r = handle_remove_rows(items, columns,
                                       "remove rows without price")
            else:
                r = handle_remove_rows(items, columns, "remove empty rows")
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "clear_column":
            r = handle_clear(items, columns,
                             f"clear {args.get('role','')}")
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "round_column":
            role = str(args.get("role", "")).lower()
            decimals = int(args.get("decimals", 2))
            r = handle_round(items, columns,
                             f"round {role} to {decimals} decimals")
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "sort":
            by = str(args.get("by", ""))
            desc = bool(args.get("descending", False))
            r = handle_sort(items, columns,
                            f"sort by {by} {'desc' if desc else 'asc'}")
            if r:
                r["applied_via"] = "ai"
                return r

        if action == "rename_column":
            old = str(args.get("from", ""))
            new = str(args.get("to", ""))
            r = handle_rename_column(items, columns, f"rename {old} to {new}")
            if r:
                r["applied_via"] = "ai"
                return r

    except Exception as e:
        logger.warning(f"AI action execution failed: {e}")
        return None
    return None


# ===========================================================================
#  PUBLIC ENTRY POINT
# ===========================================================================
def _sanitize_explanation(text):
    if not text:
        return "Done."
    if re.search(r"\b(module|import|engine|server|traceback|exception|"
                 r"not\s+loaded|failed|error)\b", text, re.I):
        return "Done."
    return text.strip()[:500]


def process_message(items, columns, message):
    safe_items = sanitize_items(items if isinstance(items, list) else [])
    safe_columns = sanitize_columns(columns if isinstance(columns, list) else [])
    safe_msg = sanitize_message(message)
    if not safe_msg:
        return NotUnderstood(safe_items, safe_columns)

    commands = split_compound(safe_msg)
    current_items = safe_items
    current_columns = safe_columns
    explanations = []
    applied_tiers = []
    clarifications = []

    for cmd in commands:
        result = run_deterministic(current_items, current_columns, cmd)
        tier = "deterministic"
        if not result:
            result = run_ai(current_items, current_columns, cmd)
            tier = "ai"
        if not result:
            explanations.append(f"Could not process: “{cmd}”.")
            applied_tiers.append("unrecognized")
            continue

        if result.get("status") == "clarify":
            clarifications.append(result)
            applied_tiers.append("clarify")
            continue

        current_items = result["items"]
        current_columns = result["columns"]
        explanations.append(_sanitize_explanation(result.get("explanation", "")))
        applied_tiers.append(tier)

    if clarifications:
        first = clarifications[0]
        return {
            "status": "clarify",
            "needs_clarification": True,
            "question": first.get("question", ""),
            "options": first.get("options", []),
            "items": safe_items,
            "columns": safe_columns,
            "pending_items": current_items,
            "pending_columns": current_columns,
            "explanations": explanations,
        }

    if not explanations or all(t == "unrecognized" for t in applied_tiers):
        return NotUnderstood(safe_items, safe_columns)

    combined = explanations[0] if len(explanations) == 1 \
        else "\n".join(f"{i+1}. {e}" for i, e in enumerate(explanations))

    if all(t == "deterministic" for t in applied_tiers):
        via = "deterministic"
    elif all(t == "ai" for t in applied_tiers):
        via = "ai"
    else:
        via = "mixed"

    return {
        "status": "ok",
        "success": True,
        "items": current_items,
        "columns": current_columns,
        "explanation": combined,
        "applied_via": via,
        "needs_clarification": False,
    }
