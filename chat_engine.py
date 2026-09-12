"""
chat_engine.py — Conversational operations for the Smart Import review page.

Architecture:
  Tier 1 (deterministic): regex + rules. Fast, offline, no rate limits.
                          Handles the most common concrete commands.
  Tier 2 (AI fallback):   Gemini -> Groq. Invoked when Tier 1 doesn't
                          match. Understands arbitrary natural language.
  Tier 3 (friendly):      If everything fails, a helpful rephrase prompt.

Compound commands are supported: "do X and then Y" is split and executed
in sequence. Each operation is applied to the result of the previous one.

Supported operations (all reachable via natural language):
  • set unit price on matching rows (or all rows)
  • calculate line totals
  • multiply a numeric column by a factor
  • add/subtract a fixed amount
  • apply a percentage discount (optionally on a filter)
  • increase prices by a percentage
  • remove a column
  • remove rows (zero quantity, missing price, empty)
  • clear a column
  • round a column
  • sort by a column
  • rename a column
  • answer read-only questions

Never invents values. Every transformation is deterministic once the
operation is chosen; the AI only decides WHAT to do, never what value to
put in.
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("chat-engine")

AI_ENABLED = os.getenv("AI_ENABLED", "0") == "1"
AI_PROVIDER = os.getenv("AI_PROVIDER", "auto").lower()
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "10"))


# ===========================================================================
#  GENERIC HELPERS
# ===========================================================================
NUMERIC_ROLES = {"quantity", "boxes", "pack_rate", "length_cm",
                 "head_size_cm", "unit_price", "total", "n"}

STOPWORDS = {"a", "an", "the", "of", "to", "for", "all", "and", "or",
             "is", "are", "on", "in", "at", "by", "with", "as", "please",
             "add", "set", "apply", "give", "assign", "each", "then",
             "every", "everything", "row", "rows"}

WILDCARD_PHRASES = {
    "all", "every", "everything", "each", "any", "all rows", "every row",
    "all flowers", "all items", "all entries", "all lines",
    "the whole list", "the entire list", "every flower", "every item",
    "everything here", "the whole thing", "the entire table",
    "all products", "all records",
}


def is_wildcard(target: str) -> bool:
    if not target:
        return False
    t = re.sub(r"\s+", " ", str(target).lower().strip())
    if t in WILDCARD_PHRASES:
        return True
    if t.startswith("all "):
        return True
    if t.startswith("every "):
        return True
    if t.startswith("each "):
        return True
    return False


def find_role_column(columns: List[Dict[str, Any]], role: str) -> Optional[str]:
    for c in columns:
        if c.get("role") == role:
            return c["key"]
    return None


def find_column_by_label(columns: List[Dict[str, Any]], needle: str) -> Optional[str]:
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
        if n in (c.get("label") or "").lower():
            return c["key"]
    n_words = [w for w in re.split(r"\s+", n) if w]
    best, best_score = None, 0
    for c in columns:
        label = (c.get("label") or "").lower()
        score = sum(1 for w in n_words if w in label)
        if score > best_score:
            best_score, best = score, c["key"]
    return best if best_score > 0 else None


def ensure_column(columns: List[Dict[str, Any]], role: str,
                  label: str) -> Tuple[str, List[Dict[str, Any]]]:
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
        "key": key, "label": label, "role": role, "source": "added",
    }]
    return key, new_cols


def row_matches(row: Dict[str, Any], column_key: str, needle: str) -> bool:
    v = row.get(column_key)
    if v is None:
        return False
    return str(needle).lower() in str(v).lower()


def text_columns(columns: List[Dict[str, Any]]) -> List[str]:
    return [c["key"] for c in columns if c.get("role") not in NUMERIC_ROLES]


def to_num(v) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        try:
            return float(str(v).replace(",", "").replace("$", "").strip())
        except (TypeError, ValueError):
            return None


def normalize_keywords(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 3]


def match_rows_by_target(items: List[Dict[str, Any]],
                         columns: List[Dict[str, Any]],
                         target: str) -> List[int]:
    """
    Return indices of rows that match the target.

    Wildcard ("all", "everything", ...) matches every row.
    Otherwise: every keyword in `target` must appear in some text column
    of the row (matched across all text columns together). If the target
    is a single word, at least one hit is enough.
    """
    if is_wildcard(target):
        return list(range(len(items)))

    words = normalize_keywords(target)
    if not words:
        return []

    text_cols = text_columns(columns)
    hits = []

    for idx, row in enumerate(items):
        # Build the row's searchable text once
        row_text = " ".join(str(row.get(ck) or "").lower() for ck in text_cols)
        if len(words) == 1:
            if words[0] in row_text:
                hits.append(idx)
        else:
            if all(w in row_text for w in words):
                hits.append(idx)
    return hits


# ===========================================================================
#  COMPOUND COMMAND SPLITTER
# ===========================================================================
def split_compound(msg: str) -> List[str]:
    """
    Split a message into individual commands. Preserves order.
    Splits on ;, " then ", " & ", and on " and " when both sides look
    like real commands.
    """
    if not msg or len(msg) < 4:
        return [msg.strip()] if msg and msg.strip() else []

    # Strong separators
    parts = re.split(r"\s*(?:;\s*|\s+then\s+|\s+&\s+)\s*", msg, flags=re.I)
    parts = [p.strip() for p in parts if p.strip()]

    command_verbs = (r"\b(?:add|set|apply|remove|delete|drop|multiply|"
                     r"calculate|compute|recalc|update|clear|round|sort|"
                     r"increase|raise|discount|rename|make|give|assign|"
                     r"show|how|count|sum|total|empty|fill|divide|halve|"
                     r"double|triple)\b")

    final: List[str] = []
    for p in parts:
        pieces = re.split(r"\s+and\s+", p, flags=re.I)
        if len(pieces) <= 1:
            final.append(p)
            continue
        merged = [pieces[0]]
        for piece in pieces[1:]:
            if re.search(command_verbs, piece, re.I):
                merged.append(piece)
            else:
                merged[-1] = merged[-1] + " and " + piece
        final.extend(merged)

    return [f.strip() for f in final if f.strip()]


# ===========================================================================
#  TIER 1 — DETERMINISTIC HANDLERS
# ===========================================================================
PRICE_RE = r"\$?\s*(\d+(?:[.,]\d+)?)"


# ------------------------------- SET PRICE -------------------------------
def handle_set_price(items, columns, msg):
    m_text = re.sub(r"\busd\b", " ", msg, flags=re.I)
    m_text = re.sub(r"\bkes\b", " ", m_text, flags=re.I)

    patterns = [
        # to all <target> add/set unit price [as/of/at/to] $X
        rf"\bto\s+(?:all\s+|the\s+)?(?P<t>[a-z0-9][a-z0-9 \-_']*?)\s+"
        rf"(?:add|set|assign|give|apply|put|make)\s+(?:a\s+|the\s+)?"
        rf"(?:unit\s*)?price\s+"
        rf"(?:as\s+|of\s+|at\s+|to\s+|=\s*)?{PRICE_RE}",

        # <target>: [add/set] [unit] price [as/of/at/to] $X
        rf"(?P<t>[a-z0-9][a-z0-9 \-_']*?)\s*[:=]\s*"
        rf"(?:add|set|assign|give|apply|put|make)?\s*(?:a\s+|the\s+)?"
        rf"(?:unit\s*)?price\s+"
        rf"(?:as\s+|of\s+|at\s+|to\s+)?{PRICE_RE}",

        # add/set/apply [a] [unit] price [of/as/at/to] $X to/for [all] <target>
        rf"\b(?:add|set|assign|give|apply|put|make)\s+(?:a\s+|the\s+)?"
        rf"(?:unit\s*)?price\s+"
        rf"(?:of\s+|as\s+|at\s+|to\s+)?{PRICE_RE}\s+"
        rf"(?:to|for|on)\s+(?:all\s+|the\s+)?"
        rf"(?P<t>[a-z0-9][a-z0-9 \-_']*?)(?:\s*$|[,.;!?])",

        # price [of/as/at] $X [to/for] [all] <target>
        rf"\bprice\s+(?:of\s+|as\s+|at\s+)?{PRICE_RE}\s+"
        rf"(?:to|for|on)\s+(?:all\s+|the\s+)?"
        rf"(?P<t>[a-z0-9][a-z0-9 \-_']*?)(?:\s*$|[,.;!?])",

        # <target> [at|with|unit] price [of/as/at] $X
        rf"(?P<t>[a-z0-9][a-z0-9 \-_']*?)\s+"
        rf"(?:at\s+|with\s+|unit\s+|the\s+)?price\s+"
        rf"(?:as\s+|of\s+|at\s+|=\s*)?{PRICE_RE}(?:\s|$)",

        # set <target>'s price to $X
        rf"\bset\s+(?P<t>[a-z0-9][a-z0-9 \-_']*?)'?s?\s+(?:unit\s*)?price\s+"
        rf"(?:as\s+|of\s+|at\s+|to\s+|=\s*)?{PRICE_RE}",

        # make <target> cost $X
        rf"\bmake\s+(?:all\s+|the\s+)?(?P<t>[a-z0-9][a-z0-9 \-_']*?)\s+"
        rf"(?:cost|price[ds]?)\s+(?:at\s+)?{PRICE_RE}",

        # all <target> @ $X
        rf"\ball\s+(?P<t>[a-z0-9][a-z0-9 \-_']*?)\s+@\s*{PRICE_RE}",

        # bare: all <target> X.YZ
        rf"\ball\s+(?P<t>[a-z][a-z0-9 \-_']*?)\s+{PRICE_RE}\s*$",
    ]

    price = None
    target = None
    for pat in patterns:
        m = re.search(pat, m_text, re.I)
        if not m:
            continue
        gd = m.groupdict()
        if gd.get("t"):
            target = gd["t"].strip()
        for g in m.groups():
            if g is None or g == target:
                continue
            try:
                val = float(str(g).replace(",", "").strip())
                price = val
                break
            except (ValueError, TypeError):
                continue
        if price is not None and target:
            break

    if price is None or not target:
        return None
    if target.lower() in STOPWORDS and not is_wildcard(target):
        return None

    price_key, columns = ensure_column(columns, "unit_price", "Unit Price")

    hits = match_rows_by_target(items, columns, target)
    if not hits:
        return None

    hit_set = set(hits)
    updated = []
    for idx, row in enumerate(items):
        r = dict(row)
        if idx in hit_set:
            r[price_key] = price
        updated.append(r)

    label = "all rows" if is_wildcard(target) else f"“{target}”"
    return (updated, columns,
            f"Set unit price to {price} on {len(hits)} row(s) matching {label}.")


# ---------------------------- CALCULATE TOTALS ----------------------------
def handle_calculate_totals(items, columns, msg):
    low = msg.lower()
    has_total = re.search(r"\b(total|totals|line\s*total|line\s*totals|amount|amounts)\b",
                          low)
    has_verb = re.search(r"\b(calc(?:ulate)?|compute|recalc(?:ulate)?|"
                         r"update|refresh|fill|add|make|create)\b", low)
    if not has_total:
        return None
    if not has_verb:
        if low.strip() not in ("totals", "line total", "total", "line totals"):
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
    return (updated, columns, f"Calculated line total for {len(updated)} row(s).")


# ------------------------------- REMOVE COLUMN -------------------------------
def handle_remove_column(items, columns, msg):
    m = re.search(
        r"\b(?:remove|delete|drop|hide|get\s+rid\s+of|take\s+out)\s+"
        r"(?:the\s+|a\s+|an\s+)?(?:column\s+|field\s+)?"
        r"([a-z0-9][a-z0-9 _\-]+?)"
        r"(?:\s+column|\s+field)?(?:\s*$|[,.;!?])",
        msg, re.I)
    if not m:
        return None
    needle = m.group(1).strip()
    if needle.lower() in STOPWORDS:
        return None
    key = find_column_by_label(columns, needle)
    if not key:
        return None
    new_cols = [c for c in columns if c["key"] != key]
    updated = [{k: v for k, v in row.items() if k != key} for row in items]
    label = next((c["label"] for c in columns if c["key"] == key), needle)
    return (updated, new_cols, f"Removed column “{label}”.")


# ------------------------------- REMOVE ROWS -------------------------------
def handle_remove_rows(items, columns, msg):
    low = msg.lower()
    if not re.search(r"\b(remove|delete|drop|filter\s+out|exclude|"
                     r"get\s+rid\s+of|hide|skip)\b.*\brows?\b", low) \
       and not re.search(r"\b(remove|delete|drop|filter|exclude)\b.*\b(items?|entries|lines?)\b",
                         low):
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
        return (kept, columns, f"Removed {removed} empty row(s).")

    m = re.search(r"\b(?:zero|0|no)\b.*\b(?:quantity|qty|stems?)\b", low)
    m2 = re.search(r"\b(?:quantity|qty|stems?)\b.*\b(?:is|equals?|=)\s*(?:zero|0)\b", low)
    if m or m2:
        if not qty_key:
            return None
        kept, removed = [], 0
        for row in items:
            q = to_num(row.get(qty_key))
            if q is None or q == 0:
                removed += 1
                continue
            kept.append(row)
        return (kept, columns, f"Removed {removed} row(s) with zero quantity.")

    if re.search(r"\b(no|missing|empty|without)\s+price\b", low):
        if not price_key:
            return None
        kept, removed = [], 0
        for row in items:
            if to_num(row.get(price_key)) is None:
                removed += 1
                continue
            kept.append(row)
        return (kept, columns, f"Removed {removed} row(s) without a price.")

    m = re.search(
        r"\b(?:where|with|that\s+have|having|containing|matching)\s+"
        r"([a-z0-9 _\-]+?)\s*(?:=|is|equals?|contains?|includes?)?\s*"
        r"([a-z0-9 _\-]+?)(?:\s*$|[,.;!?])",
        low)
    if m:
        needle = f"{m.group(1)} {m.group(2)}"
        hits = set(match_rows_by_target(items, columns, needle))
        if hits:
            kept = [r for i, r in enumerate(items) if i not in hits]
            return (kept, columns,
                    f"Removed {len(items) - len(kept)} row(s) matching “{needle}”.")
    return None


# ------------------------------- MULTIPLY -------------------------------
def handle_multiply(items, columns, msg):
    m = re.search(
        r"\bmultiply\s+(?:all\s+|the\s+)?"
        r"(quantit(?:y|ies)|qty|stems?|units?|prices?|unit\s*price|amounts?|totals?|boxes?|pack\s*rate)"
        r"\s+by\s+" + PRICE_RE,
        msg, re.I)
    if not m:
        m = re.search(
            r"\b(?:times|x)\s*" + PRICE_RE +
            r"\s+(?:on|for|to)\s+(quantit(?:y|ies)|qty|stems?|prices?|unit\s*price)",
            msg, re.I)
        if m:
            what = m.group(2).lower()
            factor = float(m.group(1))
        else:
            # bare "double prices", "halve quantities", "triple amounts"
            m2 = re.search(r"\b(double|triple|halve|half)\s+"
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
            return (updated, columns, f"{word.capitalize()}d {what}.")

    what = m.group(1).lower() if m.lastindex >= 1 else ""
    factor = float(m.group(2)) if m.lastindex >= 2 else 1.0

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
    return (updated, columns, f"Multiplied {what} by {factor}.")


def _resolve_column_for(columns: List[Dict[str, Any]], what: str) -> Optional[str]:
    w = what.lower()
    if w.startswith(("quantit", "qty", "stem", "unit")):
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


# ------------------------------- DISCOUNT -------------------------------
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
        else:
            target_rows = None  # wildcard = every row

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
    return (updated, columns, f"Applied a {pct}% discount{suffix}.")


# ---------------------------- INCREASE BY PERCENT ----------------------------
def handle_increase(items, columns, msg):
    m = re.search(
        r"\b(?:increase|raise|bump|add)\s+(?:all\s+|the\s+)?"
        r"(?:prices?|unit\s*price|amounts?|totals?)\s+by\s+"
        r"(\d+(?:\.\d+)?)\s*%",
        msg, re.I)
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
    return (updated, columns, f"Increased prices by {pct}%.")


# ---------------------------- ADD/SUBTRACT FIXED ----------------------------
def handle_add_fixed(items, columns, msg):
    m = re.search(
        r"\b(?:add|subtract)\s+\$?(\d+(?:\.\d+)?)\s+"
        r"(?:to|from)\s+(?:all\s+|the\s+)?(?:prices?|unit\s*price|amounts?|totals?)",
        msg, re.I)
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
    return (updated, columns, f"{verb} {abs(delta)} on prices.")


# ------------------------------- SORT -------------------------------
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
    return (updated, columns,
            f"Sorted by “{label}” ({'descending' if desc else 'ascending'}).")


# ------------------------------- CLEAR -------------------------------
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
    return (updated, columns, f"Cleared all {what}.")


# ------------------------------- ROUND -------------------------------
def handle_round(items, columns, msg):
    m = re.search(
        r"\bround\s+(?:all\s+|the\s+)?"
        r"(prices?|unit\s*price|quantit(?:y|ies)|qty|totals?|amounts?)"
        r"\s+to\s+(\d+)\s*decimals?",
        msg, re.I)
    if not m:
        return None
    target, decimals = m.group(1).lower(), int(m.group(2))
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
    return (updated, columns, f"Rounded {target} to {decimals} decimal(s).")


# --------------------------- SET COLUMN VALUE ---------------------------
def handle_set_column_value(items, columns, msg):
    m = re.search(
        r"\bset\s+(?:the\s+)?([a-z0-9 _\-]+?)\s+"
        r"(?:to|as|=)\s+([a-z0-9.$_\-]+)"
        r"(?:\s+on\s+(?:all\s+|the\s+)?([a-z0-9 _\-']+?))?"
        r"(?:\s*$|[,.;!?])",
        msg, re.I)
    if not m:
        return None
    col_needle = m.group(1).strip()
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
    return (updated, columns,
            f"Set {label} to {value}{suffix} on {applied} row(s).")


# --------------------------- RENAME COLUMN ---------------------------
def handle_rename_column(items, columns, msg):
    m = re.search(
        r"\brename\s+(?:the\s+)?([a-z0-9 _\-]+?)\s+(?:column\s+)?"
        r"(?:to|as)\s+([a-z0-9 _\-]+)",
        msg, re.I)
    if not m:
        return None
    old_needle, new_label = m.group(1).strip(), m.group(2).strip()
    key = find_column_by_label(columns, old_needle)
    if not key:
        return None
    new_cols = []
    for c in columns:
        if c["key"] == key:
            new_cols.append({**c, "label": new_label})
        else:
            new_cols.append(c)
    return (items, new_cols, f"Renamed “{old_needle}” to “{new_label}”.")


# --------------------------- READ-ONLY QUESTIONS ---------------------------
def handle_question(items, columns, msg):
    low = msg.lower().strip()

    qty_key = find_role_column(columns, "quantity")
    price_key = find_role_column(columns, "unit_price")
    total_key = find_role_column(columns, "total")

    if re.search(r"\b(how\s+many\s+rows?|row\s+count|count\s+(?:of\s+)?rows?|"
                 r"number\s+of\s+rows?|how\s+many\s+(?:items?|entries|lines?))\b",
                 low):
        return (items, columns, f"There are {len(items)} rows.")

    if (re.search(r"\b(total|sum|overall|how\s+many)\b", low)
        and re.search(r"\b(quantity|qty|stems?|units?|pieces?)\b", low)
        and not re.search(r"\b(add|set|calc|remove|delete|discount|multiply|"
                          r"calculate|compute)\b", low)):
        if qty_key:
            total = sum(to_num(r.get(qty_key)) or 0 for r in items)
            return (items, columns,
                    f"Total quantity is {int(total) if float(total).is_integer() else round(total, 2)}.")
        return None

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
        return (items, columns, f"Total amount is {round(total, 2)}.")

    m = re.search(r"\bhow\s+many\s+([a-z0-9 _\-']+?)(?:\s*\?|\s*$|,|\.|!)", low)
    if m:
        target = m.group(1).strip()
        hits = match_rows_by_target(items, columns, target)
        return (items, columns,
                f"I found {len(hits)} row(s) matching “{target}”.")

    m = re.search(r"\bcount\s+(?:of\s+)?([a-z0-9 _\-']+?)(?:\s*$|[,.;!?])", low)
    if m:
        target = m.group(1).strip()
        hits = match_rows_by_target(items, columns, target)
        return (items, columns, f"{len(hits)} row(s) match “{target}”.")

    m = re.search(r"\b(?:sum|total)\s+of\s+([a-z0-9 _\-]+)", low)
    if m:
        key = find_column_by_label(columns, m.group(1))
        if key:
            total = sum(to_num(r.get(key)) or 0 for r in items)
            return (items, columns,
                    f"Sum of “{m.group(1)}” is {round(total, 2)}.")
    return None


# ------------------------------- DISPATCHER -------------------------------
DETERMINISTIC_HANDLERS = [
    handle_set_price,
    handle_calculate_totals,
    handle_remove_column,
    handle_remove_rows,
    handle_multiply,
    handle_discount,
    handle_increase,
    handle_add_fixed,
    handle_sort,
    handle_clear,
    handle_round,
    handle_set_column_value,
    handle_rename_column,
    handle_question,
]


def run_deterministic(items, columns, message):
    for handler in DETERMINISTIC_HANDLERS:
        try:
            result = handler(items, columns, message)
        except Exception as e:
            logger.warning(f"{handler.__name__} raised: {e}")
            continue
        if result:
            return result
    return None


# ===========================================================================
#  TIER 2 — AI FALLBACK
# ===========================================================================
AI_SYSTEM = """You are the AI assistant for a document transformation tool.

The user has a table of line items and types a plain-English instruction.
Your job: pick the ONE operation that best matches their intent and return it
as structured JSON.

You receive:
  columns: list of {key, label, role, source}
  items:   list of rows (keyed by column.key)
  message: the user's instruction in natural language

Return ONLY valid JSON. No prose, no markdown, no code fences.

Schema:
{
  "action": "<action name>",
  "args": { ... action-specific args ... },
  "explanation": "<one short sentence describing what you did>"
}

Available actions:

1. "set_price"
   args: {"filter": "<text to match in any row>" or null for all rows, "price": <number>}
   Examples: "add unit price 0.5 to all carnations", "set price 1.20 for hydrangeas",
             "make everything $0.65", "assign 2.50 to roses",
             "to all flowers add unit price as $0.65"

2. "calculate_totals"
   args: {}
   Examples: "calculate totals", "recalculate line totals", "compute amounts"

3. "multiply"
   args: {"target": "quantity"|"unit_price"|"total", "factor": <number>}
   Examples: "multiply quantity by 1.1", "double prices", "halve quantity"

4. "add_fixed"
   args: {"target": "unit_price"|"total", "delta": <number>}
   Examples: "add 5 to all prices", "subtract 0.25 from prices"

5. "discount"
   args: {"percent": <number>, "filter": "<text>"|null}
   Examples: "apply 10% discount", "give 5% off on roses"

6. "increase_pct"
   args: {"percent": <number>}
   Examples: "increase prices by 5%", "raise prices by 10%"

7. "remove_column"
   args: {"label": "<column label to remove>"}
   Examples: "remove the color column", "delete the length column"

8. "remove_rows"
   args: {"where": "quantity_is_zero"|"price_missing"|"empty_row"}
   Examples: "remove rows with 0 quantity", "delete rows without prices"

9. "clear_column"
   args: {"role": "unit_price"|"quantity"|"total"}
   Examples: "clear all prices", "empty the quantity column"

10. "round_column"
    args: {"role": "unit_price"|"quantity"|"total", "decimals": <int>}
    Examples: "round prices to 2 decimals"

11. "sort"
    args: {"by": "<column label>", "descending": true|false}
    Examples: "sort by flower", "order by quantity descending"

12. "rename_column"
    args: {"from": "<current label>", "to": "<new label>"}
    Examples: "rename variety to Flower name"

13. "answer_question"
    args: {"text": "<the answer to the user's question in one sentence>"}
    Examples: "how many roses do we have?", "what is the total quantity?",
              "how many rows are there?", "what is the total amount?"
    IMPORTANT: for this action, COMPUTE the answer yourself from the items
    array and return it as a plain sentence in args.text.

14. "none"
    args: {}
    Use only if the instruction cannot be mapped to any of the above.

Rules:
- "all" or "every" or "all flowers" as a filter means: apply to EVERY row.
  Set filter to null for that.
- If the user mentions a category like "carnations" or "roses", set filter
  to that category word.
- Never invent prices, quantities, or column names. Only use what's given.
- If the instruction contains multiple separate actions joined by "and" or
  "then", pick the FIRST action only. The system handles compound splitting.
- Numbers must be numbers, not strings.

Example output for "to all flowers add unit price as $0.65":
{
  "action": "set_price",
  "args": {"filter": null, "price": 0.65},
  "explanation": "Set unit price to 0.65 on all rows."
}
"""


def _ai_clean_json(text: str) -> Optional[dict]:
    if not text:
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


def _ai_call(prompt: str) -> Optional[str]:
    logger.info(
        f"AI call: provider={AI_PROVIDER} "
        f"gemini_key={bool(os.getenv('GEMINI_API_KEY', '').strip())} "
        f"groq_key={bool(os.getenv('GROQ_API_KEY', '').strip())}"
    )

    if AI_PROVIDER in ("gemini", "auto"):
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
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
                    return resp.text
        except Exception as e:
            logger.warning(f"Gemini chat call failed: {e}")

    if AI_PROVIDER in ("groq", "auto"):
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if api_key:
                client = Groq(api_key=api_key)
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
                    return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq chat call failed: {e}")
    return None


def run_ai(items, columns, message):
    if not AI_ENABLED:
        return None

    payload = json.dumps({
        "columns": columns,
        "items": items[:200],
        "message": message,
    }, ensure_ascii=False)[:30000]

    raw = _ai_call(AI_SYSTEM + "\n\nINPUT:\n" + payload)
    logger.info(f"AI raw response: {(raw[:400] if raw else None)}")

    data = _ai_clean_json(raw) if raw else None
    logger.info(f"AI parsed action: {data.get('action') if data else None}")
    if not data:
        return None

    action = data.get("action")
    args = data.get("args") or {}
    explanation = data.get("explanation", "") or ""

    try:
        # ---------- set_price ----------
        if action == "set_price":
            filter_val = args.get("filter")
            price = args.get("price")
            if price is None:
                return None
            price = float(price)
            filter_str = ("" if filter_val in (None, "", "all", "everything",
                                                "each", "any")
                          else str(filter_val).strip())

            price_key, columns = ensure_column(columns, "unit_price", "Unit Price")

            if not filter_str or is_wildcard(filter_str):
                hit_set = set(range(len(items)))
            else:
                hits = match_rows_by_target(items, columns, filter_str)
                if not hits:
                    # Fuzzy fallback: match any word ≥ 4 chars in any text column
                    words = [w for w in re.split(r"\s+", filter_str.lower())
                             if len(w) >= 4]
                    for idx, row in enumerate(items):
                        for ck in text_columns(columns):
                            v = str(row.get(ck) or "").lower()
                            if any(w in v for w in words):
                                hits.append(idx)
                                break
                if not hits:
                    return None
                hit_set = set(hits)

            updated = []
            for idx, row in enumerate(items):
                r = dict(row)
                if idx in hit_set:
                    r[price_key] = price
                updated.append(r)
            label = ("all rows" if len(hit_set) == len(items)
                     else f"“{filter_str}”")
            return (updated, columns,
                    explanation or f"Set unit price to {price} on "
                                   f"{len(hit_set)} row(s) matching {label}.")

        # ---------- calculate_totals ----------
        if action == "calculate_totals":
            return _apply_calc_totals(items, columns, explanation)

        # ---------- multiply ----------
        if action == "multiply":
            target = str(args.get("target", "")).lower()
            factor = float(args.get("factor"))
            key = _resolve_column_for(columns, target)
            if not key:
                return None
            updated = []
            for row in items:
                r = dict(row)
                v = to_num(r.get(key))
                if v is not None:
                    r[key] = round(v * factor, 4)
                updated.append(r)
            return (updated, columns,
                    explanation or f"Multiplied {target} by {factor}.")

        # ---------- add_fixed ----------
        if action == "add_fixed":
            target = str(args.get("target", "")).lower()
            delta = float(args.get("delta"))
            key = _resolve_column_for(columns, target) or \
                  find_role_column(columns, "unit_price")
            if not key:
                return None
            updated = []
            for row in items:
                r = dict(row)
                v = to_num(r.get(key))
                if v is not None:
                    r[key] = round(v + delta, 4)
                updated.append(r)
            return (updated, columns,
                    explanation or f"Adjusted values by {delta}.")

        # ---------- discount ----------
        if action == "discount":
            pct = float(args.get("percent"))
            factor = 1.0 - (pct / 100.0)
            filter_val = args.get("filter")
            key = (find_role_column(columns, "unit_price")
                   or find_role_column(columns, "total"))
            if not key:
                return None
            hit_set = None
            if filter_val not in (None, "", "all", "everything"):
                hits = match_rows_by_target(items, columns, str(filter_val))
                if hits:
                    hit_set = set(hits)
            updated = []
            applied = 0
            for idx, row in enumerate(items):
                r = dict(row)
                if hit_set is None or idx in hit_set:
                    v = to_num(r.get(key))
                    if v is not None:
                        r[key] = round(v * factor, 4)
                        applied += 1
                updated.append(r)
            if applied == 0:
                return None
            return (updated, columns,
                    explanation or f"Applied {pct}% discount.")

        # ---------- increase_pct ----------
        if action == "increase_pct":
            pct = float(args.get("percent"))
            factor = 1.0 + (pct / 100.0)
            key = (find_role_column(columns, "unit_price")
                   or find_role_column(columns, "total"))
            if not key:
                return None
            updated = []
            for row in items:
                r = dict(row)
                v = to_num(r.get(key))
                if v is not None:
                    r[key] = round(v * factor, 4)
                updated.append(r)
            return (updated, columns,
                    explanation or f"Increased prices by {pct}%.")

        # ---------- remove_column ----------
        if action == "remove_column":
            label = str(args.get("label", ""))
            key = find_column_by_label(columns, label)
            if not key:
                return None
            new_cols = [c for c in columns if c["key"] != key]
            updated = [{k: v for k, v in row.items() if k != key}
                       for row in items]
            return (updated, new_cols, explanation or "Removed column.")

        # ---------- remove_rows ----------
        if action == "remove_rows":
            where = str(args.get("where", "")).lower()
            qty_key = find_role_column(columns, "quantity")
            price_key = find_role_column(columns, "unit_price")
            updated = []
            removed = 0
            for row in items:
                drop = False
                if "quantity_is_zero" in where and qty_key:
                    v = to_num(row.get(qty_key))
                    drop = (v is None or v == 0)
                elif "price_missing" in where and price_key:
                    drop = to_num(row.get(price_key)) is None
                elif "empty" in where:
                    drop = all(val in (None, "") for val in row.values())
                if drop:
                    removed += 1
                else:
                    updated.append(row)
            return (updated, columns,
                    explanation or f"Removed {removed} row(s).")

        # ---------- clear_column ----------
        if action == "clear_column":
            role = str(args.get("role", "")).lower()
            key = find_role_column(columns, role) if role else None
            if not key:
                return None
            updated = [{**row, key: None} for row in items]
            return (updated, columns, explanation or f"Cleared {role}.")

        # ---------- round_column ----------
        if action == "round_column":
            role = str(args.get("role", "")).lower()
            decimals = int(args.get("decimals", 2))
            key = find_role_column(columns, role) if role else None
            if not key:
                return None
            updated = []
            for row in items:
                r = dict(row)
                v = to_num(r.get(key))
                if v is not None:
                    r[key] = round(v, decimals)
                updated.append(r)
            return (updated, columns, explanation or f"Rounded {role}.")

        # ---------- sort ----------
        if action == "sort":
            by = str(args.get("by", ""))
            desc = bool(args.get("descending", False))
            key = find_column_by_label(columns, by)
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
            return (updated, columns, explanation or f"Sorted by “{by}”.")

        # ---------- rename_column ----------
        if action == "rename_column":
            old = str(args.get("from", ""))
            new = str(args.get("to", ""))
            key = find_column_by_label(columns, old)
            if not key or not new:
                return None
            new_cols = []
            for c in columns:
                new_cols.append({**c, "label": new} if c["key"] == key else c)
            return (items, new_cols,
                    explanation or f"Renamed “{old}” to “{new}”.")

        # ---------- answer_question ----------
        if action == "answer_question":
            text = str(args.get("text", "")).strip()
            if text:
                return (items, columns, text)

    except Exception as e:
        logger.warning(f"AI action execution failed: {e}")
        return None

    return None


def _apply_calc_totals(items, columns, explanation):
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
    return (updated, columns, explanation or "Calculated line totals.")


# ===========================================================================
#  PUBLIC ENTRY POINT
# ===========================================================================
FRIENDLY_FALLBACK = (
    "I didn't quite understand that instruction. Try things like "
    "“add unit price 0.5 to all carnations”, “calculate totals”, "
    "“remove column color”, “multiply quantity by 1.1”, or "
    "“how many roses do we have?”"
)


def _sanitize_explanation(text: str) -> str:
    """Remove any technical-sounding text before returning to the user."""
    if not text:
        return "Done."
    if re.search(
            r"\b(module|import|engine|server|traceback|exception|"
            r"not\s+loaded|failed|error)\b", text, re.I):
        return "Done."
    return text.strip()


def process_message(items: List[Dict[str, Any]],
                    columns: List[Dict[str, Any]],
                    message: str) -> Dict[str, Any]:
    """
    Split compound messages, run each part through Tier 1 then Tier 2,
    and return the final state plus a combined explanation.
    """
    msg = (message or "").strip()
    if not msg:
        return {
            "success": True, "items": items, "columns": columns,
            "explanation": "Say something first.",
            "applied_via": "none",
        }

    commands = split_compound(msg)

    current_items = items
    current_columns = columns
    explanations: List[str] = []
    applied_tiers: List[str] = []

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

        new_items, new_cols, explanation = result
        current_items = new_items
        current_columns = new_cols
        explanations.append(_sanitize_explanation(explanation))
        applied_tiers.append(tier)

    if not explanations or all(t == "unrecognized" for t in applied_tiers):
        return {
            "success": True,
            "items": items,
            "columns": columns,
            "explanation": FRIENDLY_FALLBACK,
            "applied_via": "unrecognized",
        }

    if len(explanations) == 1:
        combined = explanations[0]
    else:
        combined = "\n".join(f"{i+1}. {e}" for i, e in enumerate(explanations))

    if all(t == "deterministic" for t in applied_tiers):
        via = "deterministic"
    elif all(t == "ai" for t in applied_tiers):
        via = "ai"
    else:
        via = "mixed"

    return {
        "success": True,
        "items": current_items,
        "columns": current_columns,
        "explanation": combined,
        "applied_via": via,
    }
