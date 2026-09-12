"""
chat_engine.py — Deterministic-first conversational operations for the review page.

Design:
  Tier 1 (always, offline, <10ms): regex + rules.
  Tier 2 (fallback, optional, capped at 8s): Gemini -> Groq.
  If both fail, we return a friendly prompt, never a technical error.

The engine NEVER invents values. Every operation either transforms an
existing value with a deterministic formula or leaves it alone.

Supported Tier 1 operations (many phrasings each):
  • set unit price on matching rows
  • calculate line totals
  • add a computed column (line total, quantity × price)
  • remove a column
  • remove rows (by filter, by zero quantity, by missing price, empty rows)
  • multiply a numeric column by a factor
  • apply a percentage discount
  • increase prices by a percentage
  • add a fixed amount to prices
  • sort by a column
  • clear all prices
  • clear a column's values
  • round a numeric column to N decimals
  • set a column to a constant value for matching rows
  • rename a column
  • count rows
  • total / sum of a column
  • group totals (sum by <column>)
  • read-only questions ("how many X?", "total amount?")
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("chat-engine")

AI_ENABLED = os.getenv("AI_ENABLED", "0") == "1"
AI_PROVIDER = os.getenv("AI_PROVIDER", "auto").lower()
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "8"))


# ===========================================================================
#  GENERIC HELPERS
# ===========================================================================
NUMERIC_ROLES = {"quantity", "boxes", "pack_rate", "length_cm",
                 "head_size_cm", "unit_price", "total", "n"}

STOPWORDS = {"a", "an", "the", "of", "to", "for", "all", "and", "or",
             "is", "are", "on", "in", "at", "by", "with", "as", "please",
             "add", "set", "apply", "give", "assign", "each"}


def find_role_column(columns: List[Dict[str, Any]], role: str) -> Optional[str]:
    for c in columns:
        if c.get("role") == role:
            return c["key"]
    return None


def find_column_by_label(columns: List[Dict[str, Any]], needle: str) -> Optional[str]:
    """Match a column by label (fuzzy) or key."""
    if not needle:
        return None
    n = needle.lower().strip()
    # exact key
    for c in columns:
        if n == c.get("key", "").lower():
            return c["key"]
    # exact label
    for c in columns:
        if n == (c.get("label") or "").lower():
            return c["key"]
    # substring
    for c in columns:
        if n in (c.get("label") or "").lower():
            return c["key"]
    # token overlap
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
    """Ensure a column with the given role exists. Return (key, new_columns)."""
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
    return needle.lower() in str(v).lower()


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


def parse_number_token(s: str) -> Optional[float]:
    m = re.search(r"\d+(?:\.\d+)?", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def normalize_keywords(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 3]


def match_rows_by_target(items: List[Dict[str, Any]],
                         columns: List[Dict[str, Any]],
                         target: str) -> List[int]:
    """
    Return the indices of rows whose any-text-column contains every keyword
    of `target` (or at least one if only one keyword).
    """
    words = normalize_keywords(target)
    if not words:
        return []
    text_cols = text_columns(columns)
    hits = []
    for idx, row in enumerate(items):
        matched_words = set()
        for ck in text_cols:
            v = row.get(ck)
            if v is None:
                continue
            vl = str(v).lower()
            for w in words:
                if w in vl:
                    matched_words.add(w)
        # require all words if multiple, else at least one
        if len(words) == 1 and matched_words:
            hits.append(idx)
        elif len(matched_words) == len(words):
            hits.append(idx)
    return hits


# ===========================================================================
#  TIER 1 HANDLERS
#  Each returns (items, columns, explanation) or None.
# ===========================================================================

# ============================ SET UNIT PRICE ============================
PRICE_RE = r"\$?\s*(\d+(?:[.,]\d+)?)"


def handle_set_price(items, columns, msg):
    """
    Broadly recognizes price-assignment instructions. Handles word order
    permutations, colons, "as/of/at/to", "$", "USD", etc.
    """
    # Normalize common punctuation but keep the $ for price detection
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
        if "t" in gd and gd["t"]:
            target = gd["t"].strip()
        # find price: last group that looks numeric
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
    if target.lower().strip() in STOPWORDS:
        return None

    # Ensure the unit_price column exists
    price_key, columns = ensure_column(columns, "unit_price", "Unit Price")

    hits = match_rows_by_target(items, columns, target)
    if not hits:
        return None

    updated = []
    hit_set = set(hits)
    for idx, row in enumerate(items):
        r = dict(row)
        if idx in hit_set:
            r[price_key] = price
        updated.append(r)

    return (updated, columns,
            f"Set unit price to {price} on {len(hits)} row(s) matching “{target}”.")


# ============================ CALCULATE TOTALS ============================
def handle_calculate_totals(items, columns, msg):
    """
    calculate totals | compute line totals | recalculate | update totals
    """
    low = msg.lower()
    if not (re.search(r"\b(calc(?:ulate)?|compute|recalc(?:ulate)?|update|refresh|fill)\b",
                      low)
            and re.search(r"\b(total|totals|line\s*total|amount|subtotal)\b", low)):
        return None
    # Also match "add a line total column"
    if not re.search(r"\b(total|totals|line\s*total|amount)\b", low):
        return None

    qty_key = find_role_column(columns, "quantity")
    price_key = find_role_column(columns, "unit_price")
    if not qty_key:
        # maybe boxes × pack_rate?
        box_key = find_role_column(columns, "boxes")
        pack_key = find_role_column(columns, "pack_rate")
        if box_key and pack_key:
            qty_key = box_key  # fallback
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
    return (updated, columns,
            f"Calculated line total for {len(updated)} row(s).")


# ============================ REMOVE COLUMN ============================
def handle_remove_column(items, columns, msg):
    m = re.search(
        r"\b(?:remove|delete|drop|hide|get\s+rid\s+of|take\s+out)\s+"
        r"(?:the\s+|a\s+|an\s+)?(?:column\s+|field\s+|the\s+)?"
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


# ============================ REMOVE ROWS ============================
def handle_remove_rows(items, columns, msg):
    low = msg.lower()
    if not re.search(r"\b(remove|delete|drop|filter\s+out|exclude|"
                     r"get\s+rid\s+of|hide|skip)\b.*\brows?\b", low) \
       and not re.search(r"\b(remove|delete|drop|filter|exclude)\b.*\b(items?|entries|lines?)\b",
                         low):
        return None

    qty_key = find_role_column(columns, "quantity")
    price_key = find_role_column(columns, "unit_price")

    # empty rows
    if re.search(r"\bempty\b|\bblank\b|\bno\s+content\b", low):
        kept, removed = [], 0
        for row in items:
            if all(v in (None, "") for v in row.values()):
                removed += 1
                continue
            kept.append(row)
        return (kept, columns, f"Removed {removed} empty row(s).")

    # rows with zero quantity
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
        return (kept, columns,
                f"Removed {removed} row(s) with zero quantity.")

    # rows without price
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

    # rows where <column> = <value>
    m = re.search(
        r"\b(?:where|with|that\s+have|having|containing|matching)\s+"
        r"([a-z0-9 _\-]+?)\s*(?:=|is|equals?|contains?|includes?)?\s*"
        r"([a-z0-9 _\-]+?)(?:\s*$|[,.;!?])",
        low)
    if m:
        # try to find which text column matches
        needle = f"{m.group(1)} {m.group(2)}"
        hits = set(match_rows_by_target(items, columns, needle))
        if hits:
            kept = [r for i, r in enumerate(items) if i not in hits]
            return (kept, columns,
                    f"Removed {len(items) - len(kept)} row(s) matching “{needle}”.")

    return None


# ============================ MULTIPLY COLUMN ============================
def handle_multiply(items, columns, msg):
    m = re.search(
        r"\bmultiply\s+(?:all\s+|the\s+)?"
        r"(quantit(?:y|ies)|qty|stems?|units?|prices?|unit\s*price|amounts?|totals?|boxes?|pack\s*rate)"
        r"\s+by\s+" + PRICE_RE,
        msg, re.I)
    if not m:
        # also match "x3 on quantity" style
        m = re.search(
            r"\b(?:times|x)\s*" + PRICE_RE +
            r"\s+(?:on|for|to)\s+(quantit(?:y|ies)|qty|stems?|prices?|unit\s*price)",
            msg, re.I)
        if m:
            what = m.group(2).lower()
            factor = float(m.group(1))
        else:
            return None
    else:
        what = m.group(1).lower()
        factor = float(m.group(2))

    if what.startswith(("quantit", "qty", "stem", "unit")):
        key = find_role_column(columns, "quantity")
    elif what.startswith("price") or "price" in what:
        key = find_role_column(columns, "unit_price")
    elif what.startswith("amount") or what.startswith("total"):
        key = find_role_column(columns, "total")
    elif what.startswith("box"):
        key = find_role_column(columns, "boxes")
    elif "pack" in what:
        key = find_role_column(columns, "pack_rate")
    else:
        key = find_column_by_label(columns, what)
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


# ============================ DISCOUNT ============================
def handle_discount(items, columns, msg):
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:discount|off|less|reduction)",
        msg, re.I)
    if not m:
        m = re.search(r"\bdiscount\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", msg, re.I)
    if not m:
        return None
    pct = float(m.group(1))
    factor = 1.0 - (pct / 100.0)

    # Allow optional filter: "10% discount on all roses"
    m_filter = re.search(r"\b(?:on|for|to)\s+(?:all\s+|the\s+)?"
                         r"([a-z0-9 _\-']+?)(?:\s*$|[,.;!?])", msg, re.I)
    target_rows = None
    if m_filter:
        target = m_filter.group(1).strip()
        hits = match_rows_by_target(items, columns, target)
        if hits:
            target_rows = set(hits)

    key = find_role_column(columns, "unit_price")
    if not key:
        key = find_role_column(columns, "total")
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
        suffix = f" on “{m_filter.group(1).strip()}”"
    return (updated, columns, f"Applied a {pct}% discount{suffix}.")


# ============================ INCREASE BY PERCENT ============================
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

    key = find_role_column(columns, "unit_price")
    if not key:
        key = find_role_column(columns, "total")
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


# ============================ ADD / SUBTRACT FIXED ============================
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


# ============================ SORT ============================
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


# ============================ CLEAR ============================
def handle_clear(items, columns, msg):
    m = re.search(r"\bclear\s+(?:all\s+|the\s+)?"
                  r"(prices?|unit\s*price|quantit(?:y|ies)|qty|stems?|totals?|amounts?)",
                  msg, re.I)
    if not m:
        return None
    what = m.group(1).lower()
    if what.startswith("price"):
        key = find_role_column(columns, "unit_price")
    elif what.startswith(("quantit", "qty", "stem")):
        key = find_role_column(columns, "quantity")
    elif what.startswith("total") or what.startswith("amount"):
        key = find_role_column(columns, "total")
    else:
        key = None
    if not key:
        return None
    updated = [{**row, key: None} for row in items]
    return (updated, columns, f"Cleared all {what}.")


# ============================ ROUND ============================
def handle_round(items, columns, msg):
    m = re.search(
        r"\bround\s+(?:all\s+|the\s+)?"
        r"(prices?|unit\s*price|quantit(?:y|ies)|qty|totals?|amounts?)"
        r"\s+to\s+(\d+)\s*decimals?",
        msg, re.I)
    if not m:
        return None
    target, decimals = m.group(1).lower(), int(m.group(2))
    if target.startswith("price"):
        key = find_role_column(columns, "unit_price")
    elif target.startswith("total") or target.startswith("amount"):
        key = find_role_column(columns, "total")
    else:
        key = find_role_column(columns, "quantity")
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


# ============================ SET COLUMN VALUE ============================
def handle_set_column_value(items, columns, msg):
    """
    set <column> to <value>
    set the <column> of all <target> to <value>
    """
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

    # Skip if column needle is "unit price" — handled by set_price
    if "price" in col_needle.lower() and col_needle.lower() != "unit price":
        pass
    key = find_column_by_label(columns, col_needle)
    if not key:
        return None

    # Parse value: number if numeric, else string
    v_num = to_num(value_raw)
    value = v_num if v_num is not None else value_raw

    updated = []
    applied = 0
    hit_set = None
    if filter_target:
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
    suffix = f" on “{filter_target}”" if filter_target else ""
    return (updated, columns,
            f"Set {label} to {value}{suffix} on {applied} row(s).")


# ============================ RENAME COLUMN ============================
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


# ============================ READ-ONLY QUESTIONS ============================
def handle_question(items, columns, msg):
    low = msg.lower().strip()

    qty_key = find_role_column(columns, "quantity")
    price_key = find_role_column(columns, "unit_price")
    total_key = find_role_column(columns, "total")

    # count rows
    if re.search(r"\b(how\s+many\s+rows?|row\s+count|count\s+(?:of\s+)?rows?|"
                 r"number\s+of\s+rows?|how\s+many\s+(?:items?|entries|lines?))\b", low):
        return (items, columns, f"There are {len(items)} rows.")

    # total quantity
    if (re.search(r"\b(total|sum|overall|how\s+many)\b", low)
        and re.search(r"\b(quantity|qty|stems?|units?|pieces?)\b", low)
        and not re.search(r"\b(add|set|calc|remove|delete|discount|multiply)\b", low)):
        if qty_key:
            total = sum(to_num(r.get(qty_key)) or 0 for r in items)
            return (items, columns,
                    f"Total quantity is {int(total) if float(total).is_integer() else round(total, 2)}.")
        return None

    # total amount
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

    # how many <target>
    m = re.search(r"\bhow\s+many\s+([a-z0-9 _\-']+?)(?:\s*\?|\s*$|,|\.|!)", low)
    if m:
        target = m.group(1).strip()
        hits = match_rows_by_target(items, columns, target)
        return (items, columns,
                f"I found {len(hits)} row(s) matching “{target}”.")

    # count <target>
    m = re.search(r"\bcount\s+(?:of\s+)?([a-z0-9 _\-']+?)(?:\s*$|[,.;!?])", low)
    if m:
        target = m.group(1).strip()
        hits = match_rows_by_target(items, columns, target)
        return (items, columns, f"{len(hits)} row(s) match “{target}”.")

    # sum of <column>
    m = re.search(r"\b(?:sum|total)\s+of\s+([a-z0-9 _\-]+)", low)
    if m:
        key = find_column_by_label(columns, m.group(1))
        if key:
            total = sum(to_num(r.get(key)) or 0 for r in items)
            return (items, columns,
                    f"Sum of “{m.group(1)}” is {round(total, 2)}.")

    return None


# ============================ DISPATCHER ============================
DETERMINISTIC_HANDLERS = [
    # Order matters — most specific first
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
AI_SYSTEM = """You transform invoice tables based on a short user instruction.

You receive:
  columns: array of {key, label, role, source}
  items:   array of rows keyed by column.key
  message: the user's instruction

Return ONLY valid JSON with this exact shape:
{
  "action": "<one of: set_price | calculate_totals | multiply | add_fixed |
              discount | increase_pct | remove_column | remove_rows |
              clear_column | round_column | sort | rename_column | none>",
  "args": { ... },
  "explanation": "<one short sentence>"
}

Action arguments:
  set_price:        {"filter": "<substring>", "price": <number>}
  calculate_totals: {}
  multiply:         {"target": "quantity"|"unit_price"|"total", "factor": <number>}
  add_fixed:        {"target": "unit_price"|"total", "delta": <number>}
  discount:         {"percent": <number>, "filter": "<substring>"|null}
  increase_pct:     {"percent": <number>}
  remove_column:    {"label": "<column label or key>"}
  remove_rows:      {"where": "quantity_is_zero"|"price_missing"|"empty_row"}
  clear_column:     {"role": "unit_price"|"quantity"|"total"}
  round_column:     {"role": "unit_price"|"quantity"|"total", "decimals": <int>}
  sort:             {"by": "<column label>", "descending": true|false}
  rename_column:    {"from": "<label>", "to": "<new label>"}
  none:             {}

Do not invent prices or quantities. When unsure, use "none".
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
    """Ask whichever provider is up. Never raises."""
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
    data = _ai_clean_json(raw) if raw else None
    if not data:
        return None

    action = data.get("action")
    args = data.get("args") or {}
    explanation = data.get("explanation", "") or ""

    try:
        if action == "set_price":
            filter_ = str(args.get("filter", "")).strip()
            price = float(args.get("price"))
            if not filter_:
                return None
            return _apply_set_price(items, columns, filter_, price, explanation)

        if action == "calculate_totals":
            return _apply_calc_totals(items, columns, explanation)

        if action == "multiply":
            target = str(args.get("target", "")).lower()
            factor = float(args.get("factor"))
            role = ("quantity" if "quant" in target
                    else "unit_price" if "price" in target
                    else "total")
            key = find_role_column(columns, role)
            if not key:
                return None
            updated = []
            for row in items:
                r = dict(row)
                v = to_num(r.get(key))
                if v is not None:
                    r[key] = round(v * factor, 4)
                updated.append(r)
            return (updated, columns, explanation or f"Multiplied {target} by {factor}.")

        if action == "add_fixed":
            target = str(args.get("target", "")).lower()
            delta = float(args.get("delta"))
            role = "unit_price" if "price" in target else "total"
            key = find_role_column(columns, role)
            if not key:
                return None
            updated = []
            for row in items:
                r = dict(row)
                v = to_num(r.get(key))
                if v is not None:
                    r[key] = round(v + delta, 4)
                updated.append(r)
            return (updated, columns, explanation or f"Adjusted prices by {delta}.")

        if action == "discount":
            pct = float(args.get("percent"))
            factor = 1.0 - (pct / 100.0)
            filter_ = args.get("filter")
            key = find_role_column(columns, "unit_price") or find_role_column(columns, "total")
            if not key:
                return None
            hit_set = None
            if filter_:
                hits = match_rows_by_target(items, columns, str(filter_))
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
            return (updated, columns, explanation or f"Applied {pct}% discount.")

        if action == "increase_pct":
            pct = float(args.get("percent"))
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
            return (updated, columns, explanation or f"Increased prices by {pct}%.")

        if action == "remove_column":
            label = str(args.get("label", ""))
            key = find_column_by_label(columns, label)
            if not key:
                return None
            new_cols = [c for c in columns if c["key"] != key]
            updated = [{k: v for k, v in row.items() if k != key} for row in items]
            return (updated, new_cols, explanation or "Removed column.")

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
            return (updated, columns, explanation or f"Removed {removed} row(s).")

        if action == "clear_column":
            role = str(args.get("role", "")).lower()
            key = find_role_column(columns, role) if role else None
            if not key:
                return None
            updated = [{**row, key: None} for row in items]
            return (updated, columns, explanation or f"Cleared {role}.")

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

        if action == "rename_column":
            old = str(args.get("from", ""))
            new = str(args.get("to", ""))
            key = find_column_by_label(columns, old)
            if not key or not new:
                return None
            new_cols = []
            for c in columns:
                new_cols.append({**c, "label": new} if c["key"] == key else c)
            return (items, new_cols, explanation or f"Renamed “{old}” to “{new}”.")

    except Exception as e:
        logger.warning(f"AI action execution failed: {e}")
        return None

    return None


def _apply_set_price(items, columns, filter_, price, explanation):
    price_key, columns = ensure_column(columns, "unit_price", "Unit Price")
    hits = match_rows_by_target(items, columns, filter_)
    if not hits:
        return None
    hit_set = set(hits)
    updated = []
    for idx, row in enumerate(items):
        r = dict(row)
        if idx in hit_set:
            r[price_key] = price
        updated.append(r)
    return (updated, columns,
            explanation or f"Set unit price to {price} on {len(hits)} row(s).")


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


def process_message(items: List[Dict[str, Any]],
                    columns: List[Dict[str, Any]],
                    message: str) -> Dict[str, Any]:
    """Run Tier 1, then Tier 2. Always returns a well-formed response."""
    msg = (message or "").strip()
    if not msg:
        return {
            "success": True, "items": items, "columns": columns,
            "explanation": "Say something first.",
            "applied_via": "none",
        }

    # Tier 1 — deterministic
    result = run_deterministic(items, columns, msg)
    if result:
        new_items, new_cols, explanation = result
        # Never surface technical text to users
        if not explanation or re.search(
                r"\b(module|import|engine|server|traceback|exception)\b",
                explanation, re.I):
            explanation = "Done."
        return {
            "success": True,
            "items": new_items,
            "columns": new_cols,
            "explanation": explanation,
            "applied_via": "deterministic",
        }

    # Tier 2 — AI helper
    result = run_ai(items, columns, msg)
    if result:
        new_items, new_cols, explanation = result
        if not explanation:
            explanation = "Done."
        return {
            "success": True,
            "items": new_items,
            "columns": new_cols,
            "explanation": explanation,
            "applied_via": "ai",
        }

    # Neither worked — friendly prompt, never technical
    return {
        "success": True,
        "items": items,
        "columns": columns,
        "explanation": FRIENDLY_FALLBACK,
        "applied_via": "unrecognized",
    }
