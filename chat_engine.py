"""
chat_engine.py — Self-sufficient conversational operations for the Smart Import page.

DESIGN PRINCIPLES
  1. Tier 1 (deterministic) handles every documented operation. Works offline,
     in <10ms, and is the primary path.
  2. Tier 2 (AI) is ONLY invoked when Tier 1 has no match. It returns a
     structured action that the engine validates and executes locally.
     The AI never sees the user's raw text unescaped as SQL, HTML, or any
     executable content. It only returns a JSON object from a closed enum.
  3. Compound commands ("do X and then Y") are split and executed sequentially.
  4. Wildcards ("all", "every", "everything") match every row.
  5. All user-controlled strings are sanitized before being placed into
     anything that leaves the process. No SQL, no shell, no eval.
  6. Every handler is bounded: item counts, string lengths, recursion.

SUPPORTED OPERATIONS
  Tier 1 (always):
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
    • set a column to a constant value
    • answer read-only questions (counts, sums, totals)

  Tier 2 (AI fallback): same operations, chosen by the LLM from the same
  closed set. Validated before execution.
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

# Hard limits to bound memory and time
MAX_ITEMS = 5000
MAX_COLUMNS = 200
MAX_MESSAGE_LEN = 2000
MAX_FIELD_LEN = 500
MAX_COMPOUND_COMMANDS = 10


# ===========================================================================
#  SAFETY / INPUT VALIDATION
# ===========================================================================
_INJECTION_PATTERNS = re.compile(
    r"(?:<\s*script|javascript:|data:text/html|"
    r"\b(?:DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|EXEC|EXECUTE|"
    r"UNION|SELECT)\s+(?:TABLE|FROM|INTO|WHERE|ALL)\b|"
    r"\.\./|\bchmod\b|\bwget\b|\bcurl\b|\brm\s+-rf\b|"
    r"\beval\s*\(|\bexec\s*\(|\bsystem\s*\(|"
    r"\$\(|\`|\|\s*(?:sh|bash|cmd|powershell)\b)",
    re.IGNORECASE)

_SAFE_KEY_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,100}$")
_SAFE_LABEL_RE = re.compile(r"^[^<>\"'\\]{0,200}$")


def sanitize_message(msg: Any) -> str:
    """Strip anything that looks like an injection payload or is excessively long."""
    if not isinstance(msg, str):
        return ""
    if len(msg) > MAX_MESSAGE_LEN:
        msg = msg[:MAX_MESSAGE_LEN]
    msg = msg.replace("\x00", "")
    if _INJECTION_PATTERNS.search(msg):
        # Refuse the whole message rather than trying to clean it
        return ""
    return msg


def safe_column_key(key: Any) -> Optional[str]:
    if not isinstance(key, str) or not _SAFE_KEY_RE.match(key):
        return None
    return key


def sanitize_label(label: Any) -> str:
    """Column labels are displayed to the user — strip control chars and tags."""
    if not isinstance(label, str):
        return ""
    label = label.strip()[:MAX_FIELD_LEN]
    label = re.sub(r"[<>]", "", label)  # no angle brackets
    label = label.replace("\x00", "")
    if not _SAFE_LABEL_RE.match(label):
        return ""
    return label


def sanitize_value(v: Any) -> Any:
    """Values that live in cells. Only allow primitives; strings capped."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        # NaN / Inf are not JSON-safe
        try:
            if v != v:  # NaN
                return None
            import math
            if math.isinf(v):
                return None
        except Exception:
            return None
        return v
    if isinstance(v, str):
        return v[:MAX_FIELD_LEN]
    # Anything else (dict, list, etc.) is dropped
    return None


def sanitize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in items[:MAX_ITEMS]:
        if not isinstance(row, dict):
            continue
        clean: Dict[str, Any] = {}
        for k, v in row.items():
            kk = safe_column_key(k) or re.sub(r"[^a-zA-Z0-9_\-]", "_", str(k))[:100]
            clean[kk] = sanitize_value(v)
        out.append(clean)
    return out


def sanitize_columns(columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(columns, list):
        return []
    out: List[Dict[str, Any]] = []
    seen_keys = set()
    for col in columns[:MAX_COLUMNS]:
        if not isinstance(col, dict):
            continue
        key = safe_column_key(col.get("key"))
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        out.append({
            "key": key,
            "label": sanitize_label(col.get("label", "")) or key,
            "role": col.get("role") if isinstance(col.get("role"), str) else None,
            "source": col.get("source") if col.get("source") in ("original", "added") else "original",
        })
    return out


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
    "all products", "all records", "everything in the list",
}


def is_wildcard(target: str) -> bool:
    if not target:
        return False
    t = re.sub(r"\s+", " ", str(target).lower().strip())
    if t in WILDCARD_PHRASES:
        return True
    if t.startswith(("all ", "every ", "each ")):
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
        "key": key, "label": sanitize_label(label), "role": role, "source": "added",
    }]
    return key, new_cols


def to_num(v) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
        import math
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        try:
            f = float(str(v).replace(",", "").replace("$", "").strip())
            import math
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except (TypeError, ValueError):
            return None


def normalize_keywords(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return [w for w in words if w not in STOPWORDS and len(w) >= 3]


def text_columns(columns: List[Dict[str, Any]]) -> List[str]:
    return [c["key"] for c in columns if c.get("role") not in NUMERIC_ROLES]


def match_rows_by_target(items: List[Dict[str, Any]],
                         columns: List[Dict[str, Any]],
                         target: str) -> List[int]:
    if is_wildcard(target):
        return list(range(len(items)))
    words = normalize_keywords(target)
    if not words:
        return []
    text_cols = text_columns(columns)
    hits: List[int] = []
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
#  COMPOUND COMMAND SPLITTER
# ===========================================================================
def split_compound(msg: str) -> List[str]:
    if not msg or len(msg) < 4:
        return [msg.strip()] if msg and msg.strip() else []

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

    return [f.strip() for f in final if f.strip()][:MAX_COMPOUND_COMMANDS]


# ===========================================================================
#  TIER 1 HANDLERS
# ===========================================================================
PRICE_RE = r"\$?\s*(\d+(?:[.,]\d+)?)"


# ------------------------------- SET PRICE -------------------------------
def handle_set_price(items, columns, msg):
    m_text = re.sub(r"\busd\b", " ", msg, flags=re.I)
    m_text = re.sub(r"\bkes\b", " ", m_text, flags=re.I)

    patterns = [
        # to all <target> add/set unit price [as/of/at/to] $X
        rf"\bto\s+all\s+(?P<t>[a-z0-9][a-z0-9 \-_']*?)\s+"
        rf"(?:add|set|assign|give|apply|put|make)\s+(?:a\s+|the\s+)?"
        rf"(?:unit\s*)?price\s+"
        rf"(?:as\s+|of\s+|at\s+|to\s+|=\s*)?{PRICE_RE}",

        # to every <target> add/set price [as/of/at/to] $X
        rf"\bto\s+every\s+(?P<t>[a-z0-9][a-z0-9 \-_']*?)\s+"
        rf"(?:add|set|assign|give|apply|put|make)\s+(?:a\s+|the\s+)?"
        rf"(?:unit\s*)?price\s+"
        rf"(?:as\s+|of\s+|at\s+|to\s+|=\s*)?{PRICE_RE}",

        # to <target> add price $X (no all)
        rf"\bto\s+(?:the\s+)?(?P<t>[a-z0-9][a-z0-9 \-_']*?)\s+"
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
        rf"(?:to|for|on)\s+(?:all\s+|every\s+|the\s+)?"
        rf"(?P<t>[a-z0-9][a-z0-9 \-_']*?)(?:\s*$|[,.;!?])",

        # price [of/as/at] $X [to/for] [all] <target>
        rf"\bprice\s+(?:of\s+|as\s+|at\s+)?{PRICE_RE}\s+"
        rf"(?:to|for|on)\s+(?:all\s+|every\s+|the\s+)?"
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

    # Wildcard detection: if user said "all X" and X doesn't match any row,
    # treat "all X" as "all rows".
    wildcard = is_wildcard(target)
    if not wildcard and re.search(rf"\ball\s+{re.escape(target)}\b", msg, re.I):
        probe = match_rows_by_target(items, columns, target)
        if not probe:
            wildcard = True

    if wildcard:
        hits = list(range(len(items)))
    else:
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

    label = "all rows" if wildcard else f"“{target}”"
    return (updated, columns,
            f"Set unit price to {price} on {len(hits)} row(s) matching {label}.")


# ---------------------------- CALCULATE TOTALS ----------------------------
def handle_calculate_totals(items, columns, msg):
    low = msg.lower()
    has_total = re.search(r"\b(total|totals|line\s*total|line\s*totals|amount|amounts)\b", low)
    has_verb = re.search(r"\b(calc(?:ulate)?|compute|recalc(?:ulate)?|update|refresh|fill|add|make|create)\b", low)
    if not has_total:
        return None
    if not has_verb and low.strip() not in ("totals", "line total", "total", "line totals"):
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
       and not re.search(r"\b(remove|delete|drop|filter|exclude)\b.*\b(items?|entries|lines?)\b", low):
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

    if re.search(r"\b(?:zero|0|no)\b.*\b(?:quantity|qty|stems?)\b", low) or \
       re.search(r"\b(?:quantity|qty|stems?)\b.*\b(?:is|equals?|=)\s*(?:zero|0)\b", low):
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
        r"([a-z0-9 _\-]+?)(?:\s*$|[,.;!?])", low)
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
        r"\s+by\s+" + PRICE_RE, msg, re.I)
    if not m:
        m = re.search(
            r"\b(?:times|x)\s*" + PRICE_RE +
            r"\s+(?:on|for|to)\s+(quantit(?:y|ies)|qty|stems?|prices?|unit\s*price)",
            msg, re.I)
        if m:
            what = m.group(2).lower()
            factor = float(m.group(1))
        else:
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
    else:
        what = m.group(1).lower()
        factor = float(m.group(2))

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
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:discount|off|less|reduction)", msg, re.I)
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
    return (updated, columns, f"Applied a {pct}% discount{suffix}.")


# ---------------------------- INCREASE BY PERCENT ----------------------------
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
    return (updated, columns, f"Increased prices by {pct}%.")


# ---------------------------- ADD / SUBTRACT FIXED ----------------------------
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
        r"(?:\s+(?:order|direction))?(?:\s*$|[,.;!?])", msg, re.I)
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
        r"\s+to\s+(\d+)\s*decimals?", msg, re.I)
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
        r"(?:\s*$|[,.;!?])", msg, re.I)
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
    return (items, new_cols, f"Renamed “{old_needle}” to “{new_label}”.")


# --------------------------- READ-ONLY QUESTIONS ---------------------------
def handle_question(items, columns, msg):
    low = msg.lower().strip()

    qty_key = find_role_column(columns, "quantity")
    price_key = find_role_column(columns, "unit_price")
    total_key = find_role_column(columns, "total")

    if re.search(r"\b(how\s+many\s+rows?|row\s+count|count\s+(?:of\s+)?rows?|"
                 r"number\s+of\s+rows?|how\s+many\s+(?:items?|entries|lines?))\b", low):
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
        return (items, columns, f"I found {len(hits)} row(s) matching “{target}”.")

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
            return (items, columns, f"Sum of “{m.group(1)}” is {round(total, 2)}.")
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
#  TIER 2 — AI FALLBACK (hardened: closed action set, sanitized args)
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
   Examples: "add unit price 0.5 to all carnations",
             "to all flowers add unit price as $0.65"

2. "calculate_totals"
   args: {}

3. "multiply"
   args: {"target": "quantity"|"unit_price"|"total", "factor": <number>}

4. "add_fixed"
   args: {"target": "unit_price"|"total", "delta": <number>}

5. "discount"
   args: {"percent": <number>, "filter": "<text>"|null}

6. "increase_pct"
   args: {"percent": <number>}

7. "remove_column"
   args: {"label": "<column label to remove>"}

8. "remove_rows"
   args: {"where": "quantity_is_zero"|"price_missing"|"empty_row"}

9. "clear_column"
   args: {"role": "unit_price"|"quantity"|"total"}

10. "round_column"
    args: {"role": "unit_price"|"quantity"|"total", "decimals": <int>}

11. "sort"
    args: {"by": "<column label>", "descending": true|false}

12. "rename_column"
    args: {"from": "<current label>", "to": "<new label>"}

13. "answer_question"
    args: {"text": "<the answer to the user's question in one sentence>"}

14. "none"
    args: {}

Rules:
- "all" or "every" as a filter means every row: set filter to null.
- Never invent prices, quantities, or column names.
- Numbers must be numbers, not strings.
"""


_VALID_ACTIONS = {
    "set_price", "calculate_totals", "multiply", "add_fixed",
    "discount", "increase_pct", "remove_column", "remove_rows",
    "clear_column", "round_column", "sort", "rename_column",
    "answer_question", "none",
}


def _ai_clean_json(text: str) -> Optional[dict]:
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


def _ai_call(prompt: str) -> Optional[str]:
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


def run_ai(items, columns, message):
    if not AI_ENABLED:
        return None

    payload = json.dumps({
        "columns": columns,
        "items": items[:200],
        "message": message[:1000],
    }, ensure_ascii=False)[:30000]

    raw = _ai_call(AI_SYSTEM + "\n\nINPUT:\n" + payload)
    logger.info(f"AI raw response: {(raw[:400] if raw else None)}")

    data = _ai_clean_json(raw) if raw else None
    if not data or not isinstance(data, dict):
        return None
    action = data.get("action")
    if action not in _VALID_ACTIONS:
        logger.warning(f"AI returned invalid action: {action}")
        return None
    logger.info(f"AI parsed action: {action}")

    args = data.get("args") if isinstance(data.get("args"), dict) else {}
    explanation = str(data.get("explanation", ""))[:300]

    try:
        # ---------- set_price ----------
        if action == "set_price":
            filter_val = args.get("filter")
            price = args.get("price")
            if price is None:
                return None
            try:
                price = float(price)
            except (TypeError, ValueError):
                return None
            filter_str = ("" if filter_val in (None, "", "all", "everything", "each", "any")
                          else str(filter_val).strip()[:200])

            price_key, columns = ensure_column(columns, "unit_price", "Unit Price")

            if not filter_str or is_wildcard(filter_str):
                hit_set = set(range(len(items)))
            else:
                hits = match_rows_by_target(items, columns, filter_str)
                if not hits:
                    words = [w for w in re.split(r"\s+", filter_str.lower()) if len(w) >= 4]
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
            label = ("all rows" if len(hit_set) == len(items) else f"“{filter_str}”")
            return (updated, columns,
                    explanation or f"Set unit price to {price} on {len(hit_set)} row(s) matching {label}.")

        if action == "calculate_totals":
            return _apply_calc_totals(items, columns, explanation)

        if action == "multiply":
            target = str(args.get("target", "")).lower()
            try:
                factor = float(args.get("factor"))
            except (TypeError, ValueError):
                return None
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
            return (updated, columns, explanation or f"Multiplied {target} by {factor}.")

        if action == "add_fixed":
            target = str(args.get("target", "")).lower()
            try:
                delta = float(args.get("delta"))
            except (TypeError, ValueError):
                return None
            key = _resolve_column_for(columns, target) or find_role_column(columns, "unit_price")
            if not key:
                return None
            updated = []
            for row in items:
                r = dict(row)
                v = to_num(r.get(key))
                if v is not None:
                    r[key] = round(v + delta, 4)
                updated.append(r)
            return (updated, columns, explanation or f"Adjusted values by {delta}.")

        if action == "discount":
            try:
                pct = float(args.get("percent"))
            except (TypeError, ValueError):
                return None
            factor = 1.0 - (pct / 100.0)
            filter_val = args.get("filter")
            key = find_role_column(columns, "unit_price") or find_role_column(columns, "total")
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
            return (updated, columns, explanation or f"Applied {pct}% discount.")

        if action == "increase_pct":
            try:
                pct = float(args.get("percent"))
            except (TypeError, ValueError):
                return None
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
            label = str(args.get("label", ""))[:200]
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
            try:
                decimals = int(args.get("decimals", 2))
            except (TypeError, ValueError):
                decimals = 2
            decimals = max(0, min(decimals, 8))
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
            by = str(args.get("by", ""))[:200]
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
            old = str(args.get("from", ""))[:200]
            new = sanitize_label(str(args.get("to", "")))
            key = find_column_by_label(columns, old)
            if not key or not new:
                return None
            new_cols = []
            for c in columns:
                new_cols.append({**c, "label": new} if c["key"] == key else c)
            return (items, new_cols, explanation or f"Renamed “{old}” to “{new}”.")

        if action == "answer_question":
            text = str(args.get("text", ""))[:500]
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
    if not text:
        return "Done."
    if re.search(r"\b(module|import|engine|server|traceback|exception|"
                 r"not\s+loaded|failed|error)\b", text, re.I):
        return "Done."
    return text.strip()[:500]


def process_message(items: List[Dict[str, Any]],
                    columns: List[Dict[str, Any]],
                    message: str) -> Dict[str, Any]:
    """
    Entry point. Sanitizes all inputs. Runs Tier 1 then Tier 2 per command.
    """
    # -- Sanitize inputs (defense in depth: caller should also sanitize) --
    safe_items = sanitize_items(items if isinstance(items, list) else [])
    safe_columns = sanitize_columns(columns if isinstance(columns, list) else [])
    safe_msg = sanitize_message(message)

    if not safe_msg:
        return {
            "success": True, "items": safe_items, "columns": safe_columns,
            "explanation": "I didn't quite understand that instruction.",
            "applied_via": "unrecognized",
        }

    commands = split_compound(safe_msg)

    current_items = safe_items
    current_columns = safe_columns
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
            "items": safe_items,
            "columns": safe_columns,
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
