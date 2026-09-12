"""
chat_engine.py — Deterministic-first chat operations for the review page.

Design:
  Tier 1 (always): regex + rules. Fast, offline, no rate limits.
  Tier 2 (fallback): Gemini or Groq. Only when Tier 1 finds nothing.
                     Capped at 8 seconds. If both fail, return an
                     "unrecognized" response asking the user to rephrase.

The engine NEVER invents data. Every operation either transforms an
existing value with a deterministic formula or leaves it alone.
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
#  ROLE / COLUMN HELPERS
# ===========================================================================
def find_role_column(columns: List[Dict[str, Any]], role: str) -> Optional[str]:
    for c in columns:
        if c.get("role") == role:
            return c["key"]
    return None


def find_column_by_label(columns: List[Dict[str, Any]], needle: str) -> Optional[str]:
    n = needle.lower().strip()
    for c in columns:
        if n in (c.get("label") or "").lower() or n == c.get("key", "").lower():
            return c["key"]
    return None


def ensure_column(columns: List[Dict[str, Any]], role: str,
                  label: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Ensure a column with the given role exists; return (key, columns)."""
    key = find_role_column(columns, role)
    if key:
        return key, columns
    taken = {c["key"] for c in columns}
    base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or role
    key = base
    i = 2
    while key in taken:
        key = f"{base}_{i}"
        i += 1
    columns = list(columns) + [{
        "key": key, "label": label, "role": role, "source": "added",
    }]
    return key, columns


def row_matches(row: Dict[str, Any], column_key: str, needle: str) -> bool:
    v = row.get(column_key)
    if v is None:
        return False
    return needle.lower() in str(v).lower()


def text_columns(columns: List[Dict[str, Any]]) -> List[str]:
    """Columns likely to hold text (not numeric roles)."""
    numeric_roles = {"quantity", "boxes", "pack_rate", "length_cm",
                     "head_size_cm", "unit_price", "total", "n"}
    out = []
    for c in columns:
        if c.get("role") in numeric_roles:
            continue
        out.append(c["key"])
    return out


def to_num(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ===========================================================================
#  TIER 1 — DETERMINISTIC PATTERNS
# ===========================================================================
# Each handler returns (updated_items, updated_columns, explanation)
# or None if the pattern did not match.

PRICE_RE = r"(\d+(?:\.\d+)?)"


def handle_set_price(items, columns, msg):
    """
    Examples:
      add unit price 0.5 to all carnations
      set price 0.4 for hydrangea
      price 1.2 for roses
      hydrangea price 0.45
    """
    patterns = [
        # add/set unit price X to/for <filter>
        rf"(?:add|set|apply)\s+(?:a\s+)?(?:unit\s*)?price\s+(?:of\s+)?{PRICE_RE}\s+"
        rf"(?:to|for)\s+(?:all\s+)?([a-z0-9 \-_]+)",
        rf"price\s+{PRICE_RE}\s+(?:to|for)\s+(?:all\s+)?([a-z0-9 \-_]+)",
        rf"([a-z0-9 \-_]+)\s+(?:at|with)?\s*(?:unit\s*)?price\s+{PRICE_RE}",
    ]
    m = None
    price = None
    target = None
    for pat in patterns:
        m = re.search(pat, msg, re.I)
        if m:
            price = float(m.group(1))
            target = m.group(2).strip()
            # second pattern has price as group 1 and target as group 2
            # third pattern has target first then price
            if "price" in pat and pat.startswith(r"([a-z"):
                price = float(m.group(2))
                target = m.group(1).strip()
            break
    if not m:
        return None

    # Find the column to match on: prefer "variety", then "flower", then "product_name"
    match_col = (
        find_role_column(columns, "product_name") or
        find_role_column(columns, "variety") or
        (text_columns(columns)[0] if text_columns(columns) else None)
    )
    if not match_col:
        return None

    # Find or create unit_price column
    price_key, columns = ensure_column(columns, "unit_price", "Unit Price")

    target_words = [w for w in target.split() if len(w) >= 4]
    if not target_words:
        return None

    updated = []
    matched = 0
    for row in items:
        # try to match on the chosen column
        hit = any(row_matches(row, match_col, w) for w in target_words)
        # if that misses, try any text column
        if not hit:
            for ck in text_columns(columns):
                if any(row_matches(row, ck, w) for w in target_words):
                    hit = True
                    break
        if hit:
            r = dict(row)
            r[price_key] = price
            updated.append(r)
            matched += 1
        else:
            updated.append(row)

    if matched == 0:
        return None

    return (updated, columns,
            f"Set unit price to {price} on {matched} row(s) matching “{target}”.")


def handle_calculate_totals(items, columns, msg):
    """
    calculate totals / calculate line total / compute totals
    """
    if not re.search(r"\b(calc(?:ulate)?|compute|recalc(?:ulate)?)\b.*\btotal", msg, re.I) and \
       not re.search(r"\btotal\b.*\b(calc|compute|line)", msg, re.I):
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

    return (updated, columns,
            f"Calculated line total for {len(updated)} rows as quantity × unit price.")


def handle_remove_column(items, columns, msg):
    m = re.search(r"(?:remove|delete|drop)\s+(?:the\s+)?column\s+([a-z0-9 _\-]+)",
                  msg, re.I)
    if not m:
        return None
    needle = m.group(1).strip()
    key = find_column_by_label(columns, needle)
    if not key:
        # try fuzzy: any column whose label contains every word
        for c in columns:
            if all(w in (c.get("label") or "").lower() for w in needle.split()):
                key = c["key"]
                break
    if not key:
        return None

    new_columns = [c for c in columns if c["key"] != key]
    updated = [{k: v for k, v in row.items() if k != key} for row in items]
    return (updated, new_columns, f"Removed column “{needle}”.")


def handle_remove_rows(items, columns, msg):
    """
    remove rows with 0 quantity
    remove empty rows
    remove rows where quantity is 0
    """
    if not re.search(r"\b(remove|delete|drop|filter)\b.*\brows?", msg, re.I):
        return None

    qty_key = find_role_column(columns, "quantity")
    price_key = find_role_column(columns, "unit_price")

    if re.search(r"\bempty\b", msg, re.I):
        # Remove rows where all values are empty
        updated = []
        removed = 0
        for row in items:
            if all(v in (None, "") for v in row.values()):
                removed += 1
                continue
            updated.append(row)
        return (updated, columns, f"Removed {removed} empty row(s).")

    if qty_key and re.search(r"\b(0|zero)\b", msg, re.I):
        updated = []
        removed = 0
        for row in items:
            q = to_num(row.get(qty_key))
            if q is None or q == 0:
                removed += 1
                continue
            updated.append(row)
        return (updated, columns, f"Removed {removed} row(s) with zero quantity.")

    if price_key and re.search(r"\b(no|missing|empty)\s+price", msg, re.I):
        updated = []
        removed = 0
        for row in items:
            if to_num(row.get(price_key)) is None:
                removed += 1
                continue
            updated.append(row)
        return (updated, columns, f"Removed {removed} row(s) without price.")

    return None


def handle_multiply(items, columns, msg):
    """
    multiply quantity by 1.1
    multiply all prices by 0.9
    """
    m = re.search(r"multiply\s+(?:all\s+)?(quantit(?:y|ies)|prices?|units?)\s+by\s+"
                  rf"{PRICE_RE}", msg, re.I)
    if not m:
        return None
    what = m.group(1).lower()
    factor = float(m.group(2))

    if what.startswith("quantit") or what.startswith("unit"):
        key = find_role_column(columns, "quantity")
    else:
        key = find_role_column(columns, "unit_price")
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
            f"Multiplied {what} by {factor}.")


def handle_discount(items, columns, msg):
    """
    apply 10% discount / give 10% off / discount 15%
    """
    m = re.search(r"(?:apply|give|add)?\s*(\d+(?:\.\d+)?)\s*%\s*"
                  r"(?:discount|off)", msg, re.I)
    if not m:
        m = re.search(r"discount\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%", msg, re.I)
    if not m:
        return None

    pct = float(m.group(1))
    factor = 1.0 - (pct / 100.0)

    key = find_role_column(columns, "unit_price")
    if not key:
        return None

    updated = []
    for row in items:
        r = dict(row)
        v = to_num(r.get(key))
        if v is not None:
            r[key] = round(v * factor, 4)
        updated.append(r)
    return (updated, columns, f"Applied a {pct}% discount on unit prices.")


def handle_increase(items, columns, msg):
    """
    increase all prices by 5%
    add 5% to all prices
    """
    m = re.search(r"(?:increase|raise|add)\s+(?:all\s+)?(?:prices?|unit\s*price)\s+"
                  r"by\s+(\d+(?:\.\d+)?)\s*%", msg, re.I)
    if not m:
        return None
    pct = float(m.group(1))
    factor = 1.0 + (pct / 100.0)
    key = find_role_column(columns, "unit_price")
    if not key:
        return None
    updated = []
    for row in items:
        r = dict(row)
        v = to_num(r.get(key))
        if v is not None:
            r[key] = round(v * factor, 4)
        updated.append(r)
    return (updated, columns, f"Increased unit prices by {pct}%.")


def handle_sort(items, columns, msg):
    """
    sort by flower
    sort by quantity descending
    """
    m = re.search(r"sort\s+(?:rows?\s+)?by\s+([a-z0-9 _\-]+?)"
                  r"(?:\s+(asc(?:ending)?|desc(?:ending)?))?$", msg, re.I)
    if not m:
        return None
    needle = m.group(1).strip()
    desc = bool(m.group(2) and "desc" in m.group(2).lower())

    key = find_column_by_label(columns, needle)
    if not key:
        return None

    def key_fn(row):
        v = row.get(key)
        if v is None:
            return (1, "")
        if isinstance(v, (int, float)):
            return (0, float(v))
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v).lower())

    updated = sorted(items, key=key_fn, reverse=desc)
    return (updated, columns,
            f"Sorted by “{needle}” ({'descending' if desc else 'ascending'}).")


def handle_clear(items, columns, msg):
    """
    clear all prices / clear prices
    """
    m = re.search(r"clear\s+(?:all\s+)?(prices?|unit\s*price)", msg, re.I)
    if not m:
        return None
    key = find_role_column(columns, "unit_price")
    if not key:
        return None
    updated = [{**row, key: None} for row in items]
    return (updated, columns, "Cleared all unit prices.")


def handle_round(items, columns, msg):
    """
    round prices to 2 decimals
    round quantity
    """
    m = re.search(r"round\s+(prices?|quantit(?:y|ies)|totals?)\s+to\s+(\d+)\s*decimals?",
                  msg, re.I)
    if not m:
        return None
    target, decimals = m.group(1).lower(), int(m.group(2))
    if target.startswith("price"):
        key = find_role_column(columns, "unit_price")
    elif target.startswith("total"):
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
    return (updated, columns, f"Rounded {target} to {decimals} decimals.")


# ===========================================================================
#  INFORMATIONAL QUESTIONS (read-only)
# ===========================================================================
def handle_question(items, columns, msg):
    """Answer without modifying."""
    low = msg.lower().strip()

    qty_key = find_role_column(columns, "quantity")
    price_key = find_role_column(columns, "unit_price")
    total_key = find_role_column(columns, "total")

    # count rows
    if re.search(r"\b(how many rows|row count|count rows|number of rows)\b", low):
        return (items, columns, f"There are {len(items)} rows.")

    # total quantity
    if re.search(r"\b(total|sum)\s+(of\s+)?quantity\b", low) and qty_key:
        total = sum(to_num(r.get(qty_key)) or 0 for r in items)
        return (items, columns,
                f"Total quantity is {int(total) if total.is_integer() else total}.")

    # total amount
    if re.search(r"\b(total|grand)\s+(amount|price|value)\b", low):
        if total_key:
            total = sum(to_num(r.get(total_key)) or 0 for r in items)
        elif qty_key and price_key:
            total = sum((to_num(r.get(qty_key)) or 0) * (to_num(r.get(price_key)) or 0)
                        for r in items)
        else:
            total = 0
        return (items, columns, f"Total amount is {round(total, 2)}.")

    # count by filter
    m = re.search(r"how many\s+([a-z0-9 _\-]+)", low)
    if m:
        needle = m.group(1).strip()
        count = 0
        for row in items:
            for ck in text_columns(columns):
                if row_matches(row, ck, needle):
                    count += 1
                    break
        if count:
            return (items, columns, f"I found {count} row(s) matching “{needle}”.")

    return None


# ===========================================================================
#  TIER 1 DISPATCHER
# ===========================================================================
DETERMINISTIC_HANDLERS = [
    handle_set_price,
    handle_calculate_totals,
    handle_remove_column,
    handle_remove_rows,
    handle_multiply,
    handle_discount,
    handle_increase,
    handle_sort,
    handle_clear,
    handle_round,
    handle_question,
]


def run_deterministic(items, columns, message):
    for handler in DETERMINISTIC_HANDLERS:
        try:
            result = handler(items, columns, message)
        except Exception as e:
            logger.warning(f"{handler.__name__} failed: {e}")
            continue
        if result:
            return result
    return None


# ===========================================================================
#  TIER 2 — AI HELPER
# ===========================================================================
AI_SYSTEM = """You transform invoice tables based on short user instructions.

You receive:
  columns: [{key, label, role, source}, ...]
  items: array of rows, keyed by column.key
  message: the user's instruction

Return ONLY valid JSON with this shape:
{
  "action": "<one of: set_price | calculate_totals | multiply | remove_column | remove_rows | none>",
  "args": { ... },          // depends on action (see below)
  "explanation": "<one short sentence>"
}

Action arguments:
  set_price:          {"filter": "<substring>", "price": <number>}
  calculate_totals:   {}
  multiply:           {"target": "quantity"|"unit_price", "factor": <number>}
  remove_column:      {"label": "<column label or key>"}
  remove_rows:        {"where": "quantity_is_zero"|"price_missing"|"empty_row"}
  none:               {}   (when the instruction is unclear or unsupported)

Do not invent prices or quantities. Do not rename columns. When unsure, use "none".
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
    # Try Gemini
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

    # Then Groq
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
        "items": items[:200],   # cap to keep tokens low
        "message": message,
    }, ensure_ascii=False)[:30000]

    raw = _ai_call(AI_SYSTEM + "\n\nINPUT:\n" + payload)
    data = _ai_clean_json(raw) if raw else None
    if not data:
        return None

    action = data.get("action")
    args = data.get("args") or {}
    explanation = data.get("explanation", "")

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
            key = find_role_column(columns,
                                   "quantity" if "quant" in target else "unit_price")
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

        if action == "remove_column":
            key = find_column_by_label(columns, str(args.get("label", "")))
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

    except Exception as e:
        logger.warning(f"AI action execution failed: {e}")
        return None

    return None


def _apply_set_price(items, columns, filter_, price, explanation):
    match_cols = [find_role_column(columns, "product_name"),
                  find_role_column(columns, "variety")]
    match_cols += text_columns(columns)
    match_cols = [c for c in match_cols if c]

    price_key, columns = ensure_column(columns, "unit_price", "Unit Price")
    words = [w for w in filter_.split() if len(w) >= 3]

    updated = []
    matched = 0
    for row in items:
        hit = any(row_matches(row, ck, w) for ck in match_cols for w in words)
        if hit:
            r = dict(row)
            r[price_key] = price
            updated.append(r)
            matched += 1
        else:
            updated.append(row)
    if matched == 0:
        return None
    return (updated, columns,
            explanation or f"Set unit price to {price} on {matched} row(s).")


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

    # Tier 1
    result = run_deterministic(items, columns, msg)
    if result:
        new_items, new_cols, explanation = result
        return {
            "success": True, "items": new_items, "columns": new_cols,
            "explanation": explanation, "applied_via": "deterministic",
        }

    # Tier 2
    result = run_ai(items, columns, msg)
    if result:
        new_items, new_cols, explanation = result
        return {
            "success": True, "items": new_items, "columns": new_cols,
            "explanation": explanation, "applied_via": "ai",
        }

    # Neither worked
    return {
        "success": True, "items": items, "columns": columns,
        "explanation": "I could not understand that instruction. "
                       "Try: “add unit price 0.5 to all carnations”, "
                       "“calculate totals”, “remove column color”.",
        "applied_via": "unrecognized",
    }
