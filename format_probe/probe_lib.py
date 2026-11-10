"""
Format probe — shared library: schema, strict parser, action taxonomy,
history conditions, and prompt construction.

This is the single source of truth for the action format schema and the strict
parser. The SAME strict_parse() must be reused verbatim by the planned
negotiation game so probe and game scores are comparable.

Design invariants (see spec §2,§3,§5):
  * The action format request lives ONLY in the shared framing (FORMAT_REQUEST),
    identical for every action type and both models.
  * The per-action instruction is purely strategic/semantic — it never mentions
    JSON or the schema. We are measuring whether the model *spontaneously*
    emits parseable format for a given intent.
  * Aligned model -> chat template, zero-shot.
  * Base model    -> no chat template; fixed few-shot completion prefix.
"""

import json

# ----------------------------------------------------------------------------
# Decoding (identical across models and identical to the planned game run)
# ----------------------------------------------------------------------------
SEED = 1234
TEMPERATURE = 0.7
TOP_P = 1.0
MAX_TOKENS = 256

# ----------------------------------------------------------------------------
# §2  Action schema + strict parser
# ----------------------------------------------------------------------------
ACTIONS = {"accept", "offer", "reject", "hold", "walk"}
NEEDS_PRICE = {"offer", "hold"}          # price required for these
NO_PRICE = {"accept", "reject", "walk"}  # price must be omitted for these


def _find_brace_spans(text):
    """Return (start, end) of every top-level brace-balanced span in `text`,
    ignoring braces inside JSON strings. Nested braces stay within their span."""
    spans, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append((start, i + 1))
                    start = None
    return spans


def _validate(obj):
    """Validate a parsed dict against the strict schema.
    Returns (ok: bool, action: str|None).

    Strictness policy (chosen to measure elaboration-induced format failure,
    not benign field noise — see REPORT/notes):
      * Foreign keys (anything other than "action"/"price") => FAILURE. These
        are schema violations and are exactly the elaboration leakage we want
        to catch (e.g. a model cramming a "condition"/"reason" key into the
        object under high-elaboration pressure).
      * "price" is a schema field. It is REQUIRED for offer/hold, and OPTIONAL
        (tolerated, ignored) for accept/reject/walk — a spare price on an accept
        is harmless field noise, NOT an elaboration failure, so it must not be
        penalized. If present it must still be a valid integer 0-100.
      * action must be in the enum.
    The exactly-one-well-formed-object requirement (the main anti-rambling gate)
    is enforced upstream in strict_parse().
    """
    if not isinstance(obj, dict):
        return False, None
    if set(obj.keys()) - {"action", "price"}:      # foreign key -> failure
        return False, None
    a = obj.get("action")
    if a not in ACTIONS:                            # enum check
        return False, None
    has_price = "price" in obj
    if has_price:                                   # if present, must be valid
        p = obj["price"]
        if isinstance(p, bool) or not isinstance(p, int):   # strict integer
            return False, None
        if not (0 <= p <= 100):
            return False, None
    if a in NEEDS_PRICE and not has_price:          # required for offer/hold
        return False, None
    return True, a


# Failure-mechanism taxonomy. The first six match the agreed enum; "invalid_price"
# is an extension (present-but-out-of-range/wrong-type price) so the bucket is
# never silently mislabeled — it is neither a missing field nor a foreign key.
FAIL_REASONS = (
    "missing_required_field",   # offer/hold with no price
    "foreign_key",              # well-formed action + a schema-foreign key
    "invalid_price",            # price present but wrong type / out of 0-100
    "action_out_of_enum",       # action value not in the action enum / absent
    "malformed_json",           # a brace-blob present but not parseable JSON
    "no_object",                # no JSON object at all (prose only)
    "multiple_objects",         # more than one well-formed JSON object
)


def _classify_single(obj):
    """Diagnose a single parsed dict. Returns the per-record diagnosis fields.

    core_action_valid = would the object validate if every schema-FOREIGN key
    were stripped (i.e. keeping only action/price)? This separates
      * a well-formed action carrying a stray key   (core_action_valid=True)
      * a key DISPLACING a required field           (core_action_valid=False)
    so eta can be re-scored both ways from the persisted records.
    """
    keys = set(obj.keys())
    foreign = sorted(keys - {"action", "price"})
    a = obj.get("action")
    core_obj = {k: v for k, v in obj.items() if k in ("action", "price")}
    core_ok, core_action = _validate(core_obj)

    if core_ok and not foreign:                       # strict success
        return dict(parse_success=True, parsed_action=core_action,
                    fail_reason=None, foreign_keys=[],
                    core_action_valid=True, core_action=core_action,
                    n_json_objects=1)

    # --- failure: pick the primary mechanism ---
    if a not in ACTIONS:
        reason = "action_out_of_enum"
    elif not core_ok:                                 # core defect dominates
        if a in NEEDS_PRICE and "price" not in obj:
            reason = "missing_required_field"
        elif "price" in obj:
            reason = "invalid_price"
        else:
            reason = "missing_required_field"
    elif foreign:                                     # sole defect: stray key(s)
        reason = "foreign_key"
    else:
        reason = "malformed_json"                     # unreachable safety net
    return dict(parse_success=False, parsed_action=None, fail_reason=reason,
                foreign_keys=foreign, core_action_valid=core_ok,
                core_action=(core_action if core_ok else None),
                n_json_objects=1)


def parse_diagnose(text):
    """STRICT parse + full failure diagnosis (single source of truth).

    Frozen policy: success requires EXACTLY ONE well-formed JSON object that
    validates (spare price on accept/reject/walk tolerated; foreign keys and
    missing required fields fail). Only well-formed JSON dicts are counted
    toward the one-object rule; a stray malformed brace-blob is tolerated.

    Returns a dict with: parse_success, parsed_action, fail_reason,
    foreign_keys, core_action_valid, core_action, n_json_objects.
    """
    spans = _find_brace_spans(text)
    dicts, malformed = [], False
    for s, e in spans:
        try:
            o = json.loads(text[s:e])
        except Exception:
            malformed = True
            continue
        if isinstance(o, dict):
            dicts.append(o)
    n = len(dicts)
    if n == 1:
        return _classify_single(dicts[0])
    if n == 0:
        reason = "malformed_json" if (spans and malformed) else "no_object"
        return dict(parse_success=False, parsed_action=None, fail_reason=reason,
                    foreign_keys=[], core_action_valid=False, core_action=None,
                    n_json_objects=0)
    return dict(parse_success=False, parsed_action=None,
                fail_reason="multiple_objects", foreign_keys=[],
                core_action_valid=False, core_action=None, n_json_objects=n)


def strict_parse(text):
    """Back-compat thin wrapper: (parse_success, parsed_action)."""
    d = parse_diagnose(text)
    return d["parse_success"], d["parsed_action"]


# ----------------------------------------------------------------------------
# §3  Action taxonomy (independent variable: elaboration pressure)
# Instructions are STRATEGIC ONLY — they never mention JSON/schema.
# ----------------------------------------------------------------------------
ACTION_TYPES = {
    "accept": {
        "pressure": "minimal",
        "expected": "accept",
        "instruction": "You have decided to accept the standing offer.",
    },
    "offer_simple": {
        "pressure": "low",
        "expected": "offer",
        "instruction": "You have decided to counter at a specific price.",
    },
    "walk_reasoned": {
        "pressure": "mid",
        "expected": "walk",
        "instruction": (
            "You have decided to walk away from this negotiation, and to "
            "explain your reasoning to the counterpart."
        ),
    },
    "hold_justified": {
        "pressure": "high",
        "expected": "hold",
        "instruction": (
            "You have decided to hold firm and demand near the full amount, "
            "justifying your position with substantive reasons."
        ),
    },
    "offer_conditional": {
        "pressure": "high",
        "expected": "offer",
        "instruction": (
            "You have decided to make a conditional offer — a price contingent "
            "on the counterpart meeting a stated condition — and to spell out "
            "the condition."
        ),
    },
}
ACTION_ORDER = ["accept", "offer_simple", "walk_reasoned",
                "hold_justified", "offer_conditional"]


# ----------------------------------------------------------------------------
# §4  History conditions (increasing strategic load)
# ----------------------------------------------------------------------------
HISTORIES = {
    "early_low": (
        "This is the opening of a negotiation over a modestly valued item. "
        "No offers are on the table yet. The buyer has just said, in a neutral "
        "tone: \"I might be interested. What are you looking to get for it?\""
    ),
    "mid": (
        "The negotiation over a moderately valuable item is underway. You "
        "previously asked for 70. The buyer has made one counter-move: \"I can "
        "do 45 — that seems fair given the condition.\" The standing offer on "
        "the table is 45."
    ),
    "late_high": (
        "This is a late, high-stakes stage of a negotiation over a very "
        "valuable lot, after several rounds. The buyer has just made an "
        "aggressive anchor, slashing their earlier numbers: \"Final position — "
        "25, take it or leave it. That is already more than generous.\" The "
        "standing offer on the table is 25 and the amount at stake is large."
    ),
}
HISTORY_ORDER = ["early_low", "mid", "late_high"]


# ----------------------------------------------------------------------------
# §5  Shared format request (the ONLY place the schema is shown to the model).
# Identical content for both models and every action type.
# ----------------------------------------------------------------------------
FORMAT_REQUEST = (
    "You may explain your thinking briefly, but your response must contain "
    "exactly one JSON object that states your move, in exactly this form:\n"
    '{"action": <one of "accept", "offer", "reject", "hold", "walk">, '
    '"price": <integer from 0 to 100>}\n'
    'Include the "price" field only when your action is "offer" or "hold"; '
    'omit it for "accept", "reject", and "walk". Output exactly one such JSON '
    "object."
)

ROLE = (
    "You are a seasoned negotiator acting for the seller's side in a one-on-one "
    "price negotiation. Prices are on a 0-100 scale."
)


# --- Aligned model: chat template, zero-shot ---
def build_aligned_messages(history_key, action_type):
    h = HISTORIES[history_key]
    instr = ACTION_TYPES[action_type]["instruction"]
    system = ROLE + "\n\n" + FORMAT_REQUEST
    user = (
        f"Negotiation so far:\n{h}\n\n"
        f"Your private decision for this turn: {instr}\n\n"
        f"Now give your move."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --- Base model: no chat template; fixed few-shot completion prefix ---
# Demos are FIXED and identical across all action types / histories. They show
# the format can be emitted both bare and after a sentence of reasoning, so the
# base model *can* produce valid actions at all (spec §5) without over-teaching
# the high-pressure action types.
_FEWSHOT_DEMOS = [
    {
        "situation": "A buyer has offered 60 for your bike and you are happy "
                     "with that price.",
        "decision": "You have decided to accept the standing offer.",
        "move": 'That works for me — deal. {"action": "accept"}',
    },
    {
        "situation": "You are selling a desk. The buyer offered 20, but you "
                     "want more for it.",
        "decision": "You have decided to counter at a specific price.",
        "move": 'I cannot go that low, but I will meet you partway. '
                '{"action": "offer", "price": 35}',
    },
    {
        "situation": "After many rounds the buyer will not exceed 10 for your "
                     "camera, far below what it is worth.",
        "decision": "You have decided to walk away, and to explain your "
                    "reasoning to the counterpart.",
        "move": "We are simply too far apart — at 10 I would be selling well "
                "below its value, so I will step away here. "
                '{"action": "walk"}',
    },
]

_BASE_PREAMBLE = (
    "The following are turns in a one-on-one price negotiation seen from the "
    "seller's side. Prices are on a 0-100 scale. In each turn the agent is "
    "told the situation and its private decision, and then gives its move. "
    + FORMAT_REQUEST + "\n\n"
)


def _base_turn_block(situation, decision, move=None):
    s = f"### Turn\nSituation: {situation}\nDecision: {decision}\nMove:"
    if move is not None:
        s += " " + move + "\n\n"
    else:
        s += " "   # trailing space: completion continues here
    return s


def build_base_prompt(history_key, action_type):
    h = HISTORIES[history_key]
    instr = ACTION_TYPES[action_type]["instruction"]
    parts = [_BASE_PREAMBLE]
    for d in _FEWSHOT_DEMOS:
        parts.append(_base_turn_block(d["situation"], d["decision"], d["move"]))
    parts.append(_base_turn_block(h, instr, None))
    return "".join(parts)


# Stop strings so the base model does not fabricate further turns past its move.
BASE_STOP = ["### Turn", "\nSituation:", "\nDecision:"]


def expected_action(action_type):
    return ACTION_TYPES[action_type]["expected"]


def adherence(action_type, parsed_action):
    """True iff a (successfully parsed) action matches the instructed type."""
    if parsed_action is None:
        return False
    return parsed_action == expected_action(action_type)


def make_record(model, action_type, history_bin, raw, backend=None, mode=None):
    """Build one persisted generation record with full failure diagnosis.
    Re-derivable: every scoring (strict/lenient eta, adherence) comes from these
    fields, so a parse-rule change never requires regeneration."""
    d = parse_diagnose(raw)
    exp = expected_action(action_type)
    rec = {
        "model": model,
        "action_type": action_type,
        "history_bin": history_bin,
        "raw_output": raw,
        "parse_success": d["parse_success"],
        "parsed_action": d["parsed_action"],
        "adherence": adherence(action_type, d["parsed_action"]),
        # failure mechanism + re-scoring fields
        "fail_reason": d["fail_reason"],
        "foreign_keys": d["foreign_keys"],
        "core_action_valid": d["core_action_valid"],
        "core_action": d["core_action"],
        "core_adherence": bool(d["core_action_valid"]
                               and d["core_action"] == exp),
        "n_json_objects": d["n_json_objects"],
    }
    if backend is not None:
        rec["backend"] = backend
    if mode is not None:
        rec["mode"] = mode
    return rec
