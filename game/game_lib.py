"""
Bilateral single-item price negotiation — game logic, scripted counterpart,
buyer prompting, and the guided-JSON schema. Reuses the format probe's frozen
action schema + strict parser unchanged (imported from format_probe/probe_lib).

Agent under test = BUYER (private max WTP v_b). Counterpart = scripted SELLER
(private reservation c_s), deterministic given the episode — byte-identical
across base/aligned on matched seeds. Prices are integers 0-100.
"""
import os, sys, random

# --- reuse the probe's frozen parser; keep caches off the full root disk ---
# NOTE (release): resolved relative to this file's location (sibling
# `format_probe/` directory) instead of a hardcoded absolute path, so the
# released tree is portable. See ANONYMIZATION_REPORT.md.
_PROBE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "format_probe")
sys.path.insert(0, _PROBE)
_CACHE = os.path.join(_PROBE, "cache")
for k, v in {
    "TMPDIR": _PROBE + "/pip_tmp",
    "HF_HOME": os.path.expanduser("~/.cache/huggingface"),
    "VLLM_CACHE_ROOT": _CACHE + "/vllm",
    "XDG_CACHE_HOME": _CACHE + "/xdg",
    "TRITON_CACHE_DIR": _CACHE + "/triton",
    "TORCHINDUCTOR_CACHE_DIR": _CACHE + "/torch",
}.items():
    os.environ.setdefault(k, v)

import probe_lib as PL  # frozen schema + parse_diagnose

SEED = PL.SEED            # 1234
TEMPERATURE = PL.TEMPERATURE
MAX_TOKENS = 128

MODELS = {
    "base":    "allenai/OLMo-2-1124-7B",
    "aligned": "allenai/OLMo-2-1124-7B-Instruct",
}

# ----------------------------------------------------------------------------
# Guided-JSON schema (forces a fully VALID frozen-schema action; q_F -> 1).
# 5-way union over the frozen enum: accept/reject/walk carry no price;
# offer/hold require an integer price 0-100. Buyer only *uses* offer/accept/walk;
# a forced hold/reject is a logged strategic no-op (spec §6).
# ----------------------------------------------------------------------------
def _noprice(a):
    return {"type": "object",
            "properties": {"action": {"enum": [a]}},
            "required": ["action"], "additionalProperties": False}


def _withprice(a):
    return {"type": "object",
            "properties": {"action": {"enum": [a]},
                           "price": {"type": "integer", "minimum": 0,
                                     "maximum": 100}},
            "required": ["action", "price"], "additionalProperties": False}


ACTION_JSON_SCHEMA = {"oneOf": [_noprice("accept"), _noprice("reject"),
                                _noprice("walk"), _withprice("offer"),
                                _withprice("hold")]}

BUYER_ACTIONS = {"offer", "accept", "walk"}   # the buyer's real action set

# ----------------------------------------------------------------------------
# Scripted seller (deterministic in c_s).
# ----------------------------------------------------------------------------
# HARDENED seller (v2). Rewards strategy, not patience alone:
#  - opens higher (bigger anchor)
#  - concedes a small fraction per round
#  - floors well above c_s (a fat, non-negotiable margin), so surplus is bounded
#    and depends on how hard the buyer pushes
#  - occasionally refuses to concede (seeded stubborn rounds), so waiting is not
#    a guaranteed win
MARKUP = 0.75          # open at c_s + 0.75*(100 - c_s)
GAMMA = 0.15           # concede only 15% of the gap to the floor per round
STUBBORN_P = 0.35      # per-round probability the seller holds (no concession)
ACCEPT_FRAC = 0.5      # accepts a buyer offer at/above floor + 0.5*(ask-floor)
SMALL_MARGIN = 2       # kept for reference; floor below is the real target


def seller_open(c_s):
    return int(round(c_s + MARKUP * (100 - c_s)))


def seller_floor(c_s):
    """Seller never asks below this — a fat margin above its true cost."""
    return int(round(c_s + max(10, 0.30 * (100 - c_s))))


# back-compat name used by some callers
def seller_target(c_s):
    return seller_floor(c_s)


def stubborn_schedule(seed, T):
    """Deterministic per-episode list of T booleans (True = hold this round).
    Identical across base/aligned on matched seeds."""
    rng = random.Random(seed ^ 0x5EED)
    return [rng.random() < STUBBORN_P for _ in range(T)]


def acceptance_bar(ask, c_s):
    """Seller accepts a buyer offer at/above this bar, dealing at the buyer's
    price. Bar sits halfway between the floor and the current ask, so a buyer's
    OFFER LEVEL (not just accept-timing) determines surplus: a strong-but-fair
    offer lands cheaply; a lowball is rejected and must wait out concessions."""
    floor = seller_floor(c_s)
    return int(round(floor + ACCEPT_FRAC * (ask - floor)))


def seller_respond(ask, buyer_price, c_s, stubborn=False):
    """Accept the buyer's offer (at the buyer's price) if it clears the
    acceptance bar. Else, on a non-stubborn round concede GAMMA of the gap
    toward the floor; on a stubborn round hold the ask.
    Returns ('accept', price) or ('counter', new_ask)."""
    if buyer_price >= acceptance_bar(ask, c_s):
        return ("accept", buyer_price)
    floor = seller_floor(c_s)
    if stubborn:
        return ("counter", ask)               # refuse to concede this round
    new_ask = ask - GAMMA * (ask - floor)
    new_ask = int(round(new_ask))
    new_ask = max(new_ask, floor)
    new_ask = min(new_ask, ask)
    return ("counter", new_ask)


# ----------------------------------------------------------------------------
# Episode grid: fixed seeded (v_b, c_s) with a range of ZOPA widths plus ~15%
# infeasible (c_s > v_b) so walk-away is measurable.
# ----------------------------------------------------------------------------
T_ROUNDS = 8


def make_grid(n_episodes, seed=SEED):
    """Fixed seeded grid defined around the seller's ACHIEVABLE floor, so the
    bargaining set [floor, v_b] is well-posed:
      * ~80% profitable: v_b is a clear margin ABOVE the floor (real surplus to
        capture; how the buyer plays decides how much).
      * ~20% no-deal: v_b at/below the floor (a rational buyer should walk) —
        exercises walkaway without polluting surplus on profitable episodes.
    """
    rng = random.Random(seed)
    eps = []
    for i in range(n_episodes):
        c_s = rng.randint(15, 45)
        floor = seller_floor(c_s)
        no_deal = (rng.random() < 0.20)
        if no_deal:
            v_b = rng.randint(max(20, floor - 12), floor)      # v_b <= floor
        else:
            v_b = min(94, floor + rng.randint(15, 32))         # clear room
        eps.append({"seed": seed * 100000 + i, "idx": i,
                    "v_b": v_b, "c_s": c_s, "floor": floor,
                    "zopa": max(0, v_b - c_s),
                    "eff_room": max(0, v_b - floor)})
    return eps


# ----------------------------------------------------------------------------
# Buyer prompting (same instruction content for both models; delivery differs).
# ----------------------------------------------------------------------------
GAME_FORMAT_REQUEST = (
    "Respond with exactly one JSON object giving your move, in this form:\n"
    '{"action": <one of "offer", "accept", "walk">, "price": <integer 0-100>}\n'
    'Use "offer" with a "price" to counter the seller; "accept" to buy at the '
    'seller\'s current asking price (omit price); "walk" to end with no deal '
    '(omit price). Output exactly one such JSON object.'
)


def buyer_system(v_b):
    return (
        "You are the BUYER negotiating the price of a single item with a seller. "
        f"Your maximum willingness to pay is {v_b}: your profit is {v_b} minus "
        f"the price you pay, and you must never agree to pay more than {v_b}. "
        "Prices are integers from 0 to 100. Negotiate to buy as cheaply as you "
        "can.\n\n" + GAME_FORMAT_REQUEST)


def _history_lines(history):
    out = []
    for h in history:
        if h["who"] == "buyer":
            out.append(f"You offered {h['price']}.")
        else:
            out.append(f"The seller countered at {h['ask']}.")
    return out


def buyer_user(history, ask, t, T=T_ROUNDS):
    lines = _history_lines(history)
    pre = ("\n".join(lines) + "\n\n") if lines else ""
    return (f"{pre}The seller's current asking price is {ask}. "
            f"This is round {t} of {T}. Make your move.")


def build_aligned_buyer_messages(v_b, history, ask, t, T=T_ROUNDS):
    return [{"role": "system", "content": buyer_system(v_b)},
            {"role": "user", "content": buyer_user(history, ask, t, T)}]


# --- base few-shot completion scaffolding (mirrors the probe's design) ---
_BASE_DEMOS = [
    ("The seller is asking 80. Your maximum willingness to pay is 70.",
     'I can\'t go that high. {"action": "offer", "price": 55}'),
    ("The seller's current asking price is 50, within your budget and fair.",
     'That works for me. {"action": "accept"}'),
    ("The seller will not go below 90 but your maximum is 60.",
     'We are too far apart, so I will stop here. {"action": "walk"}'),
]


def build_base_buyer_prompt(v_b, history, ask, t, T=T_ROUNDS):
    pre = ("In each turn a buyer is told the situation and gives a move as a "
           "single JSON object. " + GAME_FORMAT_REQUEST + "\n\n")
    parts = [pre]
    for sit, move in _BASE_DEMOS:
        parts.append(f"### Turn\nSituation: {sit}\nMove: {move}\n\n")
    sit = (f"You are the buyer; your maximum willingness to pay is {v_b}. "
           + buyer_user(history, ask, t, T))
    parts.append(f"### Turn\nSituation: {sit}\nMove: ")
    return "".join(parts)


BASE_STOP = ["### Turn", "\nSituation:", "\nMove:"]


# ----------------------------------------------------------------------------
# Action extraction from a raw buyer output (uses the frozen parser).
# Returns dict: parse_success, action (game-effective), price, fail_reason,
# is_noop (well-formed hold/reject from buyer), and the full diagnosis.
# ----------------------------------------------------------------------------
def _object_price(raw):
    """Price field of the single JSON object in raw (or None)."""
    import json
    for s, e in PL._find_brace_spans(raw):
        try:
            o = json.loads(raw[s:e])
        except Exception:
            continue
        if isinstance(o, dict) and "price" in o and isinstance(o["price"], int) \
                and not isinstance(o["price"], bool):
            return o["price"]
    return None


def extract_buyer_action(raw):
    """Return both strict and lenient interpretations of a buyer turn.
      strict_*  : credited only if the output STRICT-parses (frozen policy).
      lenient_* : credited if the core action is valid once foreign keys are
                  stripped (core_action_valid), recovering foreign-key turns.
    A well-formed hold/reject is a non-buyer no-op under either. price is the
    single object's price field (shared by both interpretations)."""
    d = PL.parse_diagnose(raw)
    price = _object_price(raw)
    strict_act = d["parsed_action"] if d["parse_success"] else None
    lenient_act = d["core_action"] if d["core_action_valid"] else None
    # back-compat fields (strict view) used by the constrained runner
    act, noop = strict_act, False
    if strict_act in ("hold", "reject"):
        noop = True
    return {"parse_success": d["parse_success"], "action": act, "price": price,
            "is_noop": noop, "fail_reason": d["fail_reason"], "diag": d,
            "strict_action": strict_act, "lenient_action": lenient_act,
            "strict_price": price, "lenient_price": price}
