from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Tuple

from bidict import frozenbidict

# ── Special tokens ────────────────────────────────────────────────────────────
PAD = 0
SOS = 1
EOS = 2
EON = 3

# ── Note kind tokens ──────────────────────────────────────────────────────────
KIND_TOKENS: Dict[str, int] = {
    "Tap": 4,
    "Hold": 5,
    "Slide": 6,
    "Touch": 7,
    "TouchHold": 8,
}
KIND_INV: Dict[int, str] = {v: k for k, v in KIND_TOKENS.items()}

# ── Position tokens (41 positions) ────────────────────────────────────────────
POSITIONS: List[str] = [
    "Btn1", "Btn2", "Btn3", "Btn4", "Btn5", "Btn6", "Btn7", "Btn8",
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
    "C",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
    "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
]
POS_BASE = 9
POS_TO_ID: Dict[str, int] = {p: POS_BASE + i for i, p in enumerate(POSITIONS)}
ID_TO_POS: Dict[int, str] = {v: k for k, v in POS_TO_ID.items()}

# ── Note decoration tokens ────────────────────────────────────────────────────
DECO_NONE = 50
DECO_BREAK = 51
DECO_EX = 52
DECO_FIREWORK = 53
DECO_NAMES: Dict[str, int] = {"Break": DECO_BREAK, "Ex": DECO_EX, "Firework": DECO_FIREWORK}
DECO_INV: Dict[int, str] = {v: k for k, v in DECO_NAMES.items()}
DECO_SORT_ORDER = ["Break", "Ex", "Firework"]

# ── Slide decoration tokens ───────────────────────────────────────────────────
SLIDE_DECO_NONE = 54
SLIDE_DECO_BREAK = 55
SLIDE_DECO_NAMES: Dict[str, int] = {"Break": SLIDE_DECO_BREAK}
SLIDE_DECO_INV: Dict[int, str] = {v: k for k, v in SLIDE_DECO_NAMES.items()}

# ── Duration type markers ─────────────────────────────────────────────────────
DUR_ABS = 56
DUR_SYM = 57

# ── Divider / multiplier markers ──────────────────────────────────────────────
DIV = 58
MUL = 59

# ── Slide segment markers ─────────────────────────────────────────────────────
SEG = 60
SEG_END = 61

# ── Wait marker ───────────────────────────────────────────────────────────────
WAIT = 62

# ── Slide shape tokens ────────────────────────────────────────────────────────
SHAPES: Dict[str, int] = {
    "Straight": 63,
    "ArcLeft": 64,
    "Center": 65,
    "ArcRight": 66,
    "ZigzagS": 67,
    "ZigzagZ": 68,
    "P": 69,
    "Q": 70,
    "PP": 71,
    "QQ": 72,
    "Reflect": 73,
    "Wifi": 74,
}
SHAPE_INV: Dict[int, str] = {v: k for k, v in SHAPES.items()}

# ── Time offset encoding ──────────────────────────────────────────────────────
TIME_OFFSET_BASE = 75
MAX_DELTA_MS = 50000
MAX_DELTA_BINS = MAX_DELTA_MS // 10

# ── Integer value encoding (divider / multiplier) ─────────────────────────────
VALUE_BASE = TIME_OFFSET_BASE + MAX_DELTA_BINS  # 5075
MAX_VALUE = 8192

# ── Total vocabulary size ─────────────────────────────────────────────────────
VOCAB_SIZE = VALUE_BASE + MAX_VALUE  # 5075 + 8192 = 13267

# ── Token name -> ID lookup (built once) ──────────────────────────────────────
_TOKEN_NAMES: Dict[str, int] = {
    "PAD": PAD,
    "SOS": SOS,
    "EOS": EOS,
    "EON": EON,
    **{f"KIND_{k}": v for k, v in KIND_TOKENS.items()},
    **{f"POS_{p}": i for p, i in POS_TO_ID.items()},
    "DECO_NONE": DECO_NONE,
    "DECO_BREAK": DECO_BREAK,
    "DECO_EX": DECO_EX,
    "DECO_FIREWORK": DECO_FIREWORK,
    "SLIDE_DECO_NONE": SLIDE_DECO_NONE,
    "SLIDE_DECO_BREAK": SLIDE_DECO_BREAK,
    "DUR_ABS": DUR_ABS,
    "DUR_SYM": DUR_SYM,
    "DIV": DIV,
    "MUL": MUL,
    "SEG": SEG,
    "SEG_END": SEG_END,
    "WAIT": WAIT,
    **{f"SHAPE_{k}": v for k, v in SHAPES.items()},
}

# Build the bidirectional mapping
TOKEN_BIDICT = frozenbidict(_TOKEN_NAMES)


def is_time_token(token_id: int) -> bool:
    return TIME_OFFSET_BASE <= token_id < TIME_OFFSET_BASE + MAX_DELTA_BINS


def is_value_token(token_id: int) -> bool:
    return VALUE_BASE <= token_id < VALUE_BASE + MAX_VALUE


def decode_time_token(token_id: int) -> int:
    """Return delta in milliseconds."""
    return (token_id - TIME_OFFSET_BASE) * 10


def encode_time_token(delta_ms: float) -> int:
    bin_idx = round(delta_ms / 10.0)
    bin_idx = max(0, min(bin_idx, MAX_DELTA_BINS - 1))
    return TIME_OFFSET_BASE + bin_idx


def decode_value_token(token_id: int) -> int:
    return token_id - VALUE_BASE


def encode_value_token(value: int) -> int:
    value = max(0, min(value, MAX_VALUE - 1))
    return VALUE_BASE + value


# ═══════════════════════════════════════════════════════════════════════════════
# ChartTokenizer
# ═══════════════════════════════════════════════════════════════════════════════

_BEAT_MS_FACTOR = 600000.0   # 60000 * 10 (bpm10 = bpm * 10)
_MEASURE_MS_FACTOR = 2400000.0  # full measure: 4 * 60000 * 10


class ChartTokenizer:
    """Encodes and decodes maimai chart notes into/from token sequences.

    Parameters
    ----------
    bpm10_list : list[dict]
        The chart's ``bpm10_list`` from the processed JSON, e.g.
        ``[{"bpm10": 1750, "change_timestamp_ms": 0}, ...]``.
    """

    def __init__(self, bpm10_list: List[Dict]) -> None:
        # precompute sorted arrays for binary search
        self._bpm_times: List[int] = [r["change_timestamp_ms"] for r in bpm10_list]
        self._bpm10_values: List[int] = [r["bpm10"] for r in bpm10_list]

    # ── BPM helpers ────────────────────────────────────────────────────────

    def get_bpm10_at(self, timestamp_ms: int) -> int:
        """Return ``bpm10`` active at *timestamp_ms*."""
        if not self._bpm_times:
            return 1500  # sensible fallback: 150 BPM
        idx = bisect.bisect_right(self._bpm_times, timestamp_ms) - 1
        if idx < 0:
            idx = 0
        return self._bpm10_values[idx]

    def get_bpm_at(self, timestamp_ms: int) -> float:
        """Return BPM (float) active at *timestamp_ms*."""
        return self.get_bpm10_at(timestamp_ms) / 10.0

    # ── Default wait ───────────────────────────────────────────────────────

    def _default_wait_ms(self, timestamp_ms: int) -> float:
        """Default wait = one quarter-note beat at the current BPM."""
        bpm10 = self.get_bpm10_at(timestamp_ms)
        return _BEAT_MS_FACTOR / bpm10

    def _duration_expr_to_ms(self, expr: Dict, timestamp_ms: int) -> float:
        """Convert a DurationExpr dict to absolute milliseconds."""
        if "AbsoluteMs" in expr:
            return float(expr["AbsoluteMs"])
        # DividerMultiplier: [divider, multiplier]
        div, mul = expr["DividerMultiplier"]
        if div == 0:
            return 0.0
        bpm10 = self.get_bpm10_at(timestamp_ms)
        return _MEASURE_MS_FACTOR * mul / bpm10 / div

    # ── Duration encoding helpers ──────────────────────────────────────────

    def _encode_duration(self, tokens: List[int], expr: Dict) -> None:
        if "AbsoluteMs" in expr:
            tokens.append(DUR_ABS)
            tokens.append(encode_time_token(expr["AbsoluteMs"]))
        else:
            div, mul = expr["DividerMultiplier"]
            tokens.append(DUR_SYM)
            tokens.append(DIV)
            tokens.append(encode_value_token(div))
            tokens.append(MUL)
            tokens.append(encode_value_token(mul))

    @staticmethod
    def _decode_duration(tokens: List[int], pos: int) -> Tuple[Dict, int]:
        """Read duration tokens starting at *pos*.  Returns (duration_dict, new_pos)."""
        marker = tokens[pos]
        pos += 1
        if marker == DUR_ABS:
            ms = decode_time_token(tokens[pos])
            pos += 1
            return {"AbsoluteMs": float(ms)}, pos
        elif marker == DUR_SYM:
            # expect DIV, value, MUL, value
            if tokens[pos] != DIV:
                raise ValueError(f"Expected DIV token at position {pos}, got {tokens[pos]}")
            pos += 1
            div = decode_value_token(tokens[pos])
            pos += 1
            if tokens[pos] != MUL:
                raise ValueError(f"Expected MUL token at position {pos}, got {tokens[pos]}")
            pos += 1
            mul = decode_value_token(tokens[pos])
            pos += 1
            return {"DividerMultiplier": [div, mul]}, pos
        else:
            raise ValueError(f"Expected DUR_ABS or DUR_SYM at position {pos-1}, got {marker}")

    # ── Encoding ───────────────────────────────────────────────────────────

    def encode_notes(self, notes: List[Dict]) -> List[int]:
        """Convert a list of note dicts into a token sequence.

        The sequence starts with ``SOS`` and ends with ``EOS``.
        Each note is terminated by ``EON`` (or ``EOS`` for the final note).

        Timestamps are quantised to 10 ms bins via ``round()``.
        """
        tokens: List[int] = [SOS]

        prev_ms = 0
        last_idx = len(notes) - 1

        for i, note in enumerate(notes):
            raw_ts = note["timestamp_ms"]
            delta_ms = raw_ts - prev_ms
            # Quantise to 10 ms bins and reconstruct so the accumulated
            # ''prev_ms'' stays consistent with the quantised stream.
            quant_ms = round(delta_ms / 10.0) * 10
            cur_ms = prev_ms + quant_ms  # quantised timestamp of this note
            prev_ms = cur_ms

            # 1. Time offset
            tokens.append(encode_time_token(quant_ms))

            # 2. Kind
            kind = note["kind"]
            tokens.append(KIND_TOKENS[kind])

            # 3. Position
            tokens.append(POS_TO_ID[note["pos"]])

            # 4. Note decorations (sorted: Break, Ex, Firework)
            decos = sorted(note["deco"], key=lambda d: DECO_SORT_ORDER.index(d))
            if not decos:
                tokens.append(DECO_NONE)
            else:
                for d in decos:
                    tokens.append(DECO_NAMES[d])

            # 5. Wait (only for Slide; emit only if explicit and ≠ default)
            if kind == "Slide":
                wait_expr = note.get("wait")
                if wait_expr is not None:
                    explicit_ms = self._duration_expr_to_ms(wait_expr, cur_ms)
                    default_ms = self._default_wait_ms(cur_ms)
                    # Compare to the nearest 10 ms bin so the decision is
                    # idempotent and tolerant to the 10 ms quantisation.
                    explicit_bin = round(explicit_ms / 10.0)
                    default_bin = round(default_ms / 10.0)
                    if explicit_bin != default_bin:
                        tokens.append(WAIT)
                        self._encode_duration(tokens, wait_expr)

            # 6. Duration (Hold, TouchHold, Slide)
            if kind in ("Hold", "Slide", "TouchHold"):
                self._encode_duration(tokens, note["duration"])

            # 7. Slide segments
            if kind == "Slide":
                for seg in note["slide_segments"]:
                    tokens.append(SEG)
                    shape = seg["shape"]
                    if isinstance(shape, dict):
                        # Reflect: {"Reflect": "Btn5"}  (serde externally-tagged enum)
                        shape_name = "Reflect"
                        reflect_pos = shape["Reflect"]
                        tokens.append(SHAPES[shape_name])
                        tokens.append(POS_TO_ID[reflect_pos])
                    else:
                        tokens.append(SHAPES[shape])
                    tokens.append(POS_TO_ID[seg["end"]])
                tokens.append(SEG_END)

                # 8. Slide decorations
                slide_decos = sorted(
                    note.get("slide_deco", []),
                    key=lambda d: 0 if d == "Break" else 1,
                )
                if not slide_decos:
                    tokens.append(SLIDE_DECO_NONE)
                else:
                    for sd in slide_decos:
                        tokens.append(SLIDE_DECO_NAMES[sd])

            # 9. End-of-note
            if i == last_idx:
                tokens.append(EOS)
            else:
                tokens.append(EON)

        return tokens

    # ── Decoding ───────────────────────────────────────────────────────────

    def decode_tokens(self, tokens: List[int]) -> List[Dict]:
        """Reconstruct the original note list from a valid token sequence.

        The input should include ``SOS`` and ``EOS`` tokens.
        """
        notes: List[Dict] = []
        current_time_ms = 0

        pos = 1  # skip SOS

        while pos < len(tokens):
            tok = tokens[pos]

            if tok == EOS:
                break

            # 1. Time offset
            if not is_time_token(tok):
                raise ValueError(
                    f"Expected time offset token at position {pos}, got {tok}"
                )
            delta_ms = decode_time_token(tok)
            current_time_ms += delta_ms
            pos += 1

            # 2. Kind
            kind = KIND_INV[tokens[pos]]
            pos += 1

            # 3. Position
            note_pos = ID_TO_POS[tokens[pos]]
            pos += 1

            # 4. Note decorations
            decos: List[str] = []
            while tokens[pos] in (DECO_BREAK, DECO_EX, DECO_FIREWORK):
                decos.append(DECO_INV[tokens[pos]])
                pos += 1
            if tokens[pos] == DECO_NONE:
                pos += 1
                decos = []

            # 5. Optional wait (Slide only)
            wait = None
            if kind == "Slide" and tokens[pos] == WAIT:
                pos += 1
                wait, pos = self._decode_duration(tokens, pos)

            # 6. Duration (Hold, TouchHold, Slide)
            duration = None
            if kind in ("Hold", "Slide", "TouchHold"):
                duration, pos = self._decode_duration(tokens, pos)

            # 7. Slide segments
            slide_segments: List[Dict] = []
            slide_deco: List[str] = []
            if kind == "Slide":
                while tokens[pos] == SEG:
                    pos += 1
                    shape = SHAPE_INV[tokens[pos]]
                    pos += 1
                    if shape == "Reflect":
                        reflect_pos = ID_TO_POS[tokens[pos]]
                        pos += 1
                        end_pos = ID_TO_POS[tokens[pos]]
                        pos += 1
                        slide_segments.append(
                            {"shape": {"Reflect": reflect_pos}, "end": end_pos}
                        )
                    else:
                        end_pos = ID_TO_POS[tokens[pos]]
                        pos += 1
                        slide_segments.append({"shape": shape, "end": end_pos})
                # SEG_END
                if tokens[pos] != SEG_END:
                    raise ValueError(
                        f"Expected SEG_END at position {pos}, got {tokens[pos]}"
                    )
                pos += 1

                # 8. Slide decorations
                while tokens[pos] in (SLIDE_DECO_BREAK,):
                    slide_deco.append(SLIDE_DECO_INV[tokens[pos]])
                    pos += 1
                if tokens[pos] == SLIDE_DECO_NONE:
                    pos += 1
                    slide_deco = []

            # Build note dict
            note: Dict = {
                "timestamp_ms": current_time_ms,
                "kind": kind,
                "pos": note_pos,
                "deco": decos,
                "wait": wait,
                "duration": duration,
                "slide_segments": slide_segments,
                "slide_deco": slide_deco,
            }
            notes.append(note)

            # 9. End-of-note
            if tokens[pos] == EON:
                pos += 1
            elif tokens[pos] == EOS:
                pos += 1
                break
            else:
                raise ValueError(
                    f"Expected EON or EOS at position {pos}, got {tokens[pos]}"
                )

        return notes


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import copy
    import json

    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Time offset range: {TIME_OFFSET_BASE} – {TIME_OFFSET_BASE + MAX_DELTA_BINS - 1}")
    print(f"Value range: {VALUE_BASE} – {VALUE_BASE + MAX_VALUE - 1}")

    def _quant_note(note: Dict) -> Dict:
        """Return a copy of *note* with timestamps/durations quantised to 10 ms."""
        n = copy.deepcopy(note)
        if isinstance(n["duration"], dict) and "AbsoluteMs" in n["duration"]:
            n["duration"]["AbsoluteMs"] = float(
                round(n["duration"]["AbsoluteMs"] / 10.0) * 10
            )
        if isinstance(n.get("wait"), dict) and "AbsoluteMs" in n["wait"]:
            n["wait"]["AbsoluteMs"] = float(
                round(n["wait"]["AbsoluteMs"] / 10.0) * 10
            )
        return n

    def _quant_notes(
        notes: List[Dict], tokenizer: Optional[ChartTokenizer] = None
    ) -> List[Dict]:
        """Quantise timestamps, durations, and normalise waits.

        Timestamps are reconstructed by accumulating quantised deltas.
        If *tokenizer* is provided, waits that equal the default are set
        to ``None`` to match the tokenizer's omission behaviour.
        """
        out: List[Dict] = []
        prev = 0
        for note in notes:
            delta = note["timestamp_ms"] - prev
            delta_q = round(delta / 10.0) * 10
            prev += delta_q
            n = _quant_note(note)
            n["timestamp_ms"] = prev
            # Normalise wait: if explicit wait and default round to the same
            # 10 ms bin, the tokenizer omits the WAIT token.
            if tokenizer is not None and note["kind"] == "Slide":
                wait_expr = note.get("wait")
                if wait_expr is not None:
                    explicit_ms = tokenizer._duration_expr_to_ms(wait_expr, prev)
                    default_ms = tokenizer._default_wait_ms(prev)
                    explicit_bin = round(explicit_ms / 10.0)
                    default_bin = round(default_ms / 10.0)
                    if explicit_bin == default_bin:
                        n["wait"] = None
            out.append(n)
        return out

    # ── Test 1: basic encode/decode round-trip ─────────────────────────────
    bpm_list = [{"bpm10": 1750, "change_timestamp_ms": 0}]
    tokenizer = ChartTokenizer(bpm_list)

    notes_in = [
        {
            "timestamp_ms": 2742,
            "kind": "TouchHold",
            "pos": "C",
            "deco": [],
            "wait": None,
            "duration": {"DividerMultiplier": [1, 1]},
            "slide_segments": [],
            "slide_deco": [],
        },
        {
            "timestamp_ms": 5485,
            "kind": "Touch",
            "pos": "B1",
            "deco": [],
            "wait": None,
            "duration": None,
            "slide_segments": [],
            "slide_deco": [],
        },
    ]

    tokens = tokenizer.encode_notes(notes_in)
    print(f"\nTokens ({len(tokens)}): {tokens}")

    notes_out = tokenizer.decode_tokens(tokens)
    print(f"Decoded notes: {json.dumps(notes_out, indent=2)}")

    expected = _quant_notes(notes_in, tokenizer)
    assert notes_out == expected, (
        f"Round-trip failed!\n  Expected: {json.dumps(expected)}\n  Got:      {json.dumps(notes_out)}"
    )
    print("\n✓ Basic round-trip test passed")

    # ── Test 2: BPM lookup ─────────────────────────────────────────────────
    bpm2 = [
        {"bpm10": 1500, "change_timestamp_ms": 0},
        {"bpm10": 1800, "change_timestamp_ms": 5000},
        {"bpm10": 2000, "change_timestamp_ms": 10000},
    ]
    tok2 = ChartTokenizer(bpm2)
    assert tok2.get_bpm_at(0) == 150.0
    assert tok2.get_bpm_at(2500) == 150.0
    assert tok2.get_bpm_at(5000) == 180.0
    assert tok2.get_bpm_at(8000) == 180.0
    assert tok2.get_bpm_at(10000) == 200.0
    assert tok2.get_bpm_at(99999) == 200.0
    print("✓ BPM lookup test passed")

    # ── Test 3: decorations round-trip ─────────────────────────────────────
    notes_deco = [
        {
            "timestamp_ms": 1000,
            "kind": "Tap",
            "pos": "Btn1",
            "deco": ["Break", "Ex"],
            "wait": None,
            "duration": None,
            "slide_segments": [],
            "slide_deco": [],
        },
        {
            "timestamp_ms": 2000,
            "kind": "Tap",
            "pos": "Btn2",
            "deco": [],
            "wait": None,
            "duration": None,
            "slide_segments": [],
            "slide_deco": [],
        },
    ]
    t_deco = tokenizer.encode_notes(notes_deco)
    n_deco = tokenizer.decode_tokens(t_deco)
    assert n_deco == notes_deco, f"Deco round-trip failed: {n_deco}"
    print("✓ Decorations round-trip test passed")

    # ── Test 4: slide with segments ────────────────────────────────────────
    notes_slide = [
        {
            "timestamp_ms": 500,
            "kind": "Slide",
            "pos": "Btn1",
            "deco": ["Ex"],
            "wait": None,
            "duration": {"DividerMultiplier": [4, 2]},
            "slide_segments": [
                {"shape": "Straight", "end": "Btn4"},
                {"shape": "ArcRight", "end": "A1"},
            ],
            "slide_deco": [],
        },
        {
            "timestamp_ms": 3000,
            "kind": "Tap",
            "pos": "Btn8",
            "deco": [],
            "wait": None,
            "duration": None,
            "slide_segments": [],
            "slide_deco": [],
        },
    ]
    t_slide = tokenizer.encode_notes(notes_slide)
    n_slide = tokenizer.decode_tokens(t_slide)
    assert n_slide == notes_slide, f"Slide round-trip failed: {n_slide}"
    print("✓ Slide with segments round-trip test passed")

    # ── Test 5: absolute duration ──────────────────────────────────────────
    notes_abs = [
        {
            "timestamp_ms": 0,
            "kind": "Hold",
            "pos": "Btn3",
            "deco": ["Firework"],
            "wait": None,
            "duration": {"AbsoluteMs": 1234.5},
            "slide_segments": [],
            "slide_deco": [],
        },
        {
            "timestamp_ms": 5000,
            "kind": "TouchHold",
            "pos": "D5",
            "deco": [],
            "wait": None,
            "duration": {"AbsoluteMs": 567.0},
            "slide_segments": [],
            "slide_deco": [],
        },
    ]
    t_abs = tokenizer.encode_notes(notes_abs)
    n_abs = tokenizer.decode_tokens(t_abs)
    expected_abs = _quant_notes(notes_abs, tokenizer)
    assert n_abs == expected_abs, (
        f"Abs duration round-trip failed!\n  Expected: {json.dumps(expected_abs)}\n  Got:      {json.dumps(n_abs)}"
    )
    print("✓ Absolute duration round-trip test passed")

    # ── Test 6: explicit wait ≠ default (verify WAIT token emitted) ────────
    notes_wait = [
        {
            "timestamp_ms": 2742,
            "kind": "Slide",
            "pos": "Btn1",
            "deco": [],
            "wait": {"AbsoluteMs": 500.0},
            "duration": {"DividerMultiplier": [2, 1]},
            "slide_segments": [{"shape": "Straight", "end": "Btn2"}],
            "slide_deco": [],
        },
    ]
    t_wait = tokenizer.encode_notes(notes_wait)
    assert WAIT in t_wait, f"WAIT token missing when explicit ≠ default: {t_wait}"
    n_wait = tokenizer.decode_tokens(t_wait)
    assert n_wait == _quant_notes(notes_wait, tokenizer), f"Wait round-trip failed: {n_wait}"
    print("✓ Explicit wait test passed")

    # ── Test 7: wait equals default → no WAIT token ────────────────────────
    # Set the wait to a value that rounds to the same 10 ms bin as the default.
    default_wait = tokenizer._default_wait_ms(2742)
    default_bin = round(default_wait / 10.0)
    wait_near_default = default_bin * 10.0 + 2.0  # still rounds to the same bin
    notes_eq_wait = [
        {
            "timestamp_ms": 2742,
            "kind": "Slide",
            "pos": "Btn1",
            "deco": [],
            "wait": {"AbsoluteMs": wait_near_default},
            "duration": {"DividerMultiplier": [2, 1]},
            "slide_segments": [{"shape": "Straight", "end": "Btn2"}],
            "slide_deco": [],
        },
    ]
    t_eq_wait = tokenizer.encode_notes(notes_eq_wait)
    assert WAIT not in t_eq_wait, (
        f"WAIT token emitted when wait bin equals default bin "
        f"(default={default_wait:.1f}ms, bin={default_bin}): {t_eq_wait}"
    )
    print(f"✓ Default wait omission test passed (default={default_wait:.1f}ms)")

    # ── Test 8: idempotency (encode → decode → encode → decode) ────────────
    for name, notes in [
        ("basic", notes_in),
        ("deco", notes_deco),
        ("slide", notes_slide),
    ]:
        t1 = tokenizer.encode_notes(notes)
        n1 = tokenizer.decode_tokens(t1)
        t2 = tokenizer.encode_notes(n1)
        n2 = tokenizer.decode_tokens(t2)
        assert t1 == t2, f"Idempotency failed for {name}: tokens differ"
        assert n1 == n2, f"Idempotency failed for {name}: notes differ"
    print("✓ Idempotency test passed")

    # ── Test 9: round-trip on real processed.json files ────────────────────
    import glob as _glob
    import sys

    TEST_LIMIT = 20
    test_files = _glob.glob(
        "/Users/aromia/Creation/Programming/python/maimai-chartgpt/dataset/**/processed.json",
        recursive=True,
    )[:TEST_LIMIT]

    passed = 0
    for path in test_files:
        with open(path) as f:
            data = json.load(f)
        for chart in data["charts"]:
            if not chart["notes"]:
                continue
            rt = ChartTokenizer(chart["bpm10_list"])
            try:
                t = rt.encode_notes(chart["notes"])
                n = rt.decode_tokens(t)
                expected_q = _quant_notes(chart["notes"], rt)
                assert n == expected_q, (
                    f"Real data round-trip mismatch in {data['title']} "
                    f"[{chart['constant']}]"
                )
                # also check idempotency
                t2 = rt.encode_notes(n)
                assert t == t2
                print(
                    f"  ✓ {data['title']} [{chart['constant']}] "
                    f"({len(chart['notes'])} notes → {len(t)} tokens)"
                )
                passed += 1
            except AssertionError:
                print(
                    f"  ✗ FAILED: {data['title']} [{chart['constant']}]",
                    file=sys.stderr,
                )
                raise

    print(f"\n═══ All tokenizer tests passed ({passed} real charts) ═══")
