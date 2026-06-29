from __future__ import annotations

from typing import List, Optional, Set, Tuple

import torch

from src.tokenizer import (
    CC,
    DECO_BREAK,
    DECO_EX,
    DECO_FIREWORK,
    DECO_NONE,
    DIV,
    DUR_ABS,
    DUR_SYM,
    END_ONSET,
    EON,
    EOS,
    KIND_TOKENS,
    MAX_DELTA_BINS,
    MUL,
    ONSET,
    POS_TO_ID,
    SEG,
    SEG_END,
    SHAPES,
    SLIDE_DECO_BREAK,
    SLIDE_DECO_NONE,
    SOS,
    STAGE2_VOCAB_SIZE,
    TIME_OFFSET_BASE,
    VALUE_BASE,
    VOCAB_SIZE,
    WAIT,
    TOKEN_BIDICT,
    is_time_token,
    is_value_token,
)

# ── Position sets by note kind ────────────────────────────────────────────────
_POS_BTN = frozenset(POS_TO_ID[p] for p in POS_TO_ID if p.startswith("Btn"))
_POS_TOUCH = frozenset(POS_TO_ID[p] for p in POS_TO_ID if not p.startswith("Btn"))

_VALID_POS: dict[str, frozenset[int]] = {
    "Tap": _POS_BTN,
    "Hold": _POS_BTN,
    "Slide": _POS_BTN,
    "Touch": _POS_TOUCH,
    "TouchHold": _POS_TOUCH,
}

# ── Diagonal button pairs (1↔5, 2↔6, 3↔7, 4↔8) ───────────────────────────────
_DIAG_MAP: dict[str, str] = {
    "Btn1": "Btn5",
    "Btn2": "Btn6",
    "Btn3": "Btn7",
    "Btn4": "Btn8",
    "Btn5": "Btn1",
    "Btn6": "Btn2",
    "Btn7": "Btn3",
    "Btn8": "Btn4",
}

_DIAG_POS = frozenset(_DIAG_MAP)

# ── Decoration priority for sort-order enforcement ───────────────────────────
_DECO_PRIORITY = {
    DECO_BREAK: 0,
    DECO_EX: 1,
    DECO_FIREWORK: 2,
}

# ── All individual named token IDs (excluding ranges) ─────────────────────────
_ALL_NAMED: frozenset[int] = frozenset(
    [SOS, EOS, EON]
    + list(KIND_TOKENS.values())
    + list(POS_TO_ID.values())
    + [DECO_NONE, DECO_BREAK, DECO_EX, DECO_FIREWORK]
    + [SLIDE_DECO_NONE, SLIDE_DECO_BREAK]
    + [DUR_ABS, DUR_SYM, DIV, MUL]
    + [SEG, SEG_END, WAIT]
    + list(SHAPES.values())
)


# ═══════════════════════════════════════════════════════════════════════════════
# ChartValidator
# ═══════════════════════════════════════════════════════════════════════════════


class ChartValidator:
    """Finite‑state machine that enforces syntactically valid token streams.

    Use :meth:`valid_mask` to get a boolean mask for the next token,
    then call :meth:`advance` with the chosen token to advance state.

    Parameters
    ----------
    bpm10_list : list[dict]
        The chart's BPM list (used to compute default waits, so the
        validator knows when ``WAIT`` is optional).
    """

    def __init__(self, bpm10_list: Optional[List[dict]] = None) -> None:
        self._reset()
        # BPM for wait-default computation (lightweight – we only need
        # the BPM at note boundaries, not the full tokenizer).
        self._bpm10 = bpm10_list[0]["bpm10"] if bpm10_list else 1500

    # ── Public API ────────────────────────────────────────────────────────

    def valid_mask(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Return a ``(VOCAB_SIZE,)`` boolean mask.  ``True`` = valid next token."""
        mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)

        if self._phase == "DONE":
            return mask

        allowed: Set[int] = self._allowed_tokens()
        for tid in allowed:
            if isinstance(tid, int):
                mask[tid] = True
            elif isinstance(tid, tuple) and len(tid) == 2:
                # Numeric range: (min_id, max_id_exclusive)
                mask[tid[0] : tid[1]] = True
        return mask

    def advance(self, token_id: int) -> None:
        """Consume *token_id* and advance the FSM."""
        if self._phase == "DONE":
            raise ValueError("ChartValidator is already done (EOS received)")

        self._transition(token_id)

    def reset(self, bpm10_list: Optional[List[dict]] = None) -> None:
        """Reset to initial state for a new chart."""
        self._reset()
        if bpm10_list:
            self._bpm10 = bpm10_list[0]["bpm10"]

    # ── Internal state ─────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._phase = "EXPECT_SOS"
        self._note_kind: Optional[str] = None
        self._note_pos: Optional[str] = None
        self._prev_seg_end: Optional[str] = None  # cursor for chained segments
        self._decos_seen: Set[int] = set()
        self._last_deco_prio: int = -1
        self._wait_done: bool = False
        self._dur_done: bool = False
        self._seg_count: int = 0
        self._seg_last_shape: Optional[str] = None
        self._seg_has_wifi: bool = False
        self._in_slide_deco: bool = False
        self._slide_deco_seen: bool = False
        # Duration sub-state tracking
        self._dur_sub: int = 0
        self._dur_is_wait: bool = False

    def _begin_note(self) -> None:
        """Called after TIME is consumed; reset per‑note state."""
        self._note_kind = None
        self._note_pos = None
        self._prev_seg_end = None
        self._decos_seen.clear()
        self._last_deco_prio = -1
        self._wait_done = False
        self._dur_done = False
        self._seg_count = 0
        self._seg_last_shape = None
        self._seg_has_wifi = False
        self._in_slide_deco = False
        self._slide_deco_seen = False
        self._dur_sub = 0
        self._dur_is_wait = False

    # ── Allowed token computation ──────────────────────────────────────────

    def _allowed_tokens(self) -> Set[int]:
        phase = self._phase

        if phase == "EXPECT_SOS":
            return {SOS}

        if phase == "EXPECT_TIME":
            # Only TIME tokens (numeric range 75–5074)
            return {(TIME_OFFSET_BASE, TIME_OFFSET_BASE + 5000)}

        if phase == "EXPECT_KIND":
            return set(KIND_TOKENS.values())

        if phase == "EXPECT_POS":
            kind = self._note_kind
            if kind:
                return set(_VALID_POS.get(kind, _POS_BTN))
            return set(POS_TO_ID.values())

        if phase == "EXPECT_DECO":
            allowed: Set[int] = {DECO_NONE}
            for deco_id, prio in _DECO_PRIORITY.items():
                if deco_id not in self._decos_seen and prio >= self._last_deco_prio:
                    allowed.add(deco_id)
            # If we have seen at least one decoration, post-deco tokens
            # (WAIT / DUR / EON) can also follow directly (no DECO_NONE needed).
            if self._decos_seen:
                allowed.update(self._post_deco_allowed())
            return allowed

        if phase == "EXPECT_POST_DECO":
            return self._post_deco_allowed()

        if phase == "IN_DUR":
            return self._dur_allowed()

        if phase == "POST_DUR":
            if self._note_kind == "Slide":
                return {SEG}
            return {EON, EOS}

        if phase == "EXPECT_SEG_SHAPE":
            return set(SHAPES.values())

        if phase == "EXPECT_REFLECT_POS":
            return set(POS_TO_ID.values())

        if phase == "EXPECT_SEG_END_POS":
            shape = self._seg_last_shape
            # Wifi / ZigzagS / ZigzagZ must end at the diagonal opposite
            # of the cursor position (previous segment's end, or note start).
            if shape in ("Wifi", "ZigzagS", "ZigzagZ") and self._prev_seg_end in _DIAG_POS:
                diag_pos_name = _DIAG_MAP[self._prev_seg_end]
                return {POS_TO_ID[diag_pos_name]}
            return set(POS_TO_ID.values())

        if phase == "EXPECT_SEG_OR_END":
            # Wifi: no chaining
            if self._seg_has_wifi:
                return {SEG_END}
            return {SEG, SEG_END}

        if phase == "IN_SLIDE_DECO":
            allowed: Set[int] = {SLIDE_DECO_NONE}
            if not self._slide_deco_seen:
                allowed.add(SLIDE_DECO_BREAK)
            # If we've seen at least one slide deco, allow EON/EOS exit
            # without an explicit SLIDE_DECO_NONE.
            if self._slide_deco_seen:
                allowed.update({EON, EOS})
            return allowed

        if phase == "EXPECT_EON_EOS":
            return {EON, EOS}

        if phase == "IN_WAIT":
            if self._dur_sub == 0:
                return {DUR_ABS, DUR_SYM}
            return self._dur_allowed_impl()

        if phase == "IN_WAIT_DUR":
            return self._dur_allowed_impl()

        if phase == "POST_WAIT":
            kind = self._note_kind
            if kind in ("Hold", "Slide", "TouchHold"):
                return {DUR_ABS, DUR_SYM}
            return {EON, EOS}

        return set()

    def _post_deco_allowed(self) -> Set[int]:
        kind = self._note_kind
        if kind == "Slide":
            # Slide MUST have duration (and later segments) — no EON yet.
            allowed: Set[int] = set()
            if not self._wait_done:
                allowed.add(WAIT)
            if not self._dur_done:
                allowed.update({DUR_ABS, DUR_SYM})
            return allowed
        elif kind in ("Hold", "TouchHold"):
            # Hold / TouchHold MUST have duration — no EON without it.
            if self._dur_done:
                return {EON, EOS}
            return {DUR_ABS, DUR_SYM}
        else:  # Tap, Touch
            return {EON, EOS}

    def _dur_allowed(self) -> Set[int]:
        return self._dur_allowed_impl()

    def _dur_allowed_impl(self) -> Set[int]:
        sub = self._dur_sub
        # Entry: sub==0 means we haven't chosen DUR_ABS vs DUR_SYM yet.
        if sub == 0:
            return {DUR_ABS, DUR_SYM}

        choice = getattr(self, "_dur_choice", None)
        if choice == "abs":
            if sub == 1:
                return {(TIME_OFFSET_BASE, TIME_OFFSET_BASE + 5000)}
            return set()  # done — caller should have moved to next phase

        if choice == "sym":
            if sub == 1:
                return {DIV}
            elif sub == 2:
                return {(VALUE_BASE, VALUE_BASE + 8192)}
            elif sub == 3:
                return {MUL}
            elif sub == 4:
                return {(VALUE_BASE, VALUE_BASE + 8192)}
            return set()  # done

        return set()

    # ── Transition logic ───────────────────────────────────────────────────

    def _transition_post_deco(self, token_id: int) -> None:
        """Handle WAIT / DUR / EON after decorations are done."""
        if token_id == WAIT:
            self._wait_done = True
            self._dur_sub = 0
            self._dur_is_wait = True
            self._phase = "IN_WAIT"
            return
        if token_id in (DUR_ABS, DUR_SYM):
            self._dur_done = True
            self._dur_sub = 1
            self._dur_choice = "abs" if token_id == DUR_ABS else "sym"
            self._dur_is_wait = False
            self._phase = "IN_DUR"
            return
        if token_id in (EON, EOS):
            self._phase = "DONE" if token_id == EOS else "EXPECT_TIME"
            return
        raise ValueError(
            f"Expected WAIT/DUR/EON/EOS in post-deco, got {token_id}"
        )

    def _transition(self, token_id: int) -> None:
        phase = self._phase

        # ── EXPECT_SOS ────────────────────────────────────────────────
        if phase == "EXPECT_SOS":
            if token_id == SOS:
                self._phase = "EXPECT_TIME"
                return
            raise ValueError(f"Expected SOS, got {token_id}")

        # ── EXPECT_TIME ───────────────────────────────────────────────
        if phase == "EXPECT_TIME":
            if is_time_token(token_id):
                self._begin_note()
                self._phase = "EXPECT_KIND"
                return
            raise ValueError(f"Expected TIME token, got {token_id}")

        # ── EXPECT_KIND ───────────────────────────────────────────────
        if phase == "EXPECT_KIND":
            kind_name = {v: k for k, v in KIND_TOKENS.items()}.get(token_id)
            if kind_name is None:
                raise ValueError(f"Expected KIND token, got {token_id}")
            self._note_kind = kind_name
            self._phase = "EXPECT_POS"
            return

        # ── EXPECT_POS ───────────────────────────────────────────────
        if phase == "EXPECT_POS":
            pos_name = {v: k for k, v in POS_TO_ID.items()}.get(token_id)
            if pos_name is None:
                raise ValueError(f"Expected POS token, got {token_id}")
            self._note_pos = pos_name
            self._prev_seg_end = pos_name  # first segment starts at note position
            self._phase = "EXPECT_DECO"
            return

        # ── EXPECT_DECO ───────────────────────────────────────────────
        if phase == "EXPECT_DECO":
            if token_id == DECO_NONE:
                self._decos_seen.clear()
                self._last_deco_prio = -1
                self._phase = "EXPECT_POST_DECO"
                return
            if token_id in _DECO_PRIORITY:
                prio = _DECO_PRIORITY[token_id]
                if token_id in self._decos_seen:
                    raise ValueError(f"Duplicate decoration: {token_id}")
                if prio < self._last_deco_prio:
                    raise ValueError(
                        f"Decoration sort-order violation: "
                        f"received priority {prio} after {self._last_deco_prio}"
                    )
                self._decos_seen.add(token_id)
                self._last_deco_prio = prio
                # stay in EXPECT_DECO for more
                return
            # Post-deco exit: if we already have decorations, allow
            # WAIT / DUR / EON without an explicit DECO_NONE.
            if self._decos_seen:
                return self._transition_post_deco(token_id)
            raise ValueError(f"Expected DECO token, got {token_id}")

        # ── EXPECT_POST_DECO ──────────────────────────────────────────
        if phase == "EXPECT_POST_DECO":
            return self._transition_post_deco(token_id)

        # ── IN_WAIT ───────────────────────────────────────────────────
        if phase == "IN_WAIT":
            if token_id in (DUR_ABS, DUR_SYM):
                self._dur_sub = 1
                self._dur_choice = "abs" if token_id == DUR_ABS else "sym"
                self._phase = "IN_WAIT_DUR"
                return
            raise ValueError(f"Expected DUR_ABS/DUR_SYM after WAIT, got {token_id}")

        # ── IN_WAIT_DUR ───────────────────────────────────────────────
        if phase == "IN_WAIT_DUR":
            self._advance_dur(token_id)
            if self._dur_sub >= 5:
                # Wait duration done → back to post-deco for actual note duration
                self._phase = "EXPECT_POST_DECO"
            return

        # ── IN_DUR ────────────────────────────────────────────────────
        if phase == "IN_DUR":
            self._advance_dur(token_id)
            if self._dur_sub >= 5:
                self._phase = "POST_DUR"
            return

        # ── POST_DUR ──────────────────────────────────────────────────
        if phase == "POST_DUR":
            if token_id == SEG:
                self._seg_count += 1
                self._phase = "EXPECT_SEG_SHAPE"
                return
            if token_id in (EON, EOS):
                self._phase = "DONE" if token_id == EOS else "EXPECT_TIME"
                return
            raise ValueError(
                f"Expected SEG/EON/EOS after duration, got {token_id}"
            )

        # ── EXPECT_SEG_SHAPE ──────────────────────────────────────────
        if phase == "EXPECT_SEG_SHAPE":
            shape = {v: k for k, v in SHAPES.items()}.get(token_id)
            if shape is None:
                raise ValueError(f"Expected SHAPE token, got {token_id}")
            self._seg_last_shape = shape
            if shape == "Wifi":
                self._seg_has_wifi = True
            if shape == "Reflect":
                self._phase = "EXPECT_REFLECT_POS"
            else:
                self._phase = "EXPECT_SEG_END_POS"
            return

        # ── EXPECT_REFLECT_POS ────────────────────────────────────────
        if phase == "EXPECT_REFLECT_POS":
            if token_id not in POS_TO_ID.values():
                raise ValueError(f"Expected POS for Reflect turn, got {token_id}")
            self._phase = "EXPECT_SEG_END_POS"
            return

        # ── EXPECT_SEG_END_POS ────────────────────────────────────────
        if phase == "EXPECT_SEG_END_POS":
            if token_id not in POS_TO_ID.values():
                raise ValueError(
                    f"Expected end POS for segment, got {token_id}"
                )
            end_pos_name = {v: k for k, v in POS_TO_ID.items()}[token_id]
            # Validate diagonal constraint for Wifi / ZigzagS / ZigzagZ.
            shape = self._seg_last_shape
            if shape in ("Wifi", "ZigzagS", "ZigzagZ"):
                expected_diag = _DIAG_MAP.get(self._prev_seg_end or "")
                if end_pos_name != expected_diag:
                    raise ValueError(
                        f"{shape} end {end_pos_name} must be diagonal "
                        f"to cursor {self._prev_seg_end} (expected {expected_diag})"
                    )
            # Advance cursor for chained segments
            self._prev_seg_end = end_pos_name
            self._phase = "EXPECT_SEG_OR_END"
            return

        # ── EXPECT_SEG_OR_END ─────────────────────────────────────────
        if phase == "EXPECT_SEG_OR_END":
            if token_id == SEG:
                if self._seg_has_wifi:
                    raise ValueError("Wifi slides cannot be chained")
                self._seg_count += 1
                self._phase = "EXPECT_SEG_SHAPE"
                return
            if token_id == SEG_END:
                self._phase = "IN_SLIDE_DECO"
                return
            raise ValueError(f"Expected SEG or SEG_END, got {token_id}")

        # ── IN_SLIDE_DECO ─────────────────────────────────────────────
        if phase == "IN_SLIDE_DECO":
            if token_id == SLIDE_DECO_BREAK:
                self._slide_deco_seen = True
                self._phase = "IN_SLIDE_DECO"  # stay, allow more Break
                return
            if token_id == SLIDE_DECO_NONE:
                self._slide_deco_seen = False
                self._phase = "EXPECT_EON_EOS"
                return
            if self._slide_deco_seen and token_id in (EON, EOS):
                self._phase = "DONE" if token_id == EOS else "EXPECT_TIME"
                return
            raise ValueError(
                f"Expected SLIDE_DECO token, got {token_id}"
            )

        # ── EXPECT_EON_EOS ────────────────────────────────────────────
        if phase == "EXPECT_EON_EOS":
            if token_id == EON:
                self._phase = "EXPECT_TIME"
                return
            if token_id == EOS:
                self._phase = "DONE"
                return
            raise ValueError(f"Expected EON or EOS, got {token_id}")

        raise ValueError(
            f"Unexpected token {token_id} in phase {self._phase}"
        )

    def _advance_dur(self, token_id: int) -> None:
        """Advance the IN_DUR / IN_WAIT_DUR sub-state machine."""
        choice = self._dur_choice
        sub = self._dur_sub

        if choice == "abs":
            if sub == 1 and is_time_token(token_id):
                self._dur_sub = 5  # done
                return
            raise ValueError(f"Expected TIME after DUR_ABS, got {token_id}")

        if choice == "sym":
            if sub == 1 and token_id == DIV:
                self._dur_sub = 2
                return
            if sub == 2 and is_value_token(token_id):
                self._dur_sub = 3
                return
            if sub == 3 and token_id == MUL:
                self._dur_sub = 4
                return
            if sub == 4 and is_value_token(token_id):
                self._dur_sub = 5  # done
                return
            raise ValueError(
                f"Unexpected token {token_id} in sym-dur sub-state {sub}"
            )

        raise ValueError(f"Unknown dur choice: {choice}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage2Validator — FSM for block-format token sequences
# ═══════════════════════════════════════════════════════════════════════════════


class Stage2Validator:
    """Finite‑state machine for stage‑2 block‑format token validation.

    Handles the ``<SOS> <CC> <ONSET> … <END_ONSET> … <EOS>`` format.
    Note‑level validation within onset blocks is delegated to an internal
    ``ChartValidator`` (skipping time‑offset phases).
    """

    # Phase constants (subset of ChartValidator phases, re‑used)
    _PH_WAITS_NOTE_START = "waits_note_start"
    _PH_WAITS_KIND = "waits_kind"
    _PH_WAITS_POS = "waits_pos"
    _PH_WAITS_DECO = "waits_deco"
    _PH_WAITS_WAIT = "waits_wait"
    _PH_WAITS_DUR = "waits_dur"
    _PH_WAITS_SLIDE = "waits_slide"
    _PH_WAITS_SLIDE_DECO = "waits_slide_deco"
    _PH_WAITS_EON = "waits_eon"

    _BLOCK_PHASES = {
        _PH_WAITS_NOTE_START,
        _PH_WAITS_KIND,
        _PH_WAITS_POS,
        _PH_WAITS_DECO,
        _PH_WAITS_WAIT,
        _PH_WAITS_DUR,
        _PH_WAITS_SLIDE,
        _PH_WAITS_SLIDE_DECO,
        _PH_WAITS_EON,
    }

    def __init__(self, bpm10_list: Optional[List[dict]] = None) -> None:
        self._bpm10_list = bpm10_list
        self._reset()

    def _reset(self) -> None:
        self._phase = "waits_sos"  # top-level: waits_sos | waits_cc | waits_onset_or_eos | done
        self._kind_id: Optional[int] = None
        self._note_pos: Optional[str] = None
        self._note_decos: List[int] = []
        self._wait_emitted = False
        self._dur_started = False
        self._dur_was_wait = False  # True when the current dur is for a WAIT
        self._dur_choice: Optional[int] = None
        self._dur_sub = 0
        self._slide_seg_idx = 0
        self._slide_first_seg_end: Optional[str] = None
        self._slide_first_seg_shape: Optional[str] = None
        self._prev_seg_end: Optional[str] = None
        self._seg_is_reflect = False

    # ── valid_mask ───────────────────────────────────────────────────────

    def valid_mask(self, device: Optional["torch.device"] = None) -> "torch.Tensor":
        """Return boolean tensor of shape ``(STAGE2_VOCAB_SIZE,)``."""
        import torch

        mask = torch.zeros(STAGE2_VOCAB_SIZE, dtype=torch.bool, device=device or "cpu")

        if self._phase == "waits_sos":
            mask[SOS] = True
        elif self._phase == "waits_cc":
            mask[CC] = True
        elif self._phase == "waits_onset_or_eos":
            mask[ONSET] = True
            mask[EOS] = True
        elif self._phase == "done":
            pass
        elif self._phase in self._BLOCK_PHASES:
            self._fill_note_mask(mask)
        return mask

    def _fill_note_mask(self, mask: "torch.Tensor") -> None:
        """Fill *mask* for the current note‑parsing phase."""
        if self._phase == self._PH_WAITS_NOTE_START:
            # Start of a new note in current onset block: must be a kind
            for k_id in KIND_TOKENS.values():
                mask[k_id] = True
            mask[END_ONSET] = True  # block can end with zero notes
        elif self._phase == self._PH_WAITS_KIND:
            raise RuntimeError("waits_kind is merged into waits_note_start")
        elif self._phase == self._PH_WAITS_POS:
            assert self._kind_id is not None
            valid_pos = _VALID_POS[_inv_name_from_kind(self._kind_id)]
            for p_id in valid_pos:
                mask[p_id] = True
        elif self._phase == self._PH_WAITS_DECO:
            mask[DECO_BREAK] = True
            mask[DECO_EX] = True
            mask[DECO_FIREWORK] = True
            # Can stop decorations here → proceed to next phase
            mask[DECO_NONE] = True
            # Allow shortcuts (skip deco phase entirely)
            self._fill_post_deco_mask(mask)
        elif self._phase == self._PH_WAITS_WAIT:
            mask[WAIT] = True
            self._fill_post_wait_mask(mask)
        elif self._phase == self._PH_WAITS_DUR:
            if self._dur_choice is None:
                mask[DUR_ABS] = True
                mask[DUR_SYM] = True
            elif self._dur_choice == DUR_ABS:
                mask[TIME_OFFSET_BASE:TIME_OFFSET_BASE + MAX_DELTA_BINS] = True
            elif self._dur_choice == DUR_SYM:
                if self._dur_sub == 0:
                    mask[DIV] = True
                elif self._dur_sub == 1:
                    mask[VALUE_BASE:VALUE_BASE + MAX_VALUE] = True
                elif self._dur_sub == 2:
                    mask[MUL] = True
                elif self._dur_sub == 3:
                    mask[VALUE_BASE:VALUE_BASE + MAX_VALUE] = True
                elif self._dur_sub in (4, 5):
                    self._fill_post_dur_mask(mask)
            elif self._dur_choice is not None:
                self._fill_post_dur_mask(mask)
        elif self._phase == self._PH_WAITS_SLIDE:
            mask[SEG] = True
            if self._slide_seg_idx == 0:
                for s_id in SHAPES.values():
                    mask[s_id] = True
            else:
                # subsequent segments: any shape that can chain
                prev_shape = self._slide_first_seg_shape or ""
                for s_id in SHAPES.values():
                    shape_name = _SHAPE_INV[s_id]
                    if shape_name == "Wifi":
                        continue  # Wifi can't appear after the first segment
                    mask[s_id] = True
            mask[SEG_END] = True
        elif self._phase == self._PH_WAITS_SLIDE_DECO:
            mask[SLIDE_DECO_BREAK] = True
            mask[SLIDE_DECO_NONE] = True
            self._fill_post_slide_deco_mask(mask)
        elif self._phase == self._PH_WAITS_EON:
            mask[EON] = True
            mask[END_ONSET] = True
            mask[EOS] = True

    def _fill_post_deco_mask(self, mask: "torch.Tensor") -> None:
        kind_name = _inv_name_from_kind(self._kind_id) if self._kind_id else ""
        if kind_name in ("Hold", "Slide", "TouchHold"):
            mask[DUR_ABS] = True
            mask[DUR_SYM] = True
        if kind_name == "Slide":
            mask[WAIT] = True
            for s_id in SHAPES.values():
                mask[s_id] = True
        mask[EON] = True
        mask[END_ONSET] = True

    def _fill_post_wait_mask(self, mask: "torch.Tensor") -> None:
        kind_name = _inv_name_from_kind(self._kind_id) if self._kind_id else ""
        if kind_name in ("Hold", "Slide", "TouchHold"):
            mask[DUR_ABS] = True
            mask[DUR_SYM] = True
        if kind_name == "Slide":
            for s_id in SHAPES.values():
                mask[s_id] = True
        mask[EON] = True
        mask[END_ONSET] = True

    def _fill_post_dur_mask(self, mask: "torch.Tensor") -> None:
        kind_name = _inv_name_from_kind(self._kind_id) if self._kind_id else ""
        if kind_name == "Slide":
            for s_id in SHAPES.values():
                mask[s_id] = True
        mask[EON] = True
        mask[END_ONSET] = True

    def _fill_post_slide_deco_mask(self, mask: "torch.Tensor") -> None:
        mask[EON] = True
        mask[END_ONSET] = True

    # ── advance ──────────────────────────────────────────────────────────

    def advance(self, token_id: int) -> None:
        if self._phase == "waits_sos":
            if token_id != SOS:
                raise ValueError(f"Expected SOS at start, got {token_id}")
            self._phase = "waits_cc"
        elif self._phase == "waits_cc":
            if token_id != CC:
                raise ValueError(f"Expected CC after SOS, got {token_id}")
            self._phase = "waits_onset_or_eos"
        elif self._phase == "waits_onset_or_eos":
            if token_id == EOS:
                self._phase = "done"
            elif token_id == ONSET:
                self._phase = self._PH_WAITS_NOTE_START
                self._reset_note_state()
            else:
                raise ValueError(f"Expected ONSET or EOS, got {token_id}")
        elif self._phase == "done":
            raise ValueError("Token after EOS")
        else:
            self._advance_note(token_id)

    def _reset_note_state(self) -> None:
        self._kind_id = None
        self._note_pos = None
        self._note_decos = []
        self._wait_emitted = False
        self._dur_started = False
        self._dur_was_wait = False
        self._dur_choice = None
        self._dur_sub = 0
        self._slide_seg_idx = 0
        self._slide_first_seg_end = None
        self._slide_first_seg_shape = None
        self._prev_seg_end = None
        self._seg_is_reflect = False

    def _advance_note(self, token_id: int) -> None:
        # EON / END_ONSET / EOS can terminate a note at almost any phase
        if token_id == EON:
            self._phase = self._PH_WAITS_NOTE_START
            self._reset_note_state()
            return
        if token_id == END_ONSET:
            self._phase = "waits_onset_or_eos"
            self._reset_note_state()
            return
        if token_id == EOS:
            self._phase = "done"
            return

        if self._phase == self._PH_WAITS_NOTE_START:
            if token_id == END_ONSET:
                self._phase = "waits_onset_or_eos"
                return
            if token_id not in KIND_TOKENS.values():
                raise ValueError(
                    f"Expected note kind or END_ONSET, got {token_id}"
                )
            self._kind_id = token_id
            self._phase = self._PH_WAITS_POS
        elif self._phase == self._PH_WAITS_POS:
            kind_name = _inv_name_from_kind(self._kind_id)
            valid = _VALID_POS[kind_name]
            if token_id not in valid:
                raise ValueError(
                    f"Invalid position {token_id} for {kind_name}"
                )
            self._note_pos = TOKEN_BIDICT.inv[token_id].removeprefix("POS_")
            self._phase = self._PH_WAITS_DECO
        elif self._phase == self._PH_WAITS_DECO:
            if token_id == DECO_NONE:
                self._phase = self._PH_WAITS_WAIT
            elif token_id in _DECO_PRIORITY:
                self._note_decos.append(token_id)
                # stay in DECO phase (may have more decos)
            elif token_id == EON or token_id == END_ONSET:
                self._phase = self._PH_WAITS_WAIT
                self.advance(token_id)  # re‑enter to handle
            elif token_id in (DUR_ABS, DUR_SYM):
                self._phase = self._PH_WAITS_WAIT
                self.advance(token_id)
            elif token_id == WAIT:
                self._phase = self._PH_WAITS_WAIT
                self.advance(token_id)
            elif token_id in SHAPES.values():
                self._phase = self._PH_WAITS_WAIT
                self.advance(token_id)
            else:
                raise ValueError(
                    f"Unexpected token {token_id} in deco phase"
                )
        elif self._phase == self._PH_WAITS_WAIT:
            kind_name = _inv_name_from_kind(self._kind_id)
            if token_id == WAIT and not self._wait_emitted:
                self._wait_emitted = True
                self._dur_was_wait = True
                self._dur_choice = None
                self._dur_sub = 0
                self._phase = self._PH_WAITS_DUR
            elif token_id in (DUR_ABS, DUR_SYM):
                self._phase = self._PH_WAITS_DUR
                self.advance(token_id)
            elif token_id in SHAPES.values():
                self._phase = self._PH_WAITS_SLIDE
                self.advance(token_id)
            elif token_id == EON or token_id == END_ONSET:
                self._phase = "waits_onset_or_eos"
                # If END_ONSET: done with this onset block
                # If EON: back to waits_note_start for next note in block
                if token_id == EON:
                    self._phase = self._PH_WAITS_NOTE_START
                # When END_ONSET → back to waits_onset_or_eos
            elif kind_name not in ("Hold", "Slide", "TouchHold"):
                raise ValueError(
                    f"Unexpected token {token_id} in wait phase for {kind_name}"
                )
            else:
                raise ValueError(
                    f"Unexpected token {token_id} in wait phase"
                )
        elif self._phase == self._PH_WAITS_DUR:
            if self._dur_choice is None:
                if token_id in (DUR_ABS, DUR_SYM):
                    self._dur_choice = token_id
                    self._dur_sub = 0
                else:
                    raise ValueError(
                        f"Expected DUR_ABS or DUR_SYM, got {token_id}"
                    )
            elif self._dur_choice == DUR_ABS:
                if is_time_token(token_id):
                    if self._dur_was_wait:
                        self._dur_was_wait = False
                        self._dur_choice = None
                        self._dur_sub = 0
                        self._phase = self._PH_WAITS_WAIT
                    else:
                        self._phase = self._PH_WAITS_SLIDE
                else:
                    raise ValueError(
                        f"Expected time token, got {token_id}"
                    )
            elif self._dur_choice == DUR_SYM:
                if self._dur_sub == 0:
                    if token_id == DIV:
                        self._dur_sub = 1
                    else:
                        raise ValueError(f"Expected DIV, got {token_id}")
                elif self._dur_sub == 1:
                    if is_value_token(token_id):
                        self._dur_sub = 2
                    else:
                        raise ValueError(
                            f"Expected value token, got {token_id}"
                        )
                elif self._dur_sub == 2:
                    if token_id == MUL:
                        self._dur_sub = 3
                    else:
                        raise ValueError(f"Expected MUL, got {token_id}")
                elif self._dur_sub == 3:
                    if is_value_token(token_id):
                        self._dur_sub = 4
                        if self._dur_was_wait:
                            self._dur_was_wait = False
                            self._dur_choice = None
                            self._dur_sub = 0
                            self._phase = self._PH_WAITS_WAIT
                        else:
                            self._phase = self._PH_WAITS_SLIDE
                    else:
                        raise ValueError(
                            f"Expected value token, got {token_id}"
                        )
        elif self._phase == self._PH_WAITS_SLIDE:
            if token_id == SEG_END:
                self._phase = self._PH_WAITS_SLIDE_DECO
            elif token_id == SEG:
                # Start of a new segment — stay in slide phase, next token is shape
                pass
            elif token_id in SHAPES.values():
                shape = _SHAPE_INV[token_id]
                if shape == "Wifi" and self._slide_seg_idx > 0:
                    raise ValueError("Wifi can only be the first segment")
                self._slide_seg_idx += 1
                if self._slide_seg_idx == 1:
                    self._slide_first_seg_shape = shape
                # Seg parsing continues in waits_slide — the seg start is done
                # Now wait for position token (seg end or reflect position)
                self._seg_is_reflect = (shape == "Reflect")
                if self._seg_is_reflect:
                    # stay in slide phase, next is reflect_pos
                    pass
                else:
                    # stay in slide phase, next is end_pos
                    pass
            elif token_id in POS_TO_ID.values():
                if self._seg_is_reflect and self._prev_seg_end is None:
                    # This is the reflect position, not the end
                    self._seg_is_reflect = False
                    # Next token should be the end position
                else:
                    pos_name = TOKEN_BIDICT.inv[token_id].removeprefix("POS_")
                    self._prev_seg_end = pos_name
                    if self._slide_seg_idx == 1:
                        self._slide_first_seg_end = pos_name
                    self._seg_is_reflect = False
            else:
                raise ValueError(
                    f"Unexpected token {token_id} in slide phase"
                )
        elif self._phase == self._PH_WAITS_SLIDE_DECO:
            if token_id == SLIDE_DECO_NONE:
                self._phase = self._PH_WAITS_EON
            elif token_id == SLIDE_DECO_BREAK:
                self._phase = self._PH_WAITS_EON
            elif token_id == EON:
                self._phase = self._PH_WAITS_NOTE_START
            elif token_id == END_ONSET:
                self._phase = "waits_onset_or_eos"
            else:
                raise ValueError(
                    f"Unexpected token {token_id} in slide deco phase"
                )
        elif self._phase == self._PH_WAITS_EON:
            if token_id == EON:
                self._phase = self._PH_WAITS_NOTE_START
            elif token_id == END_ONSET:
                self._phase = "waits_onset_or_eos"
            elif token_id == EOS:
                self._phase = "done"
            else:
                raise ValueError(
                    f"Expected EON, END_ONSET, or EOS, got {token_id}"
                )


def _inv_name_from_kind(kind_id: int) -> str:
    for k, v in KIND_TOKENS.items():
        if v == kind_id:
            return k
    raise ValueError(f"Unknown kind token: {kind_id}")


_SHAPE_INV = {v: k for k, v in SHAPES.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import glob
    from src.tokenizer import ChartTokenizer

    # ── Test 1: valid chart passes validation ──────────────────────────
    notes = [
        {
            "timestamp_ms": 500,
            "kind": "Tap",
            "pos": "Btn1",
            "deco": [],
            "wait": None,
            "duration": None,
            "slide_segments": [],
            "slide_deco": [],
        },
        {
            "timestamp_ms": 1500,
            "kind": "Slide",
            "pos": "Btn1",
            "deco": ["Ex"],
            "wait": None,
            "duration": {"DividerMultiplier": [2, 1]},
            "slide_segments": [
                {"shape": "Straight", "end": "Btn4"},
                {"shape": "ArcRight", "end": "Btn8"},
            ],
            "slide_deco": [],
        },
    ]
    bpm_list = [{"bpm10": 1500, "change_timestamp_ms": 0}]
    tokenizer = ChartTokenizer(bpm_list)
    tokens = tokenizer.encode_notes(notes)
    validator = ChartValidator(bpm_list)

    for tok in tokens[:-1]:  # all except EOS
        mask = validator.valid_mask()
        assert mask[tok], f"Validator rejected token {tok} in phase {validator._phase}"
        validator.advance(tok)
    # Last token should be EOS
    mask = validator.valid_mask()
    assert mask[EOS], "Validator should accept EOS at end"
    validator.advance(EOS)
    assert validator._phase == "DONE"
    print("✓ Test 1 passed: valid chart validates")

    # ── Test 2: Wifi diagonal constraint ───────────────────────────────
    notes_wifi = [
        {
            "timestamp_ms": 500,
            "kind": "Slide",
            "pos": "Btn1",
            "deco": [],
            "wait": None,
            "duration": {"DividerMultiplier": [1, 1]},
            "slide_segments": [{"shape": "Wifi", "end": "Btn5"}],
            "slide_deco": [],
        },
    ]
    t_wifi = tokenizer.encode_notes(notes_wifi)
    v = ChartValidator(bpm_list)
    for tok in t_wifi:
        v.advance(tok)
    print("✓ Test 2 passed: Wifi Btn1→Btn5 validates")

    # ── Test 3: Wifi wrong end → reject ────────────────────────────────
    notes_wifi_bad = [
        {
            "timestamp_ms": 500,
            "kind": "Slide",
            "pos": "Btn1",
            "deco": [],
            "wait": None,
            "duration": {"DividerMultiplier": [1, 1]},
            "slide_segments": [{"shape": "Wifi", "end": "Btn3"}],
            "slide_deco": [],
        },
    ]
    t_wifi_bad = tokenizer.encode_notes(notes_wifi_bad)
    v = ChartValidator(bpm_list)
    rejected = False
    try:
        for tok in t_wifi_bad:
            v.advance(tok)
    except ValueError:
        rejected = True
    assert rejected, "Wifi Btn1→Btn3 should be rejected"
    print("✓ Test 3 passed: Wifi bad diagonal rejected")

    # ── Test 4: valid_mask shape ──────────────────────────────────────
    v = ChartValidator(bpm_list)
    mask = v.valid_mask()
    assert mask.shape == (VOCAB_SIZE,)
    assert mask.sum() > 0
    # At start, only SOS allowed
    assert mask[SOS]
    assert not mask[DECO_BREAK]
    v.advance(SOS)
    # After SOS, TIME tokens allowed
    mask = v.valid_mask()
    assert mask[TIME_OFFSET_BASE : TIME_OFFSET_BASE + 10].any()
    print(f"✓ Test 4 passed: valid_mask shape={mask.shape}, {mask.sum().item()} TIME tokens after SOS")

    # ── Test 5: ZigzagS diagonal constraint ────────────────────────────
    notes_z = [
        {
            "timestamp_ms": 500,
            "kind": "Slide",
            "pos": "Btn3",
            "deco": [],
            "wait": None,
            "duration": {"DividerMultiplier": [1, 1]},
            "slide_segments": [{"shape": "ZigzagS", "end": "Btn7"}],
            "slide_deco": [],
        },
    ]
    t_z = tokenizer.encode_notes(notes_z)
    v = ChartValidator(bpm_list)
    for tok in t_z:
        mask = v.valid_mask()
        assert mask[tok], f"ZigzagS Btn3→Btn7 rejected token {tok}"
        v.advance(tok)
    print("✓ Test 5 passed: ZigzagS diagonal validates")

    # ── Test 6: chained zigzag (Aegleseeker-style) ──────────────────────
    notes_chain = [
        {
            "timestamp_ms": 500,
            "kind": "Slide",
            "pos": "Btn1",
            "deco": [],
            "wait": None,
            "duration": {"DividerMultiplier": [1, 1]},
            "slide_segments": [
                {"shape": "ZigzagS", "end": "Btn5"},
                {"shape": "ZigzagS", "end": "Btn1"},
            ],
            "slide_deco": [],
        },
    ]
    t_chain = tokenizer.encode_notes(notes_chain)
    v = ChartValidator(bpm_list)
    for tok in t_chain:
        mask = v.valid_mask()
        assert mask[tok], (
            f"Chained zigzag Btn1→Btn5→Btn1 rejected token {tok} "
            f"at phase {v._phase}"
        )
        v.advance(tok)
    print("✓ Test 6 passed: chained zigzag Btn1→Btn5→Btn1 validates")

    # ── Test 7: non-diagonal zigzag still rejected ─────────────────────
    notes_bad_z = [
        {
            "timestamp_ms": 500,
            "kind": "Slide",
            "pos": "Btn1",
            "deco": [],
            "wait": None,
            "duration": {"DividerMultiplier": [1, 1]},
            "slide_segments": [{"shape": "ZigzagS", "end": "Btn3"}],
            "slide_deco": [],
        },
    ]
    t_bad_z = tokenizer.encode_notes(notes_bad_z)
    v = ChartValidator(bpm_list)
    rejected = False
    try:
        for tok in t_bad_z:
            v.advance(tok)
    except ValueError:
        rejected = True
    assert rejected, "ZigzagS Btn1→Btn3 (non-diagonal) should be rejected"
    print("✓ Test 7 passed: non-diagonal ZigzagS Btn1→Btn3 rejected")

    # ── Test 8: chained zigzag second segment non-diagonal → reject ────
    notes_bad_chain = [
        {
            "timestamp_ms": 500,
            "kind": "Slide",
            "pos": "Btn1",
            "deco": [],
            "wait": None,
            "duration": {"DividerMultiplier": [1, 1]},
            "slide_segments": [
                {"shape": "ZigzagS", "end": "Btn5"},   # Btn1→Btn5 ✓
                {"shape": "ZigzagS", "end": "Btn2"},   # cursor Btn5→Btn2 ✗ (not diag)
            ],
            "slide_deco": [],
        },
    ]
    t_bad_chain = tokenizer.encode_notes(notes_bad_chain)
    v = ChartValidator(bpm_list)
    rejected = False
    try:
        for tok in t_bad_chain:
            v.advance(tok)
    except ValueError:
        rejected = True
    assert rejected, (
        "Chained zigzag Btn1→Btn5→Btn2 (2nd segment non-diagonal) should be rejected"
    )
    print("✓ Test 8 passed: chained zigzag with bad 2nd segment rejected")

    # ── Test 9: chained Wifi rejected (Wifi can't chain) ──────────────
    notes_wifi_chain = [
        {
            "timestamp_ms": 500,
            "kind": "Slide",
            "pos": "Btn1",
            "deco": [],
            "wait": None,
            "duration": {"DividerMultiplier": [1, 1]},
            "slide_segments": [
                {"shape": "Wifi", "end": "Btn5"},
                {"shape": "Straight", "end": "Btn8"},
            ],
            "slide_deco": [],
        },
    ]
    t_wifi_chain = tokenizer.encode_notes(notes_wifi_chain)
    v = ChartValidator(bpm_list)
    rejected = False
    try:
        for tok in t_wifi_chain:
            v.advance(tok)
    except ValueError:
        rejected = True
    assert rejected, "Wifi chained with another segment should be rejected"
    print("✓ Test 9 passed: chained Wifi rejected")

    # ── Test 10: validate all charts in the dataset ─────────────────────
    import sys
    from tqdm import tqdm

    all_json = sorted(glob.glob("dataset/**/processed.json", recursive=True))
    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []

    for path in tqdm(all_json, desc="Validating charts"):
        try:
            with open(path) as fh:
                song = json.load(fh)
        except Exception:
            continue
        for chart in song["charts"]:
            if not chart["notes"]:
                continue
            try:
                tokenizer = ChartTokenizer(chart["bpm10_list"])
                tokens = tokenizer.encode_notes(chart["notes"])
                v = ChartValidator(chart["bpm10_list"])
                for tok in tokens:
                    v.advance(tok)
            except ValueError as exc:
                failed += 1
                failures.append(
                    (f"{song['title']} [{chart['constant']}]", str(exc))
                )
                break
            except Exception:
                failed += 1
                failures.append(
                    (f"{song['title']} [{chart['constant']}]", "unexpected error")
                )
                break
            else:
                passed += 1

    print(f"  Passed: {passed}, Failed: {failed}")
    if failures:
        print("  First 10 failures:")
        for title, err in failures[:10]:
            print(f"    {title}: {err}")
    assert failed == 0, f"{failed} charts failed validation"
    print("✓ Test 10 passed: all dataset charts validated")

    print("\n═══ All validator tests passed ═══")
