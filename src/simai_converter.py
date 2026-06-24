"""Convert processed.json (custom format) back to simai maidata.txt format."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


# ── Position mapping ─────────────────────────────────────────────────────────
POS_TO_SIMAI: Dict[str, str] = {
    "Btn1": "1", "Btn2": "2", "Btn3": "3", "Btn4": "4",
    "Btn5": "5", "Btn6": "6", "Btn7": "7", "Btn8": "8",
    "A1": "A1", "A2": "A2", "A3": "A3", "A4": "A4",
    "A5": "A5", "A6": "A6", "A7": "A7", "A8": "A8",
    "B1": "B1", "B2": "B2", "B3": "B3", "B4": "B4",
    "B5": "B5", "B6": "B6", "B7": "B7", "B8": "B8",
    "C": "C1",
    "D1": "D1", "D2": "D2", "D3": "D3", "D4": "D4",
    "D5": "D5", "D6": "D6", "D7": "D7", "D8": "D8",
    "E1": "E1", "E2": "E2", "E3": "E3", "E4": "E4",
    "E5": "E5", "E6": "E6", "E7": "E7", "E8": "E8",
}

DECO_TO_CHAR: Dict[str, str] = {
    "Break": "b",
    "Ex": "x",
    "Firework": "f",
}

SHAPE_TO_CHAR: Dict[str, str] = {
    "Straight": "-",
    "ArcLeft": "<",
    "ArcRight": ">",
    "Center": "v",
    "ZigzagS": "s",
    "ZigzagZ": "z",
    "P": "p",
    "Q": "q",
    "PP": "pp",
    "QQ": "qq",
    "Wifi": "w",
}

SLIDE_DECO_TO_CHAR: Dict[str, str] = {
    "Break": "b",
}

WIFI_ENDPOINT_SEQ: Dict[str, str] = {
    "1": "456", "2": "567", "3": "678", "4": "781",
    "5": "812", "6": "123", "7": "234", "8": "345",
}

# All dividers found in the dataset, sorted ascending.
# Used by _find_best_denominator to try candidates.
_DIVIDER_CANDIDATES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384]

_EPSILON = 1e-6


def _pos_to_simai(pos: str) -> str:
    return POS_TO_SIMAI.get(pos, pos)


def _btn_digit(pos: str) -> str:
    """Extract the digit from a button position like 'Btn3' → '3'."""
    if pos.startswith("Btn"):
        return pos[3:]
    return pos


def _find_best_denominator(diff_beat: float) -> Tuple[int, int]:
    """Find the smallest denominator that divides diff_beat evenly.

    Returns (denominator, num_slots) where num_slots = round(diff_beat * denominator).
    Tries common divisors from smallest to largest.
    """
    if diff_beat < _EPSILON:
        return (1, 0)

    for d in _DIVIDER_CANDIDATES:
        slots = round(diff_beat * d)
        if slots > 0 and abs(diff_beat - slots / d) < _EPSILON:
            return (d, slots)

    # Fallback: find the denominator with minimum error
    best_d = 1
    best_slots = max(1, round(diff_beat))
    best_err = abs(diff_beat - best_slots)
    for d in _DIVIDER_CANDIDATES:
        slots = max(1, round(diff_beat * d))
        err = abs(diff_beat - slots / d)
        if err < best_err:
            best_err = err
            best_d = d
            best_slots = slots
    return (best_d, best_slots)


def _get_fraction(diff_beat: float, base_denom: int) -> Tuple[int, int, int]:
    """Convert a beat difference to (numerator, denominator, one).

    Returns the mixed-number representation where:
      value = (one * denominator + numerator) / denominator
    """
    total_num = round(diff_beat * base_denom)
    if total_num == 0:
        return 0, 1, 0
    one = total_num // base_denom
    remainder = total_num % base_denom
    if remainder == 0:
        return 0, 1, one
    gcd_val = math.gcd(remainder, base_denom)
    numerator = remainder // gcd_val
    denominator = base_denom // gcd_val
    return numerator, denominator, one


class SimaiConverter:
    """Convert a Chart dict (from processed.json) to simai maidata.txt format."""

    def __init__(
        self,
        title: str,
        cabinet: str,
        version: str,
        bpm10_list: List[Dict],
        constant: Tuple[int, int],
        designer: str,
        notes: List[Dict],
        base_denominator: int = 1,
    ):
        self.title = title
        self.cabinet = cabinet
        self.version = version
        self.bpm10_list = sorted(bpm10_list, key=lambda r: r["change_timestamp_ms"])
        self.constant = constant
        self.designer = designer
        self.notes = sorted(notes, key=lambda n: n["timestamp_ms"])
        self.base_denominator = base_denominator

    # ── BPM helpers ───────────────────────────────────────────────────────

    def _get_bpm10_at(self, timestamp_ms: int) -> int:
        """Return bpm10 active at timestamp_ms."""
        if not self.bpm10_list:
            return 1500
        idx = 0
        for i, rec in enumerate(self.bpm10_list):
            if rec["change_timestamp_ms"] <= timestamp_ms:
                idx = i
            else:
                break
        return self.bpm10_list[idx]["bpm10"]

    def _per_comma_ms(self, bpm10: int, divider: int) -> float:
        """Duration of one comma (one beat-slot) in ms.

        From the simai spec: per-comma length = 240 / B / T (seconds)
        where B = BPM and T = divider. With bpm10 = BPM*10:
        per_comma_ms = 240 / (bpm10/10) / divider * 1000 = 2400000 / bpm10 / divider
        """
        if bpm10 == 0 or divider == 0:
            return 1000.0
        return 2400000.0 / bpm10 / divider

    def _one_beat_ms(self, bpm10: int) -> float:
        """Duration of one whole beat in ms: 2400000 / bpm10."""
        if bpm10 == 0:
            return 1000.0
        return 2400000.0 / bpm10

    # ── Duration encoding ─────────────────────────────────────────────────

    def _duration_to_bracket(self, duration_expr: Optional[Dict], timestamp_ms: int) -> str:
        """Convert DurationExpr to [div:mul] bracket string."""
        if duration_expr is None:
            return ""
        if "DividerMultiplier" in duration_expr:
            div, mul = duration_expr["DividerMultiplier"]
            return f"[{div}:{mul}]"
        if "AbsoluteMs" in duration_expr:
            ms = duration_expr["AbsoluteMs"]
            bpm10 = self._get_bpm10_at(timestamp_ms)
            one_beat = self._one_beat_ms(bpm10)
            beats = ms / one_beat
            num, denom, one = _get_fraction(beats, 4)
            total = one * denom + num
            if total == 0:
                total = 1
            return f"[{denom}:{total}]"
        return ""

    # ── Note string generation ────────────────────────────────────────────

    def _note_to_simai(self, note: Dict) -> str:
        """Convert a single note dict to its simai string representation."""
        kind = note["kind"]
        pos = note["pos"]
        deco = note.get("deco", [])
        pos_str = _pos_to_simai(pos)

        deco_order = ["Break", "Ex", "Firework"]
        deco_chars = "".join(
            DECO_TO_CHAR[d] for d in sorted(deco, key=lambda d: deco_order.index(d))
        )

        if kind == "Tap":
            return f"{pos_str}{deco_chars}"

        if kind == "Touch":
            return pos_str

        if kind == "Hold":
            dur = self._duration_to_bracket(note.get("duration"), note["timestamp_ms"])
            return f"{pos_str}{deco_chars}h{dur}"

        if kind == "TouchHold":
            dur = self._duration_to_bracket(note.get("duration"), note["timestamp_ms"])
            return f"{pos_str}{deco_chars}h{dur}"

        if kind == "Slide":
            return self._slide_to_simai(note)

        return pos_str

    def _slide_to_simai(self, note: Dict) -> str:
        """Convert a slide note to simai string."""
        pos = note["pos"]
        deco = note.get("deco", [])
        segments = note.get("slide_segments", [])
        slide_deco = note.get("slide_deco", [])
        duration = note.get("duration")

        pos_str = _pos_to_simai(pos)
        deco_order = ["Break", "Ex", "Firework"]
        deco_chars = "".join(
            DECO_TO_CHAR[d] for d in sorted(deco, key=lambda d: deco_order.index(d))
        )

        # Build segment chain
        chain_parts = []
        for seg in segments:
            shape = seg["shape"]
            end = seg["end"]
            if isinstance(shape, dict) and "Reflect" in shape:
                reflect_pos = _btn_digit(shape["Reflect"])
                chain_parts.append(f"V{reflect_pos}{_pos_to_simai(end)}")
            else:
                shape_char = SHAPE_TO_CHAR.get(shape, shape)
                chain_parts.append(f"{shape_char}{_pos_to_simai(end)}")

        chain = "*".join(chain_parts)

        # Duration
        dur = self._duration_to_bracket(duration, note["timestamp_ms"])

        # Slide decoration (after chain)
        slide_deco_str = ""
        for sd in sorted(slide_deco):
            if sd in SLIDE_DECO_TO_CHAR:
                slide_deco_str += SLIDE_DECO_TO_CHAR[sd]

        result = f"{pos_str}{deco_chars}{chain}{dur}{slide_deco_str}"

        # Try WiFi compression
        compressed = self._try_wifi_compress(note)
        if compressed:
            return compressed

        return result

    def _try_wifi_compress(self, note: Dict) -> Optional[str]:
        """Try to compress a 3-segment straight slide to WiFi syntax."""
        segments = note.get("slide_segments", [])
        duration = note.get("duration")
        deco = note.get("deco", [])
        slide_deco = note.get("slide_deco", [])

        if len(segments) != 3:
            return None
        if not all(seg["shape"] == "Straight" for seg in segments):
            return None
        if slide_deco:
            return None

        head_pos = _btn_digit(note["pos"])
        end_ids = [_btn_digit(seg["end"]) for seg in segments]
        expected = WIFI_ENDPOINT_SEQ.get(head_pos)
        if not expected:
            return None
        if sorted(end_ids) != sorted(expected):
            return None

        deco_order = ["Break", "Ex", "Firework"]
        deco_chars = "".join(
            DECO_TO_CHAR[d] for d in sorted(deco, key=lambda d: deco_order.index(d))
        )
        dur = self._duration_to_bracket(duration, note["timestamp_ms"])
        wifi_end = expected[1]

        return f"{head_pos}{deco_chars}w{wifi_end}{dur}"

    # ── Beat difference calculation ───────────────────────────────────────

    def _calculate_beat_diffs(
        self, last_time_ms: float, cur_time_ms: float
    ) -> List[Tuple[int, float]]:
        """Split the interval [last_time_ms, cur_time_ms] at BPM boundaries.

        Returns list of (bpm10, diff_beat) tuples, where diff_beat is the
        number of beats in that sub-segment.
        """
        result: List[Tuple[int, float]] = []
        seg_start = last_time_ms

        seg_idx = 0
        for i, rec in enumerate(self.bpm10_list):
            if rec["change_timestamp_ms"] <= seg_start:
                seg_idx = i
            else:
                break

        while True:
            bpm10 = self.bpm10_list[seg_idx]["bpm10"]

            if seg_idx + 1 < len(self.bpm10_list):
                next_boundary = float(self.bpm10_list[seg_idx + 1]["change_timestamp_ms"])
            else:
                next_boundary = float("inf")

            if next_boundary >= cur_time_ms:
                diff_ms = cur_time_ms - seg_start
                one_beat = self._one_beat_ms(bpm10)
                diff_beat = diff_ms / one_beat
                result.append((bpm10, diff_beat))
                break
            else:
                diff_ms = next_boundary - seg_start
                one_beat = self._one_beat_ms(bpm10)
                diff_beat = diff_ms / one_beat
                if diff_beat > _EPSILON:
                    result.append((bpm10, diff_beat))
                seg_start = next_boundary
                seg_idx += 1

        return result

    # ── Main conversion ───────────────────────────────────────────────────

    def convert(self) -> str:
        """Convert the chart to simai maidata.txt format.

        Algorithm:
        1. Sort notes by timestamp
        2. Group simultaneous notes (within 5ms)
        3. Compute beat positions for all note groups
        4. Group notes by beat floor (integer beat position)
        5. For each beat, determine denominator based on sub-beat positions
        6. Emit {N}marker, empty slots, then notes

        Format rules:
        - First line: (BPM){N}, (combines BPM marker and initial denominator)
        - All subsequent lines: {N}content, (denominator prefix always present)
        - Empty beats: {N}, (with denominator prefix)
        - Notes: {N}note, (with denominator prefix)
        - Sub-beat notes within same beat: {N}note1,note2, (comma-separated)
        """
        if not self.notes:
            return self._empty_chart_output()

        notes = self.notes
        output_lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────────
        major, minor = self.constant
        level_num = self._infer_level_num()

        output_lines.append(f"&title={self.title}")
        output_lines.append(f"&cabinet={self.cabinet}")
        output_lines.append(f"&version={self.version}")
        if self.designer:
            output_lines.append(f"&des_{level_num}={self.designer}")
        output_lines.append(f"&lv_{level_num}={major}.{minor}")
        output_lines.append(f"&inote_{level_num}=")

        # ── Group simultaneous notes ──────────────────────────────────────
        beat_groups = self._group_notes_to_beats(notes)

        # ── Compute beat positions for all groups ─────────────────────────
        group_data = []
        for time_ms, grp in beat_groups:
            bpm10 = self._get_bpm10_at(int(time_ms))
            one_beat = self._one_beat_ms(bpm10)
            beat_pos = time_ms / one_beat
            group_data.append((time_ms, grp, bpm10, beat_pos))

        if not group_data:
            return self._empty_chart_output()

        # ── Group notes by beat floor ────────────────────────────────────
        # Each beat is (beat_floor, [note_strs], beat_pos fractional parts)
        beats = []  # List of (beat_floor, [note_strs], [frac_parts], bpm10)
        
        for time_ms, grp, bpm10, beat_pos in group_data:
            one_beat = self._one_beat_ms(bpm10)
            beat_floor = int(round(time_ms / one_beat))
            frac_part = beat_pos - beat_floor
            # Format the note group (handles simultaneous notes with /)
            note_str = self._format_note_group(grp)
            
            # Check if this beat already exists
            if beats and beats[-1][0] == beat_floor:
                # Same beat - append note string
                beats[-1][1].append(note_str)
                beats[-1][2].append(frac_part)
            else:
                # New beat
                beats.append((beat_floor, [note_str], [frac_part], bpm10))

        # ── Determine denominator for each beat ──────────────────────────
        # For each beat, find the smallest denominator that divides all frac_parts
        beat_output = []  # List of (denom, bpm10, [note_strs], beat_floor)
        
        # Use larger tolerance for floating point precision
        DENOM_TOLERANCE = 1e-3
        
        for beat_floor, note_strs, frac_parts, bpm10 in beats:
            # Find the smallest denominator that works for all frac_parts
            best_denom = 1
            for d in _DIVIDER_CANDIDATES:
                # Check if all frac_parts are multiples of 1/d
                valid = True
                for frac in frac_parts:
                    # Normalize frac to [0, 1)
                    frac_norm = frac % 1.0
                    slot = round(frac_norm * d)
                    if abs(frac_norm - slot / d) > DENOM_TOLERANCE:
                        valid = False
                        break
                if valid:
                    best_denom = d
                    break
            
            beat_output.append((best_denom, bpm10, note_strs, beat_floor))

        # ── Emit output ───────────────────────────────────────────────────
        last_denom = None
        last_bpm10 = None
        last_beat_floor = -1

        for idx, (denom, bpm10, note_strs, beat_floor) in enumerate(beat_output):
            bpm_val = bpm10 / 10.0

            # Save previous state for empty beats
            prev_denom = last_denom
            prev_bpm10 = last_bpm10

            # Build prefix
            prefix = ""
            if bpm10 != last_bpm10:
                if idx == 0:
                    prefix = f"({bpm_val:.0f})"
                else:
                    prefix = f"({bpm_val:.0f})"
                last_bpm10 = bpm10
            # Always include {N} prefix
            prefix += "{" + f"{denom}" + "}"
            last_denom = denom

            if idx == 0:
                # First line: emit BPM + denominator
                output_lines.append(f"{prefix},")
                
                # Emit empty beats before first note
                # The (BPM){N}, line counts as beat 0, so we need beat_floor - 1 empty beats
                empty_beats = beat_floor - 1
                for _ in range(empty_beats):
                    output_lines.append("{" + f"{denom}" + "},")
                
                # Emit note(s) for this beat
                note_content = ",".join(note_strs)
                output_lines.append("{" + f"{denom}" + "}" + note_content + ",")
            else:
                # Emit empty beats between last beat and this beat
                # Use previous denominator for empty beats
                empty_beats = beat_floor - last_beat_floor - 1
                if prev_denom is not None:
                    empty_prefix = "{" + f"{prev_denom}" + "}"
                else:
                    empty_prefix = "{" + f"{denom}" + "}"
                for _ in range(empty_beats):
                    output_lines.append(f"{empty_prefix},")
                
                # Emit note(s) for this beat (with new prefix if denom changed)
                note_content = ",".join(note_strs)
                output_lines.append(f"{prefix}{note_content},")

            last_beat_floor = beat_floor

        # ── Footer ────────────────────────────────────────────────────────
        output_lines.append("{1},")
        output_lines.append("{1},")
        output_lines.append("{1},")
        output_lines.append("E")

        return "\n".join(output_lines) + "\n"

    def _format_note_group(self, group: List[Dict]) -> str:
        """Format a group of simultaneous notes with / separator."""
        if len(group) == 1:
            return self._note_to_simai(group[0])
        return "/".join(self._note_to_simai(n) for n in group)

    def _empty_chart_output(self) -> str:
        major, minor = self.constant
        level_num = self._infer_level_num()
        first_bpm = self.bpm10_list[0]["bpm10"] / 10.0
        lines = [
            f"&title={self.title}",
            f"&cabinet={self.cabinet}",
            f"&version={self.version}",
            f"&lv_{level_num}={major}.{minor}",
            f"&inote_{level_num}=",
            f"({first_bpm:.0f}){{1}},",
            "{{1}},",
            "{{1}},",
            "{{1}},",
            "E",
        ]
        return "\n".join(lines) + "\n"

    def _infer_level_num(self) -> int:
        """Infer the simai level number. Default to 5 (master)."""
        return 5

    # ── Beat grouping ─────────────────────────────────────────────────────

    def _group_notes_to_beats(
        self, notes: List[Dict], tolerance_ms: float = 5.0
    ) -> List[Tuple[float, List[Dict]]]:
        """Group notes that fall on the same beat (within tolerance).

        Returns list of (timestamp_ms, [notes_at_this_beat]).
        """
        if not notes:
            return []

        groups: List[Tuple[float, List[Dict]]] = []
        current_time = notes[0]["timestamp_ms"]
        current_group = [notes[0]]

        for note in notes[1:]:
            if abs(note["timestamp_ms"] - current_time) <= tolerance_ms:
                current_group.append(note)
            else:
                groups.append((float(current_time), current_group))
                current_time = note["timestamp_ms"]
                current_group = [note]

        groups.append((float(current_time), current_group))
        return groups
