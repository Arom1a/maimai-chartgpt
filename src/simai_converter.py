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


def _pos_to_simai(pos: str) -> str:
    return POS_TO_SIMAI.get(pos, pos)


def _btn_digit(pos: str) -> str:
    """Extract the digit from a button position like 'Btn3' → '3'."""
    if pos.startswith("Btn"):
        return pos[3:]
    return pos


def _get_fraction(diff_beat: float, base_denom: int) -> Tuple[int, int, int]:
    """Convert a beat difference to (numerator, denominator, one).

    Returns the mixed-number representation where:
      value = (one * denominator + numerator) / denominator

    Examples:
      0.5  → (1, 2, 0)   → 1/2
      1.0  → (0, 1, 1)   → 1
      2.25 → (1, 4, 2)   → 2 + 1/4
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
        """Duration of one comma (one beat) in ms. From parser.rs:90."""
        if bpm10 == 0 or divider == 0:
            return 1000.0
        return 2400000.0 / bpm10 / divider

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
            one_beat_ms = 2400000.0 / bpm10
            beats = ms / one_beat_ms
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

        # Decoration chars (sorted: Break, Ex, Firework)
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
        self, last_time_ms: float, cur_time_ms: float, base_denom: int
    ) -> List[Tuple[int, int, int, int]]:
        """Split the interval [last_time_ms, cur_time_ms] at BPM boundaries.

        Returns list of (bpm10, numerator, denominator, one) tuples.
        """
        result: List[Tuple[int, int, int, int]] = []
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
                one_beat_ms = self._per_comma_ms(bpm10, base_denom)
                diff_beat = diff_ms / one_beat_ms
                num, denom, one = _get_fraction(diff_beat, base_denom)
                result.append((bpm10, num, denom, one))
                break
            else:
                diff_ms = next_boundary - seg_start
                one_beat_ms = self._per_comma_ms(bpm10, base_denom)
                diff_beat = diff_ms / one_beat_ms
                num, denom, one = _get_fraction(diff_beat, base_denom)
                if (num, denom, one) != (0, 1, 0):
                    result.append((bpm10, num, denom, one))
                seg_start = next_boundary
                seg_idx += 1

        return result

    # ── Main conversion ───────────────────────────────────────────────────

    def convert(self) -> str:
        """Convert the chart to simai maidata.txt format.

        The simai format uses one line per beat slot. Each line contains:
          - Optional BPM/denominator change markers as prefixes
          - Note content (or empty for rest)
          - A trailing comma

        Between notes, empty beats are represented as lines with just a comma.
        Simultaneous notes use '/' separator.
        """
        if not self.notes:
            return self._empty_chart_output()

        notes = self.notes
        output_lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────────
        major, minor = self.constant
        level_num = self._infer_level_num()
        first_bpm = self.bpm10_list[0]["bpm10"] / 10.0

        output_lines.append(f"&title={self.title}")
        output_lines.append(f"&cabinet={self.cabinet}")
        output_lines.append(f"&version={self.version}")
        if self.designer:
            output_lines.append(f"&des_{level_num}={self.designer}")
        output_lines.append(f"&lv_{level_num}={major}.{minor}")
        output_lines.append(
            f"&inote_{level_num}=({first_bpm:.0f})" + "{" + f"{self.base_denominator}" + "},"
        )

        # ── Group simultaneous notes ──────────────────────────────────────
        beat_groups = self._group_notes_to_beats(notes)

        # ── Generate beats ────────────────────────────────────────────────
        last_theoretical_ms = 0.0
        last_bpm10 = self.bpm10_list[0]["bpm10"]
        last_denom = self.base_denominator

        for beat_time_ms, group_notes in beat_groups:
            beat_diffs = self._calculate_beat_diffs(
                last_theoretical_ms, beat_time_ms, self.base_denominator
            )

            if not beat_diffs:
                continue

            # Process each sub-segment of the beat difference
            for seg_idx, (bpm10, num, denom, one) in enumerate(beat_diffs):
                is_first_seg = (seg_idx == 0)
                is_last_seg = (seg_idx == len(beat_diffs) - 1)

                # Build prefix for BPM/denominator changes
                prefix = ""

                # BPM change: write on first sub-segment if BPM actually changed
                if bpm10 != last_bpm10:
                    prefix += f"({bpm10 / 10.0:.0f})"
                    last_bpm10 = bpm10

                # Denominator change: write if different from last
                if denom != last_denom and not (num == 0 and one == 0):
                    prefix += "{" + f"{denom}" + "}"
                    last_denom = denom

                # Calculate total beats in this sub-segment
                total_beats = one * denom + num

                if total_beats == 0:
                    continue

                if is_last_seg:
                    # Last sub-segment: the note lands on the LAST beat of
                    # this interval.  Write (total_beats - 1) empty beats
                    # before it.  The note content is written after the loop.
                    empty_before = total_beats - 1
                    if empty_before > 0:
                        if one > 0 and num > 0:
                            # Mixed number: fraction part + integer part
                            # Both the fraction and integer beats are empty
                            output_lines.append(
                                f"{prefix}{',' * num}" + "{1}" + f"{',' * one}"
                            )
                        else:
                            output_lines.append(f"{prefix}{',' * empty_before}")
                    elif prefix:
                        # No empty beats but we have a prefix; will be
                        # combined with the note line below.
                        pass
                    # Note content is written below (after the loop)
                else:
                    # Middle sub-segment: all beats are empty
                    # Write with prefix on the first comma
                    if one > 0 and num > 0:
                        # Mixed number: {denom} + commas for fraction, {1} + commas for integer
                        output_lines.append(f"{prefix}{',' * num}" + "{1}" + f"{',' * one}")
                    elif one > 0:
                        output_lines.append(f"{prefix}{',' * one}")
                    elif num > 0:
                        output_lines.append(f"{prefix}{',' * num}")

            # Update theoretical time
            last_theoretical_ms = beat_time_ms

            # Write the note content for this beat
            # Determine if we need BPM/denom prefix on the note line
            note_prefix = ""
            cur_bpm10 = self._get_bpm10_at(int(beat_time_ms))
            if cur_bpm10 != last_bpm10:
                note_prefix += f"({cur_bpm10 / 10.0:.0f})"
                last_bpm10 = cur_bpm10

            if len(group_notes) == 1:
                note_str = self._note_to_simai(group_notes[0])
            else:
                note_strs = [self._note_to_simai(n) for n in group_notes]
                note_str = "/".join(note_strs)

            output_lines.append(f"{note_prefix}{note_str},")

        # ── Footer ────────────────────────────────────────────────────────
        output_lines.append("{1},,,E")

        return "\n".join(output_lines) + "\n"

    def _empty_chart_output(self) -> str:
        major, minor = self.constant
        level_num = self._infer_level_num()
        first_bpm = self.bpm10_list[0]["bpm10"] / 10.0
        lines = [
            f"&title={self.title}",
            f"&cabinet={self.cabinet}",
            f"&version={self.version}",
            f"&lv_{level_num}={major}.{minor}",
            f"&inote_{level_num}=({first_bpm:.0f})" + "{" + f"{self.base_denominator}" + "},",
            "{1},,,E",
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
