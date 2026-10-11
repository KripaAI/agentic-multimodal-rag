"""Planet placements read from South Indian horoscope charts (Phase 10).

A VLM caption names the planets on a chart but not where they sit. In the South Indian style every
outer cell of the 4x4 grid is a fixed sign, so placements can be read exactly: OCR the labels with
their positions, put each label in its grid cell (= its sign), and compute houses from the Lagna's
cell in code. Nothing is guessed: with no Lagna mark there are signs but no houses, and a chart that
is not a South Indian grid (a North Indian diamond, a table) is reported as such and left alone.
"""

from __future__ import annotations

import bisect
import difflib
import re
from dataclasses import dataclass, field

import numpy as np

SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn",
         "Aquarius", "Pisces"]
# The South Indian layout: (row, column) of the 4x4 grid -> sign. The four centre cells hold no sign.
CELL_SIGN = {(0, 0): "Pisces", (0, 1): "Aries", (0, 2): "Taurus", (0, 3): "Gemini",
             (1, 3): "Cancer", (2, 3): "Leo",
             (3, 3): "Virgo", (3, 2): "Libra", (3, 1): "Scorpio", (3, 0): "Sagittarius",
             (2, 0): "Capricorn", (1, 0): "Aquarius"}
# Label spellings seen in the books (English abbreviations and Sanskrit names) -> canonical body.
BODIES = {
    "Lagna": ["LAGNA", "LAG", "ASC", "AS", "LG", "L"],
    "Sun": ["SUN", "SURYA", "SU", "SY", "RAVI"],
    "Moon": ["MOON", "CHANDRA", "CHAN", "MO", "CH", "NOOW", "NOON", "MOOW", "MOOM"],  # with OCR misreads
    "Mars": ["MARS", "MANGAL", "KUJA", "MA", "KU"],
    "Mercury": ["MERCURY", "MERC", "MER", "BUDHA", "BUDH", "ME", "BU"],
    "Jupiter": ["JUPITER", "JUP", "GURU", "JU", "GU", "BRIHASPATI", "SUP", "RUP", "IUP", "JUF"],  # OCR misreads
    "Venus": ["VENUS", "VEN", "SUKRA", "SHUKRA", "VE", "SK", "YEN"],
    "Saturn": ["SATURN", "SAT", "SANI", "SHANI", "SA", "BAT"],
    "Rahu": ["RAHU", "RA"],
    "Ketu": ["KETU", "KE"],
}
_SPELLINGS = sorted(((s, b) for b, ss in BODIES.items() for s in ss), key=lambda x: -len(x[0]))
_RETRO = re.compile(r"(R[XY]?)$")  # a retrograde mark after a name ("JUP RX"; OCR also reads it "RY")
ORDER = list(BODIES)


@dataclass
class ChartReading:
    style: str                      # "south_indian", or "not_south_indian"
    kind: str = "Rasi chart"        # or a divisional chart named in the centre, e.g. "Navamsa (D-9) chart"
    lagna: str | None = None        # sign of the Lagna, when marked
    placements: dict[str, str] = field(default_factory=dict)   # body -> sign
    unread: list[str] = field(default_factory=list)            # outer-cell labels that are not bodies (A.L., UL, ...)

    def house(self, body: str) -> int | None:
        if self.lagna is None or body not in self.placements:
            return None
        return (SIGNS.index(self.placements[body]) - SIGNS.index(self.lagna)) % 12 + 1

    def text(self) -> str:
        """The placements as one searchable sentence group, e.g. for a figure's search document."""
        if self.style != "south_indian" or not self.placements:
            return ""
        title = self.kind
        head = f"{title}, Lagna {self.lagna}." if self.lagna else f"{title}, Lagna not marked."
        by_sign: dict[str, list[str]] = {}
        for body in ORDER:
            if body != "Lagna" and body in self.placements:
                by_sign.setdefault(self.placements[body], []).append(body)
        parts = []
        for sign in sorted(by_sign, key=lambda s: (self.house(by_sign[s][0]) or 99, SIGNS.index(s))):
            h = self.house(by_sign[sign][0])
            suffix = f" ({_ordinal(h)} house)" if h else ""
            parts.append(f"{', '.join(by_sign[sign])} in {sign}{suffix}")
        return head + " " + "; ".join(parts) + "."


# Divisional charts (vargas) by number; the centre of a Rasi chart holds the native's birth data instead.
VARGAS = {2: "Hora", 3: "Drekkana", 4: "Chaturthamsa", 7: "Saptamsa", 9: "Navamsa", 10: "Dasamsa",
          12: "Dwadasamsa", 16: "Shodasamsa", 20: "Vimsamsa", 24: "Chaturvimsamsa", 27: "Bhamsa",
          30: "Trimsamsa", 40: "Khavedamsa", 45: "Akshavedamsa", 60: "Shashtyamsa"}


def chart_kind(centre: str) -> str:
    """"Navamsa (D-9) chart" etc. from a "D-n" or a varga name in the centre text, else "Rasi chart".
    Without this a Vimsamsa (D-20) was reported as a Rasi chart, with the wrong planets for the native."""
    t = centre.upper().replace(" ", "")
    m = re.search(r"(?<![A-Z])D\s*-?\s*(\d{1,2})(?!\d)", centre.upper())
    if m and int(m.group(1)) in VARGAS:
        n = int(m.group(1))
        return f"{VARGAS[n]} (D-{n}) chart"
    for n, name in VARGAS.items():
        if name.upper() in t or (name == "Navamsa" and "NAVAMSHA" in t):
            return f"{name} (D-{n}) chart"
    return "Rasi chart"


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def split_label(token: str) -> list[str]:
    """Bodies named in one OCR token, which may run several labels together ("VENSAT", "JUP MARS"),
    carry a retrograde mark ("JUPRX") or be slightly misread ("NOON" for MOON)."""
    t = re.sub(r"[^A-Z]", "", token.upper())
    whole = _match(t) or (_match(_RETRO.sub("", t)) if len(t) > 3 else [])
    if whole:
        return whole
    if 3 <= len(t) <= 7:  # one misread name: the closest spelling, when it is close
        close = difflib.get_close_matches(t, [sp for sp, _ in _SPELLINGS if len(sp) >= 3], n=1, cutoff=0.74)
        if close:
            return [dict(_SPELLINGS)[close[0]]]
    return []


def _match(t: str) -> list[str]:
    out, i = [], 0
    while i < len(t):
        for spelling, body in _SPELLINGS:  # longest first: SAT before SA, RAHU before RA
            if len(spelling) >= 2 and t.startswith(spelling, i):
                out.append(body)
                i += len(spelling)
                break
        else:
            return []  # part of it is no body name (A.L., UL, a word): leave the whole token unread
    return out


def _lines(dark: np.ndarray, axis: int, min_share: float) -> list[int]:
    """Positions of drawn lines: rows (axis=1) or columns (axis=0) that are dark along most of their length."""
    share = dark.mean(axis=axis)
    idx = np.where(share >= min_share)[0]
    groups, start = [], None
    for k, i in enumerate(idx):
        if start is None:
            start = i
        if k == len(idx) - 1 or idx[k + 1] != i + 1:
            groups.append((start + i) // 2)
            start = None
    return groups


def _bounds(lines: list[int], size: int) -> list[float] | None:
    """The five cell boundaries along one axis of a South Indian grid, from its four full-length lines
    (frame, inner, inner, frame). The outer cells need not be a quarter: many books draw them thin and
    the middle band wide; the middle band's two cells split it at its midpoint."""
    if len(lines) != 4:
        return None
    a0, a1, b1, b0 = lines
    span = b0 - a0
    if span < 0.5 * size:
        return None
    outer1, outer2 = a1 - a0, b0 - b1
    if min(outer1, outer2) < 0.1 * span or abs(outer1 - outer2) > 0.25 * max(outer1, outer2):
        return None  # not a symmetric frame: some other figure or a table
    return [a0, a1, (a1 + b1) / 2, b1, b0]


def grid_frame(gray: np.ndarray) -> tuple[list[float], list[float]] | None:
    """(column bounds, row bounds) of a South Indian 4x4 grid, or None. The test: exactly four
    full-length lines each way, symmetric about the centre (the centre box breaks the middle lines)."""
    dark = gray < 170
    cols, rows = _bounds(_lines(dark, 0, 0.6), gray.shape[1]), _bounds(_lines(dark, 1, 0.6), gray.shape[0])
    if cols is None or rows is None:
        return None
    return cols, rows


def read_chart(image: np.ndarray, ocr) -> ChartReading:
    """`image`: BGR or grayscale array. `ocr(image) -> [(box, text, confidence)]`, box as four (x, y)
    points (RapidOCR's output). The image is upscaled 3x for OCR: chart labels are small."""
    import cv2

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame = grid_frame(gray)
    if frame is None:
        return ChartReading(style="not_south_indian")
    xb, yb = frame
    big = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    reading = ChartReading(style="south_indian")
    centre_text = []
    for box, text, conf in ocr(big) or []:
        cx = sum(p[0] for p in box) / 12  # /4 points, /3 upscale
        cy = sum(p[1] for p in box) / 12
        col, row = bisect.bisect(xb, cx) - 1, bisect.bisect(yb, cy) - 1
        if not (0 <= row <= 3 and 0 <= col <= 3):
            continue
        sign = CELL_SIGN.get((row, col))
        if sign is None:
            centre_text.append(text)
            continue
        bodies = split_label(text)
        if not bodies:
            reading.unread.append(text)
        for body in bodies:
            if body == "Lagna":
                reading.lagna = sign
            reading.placements.setdefault(body, sign)
    reading.kind = chart_kind(" ".join(centre_text))
    return reading


def chart_records(path) -> dict[str, dict]:
    """Stored readings (charts.jsonl) by element id; an empty dict when there are none."""
    import json

    from mmrag.fsutil import jsonl_lines

    if not path.is_file():
        return {}
    return {r["element_id"]: r for r in map(json.loads, jsonl_lines(path))}
