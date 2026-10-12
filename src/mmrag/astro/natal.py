"""A Vedic birth chart computed by code (Phase 11). The model never calculates positions.

Geocentric apparent positions from NASA JPL's DE440s ephemeris (Skyfield), made sidereal with the
Lahiri (Chitrapaksha) ayanamsa as Swiss Ephemeris defines it: 23.245524743 deg on 1956-03-21 0h TT,
moved by IAU 1976 general precession, plus nutation in longitude because the positions are of the
true equinox. Rahu is the mean lunar node and Ketu is opposite. Houses are whole signs counted from
the Lagna's sign, as in the Parashari tradition. Vimshottari dashas use years of 365.25 days.

The ephemeris file (32 MB) downloads on first use into $MMRAG_EPHEMERIS_DIR, else data/ephemeris.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from mmrag.ingest.charts import SIGNS

PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
BODIES = [*PLANETS, "Rahu", "Ketu"]
_EPHEMERIS_NAME = {"Sun": "sun", "Moon": "moon", "Mars": "mars barycenter", "Mercury": "mercury",
                   "Jupiter": "jupiter barycenter", "Venus": "venus", "Saturn": "saturn barycenter"}
SIGN_LORD = {"Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon", "Leo": "Sun",
             "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Mars", "Sagittarius": "Jupiter",
             "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter"}
EXALTED = {"Sun": "Aries", "Moon": "Taurus", "Mars": "Capricorn", "Mercury": "Virgo", "Jupiter": "Cancer",
           "Venus": "Pisces", "Saturn": "Libra"}
DEBILITATED = {b: SIGNS[(SIGNS.index(s) + 6) % 12] for b, s in EXALTED.items()}
# Classical combustion distances from the Sun, in degrees: (direct, retrograde).
COMBUSTION = {"Moon": (12, 12), "Mars": (17, 17), "Mercury": (14, 12), "Jupiter": (11, 11), "Venus": (10, 8),
              "Saturn": (15, 15)}
NAKSHATRAS = ["Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu", "Pushya", "Ashlesha",
              "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha",
              "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
              "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"]
DASHA_ORDER = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
DASHA_YEARS = {"Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7, "Rahu": 18, "Jupiter": 16, "Saturn": 19,
               "Mercury": 17}
YEAR_DAYS = 365.25
NAK_SPAN = 360 / 27

_LAHIRI_T0 = (2435553.5 - 2451545.0) / 36525  # 1956-03-21 0h TT, in Julian centuries from J2000
_LAHIRI_AT_T0 = 23.250182778 - 0.004658035   # Swiss Ephemeris SE_SIDM_LAHIRI


class ChartInputError(ValueError):
    """Birth data that cannot be used (bad date, time, place or time zone)."""


def _precession_arcsec(t: float) -> float:
    """IAU 1976 general precession in longitude since J2000, in arcseconds (t in Julian centuries)."""
    return 5029.0966 * t + 1.11113 * t * t - 0.000006 * t ** 3


def lahiri_ayanamsa(jd_tt: float) -> float:
    """Mean Lahiri ayanamsa in degrees (without nutation)."""
    t = (jd_tt - 2451545.0) / 36525
    return _LAHIRI_AT_T0 + (_precession_arcsec(t) - _precession_arcsec(_LAHIRI_T0)) / 3600


def mean_node(jd_tt: float) -> float:
    """Tropical longitude of the mean ascending lunar node, degrees (Meeus, Astronomical Algorithms 47.7)."""
    t = (jd_tt - 2451545.0) / 36525
    return (125.0445479 - 1934.1362891 * t + 0.0020754 * t * t + t ** 3 / 467441 - t ** 4 / 60616000) % 360


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def dms(degrees: float) -> str:
    """12.5 -> 12°30'."""
    minutes = int(round(degrees * 60))
    return f"{minutes // 60}°{minutes % 60:02d}'"


def nakshatra_of(lon: float) -> tuple[str, int]:
    """(nakshatra, pada 1-4) of a sidereal longitude."""
    return NAKSHATRAS[int(lon // NAK_SPAN) % 27], int((lon % NAK_SPAN) // (NAK_SPAN / 4)) + 1


@dataclass
class Placement:
    body: str
    longitude: float          # sidereal, 0-360
    sign: str
    degree: float             # within the sign
    house: int                # whole-sign house from the Lagna
    nakshatra: str
    pada: int
    retrograde: bool = False
    dignity: str | None = None  # exalted, debilitated, own sign
    combust: bool = False
    lord_of: list[int] = field(default_factory=list)  # houses whose sign this planet rules


@dataclass
class Dasha:
    lord: str
    start: date
    end: date
    sub: str | None = None  # antardasha lord, for a sub-period


@dataclass
class BirthData:
    when_local: datetime      # naive local clock time
    tz: str                   # IANA zone (Asia/Kolkata) or a UTC offset (+05:30)
    latitude: float           # north positive
    longitude: float          # east positive
    name: str | None = None
    place: str | None = None

    def utc(self) -> datetime:
        m = re.fullmatch(r"(?:UTC)?\s*([+-])(\d{1,2}):?(\d{2})?", self.tz.strip())
        if m:
            sign = 1 if m.group(1) == "+" else -1
            offset = timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0)) * sign
            return (self.when_local - offset).replace(tzinfo=timezone.utc)
        try:
            zone = ZoneInfo(self.tz.strip())
        except Exception as e:  # noqa: BLE001 - zoneinfo raises several types for a bad key
            raise ChartInputError(f"unknown time zone {self.tz!r}; use a name like Asia/Kolkata or an offset "
                                  "like +05:30") from e
        return self.when_local.replace(tzinfo=zone).astimezone(timezone.utc)


def antardashas(md: Dasha) -> list[Dasha]:
    """The nine sub-periods of a mahadasha, starting with its own lord. The first mahadasha is cut
    by birth; it is measured from its full length, so its early sub-periods fall before birth."""
    years = DASHA_YEARS[md.lord]
    start = datetime.combine(md.end, datetime.min.time()) - timedelta(days=years * YEAR_DAYS)
    out, i = [], DASHA_ORDER.index(md.lord)
    for k in range(9):
        lord = DASHA_ORDER[(i + k) % 9]
        end = start + timedelta(days=years * DASHA_YEARS[lord] / 120 * YEAR_DAYS)
        out.append(Dasha(lord=md.lord, sub=lord, start=start.date(), end=end.date()))
        start = end
    return out


@dataclass
class NatalChart:
    birth: BirthData
    utc: datetime
    ayanamsa: float
    lagna: Placement
    placements: dict[str, Placement]
    dashas: list[Dasha]       # the nine mahadashas from birth
    moon_dasha_balance_years: float

    def current(self, on: date) -> tuple[Dasha | None, Dasha | None]:
        """(mahadasha, antardasha) running on `on`."""
        for md in self.dashas:
            if md.start <= on < md.end:
                return md, next((ad for ad in antardashas(md) if ad.start <= on < ad.end), None)
        return None, None

    def facts(self, on: date | None = None) -> str:
        """The chart as plain lines for the writer: every fact it may use, nothing left to work out."""
        on = on or date.today()
        b = self.birth
        who = f"{b.name}, born" if b.name else "Born"
        where = f" in {b.place}" if b.place else ""
        lines = [f"{who} {b.when_local:%Y-%m-%d %H:%M} local time ({b.tz}){where}, latitude {b.latitude:.4f}, "
                 f"longitude {b.longitude:.4f} (UTC {self.utc:%Y-%m-%d %H:%M}).",
                 f"Sidereal zodiac, Lahiri ayanamsa {dms(self.ayanamsa)}; whole-sign houses counted from the Lagna.",
                 f"Lagna (Ascendant): {self.lagna.sign} {dms(self.lagna.degree)}, {self.lagna.nakshatra} nakshatra "
                 f"pada {self.lagna.pada}; Lagna lord {SIGN_LORD[self.lagna.sign]}.",
                 "Planets:"]
        for body in BODIES:
            p = self.placements[body]
            notes = [n for n in ("retrograde" if p.retrograde and body not in ("Rahu", "Ketu") else None,
                                 p.dignity, "combust" if p.combust else None) if n]
            lords = ""
            if p.lord_of:
                lords = f"; lord of the {' and '.join(ordinal(h) for h in p.lord_of)} house" + ("s" if len(p.lord_of) > 1 else "")
            lines.append(f"- {body}: {p.sign} {dms(p.degree)}, {ordinal(p.house)} house, {p.nakshatra} nakshatra "
                         f"pada {p.pada}{lords}" + (f" ({', '.join(notes)})" if notes else ""))
        moon = self.placements["Moon"]
        lines.append(f"Moon sign (Rashi): {moon.sign}; birth nakshatra: {moon.nakshatra} pada {moon.pada}.")
        lines.append(f"Vimshottari dasha at birth: {self.dashas[0].lord} mahadasha, "
                     f"{self.moon_dasha_balance_years:.2f} years remaining.")
        md, ad = self.current(on)
        if md:
            lines.append(f"Running on {on:%Y-%m-%d}: {md.lord} mahadasha ({md.start:%Y-%m-%d} to {md.end:%Y-%m-%d})"
                         + (f", {ad.sub} antardasha ({ad.start:%Y-%m-%d} to {ad.end:%Y-%m-%d})." if ad else "."))
        lines.append("Mahadasha sequence: " + "; ".join(f"{d.lord} {d.start:%Y}-{d.end:%Y}" for d in self.dashas) + ".")
        return "\n".join(lines)


def vimshottari(moon_lon: float, birth: datetime) -> tuple[list[Dasha], float]:
    """The nine mahadashas from birth, and the years left of the first one."""
    lord = DASHA_ORDER[int(moon_lon // NAK_SPAN) % 27 % 9]
    elapsed = (moon_lon % NAK_SPAN) / NAK_SPAN
    start = birth - timedelta(days=elapsed * DASHA_YEARS[lord] * YEAR_DAYS)
    out, i = [], DASHA_ORDER.index(lord)
    for k in range(9):
        lord_k = DASHA_ORDER[(i + k) % 9]
        end = start + timedelta(days=DASHA_YEARS[lord_k] * YEAR_DAYS)
        out.append(Dasha(lord=lord_k, start=(birth if k == 0 else start).date(), end=end.date()))
        start = end
    return out, (1 - elapsed) * DASHA_YEARS[lord]


def _ephemeris_dir() -> Path:
    from mmrag.config import PROJECT_ROOT

    return Path(os.environ.get("MMRAG_EPHEMERIS_DIR") or PROJECT_ROOT / "data" / "ephemeris")


@lru_cache(maxsize=1)
def _sky():
    from skyfield.api import Loader

    folder = _ephemeris_dir()
    folder.mkdir(parents=True, exist_ok=True)
    load = Loader(str(folder), verbose=False)
    return load.timescale(builtin=True), load("de440s.bsp")


def ascendant(gast_hours: float, longitude: float, latitude: float, obliquity_rad: float) -> float:
    """Tropical longitude of the ecliptic point rising in the east (degrees), from Greenwich apparent
    sidereal time, the east longitude and latitude of the place, and the true obliquity."""
    theta = math.radians((gast_hours * 15 + longitude) % 360)
    phi = math.radians(latitude)
    return math.degrees(math.atan2(math.cos(theta), -(math.sin(theta) * math.cos(obliquity_rad)
                                                      + math.tan(phi) * math.sin(obliquity_rad)))) % 360


def compute_chart(birth: BirthData) -> NatalChart:
    """The sidereal chart for one birth."""
    from skyfield.framelib import ecliptic_frame
    from skyfield.nutationlib import iau2000b_radians, mean_obliquity

    if not -90 <= birth.latitude <= 90 or not -180 <= birth.longitude <= 180:
        raise ChartInputError("latitude must be between -90 and 90, longitude between -180 and 180 (east positive)")
    if abs(birth.latitude) > 66:
        raise ChartInputError("births beyond 66° latitude are not supported (the Lagna is ill-defined there)")
    utc = birth.utc()
    if not 1850 <= utc.year <= 2149:
        raise ChartInputError("the ephemeris covers births from 1850 to 2149")
    ts, eph = _sky()
    earth = eph["earth"]

    def ayanamsa(at) -> float:  # with nutation in longitude, as the positions are of the true equinox
        return lahiri_ayanamsa(at.tt) + math.degrees(iau2000b_radians(at)[0])

    def sidereal(body: str, at) -> float:
        _, lon, _ = earth.at(at).observe(eph[_EPHEMERIS_NAME[body]]).apparent().frame_latlon(ecliptic_frame)
        return (lon.degrees - ayanamsa(at)) % 360

    t = ts.from_datetime(utc)
    ayan = ayanamsa(t)
    lons = {b: sidereal(b, t) for b in PLANETS}
    rahu = (mean_node(t.tt) + math.degrees(iau2000b_radians(t)[0]) - ayan) % 360
    lons["Rahu"], lons["Ketu"] = rahu, (rahu + 180) % 360
    before, after = ts.tt_jd(t.tt - 0.5), ts.tt_jd(t.tt + 0.5)
    retro = {b: ((sidereal(b, after) - sidereal(b, before) + 540) % 360 - 180) < 0
             for b in ("Mars", "Mercury", "Jupiter", "Venus", "Saturn")}

    eps = math.radians(mean_obliquity(t.tdb) / 3600) + iau2000b_radians(t)[1]
    asc = (ascendant(t.gast, birth.longitude, birth.latitude, eps) - ayan) % 360
    lagna_sign = SIGNS[int(asc // 30)]

    def place(body: str, lon: float) -> Placement:
        sign = SIGNS[int(lon // 30)]
        nak, pada = nakshatra_of(lon)
        return Placement(body=body, longitude=lon, sign=sign, degree=lon % 30,
                         house=(SIGNS.index(sign) - SIGNS.index(lagna_sign)) % 12 + 1, nakshatra=nak, pada=pada)

    lagna = place("Lagna", asc)
    placements = {b: place(b, lons[b]) for b in BODIES}
    for b, p in placements.items():
        p.retrograde = retro.get(b, b in ("Rahu", "Ketu"))
        if b in EXALTED:
            p.dignity = ("exalted" if p.sign == EXALTED[b] else "debilitated" if p.sign == DEBILITATED[b]
                         else "own sign" if SIGN_LORD[p.sign] == b else None)
            p.lord_of = [h for h in range(1, 13) if SIGN_LORD[SIGNS[(SIGNS.index(lagna_sign) + h - 1) % 12]] == b]
        if b in COMBUSTION:
            gap = abs((lons[b] - lons["Sun"] + 180) % 360 - 180)
            p.combust = gap <= COMBUSTION[b][1 if p.retrograde else 0]
    dashas, balance = vimshottari(lons["Moon"], utc)
    return NatalChart(birth=birth, utc=utc, ayanamsa=ayan, lagna=lagna, placements=placements, dashas=dashas,
                      moon_dasha_balance_years=balance)


def parse_birth(date_text: str, time_text: str, tz: str, latitude: float, longitude: float,
                name: str | None = None, place: str | None = None) -> BirthData:
    """BirthData from text fields (YYYY-MM-DD, HH:MM[:SS] on the 24-hour clock)."""
    try:
        when = datetime.fromisoformat(f"{date_text.strip()}T{time_text.strip()}")
    except ValueError as e:
        raise ChartInputError("date must be YYYY-MM-DD and time HH:MM (24-hour clock)") from e
    return BirthData(when_local=when, tz=tz, latitude=latitude, longitude=longitude, name=name or None,
                     place=place or None)
