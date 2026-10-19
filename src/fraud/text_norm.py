"""Normalize the free-text-ish identity fields: device strings, OS, browser, email domain.

IEEE-CIS stores these as raw user-agent-style strings. The same phone shows up as
"SM-G930V Build/NRD90M", "SAMSUNG SM-G930V" and "SM-G930V"; the same mail provider as
"hotmail.com", "hotmail.es", "outlook.com" and "live.com.mx". Treated as raw categories,
each spelling is a separate, rarer level, which splits the signal and inflates
cardinality. This module canonicalizes them the way one would normalize word forms:
casefold, strip noise tokens (build IDs, punctuation), map surface variants onto a lemma
(brand / family / provider) and keep the informative remainder (version, country TLD).

Every function is pure and works on one string, so the batch pipeline (lookup tables over
distinct values) and the scoring API (one transaction) share exactly the same rules.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields

import pandas as pd

OTHER = "other"


def _clean(s: object) -> str | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip().casefold()
    s = re.sub(r"\s+", " ", s)
    return s or None


# ---------------------------------------------------------------------------- devices


@dataclass(frozen=True)
class Device:
    device_brand: str | None
    device_family: str | None


# (pattern, brand, family template). Order matters: the first match wins. Family
# templates may use regex groups; "{0}" is the whole match.
_DEVICE_RULES: list[tuple[str, str, str]] = [
    (r"^(?:samsung[\s-]*)?(?:sm|gt|sgh|sch)-([a-z]+\d+)", "samsung", "samsung {1}"),
    (r"^samsung", "samsung", "samsung other"),
    (r"^(?:moto(?:rola)?)\s*([gezxc])(?=\d|\b)", "motorola", "moto {1}"),
    (r"^(?:moto|motorola|xt\d{4})", "motorola", "motorola other"),
    (r"^lg[-\s]?([a-z]+\d+)", "lg", "lg {1}"),
    (r"^lg", "lg", "lg other"),
    (
        r"^(?:huawei\s*)?((?:ale|pra|was|bnd|ane|vns|cun|lld|fig|rne|hi6210)[\w]*)",
        "huawei",
        "huawei {1}",
    ),
    (r"^huawei|^honor", "huawei", "huawei other"),
    (r"^(?:redmi|xiaomi|mi\s|mi\d)", "xiaomi", "xiaomi"),
    (r"^pixel\s*(\d)?", "google", "google pixel"),
    (r"^nexus", "google", "google nexus"),
    (r"^(?:ios device|iphone|ipad)", "apple", "apple ios"),
    (r"^macos|^mac os", "apple", "apple mac"),
    (r"^windows", "microsoft", "windows desktop"),
    (r"^trident/", "microsoft", "windows desktop (ie)"),
    (r"^rv:\d+", "unknown", "gecko browser string"),
    (r"^linux", "linux", "linux desktop"),
    (r"^(?:zte|z\d{3,4}\b)", "zte", "zte"),
    (r"^lenovo", "lenovo", "lenovo"),
    (r"^(?:htc|2pq93)", "htc", "htc"),
    (r"^(?:alcatel|\d{4}[a-z]\b)", "alcatel", "alcatel"),
    (r"^(?:asus|z00\w*)", "asus", "asus"),
    (r"^(?:oneplus|a\d{4}\b)", "oneplus", "oneplus"),
    (r"^(?:nokia|ta-\d+)", "nokia", "nokia"),
    (r"^(?:sony|[cdefg]\d{4}\b)", "sony", "sony"),
    (r"^blade", "zte", "zte"),
]
_DEVICE_RX = [(re.compile(p), b, f) for p, b, f in _DEVICE_RULES]


def normalize_device(raw: object) -> Device:
    s = _clean(raw)
    if s is None:
        return Device(None, None)
    s = re.split(r"\s*build/", s)[0].strip(" -_/;")  # Android build IDs are noise
    for rx, brand, family in _DEVICE_RX:
        m = rx.search(s)
        if m:
            groups = [m.group(0), *(g or "" for g in m.groups())]
            fam = family.format(*groups).strip()
            # strip a trailing regional-variant letter: "samsung g930v" -> "samsung g930"
            fam = re.sub(r"(\d{3})[a-z]$", r"\1", fam)
            return Device(brand, fam)
    return Device(OTHER, OTHER)


# ------------------------------------------------------------------------- OS (id_30)


@dataclass(frozen=True)
class OS:
    os_family: str | None
    os_version: float | None


_WINDOWS_NAMED = {"xp": 5.1, "vista": 6.0}


def _version(text: str, sep: str = r"[._]") -> float | None:
    m = re.search(rf"(\d+)(?:{sep}(\d+))?", text)
    if not m:
        return None
    major, minor = m.group(1), m.group(2)
    return float(f"{major}.{minor}") if minor else float(major)


def normalize_os(raw: object) -> OS:
    s = _clean(raw)
    if s is None:
        return OS(None, None)
    if s.startswith("windows"):
        for name, v in _WINDOWS_NAMED.items():
            if name in s:
                return OS("windows", v)
        return OS("windows", _version(s))
    if s.startswith("ios"):
        v = _version(s)
        return OS("ios", float(int(v)) if v is not None else None)
    if s.startswith("android"):
        v = _version(s)
        return OS("android", float(int(v)) if v is not None else None)
    if s.startswith("mac"):
        return OS("macos", _version(s))  # "10_13_2" -> 10.13: the minor is the release
    if s.startswith("linux"):
        return OS("linux", None)
    return OS(OTHER, None)


# -------------------------------------------------------------------- browser (id_31)


@dataclass(frozen=True)
class Browser:
    browser_family: str | None
    browser_major: float | None
    browser_is_mobile: int | None


_BROWSER_RULES: list[tuple[str, str]] = [
    (r"samsung", "samsung"),
    (r"chrome|chromium", "chrome"),
    (r"safari", "safari"),
    (r"firefox|mozilla", "firefox"),
    (r"edge", "edge"),
    (r"\bie\b|internet explorer", "ie"),
    (r"opera", "opera"),
    (r"webview", "android webview"),
    (r"google", "google app"),
    (r"facebook|instagram", "in-app"),
    (r"android browser|android", "android browser"),
    (r"silk", "silk"),
    (
        r"puffin|cyberfox|waterfox|palemoon|seamonkey|maxthon|comodo|iron|line|aol|chrome generic",
        "niche",
    ),
]
_BROWSER_RX = [(re.compile(p), f) for p, f in _BROWSER_RULES]


def normalize_browser(raw: object) -> Browser:
    s = _clean(raw)
    if s is None:
        return Browser(None, None, None)
    mobile = int(bool(re.search(r"mobile|for android|for ios|android|webview", s)))
    family = OTHER
    for rx, fam in _BROWSER_RX:
        if rx.search(s):
            family = fam
            break
    m = re.search(r"(\d+)(?:\.\d+)?", s)
    major = float(m.group(1)) if m else None
    return Browser(family, major, mobile)


# --------------------------------------------------------------------- email domains


@dataclass(frozen=True)
class Email:
    email_provider: str | None
    email_tld: str | None
    is_free_mail: int | None
    email_region: str | None


# surface form (first domain label) -> provider lemma
_PROVIDER_ALIASES = {
    "gmail": "google", "googlemail": "google",
    "hotmail": "microsoft", "outlook": "microsoft", "live": "microsoft", "msn": "microsoft",
    "yahoo": "yahoo", "ymail": "yahoo", "rocketmail": "yahoo",
    "aol": "aol", "aim": "aol",
    "icloud": "apple", "me": "apple", "mac": "apple",
    "anonymous": "anonymous",
    "protonmail": "proton",
    "mail": "mail.com", "gmx": "gmx", "web": "web.de",
    "att": "att", "sbcglobal": "att", "bellsouth": "att", "prodigy": "telmex",
    "comcast": "comcast", "verizon": "verizon", "cox": "cox", "charter": "spectrum",
    "twc": "spectrum", "roadrunner": "spectrum", "cfl": "spectrum", "rr": "spectrum",
    "earthlink": "earthlink", "optonline": "optimum", "juno": "juno", "netzero": "juno",
    "windstream": "windstream", "frontier": "frontier", "frontiernet": "frontier",
    "centurylink": "centurylink", "embarqmail": "centurylink", "q": "centurylink",
    "suddenlink": "suddenlink", "servicios-ta": "telmex", "scranton": "edu",
}  # fmt: skip
_FREE = {"google", "microsoft", "yahoo", "aol", "apple", "proton", "mail.com", "gmx",
         "web.de", "anonymous"}  # fmt: skip
_REGION_TLDS = {"mx": "mx", "es": "es", "fr": "fr", "de": "de", "uk": "uk", "jp": "jp",
                "it": "it", "br": "br", "ca": "ca"}  # fmt: skip


def normalize_email(raw: object) -> Email:
    s = _clean(raw)
    if s is None:
        return Email(None, None, None, None)
    labels = s.split(".")
    head, tail = labels[0], labels[1:]
    provider = _PROVIDER_ALIASES.get(head, OTHER)
    if provider == OTHER and tail and tail[-1] == "edu":
        provider = "edu"
    tld = ".".join(tail) if tail else None
    region = _REGION_TLDS.get(tail[-1], "generic") if tail else "generic"
    free = int(provider in _FREE)
    return Email(provider, tld, free, region)


# --------------------------------------------------------------------- batch helpers

_FUNCS = {
    "device": normalize_device,
    "os": normalize_os,
    "browser": normalize_browser,
    "email": normalize_email,
}
_TYPES = {"device": Device, "os": OS, "browser": Browser, "email": Email}


def lookup_table(kind: str, raw_values) -> pd.DataFrame:
    """Map each distinct raw string to its normalized fields (one row per raw value)."""
    fn = _FUNCS[kind]
    raws = sorted({v for v in raw_values if v is not None and not pd.isna(v)})
    rows = [{"raw": r, **asdict(fn(r))} for r in raws]
    df = pd.DataFrame(rows, columns=["raw", *(f.name for f in fields(_TYPES[kind]))])
    # fix dtypes from the dataclass so an empty or all-missing column keeps its SQL type
    for f in fields(_TYPES[kind]):
        df[f.name] = df[f.name].astype("string" if f.type.startswith("str") else "float64")
    df["raw"] = df["raw"].astype("string")
    return df


def coverage_report(raw: pd.Series, kind: str, field: str) -> dict:
    """How much does normalization collapse the vocabulary, and how much is left unmapped?"""
    fn = _FUNCS[kind]
    s = raw.dropna()
    counts = s.value_counts()
    mapped = {r: getattr(fn(r), field) for r in counts.index}
    norm = s.map(mapped)
    unmapped = counts[[mapped[r] == OTHER for r in counts.index]]
    return {
        "raw_levels": int(counts.size),
        "normalized_levels": int(norm.nunique()),
        "rows_non_null": int(s.size),
        "row_coverage": float((norm != OTHER).mean()) if s.size else float("nan"),
        "top_unmapped": unmapped.head(10).to_dict(),
    }
