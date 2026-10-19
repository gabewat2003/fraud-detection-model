import math

import pandas as pd
import pytest

from fraud.text_norm import (
    coverage_report,
    lookup_table,
    normalize_browser,
    normalize_device,
    normalize_email,
    normalize_os,
)


@pytest.mark.parametrize(
    ("raw", "brand", "family"),
    [
        ("SM-G930V Build/NRD90M", "samsung", "samsung g930"),
        ("SAMSUNG SM-G930V", "samsung", "samsung g930"),
        ("sm-g930v", "samsung", "samsung g930"),
        ("SM-J700M", "samsung", "samsung j700"),
        ("Moto G (5) Plus Build/NPNS25.137-35-5", "motorola", "moto g"),
        ("moto e5 play", "motorola", "moto e"),
        ("XT1635-02 Build/NPNS26.118", "motorola", "motorola other"),
        ("LG-K500 Build/MXB48T", "lg", "lg k500"),
        ("HUAWEI ALE-L23 Build/HuaweiALE-L23", "huawei", "huawei ale"),
        ("ALE-L21 Build/HuaweiALE-L21", "huawei", "huawei ale"),
        ("Pixel 2 Build/OPM1.171019.011", "google", "google pixel"),
        ("Redmi Note 4", "xiaomi", "xiaomi"),
        ("iOS Device", "apple", "apple ios"),
        ("MacOS", "apple", "apple mac"),
        ("Windows", "microsoft", "windows desktop"),
        ("Trident/7.0", "microsoft", "windows desktop (ie)"),
        ("rv:11.0", "unknown", "gecko browser string"),
        ("  windows  ", "microsoft", "windows desktop"),
        ("CAM-L03 Build/HUAWEICAM-L03", "huawei", "huawei cam"),
        ("TRT-L53 Build/HUAWEITRT-L53", "huawei", "huawei trt"),
        ("something weird", "other", "other"),
    ],
)
def test_device(raw, brand, family):
    d = normalize_device(raw)
    assert (d.device_brand, d.device_family) == (brand, family)


@pytest.mark.parametrize(
    ("raw", "family", "version"),
    [
        ("Windows 10", "windows", 10.0),
        ("Windows XP", "windows", 5.1),
        ("iOS 11.2.1", "ios", 11.0),
        ("Android 7.0", "android", 7.0),
        ("Android", "android", None),
        ("Mac OS X 10_13_2", "macos", 10.13),
        ("Mac OS X 10.12", "macos", 10.12),
        ("Linux", "linux", None),
        ("func", "other", None),
    ],
)
def test_os(raw, family, version):
    o = normalize_os(raw)
    assert o.os_family == family
    assert o.os_version == version


@pytest.mark.parametrize(
    ("raw", "family", "major", "mobile"),
    [
        ("chrome 63.0", "chrome", 63.0, 0),
        ("chrome 62.0 for android", "chrome", 62.0, 1),
        ("mobile safari 11.0", "safari", 11.0, 1),
        ("ie 11.0 for desktop", "ie", 11.0, 0),
        ("samsung browser 6.2", "samsung", 6.0, 0),
        ("firefox 57.0", "firefox", 57.0, 0),
        ("edge 16.0", "edge", 16.0, 0),
        ("safari generic", "safari", None, 0),
        ("android webview 4.0", "android webview", 4.0, 1),
        ("google search application 48.0", "google app", 48.0, 0),
    ],
)
def test_browser(raw, family, major, mobile):
    b = normalize_browser(raw)
    assert (b.browser_family, b.browser_major, b.browser_is_mobile) == (family, major, mobile)


@pytest.mark.parametrize(
    ("raw", "provider", "tld", "free", "region"),
    [
        ("gmail.com", "google", "com", 1, "generic"),
        ("gmail", "google", None, 1, "generic"),
        ("hotmail.es", "microsoft", "es", 1, "es"),
        ("outlook.com", "microsoft", "com", 1, "generic"),
        ("live.com.mx", "microsoft", "com.mx", 1, "mx"),
        ("yahoo.com.mx", "yahoo", "com.mx", 1, "mx"),
        ("ymail.com", "yahoo", "com", 1, "generic"),
        ("anonymous.com", "anonymous", "com", 1, "generic"),
        ("comcast.net", "comcast", "net", 0, "generic"),
        ("sbcglobal.net", "att", "net", 0, "generic"),
        ("cfl.rr.com", "spectrum", "rr.com", 0, "generic"),
        ("sc.rr.com", "spectrum", "rr.com", 0, "generic"),
        ("scranton.edu", "edu", "edu", 0, "generic"),
        ("unknownisp.org", "other", "org", 0, "generic"),
    ],
)
def test_email(raw, provider, tld, free, region):
    e = normalize_email(raw)
    assert (e.email_provider, e.email_tld, e.is_free_mail, e.email_region) == (
        provider,
        tld,
        free,
        region,
    )


@pytest.mark.parametrize("fn", [normalize_device, normalize_os, normalize_browser, normalize_email])
@pytest.mark.parametrize("missing", [None, float("nan"), "", "   "])
def test_missing_stays_missing(fn, missing):
    # missingness is itself a signal; normalization must not invent a value
    assert all(v is None for v in vars(fn(missing)).values())


def test_lookup_table_one_row_per_distinct_value():
    t = lookup_table("email", ["gmail.com", "gmail.com", None, "hotmail.fr"])
    assert t.raw.tolist() == ["gmail.com", "hotmail.fr"]
    assert t.email_provider.tolist() == ["google", "microsoft"]


def test_coverage_report_counts_collapse():
    s = pd.Series(["hotmail.com", "hotmail.es", "outlook.com", "live.com", "mystery.io", None])
    r = coverage_report(s, "email", "email_provider")
    assert r["raw_levels"] == 5
    assert r["normalized_levels"] == 2  # microsoft + other
    assert math.isclose(r["row_coverage"], 4 / 5)
    assert r["top_unmapped"] == {"mystery.io": 1}
