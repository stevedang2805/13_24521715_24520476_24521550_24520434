# -*- coding: utf-8 -*-
"""
crawl_mogi_type_aware_lower.py
==============================

Bản đúng theo phương án đã bàn:

Crawl bộ giá tham chiếu Mogi:
    m         = giá trung bình quận/huyện từ TRANG CHUNG Mogi
    apartment = giá Căn hộ từ trang quận/huyện
    land      = giá Đất từ trang quận/huyện
    h         = giá Hẻm, ngõ từ trang quận/huyện
    f         = giá Mặt tiền, phố từ trang quận/huyện

Sau đó, nếu đưa thêm dataset Chợ Tốt vào, script sẽ tính cận dưới theo từng tin:
    Tin căn hộ / chung cư              -> u_lo = apartment
    Tin đất / đất nền / đất nông nghiệp -> u_lo = land
    Tin nhà / nhà hẻm / nhà mặt tiền    -> u_lo = min(apartment, land)
    Không rõ loại                       -> u_lo = min(apartment, land)

Lưu ý quan trọng:
    - Bảng tham chiếu KHÔNG thể có một cột l duy nhất đúng cho mọi tin.
    - l_house_unknown = min(apartment, land) chỉ là cận dưới dùng cho nhóm nhà/không rõ.
    - u_lo_type_aware mới là cận dưới đúng theo từng tin trong dataset.

Cài đặt:
    py -m pip install playwright pandas
    py -m playwright install chromium

Chạy chỉ crawl bảng tham chiếu:
    py crawl_mogi_type_aware_lower.py --summary-html "Giá Nhà Đất TPHCM Và Hà Nội Cập Nhât Mới Nhất T6_2026.html" --debug

Chạy crawl bảng tham chiếu + tính u_lo cho dataset:
    py crawl_mogi_type_aware_lower.py --summary-html "Giá Nhà Đất TPHCM Và Hà Nội Cập Nhât Mới Nhất T6_2026.html" --dataset-csv "Tin bất động sản.csv" --debug

Nếu muốn bản 48 dòng cho báo cáo:
    py crawl_mogi_type_aware_lower.py --summary-html "Giá Nhà Đất TPHCM Và Hà Nội Cập Nhât Mới Nhất T6_2026.html" --dataset-csv "Tin bất động sản.csv" --report-48 --debug

Output:
    mogi_summary_urls_complete.csv
    price_ref_hcm_hn_mogi_typeaware_complete_clean.csv
    price_ref_hcm_hn_mogi_typeaware_complete_audit.csv
    dataset_with_typeaware_lower_bound.csv   nếu có --dataset-csv

File reference clean có các cột:
    province, district, district_key, median, trend,
    apartment, land, l_house_unknown, m, h, f, url

File dataset enriched sẽ có thêm:
    property_type_ref
    u_lo_type_aware
    u_lo_rule
    ref_apartment
    ref_land
    ref_l_house_unknown
    ref_m
    ref_h
    ref_f
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import re
import statistics
import time
import unicodedata
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin


BASE_URL = "https://mogi.vn"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SummaryRow:
    province: str
    district: str
    district_key: str
    url: str
    m: Optional[float]
    trend: str
    order: int


@dataclass
class PriceRow:
    province: str
    district: str
    district_key: str

    median: Optional[float]
    trend: str

    apartment: Optional[float]        # a
    land: Optional[float]             # land
    l_house_unknown: Optional[float]  # min(apartment, land), dùng cho nhà/unknown
    m: Optional[float]                # representative price from summary page
    h: Optional[float]                # alley
    f: Optional[float]                # frontage

    m_source: str
    apartment_source: str
    land_source: str
    l_house_unknown_source: str
    h_source: str
    f_source: str
    source: str
    status: str
    url: str
    note: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_key(text: str) -> str:
    text = strip_diacritics(str(text or "")).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return compact_spaces(text)


def district_key(text: str) -> str:
    key = normalize_key(text)
    key = key.replace("tp thu duc", "thu duc")
    key = key.replace("thanh pho thu duc", "thu duc")
    key = key.replace("quan thu duc tp thu duc", "quan thu duc")
    key = key.replace("quan 2 tp thu duc", "quan 2")
    key = key.replace("quan 9 tp thu duc", "quan 9")
    return compact_spaces(key)


def province_key(text: str) -> str:
    key = normalize_key(text)
    if "ho chi minh" in key or key in {"hcm", "tp hcm", "tphcm"}:
        return "tp ho chi minh"
    if "ha noi" in key:
        return "ha noi"
    return key


def parse_num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    s = s.replace("\u00a0", "")
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def fmt_num(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{float(v):.1f}"


def strip_html_to_text(html: str) -> str:
    html = nfc(html or "")
    html = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style\b[^>]*>.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<sup[^>]*>\s*2\s*</sup>", "2", html, flags=re.S | re.I)
    html = re.sub(r"<[^>]+>", "\n", html)
    return html_lib.unescape(html)


def clean_lines(text: str) -> List[str]:
    out = []
    for line in (text or "").splitlines():
        line = compact_spaces(nfc(line))
        if line:
            out.append(line)
    return out


PRICE_RE = re.compile(
    r"(?<!\d)(\d{1,4}(?:[.,]\d{1,3})?)\s*(?:triệu|tr)\s*/?\s*m(?:2|²)?",
    re.IGNORECASE | re.UNICODE,
)


def extract_first_price(text: str) -> Optional[float]:
    m = PRICE_RE.search(text or "")
    if not m:
        return None
    return parse_num(m.group(1))


def strip_tags_inline(html: str) -> str:
    html = re.sub(r"<sup[^>]*>\s*2\s*</sup>", "2", html or "", flags=re.S | re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    return compact_spaces(html_lib.unescape(html))


def min_available(*values: Optional[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def compute_l_house_unknown(apartment: Optional[float], land: Optional[float]) -> Tuple[Optional[float], str]:
    """Cận dưới dùng cho tin nhà hoặc tin không rõ: min(apartment, land)."""
    if apartment is None and land is None:
        return None, "missing"
    if apartment is not None and land is not None:
        if land < apartment:
            return land, "min(apartment,land)=land"
        return apartment, "min(apartment,land)=apartment"
    if apartment is not None:
        return apartment, "apartment_only"
    return land, "land_only"


# ═══════════════════════════════════════════════════════════════════════════════
# FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_static_html(url: str, timeout: int = 25) -> Tuple[Optional[str], str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return nfc(resp.read().decode("utf-8", errors="replace")), "static"
    except Exception as e:
        return None, f"STATIC_FETCH_FAILED: {e}"


def fetch_rendered_with_playwright(
    url: str,
    wait_ms: int = 3000,
    timeout_ms: int = 60000,
    headless: bool = True,
) -> Tuple[Optional[str], Optional[str], str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return None, None, (
            "PLAYWRIGHT_NOT_INSTALLED. Run: py -m pip install playwright && "
            "py -m playwright install chromium. Error: " + str(e)
        )

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                locale="vi-VN",
                viewport={"width": 1366, "height": 950},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            page.wait_for_timeout(wait_ms)

            for frac in [0.25, 0.55, 0.85, 0.0]:
                page.evaluate(f"window.scrollTo(0, Math.floor(document.body.scrollHeight * {frac}));")
                page.wait_for_timeout(900)

            html = nfc(page.content())
            text = nfc(page.locator("body").inner_text(timeout=timeout_ms))

            context.close()
            browser.close()
            return html, text, "browser"
    except Exception as e:
        return None, None, f"PLAYWRIGHT_FETCH_FAILED: {e}"


def fetch_page(
    url: str,
    mode: str = "browser",
    wait_ms: int = 3000,
    headless: bool = True,
) -> Tuple[Optional[str], Optional[str], str]:
    if mode in {"browser", "auto"}:
        html, text, status = fetch_rendered_with_playwright(
            url=url,
            wait_ms=wait_ms,
            headless=headless,
        )
        if html is not None and text is not None:
            return html, text, status
        if mode == "browser":
            return None, None, status
        print(f"  ⚠ Browser fetch failed; fallback static. Reason: {status}")

    html, status = fetch_static_html(url)
    if html is None:
        return None, None, status
    return html, strip_html_to_text(html), status


def load_or_fetch_summary(args: argparse.Namespace) -> Tuple[str, str, str]:
    if args.summary_html:
        path = Path(args.summary_html)
        html = path.read_text(encoding="utf-8", errors="replace")
        return nfc(html), strip_html_to_text(html), f"local_summary_html:{path}"

    html, text, status = fetch_page(
        args.summary_url,
        mode=args.mode,
        wait_ms=args.wait_ms,
        headless=not args.show_browser,
    )
    if html is None or text is None:
        raise RuntimeError(f"Cannot fetch summary page: {status}")
    return html, text, status


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY PAGE PARSER: m + URL
# ═══════════════════════════════════════════════════════════════════════════════

def detect_province_from_title(title: str) -> str:
    t = normalize_key(title)
    if "ha noi" in t:
        return "ha noi"
    if "hcm" in t or "tp hcm" in t or "ho chi minh" in t:
        return "tp ho chi minh"
    return ""


def parse_summary_rows(summary_html: str) -> List[SummaryRow]:
    html = nfc(summary_html)

    h2_pat = re.compile(
        r'<h2[^>]*class="[^"]*mt-location-title[^"]*"[^>]*>(.*?)</h2>',
        re.S | re.I,
    )
    h2_matches = list(h2_pat.finditer(html))

    sections: List[Tuple[str, str]] = []
    if h2_matches:
        for idx, m in enumerate(h2_matches):
            title = strip_tags_inline(m.group(1))
            start = m.end()
            end = h2_matches[idx + 1].start() if idx + 1 < len(h2_matches) else len(html)
            province = detect_province_from_title(title)
            if province:
                sections.append((province, html[start:end]))
    else:
        sections.append(("", html))

    rows: List[SummaryRow] = []
    order = 0

    for province, section_html in sections:
        parts = section_html.split('<div class="mt-row clearfix">')
        for part in parts[1:]:
            block = part[:3000]

            href_m = re.search(
                r'<a[^>]+class="[^"]*link-overlay[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                block,
                re.S | re.I,
            )
            price_m = re.search(
                r'<span>\s*([0-9]+(?:[.,][0-9]+)?)\s*triệu\s*/?\s*m\s*<sup>\s*2\s*</sup>\s*</span>',
                block,
                re.S | re.I,
            )

            if not href_m or not price_m:
                continue

            url = urljoin(BASE_URL, html_lib.unescape(href_m.group(1).strip()))
            district = strip_tags_inline(href_m.group(2))
            m_val = parse_num(price_m.group(1))

            trend = ""
            sup_m = re.search(r'<sup[^>]*class="[^"]*change[^"]*"[^>]*>(.*?)</sup>', block, re.S | re.I)
            if sup_m:
                trend = strip_tags_inline(sup_m.group(1))
                trend = trend.replace(" ", "")
                trend = re.sub(r"(%)([▲▼↑↓])", r"\1 \2", trend)

            if not district or m_val is None:
                continue

            rows.append(
                SummaryRow(
                    province=province,
                    district=district,
                    district_key=district_key(district),
                    url=url,
                    m=m_val,
                    trend=trend,
                    order=order,
                )
            )
            order += 1

    seen = set()
    unique_rows = []
    for r in rows:
        if r.url in seen:
            continue
        seen.add(r.url)
        unique_rows.append(r)

    return unique_rows


def filter_summary_rows(rows: List[SummaryRow], report_48: bool = False) -> List[SummaryRow]:
    if not report_48:
        return rows

    drop_keys = {
        district_key("Quận 2 (TP. Thủ Đức)"),
        district_key("Quận 9 (TP. Thủ Đức)"),
        district_key("Thị Xã Sơn Tây"),
    }
    return [r for r in rows if r.district_key not in drop_keys]


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRICT PAGE PARSER: apartment / land / h / f
# ═══════════════════════════════════════════════════════════════════════════════

def is_label_line(kind: str, line: str) -> bool:
    lower = (line or "").lower()
    norm = normalize_key(line)

    if kind == "f":
        return "mặt tiền" in lower or "mat tien" in norm

    if kind == "apartment":
        return (
            "căn hộ" in lower
            or "can ho" in norm
            or "chung cư" in lower
            or "chung cu" in norm
        )

    if kind == "h":
        # Không dùng norm contains "ngo" vì tên đường "Ngô"/"Ngọc" dễ bị bắt nhầm.
        return (
            "hẻm" in lower
            or "hem" in norm
            or "ngõ" in lower
            or "hẻm, ngõ" in lower
        )

    if kind == "land":
        line_clean = compact_spaces(lower)
        norm_clean = normalize_key(line)
        if line_clean == "đất" or norm_clean == "dat":
            return True
        if line_clean.startswith("đất ") or norm_clean.startswith("dat "):
            return True
        return False

    return False


def is_any_type_label(line: str) -> bool:
    return any(is_label_line(k, line) for k in ("apartment", "land", "h", "f"))


def extract_price_near_label(text: str, kind: str, max_lookahead: int = 8) -> Optional[float]:
    lines = clean_lines(text)

    for i, line in enumerate(lines):
        if not is_label_line(kind, line):
            continue

        val = extract_first_price(line)
        if val is not None:
            return val

        for j in range(i + 1, min(len(lines), i + 1 + max_lookahead)):
            nxt = lines[j]

            if is_any_type_label(nxt):
                break

            val = extract_first_price(nxt)
            if val is not None:
                return val

    return None


def extract_ref_values_from_district_text(text: str) -> Dict[str, Optional[float]]:
    apartment = extract_price_near_label(text, "apartment")
    land = extract_price_near_label(text, "land")
    h = extract_price_near_label(text, "h")
    f = extract_price_near_label(text, "f")
    l_house_unknown, l_source = compute_l_house_unknown(apartment, land)

    return {
        "apartment": apartment,
        "land": land,
        "l_house_unknown": l_house_unknown,
        "l_house_unknown_source": l_source,
        "h": h,
        "f": f,
    }


def sanity_notes(
    apartment: Optional[float],
    land: Optional[float],
    l_house_unknown: Optional[float],
    m: Optional[float],
    h: Optional[float],
    f: Optional[float],
) -> List[str]:
    notes = []

    for name, val in [
        ("apartment", apartment),
        ("land", land),
        ("l_house_unknown", l_house_unknown),
        ("m", m),
        ("h", h),
        ("f", f),
    ]:
        if val is not None and not (0.1 <= float(val) <= 3000):
            notes.append(f"{name} outside broad range: {val}")

    if l_house_unknown is not None and h is not None and h < l_house_unknown:
        notes.append("warning:h<l_house_unknown")
    if h is not None and f is not None and f < h:
        notes.append("warning:f<h")
    if l_house_unknown is not None and f is not None and f < l_house_unknown:
        notes.append("warning:f<l_house_unknown")
    if apartment is not None and land is not None and land < apartment:
        notes.append("land<apartment: house/unknown lower uses land")

    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# EXISTING CSV REUSE
# ═══════════════════════════════════════════════════════════════════════════════

def load_existing_csv(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        print(f"⚠ existing-csv not found: {p}")
        return {}

    with p.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    by_key: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = row.get("district_key") or district_key(row.get("district", ""))
        key = district_key(key)
        if key:
            by_key[key] = row

        name_key = district_key(row.get("district", ""))
        if name_key:
            by_key[name_key] = row

    return by_key


def get_existing_value(existing: Dict[str, Dict[str, str]], summary_row: SummaryRow, col: str) -> Optional[float]:
    row = existing.get(summary_row.district_key)
    if not row:
        return None

    aliases = {
        "apartment": ["apartment", "a", "can_ho", "canho"],
        "land": ["land", "d", "dat"],
        "l_house_unknown": ["l_house_unknown", "l", "lower"],
        "h": ["h", "hem", "alley"],
        "f": ["f", "frontage"],
    }

    for c in aliases.get(col, [col]):
        if c in row and str(row.get(c, "")).strip():
            return parse_num(str(row.get(c, "") or ""))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def median_ratio(rows: List[PriceRow], num_col: str, default: float) -> float:
    vals = []
    for r in rows:
        num = getattr(r, num_col)
        den = r.m

        if num_col == "apartment":
            source = r.apartment_source
        elif num_col == "land":
            source = r.land_source
        elif num_col == "l_house_unknown":
            source = r.l_house_unknown_source
        elif num_col == "h":
            source = r.h_source
        elif num_col == "f":
            source = r.f_source
        else:
            source = ""

        if num is not None and den is not None and den > 0 and (
            source in {"html", "existing_csv"}
            or source.startswith("min(")
            or source in {"apartment_only", "land_only"}
        ):
            vals.append(float(num) / float(den))

    if not vals:
        return default
    return float(statistics.median(vals))


def fill_missing_by_ratio(rows: List[PriceRow]) -> Tuple[float, float, float, float, float]:
    apartment_ratio = median_ratio(rows, "apartment", default=0.42)
    land_ratio = median_ratio(rows, "land", default=0.35)
    l_house_ratio = median_ratio(rows, "l_house_unknown", default=0.35)
    h_ratio = median_ratio(rows, "h", default=0.60)
    f_ratio = median_ratio(rows, "f", default=1.20)

    for r in rows:
        if r.m is None or r.m <= 0:
            r.note = compact_spaces((r.note + " cannot_fallback_without_m").strip())
            continue

        if r.apartment is None:
            r.apartment = round(apartment_ratio * r.m, 1)
            r.apartment_source = "fallback_ratio"

        if r.land is None:
            r.land = round(land_ratio * r.m, 1)
            r.land_source = "fallback_ratio"

        l_new, l_src = compute_l_house_unknown(r.apartment, r.land)
        if l_new is not None:
            r.l_house_unknown = l_new
            if "fallback_ratio" in {r.apartment_source, r.land_source}:
                r.l_house_unknown_source = "min(apartment,land)_with_fallback"
            else:
                r.l_house_unknown_source = l_src
        elif r.l_house_unknown is None:
            r.l_house_unknown = round(l_house_ratio * r.m, 1)
            r.l_house_unknown_source = "fallback_ratio"

        if r.h is None:
            r.h = round(h_ratio * r.m, 1)
            r.h_source = "fallback_ratio"

        if r.f is None:
            r.f = round(f_ratio * r.m, 1)
            r.f_source = "fallback_ratio"

        type_sources = {
            r.apartment_source,
            r.land_source,
            r.h_source,
            r.f_source,
        }

        if "fallback_ratio" not in type_sources:
            r.source = "summary_m_plus_full_html_type_rows"
        elif "html" in type_sources or "existing_csv" in type_sources:
            r.source = "summary_m_plus_partial_type_rows_plus_fallback"
        else:
            r.source = "summary_m_plus_type_rows_fallback"

    return apartment_ratio, land_ratio, l_house_ratio, h_ratio, f_ratio


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def save_debug(debug_dir: Path, district: str, html: str, text: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    key = district_key(district).replace(" ", "_")
    (debug_dir / f"{key}.html").write_text(html or "", encoding="utf-8")
    (debug_dir / f"{key}.txt").write_text(text or "", encoding="utf-8")


def write_reference_clean_csv(rows: List[PriceRow], path: Path) -> None:
    fieldnames = [
        "province",
        "district",
        "district_key",
        "median",
        "trend",
        "apartment",
        "land",
        "l_house_unknown",
        "m",
        "h",
        "f",
        "url",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            for col in ["median", "apartment", "land", "l_house_unknown", "m", "h", "f"]:
                d[col] = fmt_num(d.get(col))
            w.writerow({k: d.get(k, "") for k in fieldnames})


def write_reference_audit_csv(rows: List[PriceRow], path: Path) -> None:
    fieldnames = [
        "province",
        "district",
        "district_key",
        "median",
        "trend",
        "apartment",
        "land",
        "l_house_unknown",
        "m",
        "h",
        "f",
        "m_source",
        "apartment_source",
        "land_source",
        "l_house_unknown_source",
        "h_source",
        "f_source",
        "source",
        "status",
        "url",
        "note",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            for col in ["median", "apartment", "land", "l_house_unknown", "m", "h", "f"]:
                d[col] = fmt_num(d.get(col))
            w.writerow({k: d.get(k, "") for k in fieldnames})


def write_json(rows: List[PriceRow], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)


def write_summary_urls_csv(rows: List[SummaryRow], path: Path) -> None:
    fieldnames = ["order", "province", "district", "district_key", "m", "trend", "url"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "order": r.order,
                "province": r.province,
                "district": r.district,
                "district_key": r.district_key,
                "m": fmt_num(r.m),
                "trend": r.trend,
                "url": r.url,
            })


# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE CRAWL
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_ref_for_row(
    sr: SummaryRow,
    args: argparse.Namespace,
    existing: Dict[str, Dict[str, str]],
) -> PriceRow:
    apartment = land = h = f = None
    apartment_source = land_source = h_source = f_source = "missing"
    status = ""
    note_parts: List[str] = []

    if args.prefer_existing_values:
        apartment = get_existing_value(existing, sr, "apartment")
        land = get_existing_value(existing, sr, "land")
        h = get_existing_value(existing, sr, "h")
        f = get_existing_value(existing, sr, "f")

        if apartment is not None:
            apartment_source = "existing_csv"
        if land is not None:
            land_source = "existing_csv"
        if h is not None:
            h_source = "existing_csv"
        if f is not None:
            f_source = "existing_csv"

    need_fetch = not args.prefer_existing_values or any(v is None for v in [apartment, land, h, f])

    if need_fetch:
        html, text, status = fetch_page(
            sr.url,
            mode=args.mode,
            wait_ms=args.wait_ms,
            headless=not args.show_browser,
        )

        if html is None or text is None:
            note_parts.append(f"district_fetch_failed:{status}")
        else:
            if args.debug:
                save_debug(Path(args.out_dir) / "debug_mogi_typeaware", sr.district, html, text)

            vals = extract_ref_values_from_district_text(text)

            if apartment is None and vals["apartment"] is not None:
                apartment = vals["apartment"]
                apartment_source = "html"

            if land is None and vals["land"] is not None:
                land = vals["land"]
                land_source = "html"

            if h is None and vals["h"] is not None:
                h = vals["h"]
                h_source = "html"

            if f is None and vals["f"] is not None:
                f = vals["f"]
                f_source = "html"

    l_house_unknown, l_house_source = compute_l_house_unknown(apartment, land)

    for msg in sanity_notes(apartment, land, l_house_unknown, sr.m, h, f):
        note_parts.append(msg)

    return PriceRow(
        province=sr.province,
        district=sr.district,
        district_key=sr.district_key,
        median=sr.m,
        trend=sr.trend,
        apartment=apartment,
        land=land,
        l_house_unknown=l_house_unknown,
        m=sr.m,
        h=h,
        f=f,
        m_source="summary_page",
        apartment_source=apartment_source,
        land_source=land_source,
        l_house_unknown_source=l_house_source,
        h_source=h_source,
        f_source=f_source,
        source="pending_before_fallback",
        status=status,
        url=sr.url,
        note="; ".join(note_parts),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET ENRICHMENT: type-aware u_lo
# ═══════════════════════════════════════════════════════════════════════════════

def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s


def row_text_norm(row) -> str:
    parts = []
    for col in [
        "title",
        "title_clean",
        "description",
        "description_clean",
        "full_text",
        "full_text_norm",
        "house_type",
        "house_type_norm",
        "land_type",
        "land_type_norm",
        "apartment_type",
        "apartment_type_norm",
        "office_type",
    ]:
        if col in row:
            parts.append(safe_str(row[col]))
    return normalize_key(" ".join(parts))


def nonempty_value(row, col: str) -> bool:
    if col not in row:
        return False
    v = safe_str(row[col]).strip()
    return v != "" and v.lower() not in {"nan", "none", "null", "0"}


def classify_property_type(row) -> str:
    """Classify listing into apartment / land / house / unknown.

    Priority:
        1. house if explicit house_type or strong house text
        2. apartment if explicit apartment_type or strong apartment text
        3. land if explicit land-only terms or land_type without house/apartment
        4. unknown

    Reason:
        Chợ Tốt house listings can still have land_type_norm="dat tho cu",
        so land_type alone must not override house_type.
    """
    text = row_text_norm(row)

    has_house_field = nonempty_value(row, "house_type") or nonempty_value(row, "house_type_norm")
    has_apartment_field = nonempty_value(row, "apartment_type") or nonempty_value(row, "apartment_type_norm")
    has_land_field = nonempty_value(row, "land_type") or nonempty_value(row, "land_type_norm")

    house_terms = [
        "ban nha", "nha ngo", "nha hem", "nha mat tien", "nha pho",
        "biet thu", "villa", "nha rieng", "nha cap 4", "hxh",
        "hem xe hoi", "mat pho", "mat tien",
    ]
    apartment_terms = [
        "can ho", "chung cu", "officetel", "condotel", "studio",
        "duplex", "penthouse", "block", "tower", "toa nha",
    ]
    land_terms = [
        "ban dat", "dat nen", "dat nong nghiep", "dat tho cu",
        "lo dat", "nen dat", "dat vuon", "dat du an", "dat mat tien",
    ]

    has_house_text = any(t in text for t in house_terms)
    has_apartment_text = any(t in text for t in apartment_terms)
    has_land_text = any(t in text for t in land_terms)

    # Explicit house wins over land_type because house ads often include land_type.
    if has_house_field or has_house_text:
        return "house"

    if has_apartment_field or has_apartment_text:
        return "apartment"

    if has_land_field or has_land_text:
        return "land"

    return "unknown"


def build_ref_lookup(ref_rows: List[PriceRow]) -> Dict[Tuple[str, str], PriceRow]:
    lookup: Dict[Tuple[str, str], PriceRow] = {}
    for r in ref_rows:
        pk = province_key(r.province)
        dk = district_key(r.district)
        lookup[(pk, dk)] = r
        lookup[("", dk)] = r
    return lookup


def find_ref_for_dataset_row(row, lookup: Dict[Tuple[str, str], PriceRow]) -> Optional[PriceRow]:
    province_candidates = []
    for col in ["province_norm", "province", "location_clean", "location"]:
        if col in row:
            pk = province_key(row[col])
            if pk:
                province_candidates.append(pk)
    province_candidates.append("")

    district_candidates = []
    for col in ["district_norm", "district", "location_clean", "location"]:
        if col in row:
            dk = district_key(row[col])
            if dk:
                district_candidates.append(dk)

    # More targeted extraction from location text if needed.
    loc_text = normalize_key(" ".join(safe_str(row.get(c, "")) for c in ["location", "location_clean"]))
    district_patterns = [
        r"(quan [0-9]+)",
        r"(quan [a-z ]+)",
        r"(huyen [a-z ]+)",
        r"(thi xa [a-z ]+)",
    ]
    for pat in district_patterns:
        m = re.search(pat, loc_text)
        if m:
            district_candidates.append(compact_spaces(m.group(1)))

    # Remove duplicates while keeping order.
    def unique(seq):
        seen = set()
        out = []
        for x in seq:
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    province_candidates = unique(province_candidates)
    district_candidates = unique(district_candidates)

    for pk in province_candidates:
        for dk in district_candidates:
            if (pk, dk) in lookup:
                return lookup[(pk, dk)]

    # Last fallback: district only with containment.
    for dk in district_candidates:
        for (pk2, dk2), ref in lookup.items():
            if pk2 != "":
                continue
            if dk == dk2 or dk in dk2 or dk2 in dk:
                return ref

    return None


def compute_type_aware_u_lo(property_type: str, ref: PriceRow) -> Tuple[Optional[float], str]:
    if property_type == "apartment":
        if ref.apartment is not None:
            return ref.apartment, "apartment -> u_lo = apartment"
        return ref.l_house_unknown, "apartment missing -> fallback l_house_unknown"

    if property_type == "land":
        if ref.land is not None:
            return ref.land, "land -> u_lo = land"
        return ref.l_house_unknown, "land missing -> fallback l_house_unknown"

    if property_type == "house":
        return ref.l_house_unknown, "house -> u_lo = min(apartment, land)"

    return ref.l_house_unknown, "unknown -> u_lo = min(apartment, land)"


def enrich_dataset_with_u_lo(dataset_csv: str, ref_rows: List[PriceRow], out_dir: Path) -> Path:
    import pandas as pd

    df = pd.read_csv(dataset_csv)
    lookup = build_ref_lookup(ref_rows)

    property_types = []
    u_los = []
    u_lo_rules = []
    matched_districts = []
    matched_provinces = []
    ref_apts = []
    ref_lands = []
    ref_l_house = []
    ref_ms = []
    ref_hs = []
    ref_fs = []
    match_statuses = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        ptype = classify_property_type(row_dict)
        ref = find_ref_for_dataset_row(row_dict, lookup)

        property_types.append(ptype)

        if ref is None:
            u_los.append(None)
            u_lo_rules.append("no reference match")
            matched_districts.append("")
            matched_provinces.append("")
            ref_apts.append(None)
            ref_lands.append(None)
            ref_l_house.append(None)
            ref_ms.append(None)
            ref_hs.append(None)
            ref_fs.append(None)
            match_statuses.append("missing_ref")
            continue

        u_lo, rule = compute_type_aware_u_lo(ptype, ref)
        u_los.append(u_lo)
        u_lo_rules.append(rule)
        matched_districts.append(ref.district)
        matched_provinces.append(ref.province)
        ref_apts.append(ref.apartment)
        ref_lands.append(ref.land)
        ref_l_house.append(ref.l_house_unknown)
        ref_ms.append(ref.m)
        ref_hs.append(ref.h)
        ref_fs.append(ref.f)
        match_statuses.append("matched")

    df["property_type_ref"] = property_types
    df["u_lo_type_aware"] = u_los
    df["u_lo_rule"] = u_lo_rules
    df["ref_match_status"] = match_statuses
    df["ref_province"] = matched_provinces
    df["ref_district"] = matched_districts
    df["ref_apartment"] = ref_apts
    df["ref_land"] = ref_lands
    df["ref_l_house_unknown"] = ref_l_house
    df["ref_m"] = ref_ms
    df["ref_h"] = ref_hs
    df["ref_f"] = ref_fs

    dataset_name = Path(dataset_csv).stem
    out_path = out_dir / f"{dataset_name}_with_typeaware_lower_bound.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Summary output
    print("\nDataset enrichment summary:")
    print(df["property_type_ref"].value_counts(dropna=False).to_string())
    print("\nReference match summary:")
    print(df["ref_match_status"].value_counts(dropna=False).to_string())

    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# ARGPARSE + MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl Mogi reference and compute type-aware lower bound.")
    parser.add_argument("--summary-url", default="https://mogi.vn/gia-nha-dat")
    parser.add_argument("--summary-html", default="", help="Optional local saved summary HTML file.")
    parser.add_argument("--dataset-csv", default="", help="Optional dataset CSV to enrich with type-aware u_lo.")
    parser.add_argument("--existing-csv", default="", help="Optional existing reference CSV to reuse values.")
    parser.add_argument(
        "--prefer-existing-values",
        action="store_true",
        help="Use apartment/land/h/f from existing reference CSV first, then only fetch missing fields.",
    )
    parser.add_argument(
        "--report-48",
        action="store_true",
        help="Drop Quận 2, Quận 9, and Thị Xã Sơn Tây for report-compatible version.",
    )
    parser.add_argument("--mode", choices=["browser", "static", "auto"], default="browser")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wait-ms", type=int, default=3000)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--show-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("SAD-VREAL — Mogi reference + TYPE-AWARE lower bound")
    print("Reference: m from summary; apartment/land/h/f from district pages.")
    print("Per listing u_lo: apartment->apartment, land->land, house/unknown->min(apartment,land).")
    print("=" * 100)

    summary_html, summary_text, summary_status = load_or_fetch_summary(args)
    print(f"Summary source: {summary_status}")

    summary_rows = parse_summary_rows(summary_html)
    print(f"Summary rows parsed from summary page: {len(summary_rows)}")

    if args.report_48:
        summary_rows = filter_summary_rows(summary_rows, report_48=True)
        print(f"After --report-48 filter: {len(summary_rows)} rows")

    if not summary_rows:
        raise RuntimeError("No summary rows parsed. Check summary HTML or page structure.")

    suffix = "report48" if args.report_48 else "complete"
    summary_urls_path = out_dir / f"mogi_summary_urls_{suffix}.csv"
    write_summary_urls_csv(summary_rows, summary_urls_path)
    print(f"Summary URL list saved: {summary_urls_path.resolve()}")

    existing = load_existing_csv(args.existing_csv)
    if existing:
        print(f"Existing reference CSV loaded keys: {len(existing)}")

    ref_rows: List[PriceRow] = []
    for idx, sr in enumerate(summary_rows, start=1):
        city = "HCM" if sr.province == "tp ho chi minh" else "HN"
        print(f"\n[{idx:02}/{len(summary_rows)}] {city} | {sr.district}")
        print(f"  m={sr.m} from summary | URL={sr.url}")

        pr = scrape_ref_for_row(sr, args, existing)
        print(
            f"  apartment={pr.apartment} ({pr.apartment_source}), "
            f"land={pr.land} ({pr.land_source}), "
            f"l_house_unknown={pr.l_house_unknown} ({pr.l_house_unknown_source}), "
            f"h={pr.h} ({pr.h_source}), "
            f"f={pr.f} ({pr.f_source})"
        )
        if pr.note:
            print(f"  ⚠ {pr.note}")

        ref_rows.append(pr)
        time.sleep(args.sleep)

    print("\n" + "=" * 100)
    print("Post-processing: fill missing apartment/land/l_house_unknown/h/f by ratio fallback")
    apartment_ratio, land_ratio, l_house_ratio, h_ratio, f_ratio = fill_missing_by_ratio(ref_rows)
    print(
        "Ratios used: "
        f"apartment/m={apartment_ratio:.4f}, "
        f"land/m={land_ratio:.4f}, "
        f"l_house_unknown/m={l_house_ratio:.4f}, "
        f"h/m={h_ratio:.4f}, "
        f"f/m={f_ratio:.4f}"
    )

    for r in ref_rows:
        extra = sanity_notes(r.apartment, r.land, r.l_house_unknown, r.m, r.h, r.f)
        if extra:
            r.note = "; ".join([x for x in [r.note, "; ".join(extra)] if x])

    ref_clean_path = out_dir / f"price_ref_hcm_hn_mogi_typeaware_{suffix}_clean.csv"
    ref_audit_path = out_dir / f"price_ref_hcm_hn_mogi_typeaware_{suffix}_audit.csv"
    ref_json_path = out_dir / f"price_ref_hcm_hn_mogi_typeaware_{suffix}.json"

    write_reference_clean_csv(ref_rows, ref_clean_path)
    write_reference_audit_csv(ref_rows, ref_audit_path)
    write_json(ref_rows, ref_json_path)

    n_missing_any = sum(1 for r in ref_rows if any(v is None for v in [r.apartment, r.land, r.l_house_unknown, r.m, r.h, r.f]))
    n_with_fallback = sum(
        1
        for r in ref_rows
        if "fallback_ratio" in {r.apartment_source, r.land_source, r.l_house_unknown_source, r.h_source, r.f_source}
        or "fallback" in r.l_house_unknown_source
    )
    n_land_lower = sum(
        1
        for r in ref_rows
        if r.apartment is not None and r.land is not None and r.land < r.apartment
    )

    print("\n" + "=" * 100)
    print("REFERENCE RESULT")
    print("=" * 100)
    print(f"Reference rows:          {len(ref_rows)}")
    print(f"Rows with fallback:      {n_with_fallback}")
    print(f"Rows where land < apt:   {n_land_lower}")
    print(f"Rows missing any value:  {n_missing_any}")
    print(f"URL list CSV:            {summary_urls_path.resolve()}")
    print(f"REFERENCE CLEAN CSV:     {ref_clean_path.resolve()}")
    print(f"REFERENCE AUDIT CSV:     {ref_audit_path.resolve()}")
    print(f"REFERENCE JSON:          {ref_json_path.resolve()}")

    if args.dataset_csv:
        dataset_path = Path(args.dataset_csv)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {dataset_path}")

        print("\n" + "=" * 100)
        print("DATASET ENRICHMENT: computing type-aware u_lo")
        print("=" * 100)
        out_dataset_path = enrich_dataset_with_u_lo(str(dataset_path), ref_rows, out_dir)
        print(f"DATASET WITH TYPE-AWARE U_LO: {out_dataset_path.resolve()}")

    print("=" * 100)


if __name__ == "__main__":
    main()
