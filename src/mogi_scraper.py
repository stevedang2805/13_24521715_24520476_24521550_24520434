import re, time, hashlib, argparse, logging, json
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import pandas as pd
from bs4 import BeautifulSoup
from unidecode import unidecode

try:
    import requests
    from requests.exceptions import (ConnectionError, Timeout,
                                     ChunkedEncodingError, RequestException)
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Retry config ─────────────────────────────────────────────────────────────
MAX_RETRIES   = 7        # thử lại tối đa 7 lần (~4 phút chờ cộng dồn)
RETRY_BASE    = 2.0      # giây chờ đầu tiên, nhân đôi mỗi lần (exponential backoff)
RETRY_MAX_WAIT= 60.0     # giới hạn trên mỗi lần chờ

# ── 104-column schema ────────────────────────────────────────────────────────
COLUMNS = [
    "title","price_raw","area_raw","location","description",
    "land_area","price_per_m2","main_door_direction","legal_documents",
    "property_features","land_type","width","length","bedrooms",
    "bathrooms","house_type","interior_status","usable_area",
    "property_status","area_detail","apartment_type","total_floors",
    "subdivision_block_tower","unit_code","floor_number",
    "balcony_direction","lot_code","apartment_features","office_type",
    "title_clean","description_clean","location_clean",
    "full_text","full_text_norm","full_text_no_price",
    "price_billion","area_m2","price_per_m2_calc",
    "land_area_num","width_num","length_num",
    "bedrooms_num","bathrooms_num","usable_area_num",
    "text_length","word_count","digit_count","exclamation_count",
    "phone_numbers","phone_like_count",
    "road_norm","ward_norm","district_norm","province_norm",
    "district","province",
    "house_type_norm","land_type_norm","legal_documents_norm",
    "property_features_norm","interior_status_norm",
    "clickbait_keyword_count","has_extreme_clickbait",
    "frontage_class","is_agri_land","is_can_ho_premium",
    "price_band_low","price_band_median","price_band_high",
    "price_band_mat_tien","price_band_source",
    "description_norm_no_phone_price","description_signature",
    "duplicate_group_size","duplicate_phone_nunique",
    "duplicate_road_nunique","duplicate_district_nunique",
    "duplicate_province_nunique","duplicate_phone_conflict",
    "duplicate_location_conflict","flag_duplicate_conflict",
    "flag_v4_parse_error","flag_v4_title_area_mismatch",
    "flag_v4_price_band_violation","flag_v4_price_band_severe",
    "flag_v4_missing_basic","flag_v4_clickbait_low_price",
    "flag_v4_missing_core_info",
    "v4_label_reason","v4_score",
    "label_suspicious_v4","label_suspicious_v4_strict",
    "label_v4_confidence","label_v4_reason",
    "label_suspicious_v4_NEW","label_suspicious_v6_NEW",
    "_v6_score","_v6_has_hard","_property_cat",
    "_u_hi","_u_lo","_threshold_source",
    "_r_typeaware","_canonical_label",
]

CLICKBAIT_KEYWORDS = [
    "vỡ nợ","vo no","cần tiền gấp","can tien gap","bán gấp","ban gap",
    "thanh lý","thanh ly","phát mại","phat mai","giải chấp","giai chap",
    "ngộp","ngop","cắt lỗ","cat lo","dưới giá thị trường",
]
EXTREME_CLICKBAIT = ["vỡ nợ","vo no","phát mại","phat mai","giải chấp","giai chap"]

PROVINCE_MAP = {
    "tphcm":                  "tp ho chi minh",
    "tp hcm":                 "tp ho chi minh",
    "ho chi minh":            "tp ho chi minh",
    "tp. ho chi minh":        "tp ho chi minh",
    "thanh pho ho chi minh":  "tp ho chi minh",
    "ha noi":                 "ha noi",
    "hanoi":                  "ha noi",
}

SLUG_TO_HOUSE_TYPE = {
    "nha-mat-tien-pho":    "nha mat pho, mat tien",
    "nha-hem-ngo":         "nha ngo, hem",
    "nha-biet-thu":        "nha biet thu",
    "biet-thu":            "nha biet thu",
    "nha-pho-lien-ke":     "nha pho lien ke",
    "nha-pho":             "nha pho lien ke",
    "shophouse":           "shophouse",
    "shop-house":          "shophouse",
    "can-ho":              "can ho chung cu",
    "can-ho-chung-cu":     "can ho chung cu",
    "chung-cu":            "can ho chung cu",
    "can-ho-du-an":        "can ho chung cu",
    "penthouse":           "can ho chung cu",
    "dat-nen":             "dat nen du an",
    "dat-nen-du-an":       "dat nen du an",
    "dat":                 "dat tho cu",
    "dat-tho-cu":          "dat tho cu",
    "dat-nong-nghiep":     "dat nong nghiep",
    "van-phong":           "van phong",
    "mat-bang":            "mat bang kinh doanh",
    "mat-bang-kinh-doanh": "mat bang kinh doanh",
    "kho-xuong":           "kho, xuong",
    "kho":                 "kho, xuong",
    "nha-tro":             "nha tro, phong tro",
    "phong-tro":           "nha tro, phong tro",
}
SLUG_TO_FRONTAGE = {
    "nha-mat-tien-pho":    "mat_tien",
    "nha-hem-ngo":         "hem",
    "shophouse":           "mat_tien",
    "shop-house":          "mat_tien",
    "mat-bang":            "mat_tien",
    "mat-bang-kinh-doanh": "mat_tien",
}
SLUG_TO_LAND_TYPE = {
    "dat-nen":             "dat nen du an",
    "dat-nen-du-an":       "dat nen du an",
    "dat":                 "dat tho cu",
    "dat-tho-cu":          "dat tho cu",
    "dat-nong-nghiep":     "dat nong nghiep",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", unidecode(str(text).lower())).strip()

def _norm_province(raw: str) -> str:
    return PROVINCE_MAP.get(_norm(raw), _norm(raw))

def _norm_district(raw: str) -> str:
    if not raw:
        return ""
    cleaned = re.sub(r"\s*\(.*?\)", "", raw).strip()
    cleaned = re.sub(r"^tp\.?\s*", "quan ", cleaned).strip()
    return _norm(cleaned)

def _remove_phone_price(text: str) -> str:
    t = re.sub(r"\b0\d{9,10}\b", "", text)
    t = re.sub(r"\b\d[\d.,]+\s*(ty|ti|trieu|billion|million|m2|m²)\b", "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()

def _parse_price(raw: str) -> Optional[float]:
    if not raw:
        return None
    raw = re.sub(r"-?\s*\d+[.,]?\d*\s*m[²2]?.*", "", raw, flags=re.I).strip()
    ty    = re.search(r"([\d.,]+)\s*(?:tỷ|tỉ|ty|ti)", raw, re.I)
    trieu = re.search(r"([\d.,]+)\s*(?:triệu|trieu)", raw, re.I)
    val = 0.0
    if ty:
        val += float(ty.group(1).replace(",", "."))
    if trieu:
        val += float(trieu.group(1).replace(",", ".")) / 1000
    return val if (ty or trieu) else None

def _parse_area(raw: str) -> Optional[float]:
    if not raw:
        return None
    m = re.search(r"([\d.,]+)\s*m[²2]?", raw, re.I)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r"([\d.,]+)\s*[xX×]\s*([\d.,]+)", raw)
    if m:
        return round(float(m.group(1)) * float(m.group(2)), 2)
    return None

def _parse_num(raw) -> Optional[float]:
    if not raw or str(raw) in ("nan", "None", ""):
        return None
    m = re.search(r"[\d.,]+", str(raw))
    return float(m.group().replace(",", ".")) if m else None

def _parse_location(s: str):
    if not s:
        return None, None, None, None
    parts = [p.strip() for p in str(s).split(",")]
    province = parts[-1] if parts else None
    district = parts[-2] if len(parts) >= 2 else None
    ward     = parts[-3] if len(parts) >= 3 else None
    road     = ", ".join(parts[:-3]) if len(parts) >= 4 else None
    return road, ward, district, province

def _extract_url_slug(url: str) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"/mua-([^/]+)/", url)
    return m.group(1) if m else None

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()

# ── HTTP fetch với auto-retry ────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Referer": "https://mogi.vn/",
}

def fetch(url: str, delay: float = 2.0) -> str:
    """
    Fetch URL với exponential backoff retry khi mất mạng.

    Retry khi gặp:
      - ConnectionError  (mất mạng, DNS fail, reset)
      - Timeout          (server không trả lời)
      - HTTP 429, 503    (rate limit / server quá tải)
      - ChunkedEncodingError (mạng chập chờn giữa chừng)

    Không retry khi gặp:
      - HTTP 404         (trang không tồn tại — bỏ qua)
      - HTTP 403         (bị block — dừng và cảnh báo)
    """
    if not REQUESTS_OK:
        raise ImportError("pip install requests")

    time.sleep(delay)  # polite delay trước mỗi request

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)

            # Không retry — lỗi cố định
            if resp.status_code == 404:
                raise ValueError(f"404 Not Found: {url}")
            if resp.status_code == 403:
                log.error(f"403 Forbidden — có thể bị block IP. Dừng lại.")
                raise ValueError(f"403 Forbidden: {url}")

            # Retry — server quá tải
            if resp.status_code in (429, 503):
                wait = min(RETRY_BASE * (2 ** (attempt - 1)), RETRY_MAX_WAIT)
                log.warning(f"  HTTP {resp.status_code} — thử lại sau {wait:.0f}s "
                            f"(lần {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text

        except (ConnectionError, Timeout, ChunkedEncodingError) as e:
            last_exc = e
            wait = min(RETRY_BASE * (2 ** (attempt - 1)), RETRY_MAX_WAIT)
            log.warning(f"  Mất kết nối ({type(e).__name__}) — "
                        f"thử lại sau {wait:.0f}s (lần {attempt}/{MAX_RETRIES}): {url}")
            time.sleep(wait)

        except ValueError:
            raise  # 404/403 — không retry, ném lên caller

        except RequestException as e:
            last_exc = e
            wait = min(RETRY_BASE * (2 ** (attempt - 1)), RETRY_MAX_WAIT)
            log.warning(f"  Request lỗi ({e}) — "
                        f"thử lại sau {wait:.0f}s (lần {attempt}/{MAX_RETRIES})")
            time.sleep(wait)

    raise ConnectionError(
        f"Thất bại sau {MAX_RETRIES} lần thử: {url}\nLỗi cuối: {last_exc}"
    )

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _ckpt_path(output: str) -> Path:
    return Path(output).with_suffix(".checkpoint.json")

def _ckpt_load(output: str) -> dict:
    p = _ckpt_path(output)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done_urls": [], "rows": []}

def _ckpt_save(output: str, done_urls: list, rows: list) -> None:
    p = _ckpt_path(output)
    p.write_text(
        json.dumps({"done_urls": done_urls, "rows": rows}, ensure_ascii=False),
        encoding="utf-8"
    )

def _ckpt_clear(output: str) -> None:
    p = _ckpt_path(output)
    if p.exists():
        p.unlink()

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_listing_page(html: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for li in soup.find_all("li"):
        info = li.find("div", class_="prop-info")
        if not info:
            continue
        a_tag     = info.find("a", class_="link-overlay")
        url       = a_tag["href"] if a_tag else None
        mogi_id   = None
        if url:
            m = re.search(r"-id(\d+)$", url)
            mogi_id = m.group(1) if m else None
        if not mogi_id:
            fav = li.find(attrs={"id": re.compile(r"^\d+$")})
            mogi_id = fav["id"] if fav else None

        h2        = info.find("h2", class_="prop-title")
        title     = h2.get_text(strip=True) if h2 else None
        addr_d    = info.find("div", class_="prop-addr")
        location  = addr_d.get_text(strip=True) if addr_d else None
        price_d   = info.find("div", class_="price")
        price_raw = price_d.get_text(strip=True) if price_d else None

        attrs    = info.find("ul", class_="prop-attr")
        area_raw = bedrooms = bathrooms = None
        if attrs:
            for item in [i.get_text(" ", strip=True) for i in attrs.find_all("li")]:
                ni = _norm(item)
                if "m2" in ni or "m²" in item.lower() or re.match(r"^\d", item.strip()):
                    if not area_raw:
                        area_raw = item.strip()
                elif "pn" in ni or "phong ngu" in ni:
                    bedrooms = item.strip()
                elif "wc" in ni or "nha tam" in ni or "toilet" in ni:
                    bathrooms = item.strip()

        extra   = li.find("div", class_="prop-extra")
        created = None
        if extra:
            cd = extra.find("div", class_="prop-created")
            created = cd.get_text(strip=True) if cd else None

        total_div = li.find("div", class_="total")
        n_photos  = None
        if total_div:
            sp = total_div.find("span")
            n_photos = sp.get_text(strip=True) if sp else None

        results.append({
            "url": url, "mogi_id": mogi_id, "title": title,
            "price_raw": price_raw, "area_raw": area_raw,
            "location": location, "bedrooms": bedrooms,
            "bathrooms": bathrooms, "created": created,
            "n_photos": n_photos, "url_slug": _extract_url_slug(url),
        })
    return results


def parse_detail_page(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    row  = {}

    h1 = soup.find("h1")
    row["title"] = h1.get_text(strip=True) if h1 else None

    price_div = soup.find("div", class_="price")
    row["price_raw"] = price_div.get_text(strip=True) if price_div else None

    addr = soup.find("div", class_="address")
    row["location"] = addr.get_text(strip=True) if addr else None

    desc_div = (soup.find("div", class_="pl-desc") or
                soup.find("div", class_="info-content-body"))
    row["description"] = desc_div.get_text("\n", strip=True) if desc_div else None

    canonical = soup.find("link", {"rel": "canonical"})
    can_url   = canonical["href"] if canonical else ""
    m = re.search(r"-id(\d+)", can_url)
    row["mogi_id"]  = m.group(1) if m else None
    row["url_slug"] = _extract_url_slug(can_url)

    attr_map = {}
    for attr_div in soup.find_all("div", class_="info-attr"):
        spans = attr_div.find_all("span")
        if len(spans) >= 2:
            attr_map[spans[0].get_text(strip=True)] = spans[1].get_text(" ", strip=True)

    KEY_MAP = {
        "Diện tích sử dụng":    "usable_area",
        "Diện tích thông thủy": "usable_area",
        "Diện tích đất":        "land_area",
        "Diện tích":            "area_detail",
        "Phòng ngủ":            "bedrooms",
        "Nhà tắm":              "bathrooms",
        "Pháp lý":              "legal_documents",
        "Ngày đăng":            "post_date",
        "Mã BĐS":               "_mogi_id_attr",
        "Số tầng":              "total_floors",
        "Hướng nhà":            "main_door_direction",
        "Hướng ban công":       "balcony_direction",
        "Tình trạng nội thất":  "interior_status",
        "Loại hình":            "house_type",
        "Loại đất":             "land_type",
        "Block/Tháp":           "subdivision_block_tower",
        "Tầng số":              "floor_number",
        "Mã căn":               "unit_code",
        "Mã lô":                "lot_code",
        "Loại căn hộ":          "apartment_type",
        "Đặc điểm":             "property_features",
        "Tiện ích căn hộ":      "apartment_features",
        "Loại văn phòng":       "office_type",
        "Tình trạng":           "property_status",
        "Giá/m²":               "price_per_m2",
    }
    for vk, col in KEY_MAP.items():
        if vk in attr_map and col not in row:
            row[col] = attr_map[vk]

    if "_mogi_id_attr" in row:
        if not row.get("mogi_id"):
            row["mogi_id"] = row["_mogi_id_attr"]
        del row["_mogi_id_attr"]

    land_raw = row.get("land_area", "") or ""
    m_wx = re.search(r"\(?([\d.,]+)\s*[xX×]\s*([\d.,]+)\)?", land_raw)
    if not m_wx:
        m_wx = re.search(r"([\d.,]+)\s*m\s*[xX×]\s*([\d.,]+)\s*m",
                         row.get("title", "") or "", re.I)
    if m_wx:
        row["width"]  = m_wx.group(1) + " m"
        row["length"] = m_wx.group(2) + " m"

    row["area_raw"] = land_raw
    phones = re.findall(r"0\d{9,10}", soup.get_text())
    row["phone_numbers"] = ",".join(set(phones)) if phones else None
    return row

# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(raw: dict) -> dict:
    r = {}

    for col in ["title","price_raw","area_raw","location","description",
                "land_area","price_per_m2","main_door_direction",
                "legal_documents","property_features","land_type",
                "width","length","bedrooms","bathrooms","house_type",
                "interior_status","usable_area","property_status",
                "area_detail","apartment_type","total_floors",
                "subdivision_block_tower","unit_code","floor_number",
                "balcony_direction","lot_code","apartment_features","office_type"]:
        v = raw.get(col)
        r[col] = "" if v is None or str(v) == "nan" else str(v)

    slug = raw.get("url_slug") or ""
    if not r["house_type"] and slug:
        r["house_type"] = SLUG_TO_HOUSE_TYPE.get(slug, "")
    if not r["land_type"] and slug:
        r["land_type"] = SLUG_TO_LAND_TYPE.get(slug, "")
    if not r["house_type"]:
        combined_kw = _norm(r["title"] + " " + r["description"][:300])
        TITLE_KW_MAP = [
            (r"\bshopp?\s*house\b",  "shophouse"),
            (r"\bpenthouse\b",       "can ho chung cu"),
            (r"\bbiet\s*thu\b",      "nha biet thu"),
            (r"\bvilla(s)?\b",       "nha biet thu"),
            (r"\bcan\s*ho\b",        "can ho chung cu"),
            (r"\bchung\s*cu\b",      "can ho chung cu"),
            (r"\bdat\s*nen\b",       "dat nen du an"),
            (r"\bnha\s*tro\b",       "nha tro, phong tro"),
            (r"\bphong\s*tro\b",     "nha tro, phong tro"),
            (r"\bvan\s*phong\b",     "van phong"),
            (r"\bkho\s*xuong\b",     "kho, xuong"),
            (r"\bmat\s*bang\b",      "mat bang kinh doanh"),
            (r"\blien\s*ke\b",       "nha pho lien ke"),
            (r"\bnha\s*pho\b",       "nha pho lien ke"),
        ]
        for pattern, ht_val in TITLE_KW_MAP:
            if re.search(pattern, combined_kw):
                r["house_type"] = ht_val
                break

    r["title_clean"]       = r["title"].lower().strip()
    r["description_clean"] = r["description"].lower().strip()  # không truncate
    r["location_clean"]    = r["location"].lower().strip()

    full_raw = re.sub(r"\s+", " ",
                      (r["title_clean"] + " " + r["description_clean"]).strip())
    r["full_text"]      = full_raw
    r["full_text_norm"] = _norm(full_raw)
    no_price = re.sub(r"\b\d[\d.,]+\s*(tỷ|triệu|ty|trieu|tỉ|billion|million)\b",
                      "", full_raw, flags=re.I)
    no_price = re.sub(r"\b0\d{9,10}\b", "", no_price)
    r["full_text_no_price"] = re.sub(r"\s+", " ", no_price).strip()

    price_b = _parse_price(r["price_raw"])
    r["price_billion"] = price_b
    area_m2 = _parse_area(r["land_area"]) or _parse_area(r["area_raw"])
    r["area_m2"] = area_m2
    r["price_per_m2_calc"] = (
        round(price_b * 1000 / area_m2, 6)
        if (price_b and area_m2 and area_m2 > 0) else None
    )
    r["land_area_num"]   = _parse_area(r["land_area"])
    r["width_num"]       = _parse_num(r["width"])
    r["length_num"]      = _parse_num(r["length"])
    r["bedrooms_num"]    = _parse_num(r["bedrooms"])
    r["bathrooms_num"]   = _parse_num(r["bathrooms"])
    usable_raw = _parse_area(r["usable_area"])
    r["usable_area_num"] = usable_raw if usable_raw else _parse_area(r["land_area"])

    desc = r["description"]
    r["text_length"]       = len(desc)
    r["word_count"]        = len(desc.split())
    r["digit_count"]       = sum(1 for c in desc if c.isdigit())
    r["exclamation_count"] = desc.count("!")
    r["phone_numbers"]     = raw.get("phone_numbers") or ""
    phones = re.findall(r"0\d{9,10}", desc)
    r["phone_like_count"]  = len(phones)

    road, ward, district, province = _parse_location(r["location"])
    r["province_norm"] = _norm_province(province)
    r["district_norm"] = _norm_district(district)
    r["road_norm"]     = _norm(road) if road else ""
    r["ward_norm"]     = _norm(ward) if ward else ""
    r["district"]      = r["district_norm"]
    r["province"]      = r["province_norm"]

    r["house_type_norm"]        = _norm(r["house_type"])
    r["land_type_norm"]         = _norm(r["land_type"])
    r["legal_documents_norm"]   = _norm(r["legal_documents"])
    r["property_features_norm"] = _norm(r["property_features"])
    r["interior_status_norm"]   = _norm(r["interior_status"])

    if slug in SLUG_TO_FRONTAGE:
        r["frontage_class"] = SLUG_TO_FRONTAGE[slug]
    else:
        ht_n = r["house_type"]
        if any(k in ht_n for k in ["mat pho","mat tien","shophouse","mat bang"]):
            r["frontage_class"] = "mat_tien"
        elif any(k in ht_n for k in ["hem","ngo"]):
            r["frontage_class"] = "hem"
        else:
            combined_fc = (r["description"][:200] + " " + r["title"]).lower()
            if any(k in combined_fc for k in
                   ["mặt tiền","mặt phố","mat tien","mat pho","mt "," 2mt","2 mt"]):
                r["frontage_class"] = "mat_tien"
            elif any(k in combined_fc for k in
                     ["hẻm","ngõ"," hem "," ngo ","hxh"]):
                r["frontage_class"] = "hem"
            else:
                r["frontage_class"] = "khac"

    lt = r["land_type_norm"]
    r["is_agri_land"]      = int("nong nghiep" in lt or "agri" in lt)
    r["is_can_ho_premium"] = int(
        r["apartment_type"] != "" and (r["usable_area_num"] or 0) > 80
    )
    combined_l = (r["title"] + " " + r["description"]).lower()
    r["clickbait_keyword_count"] = sum(1 for kw in CLICKBAIT_KEYWORDS if kw in combined_l)
    r["has_extreme_clickbait"]   = int(any(kw in combined_l for kw in EXTREME_CLICKBAIT))

    for col in ["price_band_low","price_band_median","price_band_high",
                "price_band_mat_tien","price_band_source"]:
        r[col] = None

    desc_norm_clean = _remove_phone_price(_norm(r["description"]))
    r["description_norm_no_phone_price"] = desc_norm_clean[:200]
    r["description_signature"] = _md5(desc_norm_clean)  # không truncate

    r["duplicate_group_size"]        = 1
    r["duplicate_phone_nunique"]     = len(set(phones))
    r["duplicate_road_nunique"]      = 1
    r["duplicate_district_nunique"]  = 1
    r["duplicate_province_nunique"]  = 1
    r["duplicate_phone_conflict"]    = 0
    r["duplicate_location_conflict"] = 0
    r["flag_duplicate_conflict"]     = 0

    for col in ["flag_v4_parse_error","flag_v4_title_area_mismatch",
                "flag_v4_price_band_violation","flag_v4_price_band_severe",
                "flag_v4_missing_basic","flag_v4_clickbait_low_price",
                "flag_v4_missing_core_info"]:
        r[col] = 0
    r["v4_label_reason"] = None
    r["v4_score"]        = 0
    for col in ["label_suspicious_v4","label_suspicious_v4_strict",
                "label_v4_confidence","label_v4_reason",
                "label_suspicious_v4_NEW","label_suspicious_v6_NEW",
                "_v6_score","_v6_has_hard","_property_cat",
                "_u_hi","_u_lo","_threshold_source",
                "_r_typeaware","_canonical_label"]:
        r[col] = None

    return r

# ── Dedup signals ─────────────────────────────────────────────────────────────

def compute_dup_signals(df: pd.DataFrame) -> pd.DataFrame:
    sig_counts = df["description_signature"].value_counts()
    df["duplicate_group_size"] = (
        df["description_signature"].map(sig_counts).fillna(1).astype(int)
    )
    for sig, grp in df.groupby("description_signature"):
        if len(grp) <= 1:
            continue
        phones = grp["phone_numbers"].dropna().unique()
        dists  = grp["district_norm"].dropna().unique()
        df.loc[grp.index, "duplicate_phone_conflict"]    = int(len(phones) > 1)
        df.loc[grp.index, "duplicate_location_conflict"] = int(len(dists) > 1)
    df["flag_duplicate_conflict"] = (
        (df["duplicate_phone_conflict"] == 1) |
        (df["duplicate_location_conflict"] == 1)
    ).astype(int)
    return df

# ── Core scraping logic ───────────────────────────────────────────────────────

TARGET_PROVINCES = {"tp ho chi minh", "ha noi"}

LISTING_URLS = {
    # Trang tổng hợp mua bán nhà đất. Mogi hiện phân trang bằng ?cp=2, ?cp=3...
    "hcm": "https://mogi.vn/ho-chi-minh/mua-nha-dat",
    "hn":  "https://mogi.vn/ha-noi/mua-nha-dat",
}

def make_page_url(base_url: str, page: int, page_param: str = "cp") -> str:
    """
    Tạo URL phân trang an toàn.

    Mogi dùng `?cp=2`, `?cp=3`, ... cho listing pages.
    Hàm này vẫn xử lý được nếu base_url đã có query string.
    """
    if page <= 1:
        return base_url
    parsed = urlparse(base_url)
    query = parse_qs(parsed.query)
    query[page_param] = [str(page)]
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def _save_partial_csv(rows: list, output: str) -> None:
    """Ghi CSV tạm để đang crawl vẫn mở file thấy dữ liệu."""
    if not rows:
        return
    try:
        _finalize(rows).to_csv(output, index=False, encoding="utf-8-sig")
    except Exception as e:
        log.warning(f"  Không thể lưu CSV tạm {output}: {e}")


def _output_with_suffix(output: str, suffix: str) -> str:
    """Tạo tên output phụ, ví dụ out.csv -> out_hcm.csv."""
    p = Path(output)
    if p.suffix.lower() == ".csv":
        return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))
    return output + f"_{suffix}.csv"

def get_detail_urls(listing_html: str) -> list:
    soup = BeautifulSoup(listing_html, "html.parser")
    urls = []
    for a in soup.find_all("a", class_="link-overlay"):
        href = a.get("href", "")
        if href and href.startswith("http"):
            urls.append(href)
    return list(dict.fromkeys(urls))

def _finalize(all_rows: list) -> pd.DataFrame:
    if not all_rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(all_rows)
    df = compute_dup_signals(df)
    cols_present = [c for c in COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in COLUMNS]
    return df[cols_present + extra]

def _scrape_one_url(base_url: str, label: str, pages: int,
                    delay: float, output: str, resume: bool) -> list:
    """
    Scrape một URL listing qua nhiều trang.
    Tự động lưu checkpoint sau mỗi tin rao.
    Nếu resume=True, bỏ qua các URL đã done từ checkpoint.
    """
    # Load checkpoint
    ckpt = _ckpt_load(output) if resume else {"done_urls": [], "rows": []}
    done_urls = set(ckpt["done_urls"])
    all_rows  = list(ckpt["rows"])

    if done_urls:
        log.info(f"  Resume: đã có {len(done_urls)} URL done, {len(all_rows)} rows từ checkpoint")

    log.info(f"\n{'='*50}\n{label} — {pages} trang\n{'='*50}")

    for page in range(1, pages + 1):
        page_url = make_page_url(base_url, page, page_param="cp")
        log.info(f"  Trang {page}: {page_url}")

        try:
            listing_html = fetch(page_url, delay=delay)
        except ConnectionError as e:
            log.error(f"  Bỏ qua trang {page} sau nhiều lần thử: {e}")
            continue

        stubs       = {s["url"]: s for s in parse_listing_page(listing_html)}
        detail_urls = [u for u in get_detail_urls(listing_html)
                       if u not in done_urls]  # bỏ qua đã done

        skipped = len(get_detail_urls(listing_html)) - len(detail_urls)
        log.info(f"  {len(detail_urls)} tin mới (bỏ qua {skipped} đã cào)")

        iter_urls = (tqdm(detail_urls, desc=f"{label} p{page}")
                     if TQDM_OK else detail_urls)

        for url in iter_urls:
            try:
                detail_html = fetch(url, delay=delay)
                raw  = parse_detail_page(detail_html)
                stub = stubs.get(url, {})
                for k, v in stub.items():
                    if k not in raw or not raw[k]:
                        raw[k] = v
                row = engineer_features(raw)

                if row.get("province_norm") not in TARGET_PROVINCES:
                    log.info(f"    Bỏ qua (tỉnh ngoài HCM+HN): {url}")
                    done_urls.add(url)
                    _ckpt_save(output, list(done_urls), all_rows)
                    continue

                all_rows.append(row)
                done_urls.add(url)

                # Lưu checkpoint + CSV tạm sau mỗi tin hợp lệ
                _ckpt_save(output, list(done_urls), all_rows)
                _save_partial_csv(all_rows, output)

            except ValueError as e:
                # 404/403 — ghi nhận và bỏ qua
                log.warning(f"    Bỏ qua ({e}): {url}")
                done_urls.add(url)
                _ckpt_save(output, list(done_urls), all_rows)
            except ConnectionError as e:
                # Hết retry — bỏ qua tin này, tiếp tục
                log.error(f"    Không thể fetch sau {MAX_RETRIES} lần: {url}")
            except Exception as e:
                log.warning(f"    Lỗi không xác định: {e} — {url}")

    return all_rows


def scrape_live(pages: int, delay: float,
                custom_url: Optional[str], output: str,
                resume: bool) -> pd.DataFrame:
    all_rows = []
    if custom_url:
        label    = custom_url.rstrip("/").split("/")[-1]
        all_rows = _scrape_one_url(custom_url, label, pages, delay, output, resume)
    else:
        for city, url in LISTING_URLS.items():
            # Dùng city-specific checkpoint khi scrape cả 2 thành phố
            city_output = _output_with_suffix(output, city)
            rows = _scrape_one_url(url, city.upper(), pages, delay,
                                   city_output, resume)
            all_rows.extend(rows)
            # Lưu partial output từng thành phố
            if rows:
                _finalize(rows).to_csv(city_output, index=False, encoding="utf-8-sig")
                log.info(f"  Đã lưu {len(rows)} rows HCM/HN → {city_output}")

    df = _finalize(all_rows)
    _ckpt_clear(output)  # xóa checkpoint khi hoàn thành
    return df


def scrape_offline(listing_htmls: list, detail_htmls: list) -> pd.DataFrame:
    stubs = {}
    for lpath in listing_htmls:
        html = Path(lpath).read_text(encoding="utf-8", errors="replace")
        for s in parse_listing_page(html):
            if s.get("url"):
                stubs[s["url"]] = s
    all_rows = []
    for dpath in detail_htmls:
        html = Path(dpath).read_text(encoding="utf-8", errors="replace")
        raw  = parse_detail_page(html)
        soup = BeautifulSoup(html, "html.parser")
        can  = soup.find("link", {"rel": "canonical"})
        url  = can["href"] if can else None
        stub = stubs.get(url, {})
        for k, v in stub.items():
            if k not in raw or not raw[k]:
                raw[k] = v
        all_rows.append(engineer_features(raw))
    return _finalize(all_rows)

# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape mogi.vn → Chợ Tốt 104-column format (v3_modified, fixed pagination ?cp=)"
    )
    parser.add_argument("--mode", choices=["live", "offline"], required=True)
    parser.add_argument("--pages",       type=int,   default=15)
    parser.add_argument("--delay",       type=float, default=2.0,
                        help="Giây delay giữa mỗi request")
    parser.add_argument("--listing-url", default=None,
                        help="URL listing tùy chỉnh (per-district, per-type)")
    parser.add_argument("--resume",      action="store_true",
                        help="Tiếp tục từ checkpoint nếu bị ngắt giữa chừng")
    parser.add_argument("--listing-html", nargs="+", default=[])
    parser.add_argument("--detail-html",  nargs="+", default=[])
    parser.add_argument("--output",      default="mogi_dataset.csv")
    args = parser.parse_args()

    if args.mode == "live":
        df = scrape_live(pages=args.pages, delay=args.delay,
                         custom_url=args.listing_url,
                         output=args.output, resume=args.resume)
    else:
        df = scrape_offline(args.listing_html, args.detail_html)

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    log.info(f"Saved {len(df)} rows × {len(df.columns)} cols → {args.output}")

    print(f"\n{'='*55}")
    print(f"  Tổng: {len(df)} tin")
    print(f"  Province:\n{df['province_norm'].value_counts().to_string()}")
    print(f"  house_type_norm (top5):\n{df['house_type_norm'].value_counts().head(5).to_string()}")
    print(f"  frontage_class:\n{df['frontage_class'].value_counts().to_string()}")
    print(f"  duplicate_group_size > 1: {(df['duplicate_group_size']>1).sum()} rows")
    print(f"  Null: price_per_m2_calc={df['price_per_m2_calc'].isna().mean()*100:.1f}%  "
          f"house_type_norm={(df['house_type_norm']=='').mean()*100:.1f}%")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
