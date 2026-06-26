"""
STEP1_mogi.py
=============
Pipeline gán nhãn weak-supervision cho dataset mogi.vn.
Dùng với file: mogi_clean.csv (66 cột, chưa có label)
Tham chiếu:    price_ref_hcm_hn_mogi.csv (bảng giá 51 quận/huyện)

Chạy:
    python STEP1_mogi.py

Output:
    mogi_labeled.csv   — dataset đầy đủ đã có label + tất cả cột trung gian
    mogi_model_input.csv — chỉ giữ các cột cần thiết cho STEP2 (model training)

Yêu cầu:
    pip install pandas numpy unidecode
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from unidecode import unidecode

# ══════════════════════════════════════════════════════════════════
# CẤU HÌNH — chỉnh đường dẫn nếu cần
# ══════════════════════════════════════════════════════════════════
INPUT_CSV    = "mogi_clean.csv"
PRICE_REF    = "price_ref_hcm_hn_mogi.csv"
OUTPUT_FULL  = "mogi_labeled.csv"          # toàn bộ cột + label
OUTPUT_MODEL = "mogi_model_input.csv"      # chỉ cột cần cho STEP2

# Ngưỡng r(x) — tự động tính từ phân phối thực nghiệm của dataset đầu vào.
# Không hardcode giá trị tuyệt đối; thay vào đó giữ cố định PERCENTILE.
# Hai hằng số dưới đây là PERCENTILE (0–100), không phải giá trị r(x).
# Giá trị tuyệt đối THR_SEVERE / THR_VIOLATION được tính tại BƯỚC 5
# sau khi r(x) đã được tính trên toàn dataset.
#
#   PCTL_SEVERE    = 90  → ~10% ads lệch giá cực đoan nhất → hard signal
#   PCTL_VIOLATION = 82  → ~8% kế tiếp → soft signal
#
# Ý nghĩa: giá trị tuyệt đối thay đổi theo dataset (vd: 2.5 trên Chợ Tốt,
# 2.65 trên mogi) nhưng selectivity (top 10% / top 8%) luôn nhất quán.
PCTL_SEVERE    = 90   # percentile cho hard signal (price_band_severe)
PCTL_VIOLATION = 82   # percentile cho soft signal (price_band_violation)
THR_SOFT_MIN   = 5    # tổng điểm soft để label suspicious

# Hệ số nới lỏng cho căn hộ cao cấp (premium apartment)
PREMIUM_RELAX = 1.3

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def norm(s):
    """Lowercase + remove diacritics + collapse spaces."""
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", unidecode(str(s).lower())).strip()


def dist_key(d):
    """Strip prefix quan/huyen/thi xa khỏi tên quận."""
    s = norm(d)
    return re.sub(r"^(quan|huyen|thi xa|thanh pho|tp)\s+", "", s).strip()


# ══════════════════════════════════════════════════════════════════
# BƯỚC 1 — Load data
# ══════════════════════════════════════════════════════════════════

print("=" * 60)
print("STEP1_mogi — Weak-supervision labeling pipeline")
print("=" * 60)

df  = pd.read_csv(INPUT_CSV, low_memory=False)
ref = pd.read_csv(PRICE_REF)

print(f"\n[1] Loaded: {len(df)} rows × {len(df.columns)} cols")
print(f"    Price ref: {len(ref)} districts")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 2 — Preprocessing (dedup + price filter)
# ══════════════════════════════════════════════════════════════════

n0 = len(df)

# 2a. Exact duplicate: cùng full_text + price + area
df = df.drop_duplicates(
    subset=["full_text", "price_billion", "area_m2"], keep="first"
)
n1 = len(df)

# 2b. Near-duplicate: cùng description_clean + price + area
df = df.drop_duplicates(
    subset=["description_clean", "price_billion", "area_m2"], keep="first"
)
n2 = len(df)

# 2c. Loại bỏ giá/m² phi thực tế (< 1 hoặc > 2000 triệu/m²)
rho = df["price_per_m2_calc"]
df  = df[(rho >= 1) & (rho <= 2000)].copy().reset_index(drop=True)
n3  = len(df)

print(f"\n[2] Preprocessing: {n0} → {n1} → {n2} → {n3}")
print(f"    Removed: {n0-n1} exact dup | {n1-n2} near-dup | {n2-n3} bad price")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 3 — Build price reference lookup
# ══════════════════════════════════════════════════════════════════

lookup = {}
for _, r in ref.iterrows():
    prov = norm(r["province"])
    dk   = dist_key(r["district"])
    lookup[(prov, dk)] = {
        "a":   float(r["apartment"]),
        "lnd": float(r["land"]),
        "h":   float(r["h"]),
        "f":   float(r["f"]),
        "m":   float(r["m"]),
    }

# Alias cho Quận 2, 9, Thủ Đức (ref có suffix "(TP. Thủ Đức)" trong tên)
for suffix, short in [
    ("2 (tp. thu duc)", "2"),
    ("9 (tp. thu duc)", "9"),
    ("thu duc (tp. thu duc)", "thu duc"),
]:
    key_src = ("tp ho chi minh", suffix)
    key_dst = ("tp ho chi minh", short)
    if key_src in lookup:
        lookup[key_dst] = lookup[key_src]

print(f"\n[3] Price lookup: {len(lookup)} keys")


def get_threshold(prov: str, dist: str):
    """
    Tra bảng giá cho (tỉnh, quận).
    Trả về (dict giá, nguồn): exact / fuzzy / province_mean / no_match
    """
    dk  = dist_key(dist)
    pv  = norm(prov)
    key = (pv, dk)
    if key in lookup:
        return lookup[key], "exact"
    # Fuzzy: so 8 ký tự đầu
    for (p, d), v in lookup.items():
        if p == pv and d[:8] == dk[:8] and len(dk) >= 4:
            return v, "fuzzy"
    # Fallback: trung bình cấp tỉnh
    prov_vals = [v for (p, _), v in lookup.items() if p == pv]
    if prov_vals:
        return (
            {k: float(np.mean([x[k] for x in prov_vals]))
             for k in ("a", "lnd", "h", "f", "m")},
            "province_mean",
        )
    return None, "no_match"

# ══════════════════════════════════════════════════════════════════
# BƯỚC 4 — Property classification từ house_type_norm
#
# Mogi đã cung cấp house_type rõ ràng cho từng loại BĐS.
# Mapping trực tiếp house_type_norm → _property_cat
# (không cần keyword heuristic như Chợ Tốt)
# ══════════════════════════════════════════════════════════════════

HT_TO_CAT = {
    # Nhà ở
    "nha mat pho, mat tien": "nha_mat_tien",
    "nha ngo, hem":          "nha_hem",
    "nha biet thu":          "nha_mat_tien",   # biệt thự → ngưỡng mặt tiền
    "nha pho lien ke":       "nha_mat_tien",   # liền kề → ngưỡng mặt tiền
    "shophouse":             "nha_mat_tien",   # shophouse → ngưỡng mặt tiền
    "mat bang kinh doanh":   "nha_mat_tien",   # mặt bằng → ngưỡng mặt tiền
    # Căn hộ
    "can ho chung cu":       "can_ho",
    # Đất
    "dat tho cu":            "dat",
    "dat nen du an":         "dat",
    "dat nong nghiep":       "dat",
    # Khác
    "nha tro, phong tro":    "nha_hem",        # nhà trọ → ngưỡng hẻm
    "van phong":             "other",
    "kho, xuong":            "other",
}

# Fallback cho 154 rows NaN house_type_norm
APT_KW  = re.compile(
    r"\b(can ho|chung cu|ccmn|officetel|penthouse|duplex)\b"
)
LAND_KW = re.compile(
    r"\b(dat nen|lo dat|dat tho cu|dat o |dat phan lo|"
    r"dat vuon|dat trong|dat nong nghiep|ban dat)\b"
)


def classify(row) -> str:
    ht = norm(str(row.get("house_type_norm", "")))

    # Direct mapping (96%+ rows)
    if ht and ht in HT_TO_CAT:
        return HT_TO_CAT[ht]

    # Fallback: keyword trong title + description
    text = norm(str(row.get("title_clean", ""))) + " " + \
           norm(str(row.get("full_text", ""))[:300])
    if APT_KW.search(text):
        return "can_ho"
    if LAND_KW.search(text):
        return "dat"

    # Cuối cùng: frontage_class
    fc = str(row.get("frontage_class", ""))
    if fc == "mat_tien":
        return "nha_mat_tien"
    if fc == "hem":
        return "nha_hem"

    return "nha_mat_tien"  # safe default


df["_property_cat"] = df.apply(classify, axis=1)

print(f"\n[4] Property classification:")
for cat, n in df["_property_cat"].value_counts().items():
    pct = n / len(df) * 100
    print(f"    {cat:<20} {n:>5}  ({pct:.1f}%)")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 5 — Tính r(x) và u_hi / u_lo
# ══════════════════════════════════════════════════════════════════

u_hi_list, u_lo_list, src_list = [], [], []

for _, row in df.iterrows():
    thr, src = get_threshold(
        row.get("province_norm", ""),
        row.get("district_norm", ""),
    )
    if thr is None:
        u_hi_list.append(np.nan)
        u_lo_list.append(np.nan)
        src_list.append(src)
        continue

    # u_hi: ngưỡng trên theo loại BĐS
    u_hi_map = {
        "nha_mat_tien": thr["f"],
        "nha_hem":      thr["h"],
        "can_ho":       thr["a"],
        "dat":          thr["lnd"],
        "other":        thr["m"],
    }
    u_hi = u_hi_map[row["_property_cat"]]

    # is_can_ho_premium: nới lỏng ngưỡng 1.3× cho căn hộ diện tích > 80m²
    if row["_property_cat"] == "can_ho" and float(row.get("usable_area_num") or 0) > 80:
        u_hi = u_hi * PREMIUM_RELAX

    # u_lo: ngưỡng sàn = giá thấp nhất toàn quận (adaptive floor)
    u_lo = min(thr["a"], thr["lnd"], thr["h"], thr["f"])

    u_hi_list.append(u_hi)
    u_lo_list.append(u_lo)
    src_list.append(src)

df["_u_hi"]             = u_hi_list
df["_u_lo"]             = u_lo_list
df["_threshold_source"] = src_list

# Tính r(x) = max(ρ/u_hi, u_lo/ρ)
rho = df["price_per_m2_calc"].astype(float)
df["_r_typeaware"] = np.maximum(rho / df["_u_hi"], df["_u_lo"] / rho)

# ── Tự động tính ngưỡng từ phân phối r(x) của dataset này ──────────────
# Giữ cố định PERCENTILE (PCTL_SEVERE/PCTL_VIOLATION) thay vì giá trị
# tuyệt đối, để selectivity nhất quán bất kể phân phối r(x) của dataset.
# Kết quả: THR_SEVERE ≈ 2.5 trên Chợ Tốt, ≈ 2.65 trên mogi — khác nhau
# nhưng đều chọn ~top 10% / top 8% ads lệch giá nhất.
r_valid       = df["_r_typeaware"].dropna()
THR_SEVERE    = float(np.percentile(r_valid, PCTL_SEVERE))
THR_VIOLATION = float(np.percentile(r_valid, PCTL_VIOLATION))
# ────────────────────────────────────────────────────────────────────────

print(f"\n[5] r(x) computed:")
print(f"    District match — exact:        {(df['_threshold_source']=='exact').sum()}")
print(f"    District match — fuzzy:        {(df['_threshold_source']=='fuzzy').sum()}")
print(f"    District match — province_mean:{(df['_threshold_source']=='province_mean').sum()}")
print(f"    District match — no_match:     {(df['_threshold_source']=='no_match').sum()}")
r = df["_r_typeaware"].dropna()
print(f"    r(x): median={r.median():.3f} | "
      f"P{PCTL_VIOLATION}={r.quantile(PCTL_VIOLATION/100):.3f} | "
      f"P{PCTL_SEVERE}={r.quantile(PCTL_SEVERE/100):.3f}")
print(f"    THR_VIOLATION (P{PCTL_VIOLATION}) = {THR_VIOLATION:.4f}  [soft signal]")
print(f"    THR_SEVERE    (P{PCTL_SEVERE}) = {THR_SEVERE:.4f}  [hard signal]")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 6 — Tính các signal flags
# ══════════════════════════════════════════════════════════════════

r = df["_r_typeaware"]

# Hard signals
df["flag_v4_price_band_severe"] = (r > THR_SEVERE).astype(int)

# Title–area mismatch: kích thước ghi trong title khác với area_m2 khai báo
def detect_title_area_mismatch(row) -> int:
    title = str(row.get("title", ""))
    area  = row.get("area_m2")
    if pd.isna(area):
        return 0
    # Tìm WxL trong title: "4x15", "5x20m", "4,5x18"
    m = re.search(r"(\d+[.,]?\d*)\s*[xX×]\s*(\d+[.,]?\d*)", title)
    if not m:
        return 0
    try:
        w = float(m.group(1).replace(",", "."))
        l = float(m.group(2).replace(",", "."))
        area_title = w * l
        # Mismatch nếu lệch > 20%
        # Ngưỡng 100%: chỉ flag khi WxL khác area_m2 gấp đôi (tránh false positive nở hậu, đất hình thang)
        return int(abs(area_title - float(area)) / float(area) > 1.00)
    except (ValueError, ZeroDivisionError):
        return 0

df["flag_v4_title_area_mismatch"] = df.apply(detect_title_area_mismatch, axis=1)

# Soft signals
CLICKBAIT_KW = [
    "vỡ nợ", "vo no", "cần tiền gấp", "can tien gap",
    "thanh lý", "thanh ly",
    "phát mại", "phat mai", "giải chấp", "giai chap",
    "ngộp", "ngop", "cắt lỗ", "cat lo",
    # "bán gấp" / "ban gap" bị loại — quá phổ biến trong tin hợp lệ,
    # không đủ tính phân biệt để dùng làm soft signal
]

def has_clickbait_low_price(row) -> int:
    text  = (str(row.get("title", "")) + " " + str(row.get("description", ""))).lower()
    rho_v = row.get("price_per_m2_calc")
    u_lo  = row.get("_u_lo")
    if pd.isna(rho_v) or pd.isna(u_lo):
        return 0
    has_kw = any(kw in text for kw in CLICKBAIT_KW)
    return int(has_kw and float(rho_v) < float(u_lo))

df["flag_v4_price_band_violation"] = ((r > THR_VIOLATION) & (r <= THR_SEVERE)).astype(int)
df["flag_v4_clickbait_low_price"]  = df.apply(has_clickbait_low_price, axis=1)
df["flag_v4_missing_basic"]        = (
    df["price_billion"].isna() | df["area_m2"].isna()
).astype(int)
df["flag_v4_missing_core_info"]    = (
    df["road_norm"].fillna("").eq("") | (df["word_count"].fillna(0) < 20)
).astype(int)

print(f"\n[6] Signal activation:")
for col in [
    "flag_v4_price_band_severe",
    "flag_v4_title_area_mismatch",
    "flag_v4_price_band_violation",
    "flag_v4_clickbait_low_price",
    "flag_v4_missing_basic",
    "flag_v4_missing_core_info",
]:
    n = int(df[col].sum())
    print(f"    {col:<42} {n:>5}  ({n/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 7 — Gán nhãn
# ══════════════════════════════════════════════════════════════════

# Soft score
SOFT_WEIGHTS = {
    "flag_v4_price_band_violation":  3,
    "flag_v4_clickbait_low_price":   3,
    "flag_v4_missing_basic":         2,
    "flag_v4_missing_core_info":     2,
}
df["_v6_score"] = sum(df[c] * w for c, w in SOFT_WEIGHTS.items())

# Hard signal = severe OR mismatch
df["_v6_has_hard"] = (
    (df["flag_v4_price_band_severe"] == 1) |
    (df["flag_v4_title_area_mismatch"] == 1)
).astype(int)

# Label: hard signal HOẶC soft score ≥ ngưỡng
df["label_suspicious_v6_NEW"] = (
    (df["_v6_has_hard"] == 1) | (df["_v6_score"] >= THR_SOFT_MIN)
).astype(int)

n_sus   = int(df["label_suspicious_v6_NEW"].sum())
n_hard  = int(df["_v6_has_hard"].sum())
n_soft  = int(((df["_v6_has_hard"] == 0) & (df["label_suspicious_v6_NEW"] == 1)).sum())
sus_pct = n_sus / len(df) * 100

print(f"\n[7] Labeling results:")
print(f"    n total       = {len(df):,}")
print(f"    n suspicious  = {n_sus:,}  ({sus_pct:.2f}%)")
print(f"      └ hard only = {n_hard:,}")
print(f"      └ soft only = {n_soft:,}")

print(f"\n    Suspicious rate by property_cat:")
for cat, grp in df.groupby("_property_cat"):
    sr = grp["label_suspicious_v6_NEW"].mean() * 100
    print(f"      {cat:<20} {sr:>5.1f}%  (n={len(grp)})")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 8 — Duplicate group signals
# ══════════════════════════════════════════════════════════════════

sig_counts = df["description_signature"].value_counts()
df["duplicate_group_size"] = (
    df["description_signature"].map(sig_counts).fillna(1).astype(int)
)

df["duplicate_phone_conflict"]    = 0
df["duplicate_location_conflict"] = 0

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

df["duplicate_road_nunique"]     = df.groupby("description_signature")["road_norm"].transform("nunique")
df["duplicate_district_nunique"] = df.groupby("description_signature")["district_norm"].transform("nunique")
df["duplicate_province_nunique"] = df.groupby("description_signature")["province_norm"].transform("nunique")
df["duplicate_phone_nunique"]    = df.groupby("description_signature")["phone_numbers"].transform("nunique")

dup_gt1 = (df["duplicate_group_size"] > 1).sum()
print(f"\n[8] Duplicate signals: {dup_gt1} rows in groups > 1")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 9 — Canonical label
# ══════════════════════════════════════════════════════════════════

df["_canonical_label"] = df["label_suspicious_v6_NEW"]

# ══════════════════════════════════════════════════════════════════
# BƯỚC 10 — Lưu output đầy đủ
# ══════════════════════════════════════════════════════════════════

df.to_csv(OUTPUT_FULL, index=False, encoding="utf-8-sig")
print(f"\n[9] Saved full output → {OUTPUT_FULL}  ({len(df)} rows × {len(df.columns)} cols)")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 11 — Chọn cột cho STEP2 (model training)
#
# Giữ đúng 27 tabular features + 2 text features + label
# Loại bỏ:
#   - Các cột raw text gốc (title, description, location...)
#   - Các cột all-null trong dataset này
#   - Các cột nội bộ pipeline (_u_hi, _u_lo, _r_typeaware...)
#   - Các cột chỉ dùng để debug (v4_*, label_v4_*)
# ══════════════════════════════════════════════════════════════════

# 27 tabular features cho M2 (khớp đúng Table 6 bài báo)
FEAT_NUM = [
    "price_billion",
    "area_m2",
    "price_per_m2_calc",
    "bedrooms_num",
    "bathrooms_num",
    "usable_area_num",
    "length_num",
    "width_num",
    "land_area_num",
    "word_count",
    "digit_count",
    "exclamation_count",
    "text_length",
    "phone_like_count",
    "is_agri_land",
    "is_can_ho_premium",
    "has_extreme_clickbait",
    "clickbait_keyword_count",
    "flag_v4_parse_error",
    "flag_v4_title_area_mismatch",
    "flag_v4_missing_core_info",
    "duplicate_group_size",
    "duplicate_road_nunique",
    "duplicate_district_nunique",
    "duplicate_phone_nunique",
]

FEAT_CAT = [
    "district_norm",
    "province_norm",
    "frontage_class",
    "house_type_norm",
    "land_type_norm",
    "legal_documents_norm",
]

# Text features cho M1/M3/M4/M5/M6/M7
TEXT_COLS = [
    "full_text_no_price",   # TF-IDF input
    "full_text_norm",       # PhoBERT input
]

# Metadata (giữ để trace lại dễ hơn)
META_COLS = [
    "title",
    "location",
    "district",
    "province",
    "district_norm",
    "province_norm",
    "description_signature",
]

# Label
LABEL_COLS = [
    "label_suspicious_v6_NEW",
    "_canonical_label",
    "_property_cat",
    "_v6_has_hard",
    "_v6_score",
    "_threshold_source",
]

# Signal flags (hữu ích cho analysis, loại trước khi train)
FLAG_COLS = [
    "flag_v4_price_band_severe",
    "flag_v4_price_band_violation",
    "flag_v4_clickbait_low_price",
    "flag_v4_title_area_mismatch",
    "flag_v4_missing_basic",
    "flag_v4_missing_core_info",
    "flag_duplicate_conflict",
    "duplicate_phone_conflict",
    "duplicate_location_conflict",
]

# Tổng hợp các cột cần giữ
keep_cols = list(dict.fromkeys(
    META_COLS + FEAT_NUM + FEAT_CAT + TEXT_COLS + FLAG_COLS + LABEL_COLS
))

# Chỉ giữ cột thực sự tồn tại trong df
keep_cols = [c for c in keep_cols if c in df.columns]

# Thêm is_can_ho_premium, has_extreme_clickbait, clickbait_keyword_count
# nếu chưa có trong df (scraper v2 chưa tính)
for col in ["has_extreme_clickbait", "clickbait_keyword_count", "flag_v4_parse_error"]:
    if col not in df.columns:
        df[col] = 0

# Re-check sau khi fill
keep_cols = [c for c in keep_cols if c in df.columns]

df_model = df[keep_cols].copy()
df_model.to_csv(OUTPUT_MODEL, index=False, encoding="utf-8-sig")

print(f"\n[10] Model input saved → {OUTPUT_MODEL}")
print(f"     {len(df_model)} rows × {len(df_model.columns)} cols")
print(f"\n     Cột được giữ:")
print(f"       Metadata:       {len(META_COLS)} cols")
print(f"       Numeric feat:   {len([c for c in FEAT_NUM if c in df.columns])} cols")
print(f"       Categorical:    {len([c for c in FEAT_CAT if c in df.columns])} cols")
print(f"       Text feat:      {len([c for c in TEXT_COLS if c in df.columns])} cols")
print(f"       Signal flags:   {len([c for c in FLAG_COLS if c in df.columns])} cols")
print(f"       Labels:         {len([c for c in LABEL_COLS if c in df.columns])} cols")

print(f"\n     Cột bị loại bỏ ({len(df.columns) - len(df_model.columns)}):")
dropped = [c for c in df.columns if c not in keep_cols]
for c in dropped:
    print(f"       - {c}")

print("\n" + "=" * 60)
print("STEP1 hoàn thành.")
print(f"  → Gửi file '{OUTPUT_MODEL}' vào STEP2 để train model.")
print("=" * 60)
