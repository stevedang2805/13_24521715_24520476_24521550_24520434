# SAD-VREAL

**Detecting Suspicious Real Estate Advertisements via Multimodal Machine Learning with Weak Supervision on Vietnamese Data**

Đặng Nguyên Thọ · Nguyễn Trọng Phúc Hiền · Nguyễn Anh Tài · Lê Minh Hải  
Khoa Hệ thống Thông tin — Trường ĐH Công nghệ Thông tin, ĐHQG-HCM

---

## Tổng quan

SAD-VREAL là hệ thống phát hiện quảng cáo bất động sản đáng ngờ trên mogi.vn bằng **Weak Supervision** — không cần nhãn chuyên gia.

**Kết quả chính (mogi.vn, n=4,866):**
- M2 Tabular (HGB): F₁ = 0.799, AUC = 0.968
- Tỷ lệ suspicious: 501/4,866 = 10.30%

---

## Cấu trúc thư mục

```
SAD-VREAL/
├── data/
│   └── price_ref_hcm_hn_mogi.csv   # Bảng giá tham chiếu 51 quận/huyện từ mogi.vn
│
├── src/
│   ├── mogi_scraper.py              # Thu thập quảng cáo từ mogi.vn
│   ├── crawl_price_reference.py     # Thu thập bảng giá tham chiếu từ mogi.vn
│   ├── STEP1_labeling.py            # Tiền xử lý + gắn nhãn weak supervision
│   └── STEP2_training.py            # Huấn luyện và đánh giá 7 mô hình
│
├── notebook/
│   └── SAD_VREAL_pipeline.ipynb     # Notebook tổng hợp toàn bộ pipeline
│
├── results/                         # Output sau khi chạy (tự sinh)
│   ├── mogi_labeled.csv
│   ├── mogi_model_input.csv
│   ├── step2_mogi_output.txt
│   ├── roc_data_mogi.csv
│   ├── feature_importance_mogi.csv
│   └── figures/
│
└── README.md
```

> **Lưu ý:** `mogi_clean.csv` (21MB) không được đưa lên GitHub do kích thước lớn.  
> Xem hướng dẫn thu thập dữ liệu bên dưới.

---

## Cài đặt

```bash
pip install pandas numpy scikit-learn unidecode scipy matplotlib
pip install transformers torch                  # chỉ cần cho M6/M7 (PhoBERT)
pip install playwright && playwright install chromium  # chỉ cần cho crawl
```

---

## Cách chạy

### 1. Thu thập dữ liệu quảng cáo
```bash
python src/mogi_scraper.py --mode live --pages 130 --output mogi_clean.csv
```

### 2. Thu thập bảng giá tham chiếu
```bash
python src/crawl_price_reference.py
# Output: data/price_ref_hcm_hn_mogi.csv
```

### 3. Gắn nhãn (Weak Supervision)
```bash
python src/STEP1_labeling.py
# Input:  mogi_clean.csv, data/price_ref_hcm_hn_mogi.csv
# Output: mogi_labeled.csv, mogi_model_input.csv
```

### 4. Huấn luyện và đánh giá mô hình
```bash
python src/STEP2_training.py
# Input:  mogi_model_input.csv
# Output: results/step2_mogi_output.txt, roc_data_mogi.csv, feature_importance_mogi.csv
```

### 5. Tạo figures
Chạy **STEP 3** trong notebook `notebook/SAD_VREAL_pipeline.ipynb`

---

## Pipeline tổng quan

```
mogi.vn  ──►  mogi_scraper.py  ──►  mogi_clean.csv
                                            │
mogi.vn  ──►  crawl_price_reference.py  ──►  price_ref_hcm_hn_mogi.csv
                                            │
                                    STEP1_labeling.py
                                            │
                              mogi_labeled.csv + mogi_model_input.csv
                                            │
                                    STEP2_training.py
                                            │
                              F₁=0.799 (M2) · AUC=0.968
```

---

## Kết quả

| Model | F₁ | AUC |
|-------|-----|-----|
| M1 Text (word) | 0.451 | 0.860 |
| **M2 Tabular (HGB)** ★ | **0.799** | **0.968** |
| M3 Multimodal | 0.729 | 0.955 |
| M4 Text (word+char) | 0.470 | 0.857 |
| M5 Multimodal full | 0.706 | 0.949 |
| M6 PhoBERT (text) | 0.322 | 0.732 |
| M7 PhoBERT+tab | 0.656 | 0.937 |
