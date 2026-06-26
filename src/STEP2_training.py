#!/usr/bin/env python3
"""
STEP2_mogi.py — SAD-VREAL (Mogi dataset)
═══════════════════════════════════════════════════════════════════
Chạy 7 mô hình trên dataset mogi đã gán nhãn bởi STEP1_mogi.py.
Giao thức CV 5-fold stratified, cùng kiến trúc với bài báo gốc.

Cách chạy:
    python STEP2_mogi.py                   # chạy M1–M5 (không cần GPU)
    python STEP2_mogi.py --skip-phobert   # bỏ M6/M7

Input:  mogi_model_input.csv   (output của STEP1_mogi.py)
Output: step2_mogi_output.txt
        roc_data_mogi.csv
        feature_importance_mogi.csv

Yêu cầu:
    pip install pandas numpy scikit-learn scipy
    pip install transformers torch underthesea   # chỉ cho M6/M7
═══════════════════════════════════════════════════════════════════
"""

import sys, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.inspection import permutation_importance

# ── CONFIG ────────────────────────────────────────────────────
DATA_PATH      = 'mogi_model_input.csv'
LABEL_COL      = 'label_suspicious_v6_NEW'
TEXT_COL       = 'full_text_no_price'
N_SPLITS       = 5
RANDOM_STATE   = 42
SKIP_PHOBERT   = '--skip-phobert' in sys.argv
PHOBERT_MODEL  = 'vinai/phobert-base'
PHOBERT_MAX_LEN = 128

# 22 numeric features (loại has_extreme_clickbait / clickbait_keyword_count
# / flag_v4_parse_error vì không có trong mogi dataset)
# 20 numeric features
# Ghi chú: flag_v4_title_area_mismatch và flag_v4_missing_core_info
# bị loại vì là thành phần của công thức gán nhãn STEP1
# (flag_v4_title_area_mismatch là hard signal trực tiếp tạo label,
#  flag_v4_missing_core_info là soft signal trọng số 2)
# → đưa vào FEAT_NUM = label leak. Paper gốc không dùng flag_v4_* nào.
FEAT_NUM = [
    'price_billion', 'area_m2', 'price_per_m2_calc',
    'bedrooms_num', 'bathrooms_num', 'usable_area_num',
    'length_num', 'width_num', 'land_area_num',
    'word_count', 'digit_count', 'exclamation_count',
    'text_length', 'phone_like_count',
    'is_agri_land', 'is_can_ho_premium',
    'duplicate_group_size', 'duplicate_road_nunique',
    'duplicate_district_nunique', 'duplicate_phone_nunique',
]

# 6 categorical features (giống bài báo gốc)
FEAT_CAT = [
    'frontage_class', 'district_norm', 'province_norm',
    'house_type_norm', 'land_type_norm', 'legal_documents_norm',
]

# ── HELPERS ───────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df[LABEL_COL] = df[LABEL_COL].fillna(0).astype(int)
    for c in FEAT_NUM:
        df[c] = pd.to_numeric(df.get(c, 0), errors='coerce').fillna(0)
    for c in FEAT_CAT:
        df[c] = df[c].fillna('unknown').astype(str) if c in df.columns else 'unknown'
    if TEXT_COL not in df.columns:
        df[TEXT_COL] = ''
    df[TEXT_COL] = df[TEXT_COL].fillna('')
    return df

def build_tabular(df):
    num_arr   = df[[c for c in FEAT_NUM if c in df.columns]].values.astype(float)
    cat_parts = []
    for c in FEAT_CAT:
        col = df[c].astype(str) if c in df.columns else pd.Series(['unknown']*len(df))
        le  = LabelEncoder()
        cat_parts.append(le.fit_transform(col).reshape(-1,1).astype(float))
    X_tab = np.hstack([num_arr] + cat_parts)
    feat_names = [c for c in FEAT_NUM if c in df.columns] + FEAT_CAT
    return X_tab, feat_names

def get_metrics(y_true, y_pred, y_prob):
    return {
        'prec': precision_score(y_true, y_pred, zero_division=0),
        'rec':  recall_score(y_true, y_pred, zero_division=0),
        'f1':   f1_score(y_true, y_pred, zero_division=0),
        'acc':  accuracy_score(y_true, y_pred),
        'auc':  roc_auc_score(y_true, y_prob),
    }

def run_cv(model_fn, X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    all_preds = np.zeros(len(y), dtype=int)
    all_probs = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = model_fn()
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        prob = clf.predict_proba(X[te])[:, 1]
        all_preds[te] = pred
        all_probs[te] = prob
        fold_metrics.append(get_metrics(y[te], pred, prob))
    result = {}
    for k in ['prec', 'rec', 'f1', 'acc', 'auc']:
        vals = [m[k] for m in fold_metrics]
        result[f'{k}_mean'] = np.mean(vals)
        result[f'{k}_std']  = np.std(vals, ddof=1)
    result['all_preds'] = all_preds
    result['all_probs'] = all_probs
    return result

def mcnemar_test(y_true, preds_a, preds_b):
    ca = (preds_a == y_true); cb = (preds_b == y_true)
    b  = np.sum(ca & ~cb);    c  = np.sum(~ca & cb)
    if b + c == 0: return 1.0
    stat = (abs(b - c) - 1)**2 / (b + c)
    return 1 - stats.chi2.cdf(stat, df=1)

def bootstrap_ci(y_true, all_preds, all_probs, n_boot=1000, alpha=0.95):
    rng = np.random.default_rng(RANDOM_STATE)
    n   = len(y_true)
    f1s, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp, yprob = y_true[idx], all_preds[idx], all_probs[idx]
        if len(np.unique(yt)) < 2: continue
        f1s.append(f1_score(yt, yp, zero_division=0))
        aucs.append(roc_auc_score(yt, yprob))
    lo = (1 - alpha) / 2
    return {
        'f1_ci_lo':  np.quantile(f1s, lo),
        'f1_ci_hi':  np.quantile(f1s, 1-lo),
        'auc_ci_lo': np.quantile(aucs, lo),
        'auc_ci_hi': np.quantile(aucs, 1-lo),
    }

def roc_df(y_true, all_probs, name):
    fpr, tpr, _ = roc_curve(y_true, all_probs)
    idx = np.round(np.linspace(0, len(fpr)-1, 200)).astype(int)
    return pd.DataFrame({'model': name, 'fpr': fpr[idx], 'tpr': tpr[idx]})

# ── PHOBERT ───────────────────────────────────────────────────

def extract_phobert(texts, batch_size=32):
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        print("  [LỖI] pip install transformers torch"); sys.exit(1)
    print(f"  Đang tải {PHOBERT_MODEL}...")
    tok = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
    mdl = AutoModel.from_pretrained(PHOBERT_MODEL); mdl.eval()
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    mdl.to(device)
    print(f"  Device: {device}")
    try:
        from underthesea import word_tokenize
        texts = [word_tokenize(t, format='text') for t in texts]
        print("  Tách từ: underthesea ✓")
    except ImportError:
        print("  [Cảnh báo] underthesea không có, dùng text thô.")
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=PHOBERT_MAX_LEN, return_tensors='pt').to(device)
        with __import__('torch').no_grad():
            out = mdl(**enc)
        vecs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        print(f"  PhoBERT: {min(100, int(100*(i+batch_size)/len(texts)))}%...", end='\r')
    print()
    return np.vstack(vecs)

# ── MAIN ──────────────────────────────────────────────────────

def main():
    t0 = time.time()
    lines = []
    def pr(s=''):
        print(s); lines.append(str(s))

    pr('='*65)
    pr('SAD-VREAL — STEP2 MOGI (CV 5-fold, 7 mô hình)')
    pr('='*65)

    # 1. Load
    pr(f'\nĐọc {DATA_PATH}...')
    df = load_data(DATA_PATH)
    y  = df[LABEL_COL].values
    n_sus, n_tot = int(y.sum()), len(y)
    pr(f'  n_total = {n_tot:,}   n_sus = {n_sus:,}  ({100*n_sus/n_tot:.2f}%)')
    pr(f'  HCM: {(df["province_norm"]=="tp ho chi minh").sum():,}'
       f'  HN: {(df["province_norm"]=="ha noi").sum():,}')

    # 2. Build features
    pr('\nXây dựng feature matrices...')
    X_tab, feat_names_tab = build_tabular(df)
    pr(f'  X_tab: {X_tab.shape}  ({len([c for c in FEAT_NUM if c in df.columns])} num + {len(FEAT_CAT)} cat)')

    pr('  TF-IDF word-level...')
    tfidf_w   = TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True, min_df=2)
    X_tfidf_w = tfidf_w.fit_transform(df[TEXT_COL])
    X_svd     = TruncatedSVD(n_components=100, random_state=RANDOM_STATE).fit_transform(X_tfidf_w)

    pr('  TF-IDF char-level...')
    tfidf_c   = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=8000, sublinear_tf=True, min_df=3)
    X_tfidf_c = tfidf_c.fit_transform(df[TEXT_COL])
    X_svd_c   = TruncatedSVD(n_components=80, random_state=RANDOM_STATE).fit_transform(X_tfidf_c)

    feats = {
        'y':     y,
        'X_tab': X_tab,
        'X_m1':  X_tfidf_w.toarray().astype(np.float32),
        'X_m3':  np.hstack([X_svd, X_tab]),
        'X_m4':  np.hstack([X_tfidf_w.toarray(), X_tfidf_c.toarray()]).astype(np.float32),
        'X_m5':  np.hstack([X_svd, X_svd_c, X_tab]),
        'X_svd': X_svd,
        'feat_names_tab': feat_names_tab,
    }

    # 3. PhoBERT
    if not SKIP_PHOBERT:
        pr('\nTrích đặc trưng PhoBERT M6/M7...')
        X_pb = extract_phobert(df[TEXT_COL].tolist())
        feats['X_m6'] = X_pb
        feats['X_m7'] = np.hstack([X_pb, X_tab])
        pr(f'  PhoBERT: {X_pb.shape}')
    else:
        pr('\n[Bỏ qua M6/M7 theo --skip-phobert]')

    # 4. Model factories
    def lr():  return LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced',
                                          random_state=RANDOM_STATE, n_jobs=-1)
    def hgb(): return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                        max_leaf_nodes=31, min_samples_leaf=20,
                        class_weight='balanced', random_state=RANDOM_STATE)

    models = {
        'M1 Văn bản (từ)':           ('X_m1',  lr),
        'M2 Bảng (HGB)':             ('X_tab', hgb),
        'M3 Đa phương thức':         ('X_m3',  hgb),
        'M4 Văn bản (từ+ký tự)':     ('X_m4',  lr),
        'M5 Đa phương thức đầy đủ':  ('X_m5',  hgb),
    }
    if not SKIP_PHOBERT and 'X_m6' in feats:
        models['M6 PhoBERT (văn bản)']    = ('X_m6', lr)
        models['M7 PhoBERT + bảng (HGB)'] = ('X_m7', hgb)

    # 5. CV
    pr('\n' + '-'*65)
    pr('BẢNG KẾT QUẢ — CV 5-fold')
    pr('-'*65)
    pr(f"{'Mô hình':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6} {'AUC':>6}")
    pr(f"{'':35} {'±std':>6} {'±std':>6} {'±std':>6} {'±std':>6} {'±std':>6}")
    pr('-'*65)

    cv_results = {}; roc_dfs = []
    for name, (xkey, mfn) in models.items():
        t1 = time.time()
        print(f'  {name}...', end=' ', flush=True)
        res = run_cv(mfn, feats[xkey], y)
        ci  = bootstrap_ci(y, res['all_preds'], res['all_probs'])
        res.update(ci); cv_results[name] = res
        elapsed = time.time() - t1
        pr(f"{name:<35} {res['prec_mean']:>6.3f} {res['rec_mean']:>6.3f} "
           f"{res['f1_mean']:>6.3f} {res['acc_mean']:>6.3f} {res['auc_mean']:>6.3f}")
        pr(f"{'':35} {res['prec_std']:>6.3f} {res['rec_std']:>6.3f} "
           f"{res['f1_std']:>6.3f} {res['acc_std']:>6.3f} {res['auc_std']:>6.3f}")
        print(f'({elapsed:.0f}s)')
        roc_dfs.append(roc_df(y, res['all_probs'], name))

    # 6. Bootstrap CI
    pr('\n' + '-'*65)
    pr('BOOTSTRAP 95% CI — F1 và AUC  (n_boot=1000)')
    pr('-'*65)
    pr(f"{'Mô hình':<35} {'F1 [lo–hi]':>20} {'AUC [lo–hi]':>20}")
    pr('-'*65)
    for name, res in cv_results.items():
        pr(f"{name:<35} [{res['f1_ci_lo']:.3f} – {res['f1_ci_hi']:.3f}]"
           f"   [{res['auc_ci_lo']:.3f} – {res['auc_ci_hi']:.3f}]")

    # 7. McNemar — M2 vs tất cả
    pr('\n' + '-'*65)
    pr('MCNEMAR TEST — M2 vs. các cấu hình')
    pr('-'*65)
    if 'M2 Bảng (HGB)' in cv_results:
        m2_preds = cv_results['M2 Bảng (HGB)']['all_preds']
        pr(f"{'So sánh':<45} {'p-value':>10}  Kết luận"); pr('-'*65)
        for name, res in cv_results.items():
            if name == 'M2 Bảng (HGB)': continue
            p = mcnemar_test(y, m2_preds, res['all_preds'])
            conclude = 'Khác biệt đáng kể (p<0.05)' if p < 0.05 else 'Không đáng kể'
            pr(f"{'M2 vs ' + name:<45} {p:>10.4f}  {conclude}")

    # 8. Ablation M3
    pr('\n' + '-'*65)
    pr('ABLATION — Đóng góp phương thức (M3, CV 5-fold)')
    pr('-'*65)
    n_num = len([c for c in FEAT_NUM if c in df.columns])
    ablation = {
        'Đầy đủ (văn bản + bảng)':                  feats['X_m3'],
        'Chỉ bảng (bỏ văn bản)':                    feats['X_tab'],
        'Chỉ văn bản (bỏ bảng)':                    feats['X_svd'],
        f'Non-categorical tabular ({n_num} features)': feats['X_tab'][:, :n_num],
    }
    pr(f"{'Cấu hình':<42} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6}"); pr('-'*65)
    for cfg, X_cfg in ablation.items():
        r = run_cv(hgb, X_cfg, y)
        pr(f"{cfg:<42} {r['prec_mean']:>6.3f} {r['rec_mean']:>6.3f} "
           f"{r['f1_mean']:>6.3f} {r['acc_mean']:>6.3f}")

    # 9. Permutation importance M3
    pr('\n' + '-'*65)
    pr('FEATURE IMPORTANCE — Permutation (M3, fold 1, n_repeats=10)')
    pr('-'*65)
    skf_fi = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    tr_i, te_i = next(skf_fi.split(feats['X_m3'], y))
    clf_fi = hgb(); clf_fi.fit(feats['X_m3'][tr_i], y[tr_i])
    pi = permutation_importance(clf_fi, feats['X_m3'][te_i], y[te_i],
                                n_repeats=10, random_state=RANDOM_STATE,
                                scoring='f1', n_jobs=-1)
    fi_names = [f'tfidf_svd_{i}' for i in range(100)] + feats['feat_names_tab']
    fi_df = (pd.DataFrame({'feature': fi_names,
                            'importance': pi.importances_mean,
                            'std': pi.importances_std})
               .sort_values('importance', ascending=False).head(20))
    pr(f"{'Rank':<5} {'Feature':<35} {'Importance':>12} {'±Std':>8}"); pr('-'*65)
    fi_records = []
    for rank, (_, row) in enumerate(fi_df.iterrows(), 1):
        pr(f"{rank:<5} {row['feature']:<35} {row['importance']:>12.4f} {row['std']:>8.4f}")
        fi_records.append({'rank': rank, 'feature': row['feature'],
                           'importance': row['importance'], 'std': row['std']})

    # 10. Per-segment M3
    pr('\n' + '-'*65)
    pr('PHÂN TÍCH THEO PHÂN KHÚC GIÁ — M3')
    pr('-'*65)
    m3_preds = cv_results['M3 Đa phương thức']['all_preds']
    m3_probs = cv_results['M3 Đa phương thức']['all_probs']
    bins   = [0, 1, 3, 10, 50, 1e9]
    labels = ['<1 tỷ', '1–3 tỷ', '3–10 tỷ', '10–50 tỷ', '>50 tỷ']
    df['price_seg'] = pd.cut(df['price_billion'], bins=bins, labels=labels)
    pr(f"{'Phân khúc':<12} {'n':>6} {'%sus':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}"); pr('-'*65)
    for seg in labels:
        mask = (df['price_seg'] == seg).values
        if mask.sum() == 0: continue
        yt, yp = y[mask], m3_preds[mask]
        f1   = f1_score(yt, yp, zero_division=0)
        prec = precision_score(yt, yp, zero_division=0)
        rec  = recall_score(yt, yp, zero_division=0)
        pr(f"{seg:<12} {mask.sum():>6} {100*yt.mean():>7.1f}% "
           f"{f1:>8.3f} {prec:>8.3f} {rec:>8.3f}")

    # 11. HCM vs HN M3
    pr('\n' + '-'*65)
    pr('PHÂN TÍCH THEO ĐỊA LÝ — M3 (HCM vs HN)')
    pr('-'*65)
    pr(f"{'Vùng':<20} {'n':>6} {'%sus':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}"); pr('-'*65)
    for prov_key, prov_label in [('tp ho chi minh','TP.HCM'), ('ha noi','Hà Nội')]:
        mask = (df['province_norm'] == prov_key).values
        if mask.sum() == 0: continue
        yt, yp = y[mask], m3_preds[mask]
        f1   = f1_score(yt, yp, zero_division=0)
        prec = precision_score(yt, yp, zero_division=0)
        rec  = recall_score(yt, yp, zero_division=0)
        pr(f"{prov_label:<20} {mask.sum():>6} {100*yt.mean():>7.1f}% "
           f"{f1:>8.3f} {prec:>8.3f} {rec:>8.3f}")

    # 12. Confusion matrix M3
    pr('\n' + '-'*65)
    pr('CONFUSION MATRIX — M3 (aggregated OOF)')
    pr('-'*65)
    cm = confusion_matrix(y, m3_preds)
    tn, fp, fn, tp_ = cm.ravel()
    pr(f'  TN={tn}  FP={fp}  FN={fn}  TP={tp_}')
    pr(f'  FPR = {fp/(tn+fp):.3f}   FNR = {fn/(fn+tp_):.3f}')

    # Summary
    pr('\n' + '='*65)
    pr(f'TỔNG THỜI GIAN: {time.time()-t0:.0f} giây')
    pr('='*65)

    # Save
    with open('step2_mogi_output.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    pd.concat(roc_dfs, ignore_index=True).to_csv('roc_data_mogi.csv', index=False)
    pd.DataFrame(fi_records).to_csv('feature_importance_mogi.csv', index=False)

    print('\nĐã lưu:')
    print('  step2_mogi_output.txt')
    print('  roc_data_mogi.csv')
    print('  feature_importance_mogi.csv')
    print('\nGửi 3 file này cho Claude để xử lý tiếp.')

if __name__ == '__main__':
    main()
