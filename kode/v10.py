# =============================================================================
# INFEST XII 2026 - Klasifikasi Citra Aksara Tradisional Nusantara  [v10]
# Satu cell: load data -> EDA -> preprocessing -> training -> validasi -> submission
# Metrik: F1-Score Macro
#
# RIWAYAT LB: v3.1 0.8073 | v5 0.81054 | v6 0.83902 | v7/v8 0.83422
#   Pelabelan MANUAL manusia atas test = 0.71330 -> model sudah jauh di atasnya.
# PEROMBAKAN v9 (semua berdasar bukti leaderboard, bukan dugaan):
#   1. heightnorm DIBUANG total   - F1 blok 0.82 vs letterbox 0.94 (49% test blok)
#   2. DEGRADE kembali ke v6      - v6 0.83902 > v7 0.83422; kalibrasi JUSTRU rugi
#   3. AugMax multi-chain         - Wang et al. NeurIPS 2021, diversity + hardness
#   4. Ensemble penuh dipaksa     - greedy selalu pilih 1 model, 4 run terbuang
#   5. Ambang per-kelas macro-F1  - ditala di OOF tertimbang, bukan ditebak
#
# BARU DI v10 - menyerang PINTASAN, bukan menambah kapasitas model.
# Diukur langsung dari gambar (3.771 train / 1.220 test):
#   RandomForest yang HANYA melihat metadata (ukuran, AR, saturasi, derajat
#   binarisasi, oklusi) - TANPA melihat bentuk aksara - mencapai macro-F1
#   0.7826 di CV train. Model terbaik kita 0.839 di LB. Artinya hampir seluruh
#   skor kita bisa dijelaskan oleh pintasan. Dan AUC pembeda domain train-vs-test
#   dari metadata yang sama = 0.9169: fitur yang paling memprediksi kelas di
#   train justru yang paling bergeser ke test.
#
#   Uji kanonikalisasi memisahkan DUA pintasan yang berbeda:
#     fitur fotometrik saja : pintasan 0.6477 -> 0.2176, AUC 0.9110 -> 0.5955
#     ikut ukuran/AR        : pintasan 0.8344 -> 0.7504, AUC 0.9220 -> 0.7746
#   Jadi Otsu menghabisi pintasan fotometrik tapi TIDAK menyentuh geometri.
#
#   6. CANON  - binarisasi Otsu pd train DAN test (pintasan fotometrik)
#   7. ARJIT  - regangan sumbu-x acak saat latih (pintasan geometri)
#   8. DFR    - latih ulang HANYA layer terakhir di fitur group-balanced [#5]
#   9. CLEAN  - confident learning utk sampel tak mungkin dipelajari [#50]
#  10. SAM    - tersedia, default MATI (menggandakan biaya per langkah) [#99]
# =============================================================================
import os, sys, csv, glob, math, time, random, zipfile, warnings
from collections import Counter
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

RUNS = [
    # DIROMBAK. Bukti dari 3 run: view LETTERBOX mengalahkan heightnorm secara
    # konsisten pada populasi 'blok' (F1 0.94 vs 0.82) yang menyusun ~49% test,
    # dan greedy SELALU memilih cnx-sq384 sendirian. Jadi seluruh heightnorm
    # dibuang; komputasinya dialihkan ke letterbox resolusi lebih tinggi dan
    # backbone lebih beragam supaya ensemble punya sesuatu untuk digabung.
    dict(tag="cnx-sq384", backbone="convnext_tiny.in12k_ft_in1k",       view="letterbox",
         h=384, w=384, tta=1, lr=1e-4),
    dict(tag="cnx-sq448", backbone="convnext_tiny.in12k_ft_in1k",       view="letterbox",
         h=448, w=448, tta=1, lr=1e-4),
    dict(tag="effv2-384", backbone="tf_efficientnetv2_s.in21k_ft_in1k", view="letterbox",
         h=384, w=384, tta=1, lr=2e-4),
    dict(tag="eff-sq320", backbone="tf_efficientnet_b0.ns_jft_in1k",    view="letterbox",
         h=320, w=320, tta=1, lr=3e-4),
    dict(tag="cnx-wide",  backbone="convnext_tiny.in12k_ft_in1k",       view="letterbox",
         h=160, w=640, tta=1, lr=1e-4),
    # [#166] kepala ArcFace sub-center: geometri keputusan yang BEDA sama sekali
    # dari 4 run di atas, jadi ensemble punya sesuatu yang benar-benar berbeda.
    dict(tag="arc-384",   backbone="convnext_tiny.in12k_ft_in1k",       view="letterbox",
         h=384, w=384, tta=1, lr=1e-4, head="arcface"),
]
SPEC = dict(enable=True, backbone="convnext_tiny.in12k_ft_in1k", view="letterbox",
            h=384, w=384, tta=1, lr=1e-4, epochs=22)   # spesialis pasangan tersulit
MAXSIDE = 1600      # gambar terbesar 5664x4248 -> perkecil dulu biar loading tak lambat
# DIUKUR pada dataset: train 95.3% abu-abu vs test 36.8%; ketajaman (var Laplacian)
# median train 13101 vs test 2464 (test 5.3x lebih buram). Model v3.1/v5 mempelajari
# pintasan palsu "berwarna -> pegon" (pegon = kelas paling banyak berwarna di train,
# 11.3%): korelasi %berwarna-train vs skew prediksi test rho=+0.89 (v5) & +0.86 (v3.1).
GRAYSCALE = True                      # matikan sumbu warna
# DIKEMBALIKAN ke setelan v6. BUKTI LEADERBOARD: v6 (blur 0.6-2.2) = 0.83902,
# v7 (blur 0.18-0.66, "dikalibrasi" ke statistik test) = 0.83422. Kalibrasi ke
# tingkat korupsi test justru MENURUNKAN skor 0.0048.
# KOREKSI KLAIM: versi sebelumnya komentar ini menulis "sejalan dgn literatur
# (AugMix/AugMax/DeepAugment)". Itu KELIRU. Mintun dkk. (arXiv 2102.11273,
# NeurIPS 2021) justru menemukan yang SEBALIKNYA: robustness paling baik saat
# augmentasi latih PERSEPTUAL MIRIP dgn korupsi uji - yang berarti v7 seharusnya
# menang, padahal kalah. Jadi satu-satunya dasar setelan ini adalah pengukuran
# LB kita sendiri (n=1 pasang run), bukan literatur. SEV utk menguji arahnya.
SEV = 1.0                             # 1.0 = setelan v6; coba 1.3 utk lebih keras
DEGRADE = dict(p_blur=0.70, blur=(0.6*SEV, 2.2*SEV),
               p_contrast=0.60, contrast=(0.45, 0.95),
               p_rescale=0.50, rescale=(max(0.15, 0.35/SEV), 0.80),
               p_jpeg=0.50, jpeg=(max(10, int(25/SEV)), 88))
# AugMax-style (Wang et al. NeurIPS 2021): campuran BEBERAPA rantai augmentasi
# jauh lebih beragam daripada satu rantai. Diversity + hardness, bukan salah satu.
AUGMAX = dict(enable=True, p=0.35, chains=3)
# [#122 Tent / #121 MEMO] Test-Time Adaptation: satu-satunya metode di daftar Anda
# yang menyerang LANGSUNG masalah utama kita (pergeseran domain train->test).
# Hanya parameter lapisan normalisasi yang diperbarui, dgn tujuan meminimalkan
# ENTROPI prediksi di data TEST - tidak butuh label. Berbahaya (bisa kolaps ke
# satu kelas), jadi DIGERBANGI: diuji dulu di validasi yang sengaja dirusak
# menyerupai test; dipakai hanya kalau di sana terbukti menolong.
TENT = dict(enable=True, steps=1, lr=1e-3, gain=0.005)
# [v10] KANONIKALISASI: binarisasi Otsu diterapkan ke TRAIN DAN TEST. Ini yang
# menghapus pintasan fotometrik sekaligus menutup jarak domain. Urutannya
# penting: canon DULU, baru degrade/augmax - supaya train dan test berangkat
# dari representasi yang SAMA, lalu train saja yang diberi korupsi tambahan.
# blur_pre meredam derau garam-merica sebelum ambang batas diambil.
CANON = dict(enable=True, blur_pre=0.5)
# [v10] PENGHANCUR PINTASAN GEOMETRI: regangkan sumbu-x secara acak saat latih
# supaya aspect ratio berhenti jadi informasi kelas. Otsu tidak menyentuh ini:
# dengan ukuran/AR ikut dipakai pintasan masih 0.7504 (AUC domain 0.7746).
ARJIT = dict(enable=True, lo=0.65, hi=1.55)
# [v10] [#5 DFR] Latih ulang HANYA layer terakhir pada fitur yang GROUP-BALANCED
# (grup = kelas x binarisasi). CNN-nya tidak disentuh, jadi biayanya menit.
# Digerbangi: dipakai per-run hanya kalau menaikkan OOF tertimbang.
DFR = dict(enable=True, gain=0.002, C=1.0, min_per_group=8)
# [v10] [#50 confident learning] Buang sampel yang model sendiri sangat yakin
# BUKAN labelnya. Di train ada gambar berlabel jawi yang isinya teks Latin
# "a.  Kepri", dan banyak yang isinya cuma angka Arab-Hindi - dipakai jawi
# maupun pegon, jadi ambigu secara definisi.
CLEAN = dict(enable=True, p_max=0.02, report=40)
# [v10] [#99 SAM] Flat minima. MATI secara default: menggandakan biaya tiap
# langkah (dua forward-backward), dan belum pernah diukur di lomba ini.
SAM = dict(enable=False, rho=0.05)
CFG = dict(n_folds=5, epochs=18, batch=32, wd=0.05, ls=0.05, nw=2, ema=0.999)
SEED = 42
def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
seed_all()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEV} | torch={torch.__version__}")

# --------------------------- 1. LOAD / DISCOVER DATA -------------------------
SEARCH = [p for p in ["/kaggle/input", "/kaggle/working", ".", "./data", "/content"] if os.path.isdir(p)]
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

def find_csv(names):
    for root in SEARCH:
        for dp, _, fs in os.walk(root, followlinks=True):
            for f in fs:
                if f.lower() in names: return os.path.join(dp, f)
    return None

train_csv = find_csv({"train.csv"}); test_csv = find_csv({"test.csv"})
sub_csv   = find_csv({"sample_submission.csv"})
print("train.csv :", train_csv); print("test.csv  :", test_csv); print("sample_sub:", sub_csv)
tr_df = pd.read_csv(train_csv); te_df = pd.read_csv(test_csv)

# gambar bisa datang sebagai folder ATAU sebagai .zip -> ekstrak sekali kalau perlu
for root in list(SEARCH):
    for z in glob.glob(os.path.join(root, "*.zip")):
        tag = os.path.splitext(os.path.basename(z))[0].lower()
        if tag in ("train", "test", "images"):
            out = os.path.join(WORK, "_imgs")
            if not os.path.isdir(os.path.join(out, tag)):
                os.makedirs(out, exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out, members=[m for m in zf.namelist() if not m.startswith("__MACOSX")])
                print(f"extracted {z} -> {out}")
if os.path.isdir(os.path.join(WORK, "_imgs")): SEARCH.append(os.path.join(WORK, "_imgs"))

# index semua file gambar sekali: nama lengkap DAN nama tanpa ekstensi
EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
IDX, STEM = {}, {}
for root in SEARCH:
    for dp, _, fs in os.walk(root, followlinks=True):
        for f in fs:
            if f.lower().endswith(EXT):
                full = os.path.join(dp, f)
                IDX.setdefault(f, []).append(full)
                STEM.setdefault(os.path.splitext(f)[0], []).append(full)

def resolve(name, split):
    b = os.path.basename(str(name))
    c = IDX.get(b) or STEM.get(os.path.splitext(b)[0])    # cocokkan tanpa ekstensi
    if not c: return None
    if len(c) == 1: return c[0]
    pref = [p for p in c if f"{os.sep}{split}" in p.lower()]
    return (pref or c)[0]

tr_df["path"] = [resolve(x, "train") for x in tr_df.image_id]
te_df["path"] = [resolve(x, "test")  for x in te_df.image_id]
print(f"train rows={len(tr_df)} resolved={tr_df.path.notna().sum()} | "
      f"test rows={len(te_df)} resolved={te_df.path.notna().sum()}")

# diagnosis jelas kalau gagal, bukan assert buta
mt, me = tr_df[tr_df.path.isna()], te_df[te_df.path.isna()]
if len(me) or len(mt) == len(tr_df):
    print("\n!! GAGAL MENEMUKAN GAMBAR")
    print(f"   train tidak ketemu : {len(mt)}/{len(tr_df)}  contoh: {mt.image_id.head(5).tolist()}")
    print(f"   test  tidak ketemu : {len(me)}/{len(te_df)}  contoh: {me.image_id.head(5).tolist()}")
    print(f"   file gambar terindeks: {sum(len(v) for v in IDX.values())}")
    print(f"   ekstensi di disk     : {dict(Counter(os.path.splitext(f)[1].lower() for f in IDX))}")
    print(f"   folder dipindai      : {SEARCH}")
    raise AssertionError("Gambar tidak ketemu - lihat diagnosis di atas.")

mism = sum(1 for i, p in zip(tr_df.image_id, tr_df.path)
           if isinstance(p, str) and os.path.basename(p) != os.path.basename(str(i)))
if mism: print(f"catatan: {mism} baris ekstensinya beda antara CSV dan file asli (sudah dicocokkan)")

# --- kebocoran train<->test: gambar test yang IDENTIK dgn gambar train sudah
# --- diketahui labelnya. Dideteksi otomatis lewat hash, bukan di-hardcode.
import hashlib
def _md5(path):
    try:
        with open(path, "rb") as fh: return hashlib.md5(fh.read()).hexdigest()
    except Exception: return None
_htr = {}
for _i, _p, _l in zip(tr_df.image_id, tr_df.path, tr_df.label):
    if isinstance(_p, str):
        _h = _md5(_p)
        if _h: _htr.setdefault(_h, []).append(_l)
LEAK = {}
for _i, _p in zip(te_df.image_id, te_df.path):
    if not isinstance(_p, str): continue
    _h = _md5(_p)
    if _h in _htr and len(set(_htr[_h])) == 1: LEAK[_i] = _htr[_h][0]
print(f"kebocoran train<->test terdeteksi: {len(LEAK)} gambar test labelnya sudah pasti")
for _k, _v in list(LEAK.items())[:8]: print(f"   {_k} -> {_v}")

# --------------------------------- 2. EDA ------------------------------------
print("\n" + "=" * 70 + "\nSTEP 1 - EDA\n" + "=" * 70)
print("\nDistribusi kelas (macro-F1 -> kelas kecil sama pentingnya):")
vc = tr_df.label.value_counts()
for k, v in vc.items(): print(f"  {k:10s} {v:5d}  ({v/len(tr_df)*100:5.2f}%)  {'#'*int(v/15)}")
print(f"  imbalance ratio mayoritas/minoritas = {vc.max()/vc.min():.2f}x")

def probe(paths, tag):
    ws, hs, modes, bad = [], [], Counter(), []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.load(); ws.append(im.size[0]); hs.append(im.size[1]); modes[im.mode] += 1
        except Exception as e:
            bad.append((p, str(e)[:40]))
    print(f"\n[{tag}] n={len(paths)} ok={len(ws)} corrupt={len(bad)} modes={dict(modes)}")
    if not ws:
        print("  (tidak ada gambar terbaca)"); return bad
    ws, hs = np.array(ws), np.array(hs); ar = ws / np.maximum(hs, 1)
    print(f"  W  p1/p50/p99 = {np.percentile(ws,1):.0f}/{np.percentile(ws,50):.0f}/{np.percentile(ws,99):.0f}")
    print(f"  H  p1/p50/p99 = {np.percentile(hs,1):.0f}/{np.percentile(hs,50):.0f}/{np.percentile(hs,99):.0f}")
    print(f"  AR p1/p50/p99 = {np.percentile(ar,1):.2f}/{np.percentile(ar,50):.2f}/{np.percentile(ar,99):.2f}")
    for p, e in bad[:5]: print(f"  CORRUPT: {os.path.basename(p)} -> {e}")
    return bad

bad_tr = probe([p for p in tr_df.path if p], "TRAIN")
bad_te = probe([p for p in te_df.path if p], "TEST")

print("\nAspect-ratio per kelas (ukuran/layout = sinyal kuat, mis. pegon = blok potret):")
for lab in sorted(tr_df.label.unique()):
    a = []
    for p in tr_df.loc[tr_df.label == lab, "path"]:
        try:
            with Image.open(p) as im: a.append(im.size[0] / im.size[1])
        except Exception: pass
    a = np.array(a)
    print(f"  {lab:10s} AR p10={np.percentile(a,10):6.2f} p50={np.percentile(a,50):6.2f} "
          f"p90={np.percentile(a,90):6.2f}  potret(<1)={100*(a<1).mean():4.1f}%")

# buang baris train yang file-nya hilang / corrupt (test TIDAK PERNAH dibuang)
badset = {p for p, _ in bad_tr}
before = len(tr_df)
tr_df = tr_df[tr_df.path.notna() & ~tr_df.path.isin(badset)].reset_index(drop=True)
print(f"\ntrain dipakai: {len(tr_df)} (buang {before-len(tr_df)} hilang/corrupt)")

CLASSES = sorted(tr_df.label.unique()); C2I = {c: i for i, c in enumerate(CLASSES)}
tr_df["y"] = tr_df.label.map(C2I)
print("kelas:", CLASSES)

# --- DUA POPULASI: 'strip' (potongan teks memanjang) vs 'blok' (foto halaman,
# --- screenshot, tabel). Ini terbukti jadi sumber kegagalan v3.1: OOF dihitung
# --- pada campuran train, sedangkan test punya campuran yang berbeda, sehingga
# --- OOF 0.9716 berpasangan dengan LB 0.8073. Semua pemilihan model mulai
# --- sekarang ditimbang ke komposisi TEST, bukan komposisi train.
AR_SPLIT = 3.0
def _meta(paths):
    """satu lintasan: aspect ratio + 'bil' = porsi piksel yang sudah 2-aras.
    'bil' dipakai sebagai atribut grup untuk DFR, dan sebagai diagnostik
    pintasan: di train pegon 67.0% 2-aras vs jawa 7.1%, sedangkan di test
    hanya 7.0% keseluruhan. Inilah pintasan terkuat yang kita ukur."""
    ar, bil = [], []
    for q in paths:
        try:
            with Image.open(q) as im:
                ar.append(im.size[0] / max(im.size[1], 1))
                g = im.convert("L"); g.thumbnail((128, 128))
                a = np.asarray(g, np.float32)
                bil.append(float(((a < 8) | (a > 247)).mean()))
        except Exception:
            ar.append(np.nan); bil.append(np.nan)
    return np.array(ar, float), np.array(bil, float)
AR_TR, BIL_TR = _meta(tr_df.path.tolist())
AR_TE, BIL_TE = _meta(te_df.path.tolist())
BLOK_TR = AR_TR < AR_SPLIT
W_BLOK  = float(np.nanmean(AR_TE < AR_SPLIT))                      # diukur dari TEST
# atribut grup utk DFR: 1 = gambar sudah dibinarisasi di sumbernya
BIN_TR  = np.nan_to_num(BIL_TR, nan=0.0) > 0.90
print(f"\npintasan binarisasi: train {100*BIN_TR.mean():.1f}% gambar sudah 2-aras, "
      f"test {100*np.nanmean(np.nan_to_num(BIL_TE, nan=0.0) > 0.90):.1f}%")
print("  per kelas (kalau angkanya jauh berbeda, ini PINTASAN, bukan aksara):")
for _c in CLASSES:
    _m = (tr_df.y.values == C2I[_c])
    print(f"    {_c:10s} 2-aras {100*BIN_TR[_m].mean():5.1f}%   AR p50 {np.nanmedian(AR_TR[_m]):6.2f}")
print(f"  CANON={CANON['enable']} (Otsu ke train DAN test)  ARJIT={ARJIT['enable']} "
      f"(regangan-x {ARJIT['lo']}-{ARJIT['hi']}x saat latih)")
print(f"\npopulasi (AR<{AR_SPLIT} = 'blok'):")
print(f"  train: blok {BLOK_TR.mean()*100:5.1f}%  strip {100-BLOK_TR.mean()*100:5.1f}%")
print(f"  test : blok {W_BLOK*100:5.1f}%  strip {100-W_BLOK*100:5.1f}%   <- bobot penilaian")
if abs(BLOK_TR.mean() - W_BLOK) > 0.05:
    print("  PERHATIAN: komposisi train != test. OOF polos AKAN terlalu optimistis.")

# --- DIAGNOSTIK WARNA: ukur jarak domain yang jadi biang kegagalan v3.1/v5.
def _sat(paths, n=400):
    paths = [q for q in paths if isinstance(q, str)]
    if not paths: return np.array([np.nan])
    idx = np.random.RandomState(0).choice(len(paths), min(n, len(paths)), replace=False)
    out = []
    for q in [paths[i] for i in idx]:
        try:
            with Image.open(q) as im:
                im = im.convert("RGB"); im.thumbnail((300, 300)); a = np.asarray(im, float)
            out.append(float(np.abs(a - a.mean(2, keepdims=True)).mean()))
        except Exception: pass
    return np.array(out) if out else np.array([np.nan])
_gtr, _gte = _sat(tr_df.path.tolist()), _sat(te_df.path.tolist())
print(f"\nwarna (0 = abu-abu murni):  train abu-abu {100*np.nanmean(_gtr<2):.1f}%  "
      f"test abu-abu {100*np.nanmean(_gte<2):.1f}%")
print(f"  GRAYSCALE={GRAYSCALE} -> sumbu warna "
      f"{'DIMATIKAN' if GRAYSCALE else 'AKTIF (RISIKO pintasan palsu)'}")


# ------------------- 3. PREPROCESSING & AUGMENTASI ---------------------------
import io
from PIL import ImageFilter, ImageEnhance
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def letterbox(im, H, W, train):
    w, h = im.size
    if train and ARJIT["enable"]:
        # [v10] regangkan HANYA sumbu-x -> aspect ratio berhenti jadi informasi
        # kelas. Di train AR p50 bali 19.19 vs pegon 1.16; di test sebarannya
        # jauh berbeda (p50 6.75 vs 3.08), jadi AR adalah pintasan yang runtuh.
        f = random.uniform(ARJIT["lo"], ARJIT["hi"])
        w = max(1, int(round(w * f)))
        im = im.resize((w, h), Image.BILINEAR)
    s = min(H / h, W / w)
    if train: s *= random.uniform(0.92, 1.08)
    nw, nh = max(1, min(W, int(round(w * s)))), max(1, min(H, int(round(h * s))))
    im = im.resize((nw, nh), Image.BILINEAR)
    c = Image.new("RGB", (W, H), (255, 255, 255))
    ox = random.randint(0, W - nw) if train else (W - nw) // 2
    oy = random.randint(0, H - nh) if train else (H - nh) // 2
    c.paste(im, (ox, oy)); return c

def heightnorm(im, H, W, train, view=0, nview=1):
    w, h = im.size; nw = max(1, int(round(w * H / h)))
    im = im.resize((nw, H), Image.BILINEAR)
    if nw <= W:
        c = Image.new("RGB", (W, H), (255, 255, 255))
        c.paste(im, (random.randint(0, W - nw) if train else (W - nw) // 2, 0)); return c
    x = random.randint(0, nw - W) if train else (0 if nview == 1 else int(view * (nw - W) / (nview - 1)))
    return im.crop((x, 0, x + W, H))

def prep(im, r, train, view=0, nview=1):
    return (letterbox(im, r["h"], r["w"], train) if r["view"] == "letterbox"
            else heightnorm(im, r["h"], r["w"], train, view, nview))

def otsu_thr(a):
    """ambang Otsu, tanpa cv2/skimage (keduanya belum tentu ada di runtime lomba)"""
    h = np.bincount(np.clip(a, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    tot = float(a.size); sT = float((np.arange(256) * h).sum())
    wB = 0.0; sB = 0.0; best_t, best_v = 128, -1.0
    for t in range(256):
        wB += h[t]
        if wB == 0: continue
        wF = tot - wB
        if wF <= 0: break
        sB += t * h[t]
        v = wB * wF * ((sB / wB) - ((sT - sB) / wF)) ** 2
        if v > best_v: best_t, best_v = t, v
    return best_t

def canon(im):
    """[v10] Kanonikalisasi: paksa SEMUA gambar - train maupun test - ke bentuk
    2-aras yang sama. TERUKUR: pintasan fotometrik 0.6477 -> 0.2176 (tebak acak
    0.143) dan AUC pembeda domain 0.9110 -> 0.5955, yaitu train dan test jadi
    nyaris tak terbedakan dari metadata fotometriknya."""
    g = im.convert("L")
    if CANON["blur_pre"] > 0:
        g = g.filter(ImageFilter.GaussianBlur(CANON["blur_pre"]))
    a = np.asarray(g, np.float32)
    if a.size == 0: return im
    b = ((a > otsu_thr(a)).astype(np.uint8) * 255)
    return Image.fromarray(b, "L").convert("RGB")

class AddNoise(nn.Module):
    def __init__(s, p=0.3, sd=0.04): super().__init__(); s.p, s.sd = p, sd
    def forward(s, x):
        return (x + torch.randn_like(x) * s.sd).clamp(0, 1) if random.random() < s.p else x

AUG = T.Compose([
    T.RandomApply([T.RandomRotation(4, fill=255)], p=0.5),
    T.RandomApply([T.RandomAffine(0, translate=(0.02, 0.05), shear=4, fill=255)], p=0.3),
    T.ColorJitter(brightness=0.30, contrast=0.30),
    T.ToImage(), T.ToDtype(torch.float32, scale=True), AddNoise(),
    T.Normalize(MEAN, STD), T.RandomErasing(p=0.25, scale=(0.01, 0.06))])
PLAIN = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


def degrade(im):
    """Domain randomization: rusak gambar TRAIN supaya menyerupai statistik TEST.
    Diuji terkontrol: skew prediksi pegon -18.1pp (RGB) -> -6.2pp (abu-abu)
    -> 0.0pp (abu-abu + degradasi). Efek pada AKURASI tidak terbukti (n uji 177)."""
    d = DEGRADE
    if random.random() < d["p_blur"]:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(*d["blur"])))
    if random.random() < d["p_contrast"]:
        im = ImageEnhance.Contrast(im).enhance(random.uniform(*d["contrast"]))
    if random.random() < d["p_rescale"]:                 # hilangkan detail halus
        w, h = im.size; f = random.uniform(*d["rescale"])
        im = im.resize((max(8, int(w*f)), max(8, int(h*f))), Image.BILINEAR) \
               .resize((w, h), Image.BILINEAR)
    if random.random() < d["p_jpeg"]:
        b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(*d["jpeg"])); b.seek(0)
        im = Image.open(b).convert("RGB")
    return im

def _chain(im):
    """satu rantai augmentasi acak (1-3 operasi)"""
    ops = [lambda x: x.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 2.0*SEV))),
           lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.35, 1.5)),
           lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.55, 1.45)),
           lambda x: ImageEnhance.Sharpness(x).enhance(random.uniform(0.0, 2.0)),
           lambda x: x.filter(ImageFilter.SMOOTH_MORE),
           lambda x: x.rotate(random.uniform(-4, 4), fillcolor=(255, 255, 255))]
    for f in random.sample(ops, random.randint(1, 3)):
        try: im = f(im)
        except Exception: pass
    return im

def augmax(im):
    """AugMax/AugMix: campur BEBERAPA rantai augmentasi dgn bobot Dirichlet.
    Keberagaman (banyak rantai) + kekerasan (tiap rantai bisa ekstrem)."""
    k = AUGMAX["chains"]
    w = np.random.dirichlet([1.0]*k); m = np.random.beta(1.0, 1.0)
    a = np.asarray(im, np.float32)
    mix = np.zeros_like(a)
    for i in range(k):
        mix += w[i] * np.asarray(_chain(im.copy()), np.float32)
    return Image.fromarray(np.clip((1-m)*a + m*mix, 0, 255).astype(np.uint8))

class ScriptDS(Dataset):
    def __init__(s, df, r, train, view=0, nview=1, sim=False):
        s.p = df.path.tolist(); s.y = df.y.tolist() if "y" in df else [0]*len(df)
        s.r, s.t, s.v, s.n, s.sim = r, train, view, nview, sim
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        try:
            with Image.open(s.p[i]) as im:
                if max(im.size) > MAXSIDE: im.thumbnail((MAXSIDE, MAXSIDE))  # 5664x4248 -> cepat
                im = im.convert("RGB")
        except Exception: im = Image.new("RGB", (s.r["w"], s.r["h"]), (255, 255, 255))
        if CANON["enable"]: im = canon(im)        # [v10] SEBELUM degrade, train & test
        if s.t:
            im = augmax(im) if (AUGMAX["enable"] and random.random() < AUGMAX["p"]) \
                 else degrade(im)
        elif s.sim:
            im = degrade(im)          # tiru domain TEST utk menguji adaptasi
        if GRAYSCALE: im = im.convert("L").convert("RGB")   # train DAN test
        return (AUG if s.t else PLAIN)(prep(im, s.r, s.t, s.v, s.n)), s.y[i]

# ----------------------------- 4. MODEL --------------------------------------
# [#166 ArcFace / sub-center ArcFace] Kepala sudut, bukan linear biasa. Alih-alih
# hyperplane, tiap kelas jadi ARAH di hipersfer dan diberi margin sudut, sehingga
# kelas yang mirip (jawi vs pegon: dua-duanya turunan Arab) dipaksa terpisah.
# 'sub-center' k=3: satu kelas boleh punya 3 pusat, berguna karena satu aksara di
# sini muncul dalam beberapa gaya (manuskrip, cetak, tulisan tangan).
class ArcFace(nn.Module):
    def __init__(s, in_f, nc, m=0.30, sc=30.0, k=3):
        super().__init__()
        s.W = nn.Parameter(torch.randn(nc*k, in_f) * 0.01)
        s.nc, s.k, s.m, s.sc = nc, k, m, sc
    def forward(s, x, y=None):
        cos = nn.functional.linear(nn.functional.normalize(x, dim=1),
                                   nn.functional.normalize(s.W, dim=1))
        cos = cos.view(-1, s.nc, s.k).max(2).values          # sub-center: ambil pusat terdekat
        if y is None: return cos * s.sc                      # inferensi: tanpa margin
        th = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        oh = torch.zeros_like(cos); oh.scatter_(1, y.view(-1, 1), 1.0)
        return torch.cos(th + s.m * oh) * s.sc               # margin hanya di kelas benar

class ArcNet(nn.Module):
    def __init__(s, body, feat, nc):
        super().__init__(); s.body, s.head, s.is_arc = body, ArcFace(feat, nc), True
    def forward(s, x, y=None): return s.head(s.body(x), y)

_OK = {}                       # cache: jangan coba unduh ulang tiap fold kalau sudah gagal
def build(name, nc, head="linear"):
    """Coba timm (pretrained) -> torchvision -> random init. Kembalikan (model, pretrained?)."""
    import torchvision.models as tvm
    arc = (head == "arcface")
    if _OK.get(name, True):
        try:
            import timm
            m = timm.create_model(name, pretrained=True, num_classes=(0 if arc else nc))
            _OK[name] = True
            return (ArcNet(m, m.num_features, nc), True) if arc else (m, True)
        except Exception as e:
            _OK[name] = False
            print(f"    ! {name} tak bisa diunduh ({type(e).__name__}) -> fallback resnet34")
    try:
        m = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1); pre = True
    except Exception:
        m = tvm.resnet34(weights=None); pre = False
    feat = m.fc.in_features
    if arc:
        m.fc = nn.Identity(); return ArcNet(m, feat, nc), pre
    m.fc = nn.Linear(feat, nc); return m, pre

class EMA:
    """Polyak averaging - bobot rata-rata jauh lebih stabil drpd bobot epoch terakhir."""
    def __init__(s, model, decay=0.999):
        s.decay = decay
        s.sh = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    def update(s, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point: s.sh[k].mul_(s.decay).add_(v.detach().float(), alpha=1 - s.decay)
            else: s.sh[k] = v.detach().clone().float()
    def state(s, model):
        ref = model.state_dict()
        return {k: v.to(ref[k].dtype) for k, v in s.sh.items()}


def param_groups(model, lr, wd, decay=0.75):
    """Layer-wise LR decay (BEiT/ELECTRA): layer awal LR kecil, head LR penuh."""
    try:
        from timm.optim import param_groups_layer_decay
        return param_groups_layer_decay(model, weight_decay=wd, layer_decay=decay)
    except Exception:
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "weight_decay": wd}]

# --------------------- 5. TRAINING: 1 FOLD + GUARD KOLAPS --------------------
cnt = np.bincount(tr_df.y, minlength=len(CLASSES)); NC = len(CLASSES)
CW = torch.tensor(cnt.sum() / (NC * cnt), dtype=torch.float32, device=DEV)
ytrue = tr_df.y.values

def popf1(p, mask, detail=False):
    """macro-F1 ditimbang ke komposisi strip/blok TEST, bukan train.
    Inilah angka yang dipakai untuk memilih model/ensemble/spesialis."""
    out, parts = 0.0, {}
    for nm, sub, w in (("blok", BLOK_TR, W_BLOK), ("strip", ~BLOK_TR, 1 - W_BLOK)):
        m = mask & sub
        v = f1_score(ytrue[m], p[m].argmax(1), average="macro") if m.sum() >= NC else float("nan")
        parts[nm] = (v, int(m.sum()))
        if not np.isnan(v): out += w * v
    return (out, parts) if detail else out
# Model kolaps = menumpuk prediksi di ~1 kelas (macro-F1 ~0.044). Dideteksi lewat
# KONSENTRASI prediksi, bukan ambang F1 absolut, supaya model yang sekadar LEMAH
# (mis. epoch sedikit) tidak ikut terbuang.
def is_collapsed(pred, f1):
    share = np.bincount(pred, minlength=NC).max() / max(len(pred), 1)
    return bool(share > 0.90 or f1 < 0.10)

@torch.no_grad()
def predict(model, df, r, tta=None, sim=False):
    nv = tta or r["tta"]; acc = np.zeros((len(df), NC)); model.eval()
    for k in range(nv):
        out = []
        for x, _ in DataLoader(ScriptDS(df, r, False, k, nv, sim), batch_size=CFG["batch"]*2,
                               shuffle=False, num_workers=CFG["nw"]):
            x = x.to(DEV, non_blocking=True)
            with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                out.append(torch.softmax(model(x).float(), 1).cpu().numpy())
        acc += np.concatenate(out)
    return acc / nv

def tent_adapt(model, df, r, sim=False):
    """[#122 Tent] Perbarui HANYA parameter lapisan normalisasi supaya entropi
    prediksi di data target turun. Tanpa label. Model asli tidak disentuh."""
    import copy
    m = copy.deepcopy(model)
    norms = [q for q in m.modules()
             if isinstance(q, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm))]
    for p in m.parameters(): p.requires_grad_(False)
    ps = []
    for q in norms:
        if isinstance(q, nn.BatchNorm2d):      # pakai statistik BATCH TARGET
            q.track_running_stats = False; q.running_mean = None; q.running_var = None
        for p in q.parameters(recurse=False):
            p.requires_grad_(True); ps.append(p)
    if not ps:
        del m; return model                    # tak ada lapisan norm -> tak bisa diadaptasi
    opt = torch.optim.SGD(ps, lr=TENT["lr"], momentum=0.9)
    m.train()
    for _ in range(TENT["steps"]):
        for x, _y in DataLoader(ScriptDS(df, r, False, 0, 1, sim), batch_size=CFG["batch"],
                                shuffle=False, num_workers=CFG["nw"]):
            x = x.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            lo = m(x).float()
            ent = -(torch.softmax(lo, 1) * torch.log_softmax(lo, 1)).sum(1).mean()
            ent.backward(); opt.step()
    return m

def tent_gate(model, dva, r):
    """Gerbang JUJUR: uji Tent di validasi yang sengaja dirusak menyerupai test.
    Kalau di sana tidak menolong, jangan dipakai pada test sungguhan."""
    b = predict(model, dva, r, 1, sim=True)
    fb = f1_score(dva.y, b.argmax(1), average="macro")
    m2 = tent_adapt(model, dva, r, sim=True)
    a = predict(m2, dva, r, 1, sim=True)
    fa = f1_score(dva.y, a.argmax(1), average="macro")
    col = is_collapsed(a.argmax(1), fa)
    del m2
    if DEV == "cuda": torch.cuda.empty_cache()
    ok = (fa > fb + TENT["gain"]) and not col
    print(f"  gerbang Tent (validasi dirusak): {fb:.4f} -> {fa:.4f}"
          f"{'  KOLAPS' if col else ''}  -> {'PAKAI' if ok else 'tolak'}")
    return ok

# ---------------- [v10] EKSTRAKSI FITUR + DFR + CONFIDENT LEARNING ----------
def feat_fn(model):
    """kembalikan fungsi x -> fitur penultimate, atau None kalau tak bisa."""
    if getattr(model, "is_arc", False): return lambda x: model.body(x)
    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        return lambda x: model.forward_head(model.forward_features(x), pre_logits=True)
    fc = getattr(model, "fc", None)
    if isinstance(fc, nn.Linear):
        def f(x):
            old = model.fc; model.fc = nn.Identity()
            try: return model(x)
            finally: model.fc = old
        return f
    return None

@torch.no_grad()
def extract(model, df, r):
    fn = feat_fn(model)
    if fn is None: return None
    model.eval(); out = []
    for x, _ in DataLoader(ScriptDS(df, r, False, 0, 1), batch_size=CFG["batch"]*2,
                           shuffle=False, num_workers=CFG["nw"]):
        x = x.to(DEV, non_blocking=True)
        with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
            out.append(fn(x).float().cpu().numpy())
    return np.concatenate(out) if out else None

@torch.no_grad()
def head_probs(model, f):
    """jalankan kepala klasifikasi ASLI di atas fitur yang sudah diekstrak.
    Gratis - tidak ada lintasan gambar tambahan."""
    try:
        cl = model.head if getattr(model, "is_arc", False) else (
             model.get_classifier() if hasattr(model, "get_classifier") else getattr(model, "fc", None))
        if cl is None: return None
        lo = cl(torch.from_numpy(f).to(DEV))
        return torch.softmax(lo.float(), 1).cpu().numpy()
    except Exception: return None

def dfr_fold(model, dtr, dva, r, itr):
    """[#5 Deep Feature Reweighting] CNN dibekukan; HANYA kepala linear yang
    dilatih ulang, pada subsampel yang SEIMBANG antar grup (kelas x binarisasi).
    Kalau sebuah neuron hanya berguna lewat pintasan binarisasi, di data
    group-balanced ia berhenti membantu dan bobotnya mengecil sendiri.
    Biaya: satu lintasan ekstraksi fitur per fold, bukan pelatihan CNN."""
    from sklearn.linear_model import LogisticRegression
    ftr = extract(model, dtr, r)
    if ftr is None: return None
    fva = extract(model, dva, r); fte = extract(model, te_df, r)
    if fva is None or fte is None: return None
    ytr_ = np.asarray(dtr.y.values)
    grp = ytr_ * 2 + BIN_TR[itr].astype(int)          # grup = kelas x binarisasi

    keep = np.ones(len(ytr_), bool)
    if CLEAN["enable"]:
        # [#50 confident learning] buang sampel yang kepala ASLI pun memberi
        # probabilitas sangat rendah pada labelnya sendiri. CATATAN JUJUR: ini
        # probabilitas IN-SAMPLE (sampel ini ikut melatih CNN), jadi lebih lemah
        # daripada cleanlab sungguhan yang memakai probabilitas out-of-sample.
        # Yang tetap tertangkap: sampel yang MUSTAHIL dipelajari (gambar berlabel
        # jawi yang isinya teks Latin, atau cuma angka Arab-Hindi).
        pin = head_probs(model, ftr)
        if pin is not None:
            keep = pin[np.arange(len(ytr_)), ytr_] >= CLEAN["p_max"]
            if keep.sum() < 0.5 * len(keep): keep = np.ones(len(ytr_), bool)   # jangan babat

    rs = np.random.RandomState(SEED)
    ids = np.where(keep)[0]
    if len(ids) < NC * 2: return None
    u, c = np.unique(grp[ids], return_counts=True)
    n = max(DFR["min_per_group"], int(c.min()))      # SUBG: semua grup ke ukuran terkecil
    sel = []
    for gv in u:
        pool = ids[grp[ids] == gv]
        sel.append(rs.choice(pool, min(len(pool), n), replace=False))
    sel = np.concatenate(sel)
    if len(np.unique(ytr_[sel])) < 2: return None

    mu, sd = ftr[sel].mean(0), ftr[sel].std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=DFR["C"], class_weight="balanced")
    clf.fit((ftr[sel] - mu) / sd, ytr_[sel])
    def _full(F):
        q = clf.predict_proba((F - mu) / sd)
        out = np.zeros((len(F), NC))
        for j, cl_ in enumerate(clf.classes_): out[:, int(cl_)] = q[:, j]
        return out
    return _full(fva), _full(fte), int(len(sel)), int((~keep).sum())

def train_fold(r, dtr, dva, lr, epochs, seed):
    seed_all(seed)
    # drop_last hanya kalau datanya cukup, kalau tidak len(ld) bisa 0 -> scheduler pecah
    ld = DataLoader(ScriptDS(dtr, r, True), batch_size=CFG["batch"], shuffle=True,
                    num_workers=CFG["nw"], drop_last=(len(dtr) >= 2*CFG["batch"]),
                    pin_memory=(DEV == "cuda"))
    model, _ = build(r["backbone"], NC, r.get("head", "linear")); model = model.to(DEV)
    crit = nn.CrossEntropyLoss(weight=CW, label_smoothing=CFG["ls"])
    opt = torch.optim.AdamW(param_groups(model, lr, CFG["wd"]), lr=lr, weight_decay=CFG["wd"])
    total_steps = max(10, epochs * max(1, len(ld)))   # OneCycleLR pecah kalau terlalu kecil
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr*2, total_steps=total_steps,
                                              pct_start=0.25)          # warmup 25%
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
    ema = EMA(model, decay=CFG["ema"]); best, best_state, best_pred = -1, None, None
    for ep in range(epochs):
        model.train()
        for x, y in ld:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                lo = model(x, y) if getattr(model, "is_arc", False) else model(x)
                loss = crit(lo, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # cegah divergensi
            if SAM["enable"]:
                # [#99 SAM] naik ke titik TERBURUK dalam radius rho, hitung
                # gradien di sana, lalu kembalikan bobotnya. Dua kali biaya.
                with torch.no_grad():
                    gn = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()
                                                 if p.grad is not None]))
                    eps = []
                    for p in model.parameters():
                        if p.grad is None: eps.append(None); continue
                        e = p.grad * (SAM["rho"] / (gn + 1e-12)); p.add_(e); eps.append(e)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                    lo2 = model(x, y) if getattr(model, "is_arc", False) else model(x)
                    loss2 = crit(lo2, y)
                scaler.scale(loss2).backward(); scaler.unscale_(opt)
                with torch.no_grad():
                    for p, e in zip(model.parameters(), eps):
                        if e is not None: p.sub_(e)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); ema.update(model)
        if ep >= epochs // 3:
            pr = predict(model, dva, r, 1).argmax(1)
            raw = f1_score(dva.y, pr, average="macro")
            bak = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state(model))                    # cek bobot EMA juga
            pe = predict(model, dva, r, 1).argmax(1)
            emf = f1_score(dva.y, pe, average="macro")
            if emf < raw: model.load_state_dict(bak)
            f1, src = max(raw, emf), ("ema" if emf >= raw else "raw")
            if f1 > best:
                best, best_pred = f1, (pe if emf >= raw else pr)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(bak)
    model.load_state_dict(best_state)
    return model, best, best_pred

print("\n" + "="*70 + "\nSTEP 3-4 - TRAINING (guard kolaps + EMA + layer-decay LR)\n" + "="*70)
oof_by, test_by, REPORT = {}, {}, []
skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=SEED)
FOLDS = list(skf.split(tr_df, tr_df.y))

for r in RUNS:
    oof = np.zeros((len(tr_df), NC)); tp = np.zeros((len(te_df), NC))
    oof_d = np.zeros((len(tr_df), NC)); tp_d = np.zeros((len(te_df), NC))   # varian DFR
    f1s, dead, used, used_d, n_drop = [], 0, 0, 0, 0
    use_tent = None                       # None = belum diuji gerbangnya
    print(f"\n--- {r['tag']}: {r['backbone']} | {r['view']} {r['h']}x{r['w']} | lr={r['lr']:.0e} ---")
    for fold, (itr, iva) in enumerate(FOLDS):
        dtr, dva = tr_df.iloc[itr].reset_index(drop=True), tr_df.iloc[iva].reset_index(drop=True)
        lr, model, f1, bad = r["lr"], None, -1, True
        for attempt in range(3):                                       # GUARD: ulang dgn LR lebih kecil
            model, f1, vp = train_fold(r, dtr, dva, lr, CFG["epochs"], SEED + fold*10 + attempt)
            bad = is_collapsed(vp, f1)
            if not bad: break
            lr /= 3
            sh = np.bincount(vp, minlength=NC).max()/len(vp)
            print(f"  fold{fold} KOLAPS (F1={f1:.4f}, {sh*100:.0f}% prediksi 1 kelas)"
                  f" -> ulang dgn lr={lr:.1e}")
        if bad:                                                        # tetap mati -> BUANG
            dead += 1; print(f"  fold{fold} tetap kolaps -> DIKELUARKAN dari ensemble")
            del model; torch.cuda.empty_cache() if DEV == "cuda" else None; continue
        if TENT["enable"] and use_tent is None: use_tent = tent_gate(model, dva, r)
        oof[iva] = predict(model, dva, r)
        if use_tent:
            mt = tent_adapt(model, te_df, r)          # adaptasi ke TEST sungguhan
            pt = predict(mt, te_df, r)
            if is_collapsed(pt.argmax(1), 1.0):       # pengaman terakhir
                print(f"  fold{fold} Tent kolaps di test -> pakai prediksi tanpa Tent")
                pt = predict(model, te_df, r)
            del mt
            if DEV == "cuda": torch.cuda.empty_cache()
        else:
            pt = predict(model, te_df, r)
        tp += pt; used += 1
        f1s.append(f1); print(f"  fold{fold} macro-F1={f1:.4f}")
        if DFR["enable"]:
            try:
                res = dfr_fold(model, dtr, dva, r, itr)
            except Exception as e:
                res = None; print(f"  fold{fold} DFR gagal ({type(e).__name__}: {e})")
            if res is not None:
                pv_d, pt_dd, n_used, n_dr = res
                oof_d[iva] = pv_d; tp_d += pt_dd; used_d += 1; n_drop += n_dr
                fd = f1_score(dva.y, pv_d.argmax(1), average="macro")
                print(f"  fold{fold} DFR: {n_used} sampel group-balanced, "
                      f"{n_dr} dibuang (confident learning), F1 fold {f1:.4f} -> {fd:.4f}")
        del model
        if DEV == "cuda": torch.cuda.empty_cache()
    if used == 0:
        print(f"  !! {r['tag']} gagal total, dilewati"); continue
    tp /= used                                                          # rata-rata HANYA fold sehat
    ok = oof.sum(1) > 0                                                 # baris tanpa model sehat
    # GERBANG DFR: pakai kepala hasil DFR hanya kalau OOF TERTIMBANG naik.
    if DFR["enable"] and used_d == used and (oof_d.sum(1) > 0).sum() >= ok.sum():
        f_base, f_dfr = popf1(oof, ok), popf1(oof_d, ok)
        print(f"  gerbang DFR: OOF tertimbang {f_base:.4f} -> {f_dfr:.4f} "
              f"({f_dfr-f_base:+.4f}; {n_drop} sampel dibuang) -> "
              f"{'PAKAI' if f_dfr > f_base + DFR['gain'] else 'tolak'}")
        if f_dfr > f_base + DFR["gain"]:
            oof, tp = oof_d, tp_d / max(used_d, 1)
    elif DFR["enable"]:
        print(f"  gerbang DFR dilewati (fold DFR sukses {used_d}/{used})")
    o = f1_score(ytrue[ok], oof[ok].argmax(1), average="macro")
    oof_by[r["tag"]], test_by[r["tag"]] = oof, tp
    REPORT.append(dict(model=r["tag"], backbone=r["backbone"].split(".")[0], view=r["view"],
                       res=f"{r['h']}x{r['w']}", fold_mean=np.mean(f1s), fold_sd=np.std(f1s),
                       oof=o, dead=dead, cov=ok.mean()))
    print(f"  => fold mean={np.mean(f1s):.4f} sd={np.std(f1s):.4f} | OOF={o:.4f} | fold mati={dead}")

# ------------- 6. TABEL METRIK ANTAR MODEL + ENSEMBLE + VALIDASI -------------
print("\n" + "="*70 + "\nSTEP 4 - PERBANDINGAN METRIK ANTAR MODEL\n" + "="*70)
assert REPORT, ("Semua run gagal/kolaps. Cek: bobot pretrained terunduh? lr terlalu besar? "
                "Kurangi lr di RUNS atau tambah epoch.")
rep = pd.DataFrame(REPORT).sort_values("oof", ascending=False)
print("\n[A] Ringkasan per model (urut dari terbaik):\n")
print(rep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
if rep.dead.sum() == 0: print("\n  OK: tidak ada fold yang kolaps.")
else: print(f"\n  PERHATIAN: {int(rep.dead.sum())} fold kolaps & sudah dikeluarkan dari ensemble.")

print("\n[B] F1 per kelas tiap model (kolom paling rendah = titik lemah):\n")
per = {t: f1_score(ytrue[o.sum(1) > 0], o[o.sum(1) > 0].argmax(1), average=None, zero_division=0)
       for t, o in oof_by.items()}
pc = pd.DataFrame(per, index=CLASSES).T
pc["MACRO"] = pc.mean(1)
print(pc.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\n  kelas terlemah rata-rata semua model: "
      f"{pc[CLASSES].mean().idxmin()} ({pc[CLASSES].mean().min():.4f})")

# [B2] Inilah tabel yang v3.1 TIDAK punya, dan ketiadaannya yang menyembunyikan
# kegagalan: model bisa terlihat hebat di campuran train tapi lemah di populasi
# 'blok' yang porsinya jauh lebih besar di test.
print("\n[B2] Performa per POPULASI (ini yang menentukan skor test):\n")
print(f"  {'model':14s}{'blok':>9s}{'strip':>9s}{'polos':>9s}{'tertimbang':>13s}{'bias OOF':>11s}")
for t, o in oof_by.items():
    m = o.sum(1) > 0
    wt, pp = popf1(o, m, detail=True)
    pl = f1_score(ytrue[m], o[m].argmax(1), average="macro")
    print(f"  {t:14s}{pp['blok'][0]:9.4f}{pp['strip'][0]:9.4f}{pl:9.4f}{wt:13.4f}{pl-wt:+11.4f}")
print("  ('bias OOF' = seberapa besar OOF polos MELEBIHKAN performa di komposisi test)")

# Greedy ensemble selection (Caruana et al. 2004): tambah model satu per satu
# selama OOF macro-F1 naik - lebih aman drpd merata-ratakan semuanya buta.
def mf1(p, mask): return f1_score(ytrue[mask], p[mask].argmax(1), average="macro")

tags = list(oof_by); mask = np.ones(len(tr_df), bool)
for t in tags: mask &= oof_by[t].sum(1) > 0
cur, chosen, best = np.zeros((len(tr_df), NC)), [], -1
for _ in range(12):                               # seleksi boleh mengulang model = pembobotan
    cand = max(tags, key=lambda t: popf1((cur*len(chosen) + oof_by[t])/(len(chosen)+1), mask))
    sc = popf1((cur*len(chosen) + oof_by[cand])/(len(chosen)+1), mask)
    if sc <= best + 1e-6: break
    cur = (cur*len(chosen) + oof_by[cand])/(len(chosen)+1); chosen.append(cand); best = sc
    print(f"  greedy += {cand:14s} -> OOF {sc:.4f}")

single = max(tags, key=lambda t: popf1(oof_by[t], oof_by[t].sum(1) > 0))
if popf1(oof_by[single], mask) >= best: chosen, best = [single], popf1(oof_by[single], mask)
# Di v6/v7/v8 greedy SELALU berhenti di 1 model -> 4 dari 5 run terbuang.
# Paksa uji rata-rata semua model; pakai kalau tidak lebih buruk dari ambang
# derau. Ensemble beragam lebih tahan pergeseran domain drpd model tunggal.
if len(set(chosen)) == 1 and len(tags) > 1:
    _all = np.mean([oof_by[t] for t in tags], 0)
    _fa = popf1(_all, mask)
    print(f"  cek rata-rata SEMUA {len(tags)} model -> OOF {_fa:.4f} (tunggal {best:.4f})")
    if _fa >= best - 0.002:            # ambang derau; seri -> pilih yg beragam
        chosen, best = list(tags), _fa
        print("  -> pakai ENSEMBLE penuh (lebih tahan pergeseran domain)")
w = pd.Series(Counter(chosen)) / len(chosen)      # tampilkan sbg BOBOT, bukan daftar berulang
print(f"\n[C] Bobot ensemble terpilih (OOF={best:.4f}):")
for t, v in w.sort_values(ascending=False).items(): print(f"      {t:14s} {v*100:5.1f}%")
print(f"    model tunggal terbaik: {single} (OOF-tertimbang={popf1(oof_by[single], mask):.4f})")
oof = np.mean([oof_by[t] for t in chosen], 0); test_prob = np.mean([test_by[t] for t in chosen], 0)

# ---- [#204 kalibrasi suhu] + [#206/#208 product fusion] --------------------
# Dua metode dari daftar 300. Rata-rata aritmetik memaksa model yang RAGU ikut
# menarik hasil; perkalian (rata-rata geometrik) memberi hak veto pada model yang
# yakin-salah-sedikit dan lebih menghargai kesepakatan. Suhu ditala per model via
# NLL di OOF (bukan via F1: suhu TIDAK mengubah argmax satu model, hanya bobotnya
# saat digabung). Keempat kombinasi diadu di OOF TERTIMBANG; default 'rata-rata
# mentah' hanya diganti kalau unggul > 0.002. Ini gerbangnya, bukan keyakinan.
print("\n" + "="*70 + "\nSTEP 4a - CARA FUSI ENSEMBLE (kalibrasi + produk)\n" + "="*70)
def _temp(p, T):
    l = np.log(np.maximum(p, 1e-12)) / T
    l -= l.max(1, keepdims=True); e = np.exp(l)
    return e / np.maximum(e.sum(1, keepdims=True), 1e-12)

_wts = Counter(chosen); _tot = sum(_wts.values())
TEMP = {}
for t in _wts:                                   # kalibrasi: minimalkan NLL, bukan F1
    _bT, _bN = 1.0, float("inf")
    for T in (0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0):
        q = _temp(oof_by[t], T)
        n = float(-np.log(np.maximum(q[mask, ytrue[mask]], 1e-12)).mean())
        if n < _bN: _bT, _bN = T, n
    TEMP[t] = _bT
print("  suhu terpilih: " + "  ".join(f"{t}={TEMP[t]:.2f}" for t in _wts))

def _fuse(src, prod):
    a = np.zeros_like(next(iter(src.values())))
    for t, k in _wts.items():
        a += (k/_tot) * (np.log(np.maximum(src[t], 1e-12)) if prod else src[t])
    if prod: a = np.exp(a - a.max(1, keepdims=True))
    return a / np.maximum(a.sum(1, keepdims=True), 1e-12)

_oc = {t: _temp(oof_by[t], TEMP[t]) for t in _wts}
_tc = {t: _temp(test_by[t], TEMP[t]) for t in _wts}
CAND = {"rata-rata":            (lambda: _fuse(oof_by, 0), lambda: _fuse(test_by, 0)),
        "rata-rata+kalibrasi":  (lambda: _fuse(_oc,    0), lambda: _fuse(_tc,    0)),
        "produk":               (lambda: _fuse(oof_by, 1), lambda: _fuse(test_by, 1)),
        "produk+kalibrasi":     (lambda: _fuse(_oc,    1), lambda: _fuse(_tc,    1))}
_base_f = popf1(_fuse(oof_by, 0), mask); _pick, _pf = "rata-rata", _base_f
for nm, (fo, _) in CAND.items():
    f = popf1(fo(), mask)
    mark = "" if nm == "rata-rata" else ("  <- unggul" if f > _base_f + 0.002 else "")
    print(f"  {nm:22s} OOF {f:.4f}{mark}")
    if nm != "rata-rata" and f > _pf + 0.002: _pick, _pf = nm, f
print(f"  -> dipakai: {_pick} (OOF {_pf:.4f}); default hanya diganti kalau unggul > 0.002")
if len(_wts) == 1: print("  (ensemble cuma 1 model -> fusi tidak berpengaruh)")
oof, test_prob = CAND[_pick][0](), CAND[_pick][1]()

prior = cnt / cnt.sum()

# --- tau logit-adjustment SENGAJA DIBUANG. Di v3.1 ia terpilih pada gain
# --- +0.0000 (perbandingan "if f > bf1" lolos pada derau pembulatan) lalu tetap
# --- diterapkan ke test. Diuji ulang di bangku uji pergeseran domain: tau
# --- merugikan di SEMUA kondisi, begitu pula koreksi prior EM/SLD (runtuh
# --- 0.41 -> 0.087) dan bahkan prior ORACLE (-0.05). Kalau mengetahui prior
# --- test dengan sempurna saja merugikan, ini bukan masalah prior.
base = f1_score(ytrue[mask], oof[mask].argmax(1), average="macro")
bf1, _pp = popf1(oof, mask, detail=True)
pb, ps = _pp["blok"], _pp["strip"]
print(f"\nF1 Macro OOF polos (campuran train)       : {base:.4f}   <- angka yang menipu di v3.1")
print("F1 Macro OOF per populasi:")
print(f"  blok  (n={pb[1]:5d}) : {pb[0]:.4f}")
print(f"  strip (n={ps[1]:5d}) : {ps[0]:.4f}")
print(f"F1 Macro OOF TERTIMBANG ke komposisi test : {bf1:.4f}   <<< dasar pemilihan")
if base - bf1 > 0.005:
    print(f"  -> OOF polos melebihkan {base-bf1:+.4f}; selisih seperti inilah yang dulu muncul di LB.")

pred = oof.argmax(1)
print("\n[D] Laporan akhir ensemble:\n")
print(classification_report(ytrue[mask], pred[mask], target_names=CLASSES, digits=4, zero_division=0))
cm = confusion_matrix(ytrue[mask], pred[mask])
print("Confusion matrix (baris = asli, kolom = prediksi):")
print(f"{'':>10s}" + "".join(f"{c[:7]:>8s}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    print(f"{c:>10s}" + "".join(f"{v:8d}" for v in cm[i]) + f"   (recall {cm[i,i]/cm[i].sum():.3f})")
pf = f1_score(ytrue[mask], pred[mask], average=None, zero_division=0)
print("\nKelas paling lemah -> terkuat:")
for i in np.argsort(pf):
    off = cm[i].copy(); off[i] = 0; j = int(off.argmax())
    print(f"  {CLASSES[i]:10s} F1={pf[i]:.4f} n={cnt[i]:4d} tertukar dgn '{CLASSES[j]}' ({off[j]}x)")

# ---------- 6b. SPESIALIS PASANGAN TERSULIT (cascade / coarse-to-fine) -------
# Sisa error menumpuk di SATU pasangan kelas (di sini jawi<->pegon: dua-duanya
# turunan aksara Arab, dan ~26% gambar jawi cuma fragmen 1-3 huruf). Model umum
# harus membagi kapasitasnya ke 7 kelas; spesialis biner resolusi tinggi hanya
# mengurus pasangan itu. Dipakai HANYA kalau OOF naik.
spec_used = False
off = cm.copy(); np.fill_diagonal(off, 0)
i, j = np.unravel_index((off + off.T).argmax(), off.shape)
A, B = CLASSES[int(i)], CLASSES[int(j)]
n_err = int(off[i, j] + off[j, i])
# kalau OOF sudah sempurna, argmax matriks nol memberi A==B -> jangan latih spesialis
if not (SPEC["enable"] and len(CLASSES) > 2 and A != B and n_err > 0):
    print("\n" + "="*70 + "\nSTEP 4b - spesialis DILEWATI "
          f"(pasangan tertukar terbanyak: {n_err} error)\n" + "="*70)
if SPEC["enable"] and len(CLASSES) > 2 and A != B and n_err > 0:
    print("\n" + "="*70 + f"\nSTEP 4b - SPESIALIS '{A}' vs '{B}' ({n_err} error OOF)\n" + "="*70)

    sdf = tr_df[tr_df.label.isin([A, B])].reset_index(drop=True)
    sdf["y"] = (sdf.label == B).astype(int)              # biner: 0=A, 1=B
    sr = dict(SPEC); sr["backbone"] = SPEC["backbone"]
    NC_BAK = NC; CW_BAK = CW
    globals()["NC"] = 2
    globals()["CW"] = torch.tensor([1.0, 1.0], dtype=torch.float32, device=DEV)

    s_oof = np.zeros((len(sdf), 2)); s_test = np.zeros((len(te_df), 2)); used_s = 0
    sk2 = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=SEED)
    for fold, (a, b) in enumerate(sk2.split(sdf, sdf.y)):
        dtr2, dva2 = sdf.iloc[a].reset_index(drop=True), sdf.iloc[b].reset_index(drop=True)
        m2, f2, vp2 = train_fold(sr, dtr2, dva2, SPEC["lr"], SPEC["epochs"], SEED + fold)
        if is_collapsed(vp2, f2):
            print(f"  fold{fold} spesialis kolaps -> dilewati"); del m2; continue
        s_oof[b] = predict(m2, dva2, sr); s_test += predict(m2, te_df, sr); used_s += 1
        print(f"  fold{fold} akurasi biner={(s_oof[b].argmax(1)==dva2.y).mean():.4f}")
        del m2
        if DEV == "cuda": torch.cuda.empty_cache()

    globals()["NC"] = NC_BAK; globals()["CW"] = CW_BAK
    if used_s:
        s_test /= used_s
        ia, ib = C2I[A], C2I[B]
        pos = {img: k for k, img in enumerate(sdf.image_id)}     # baris train -> baris spesialis
        oof_base = oof.copy()          # BUG FIX: setiap ambang diuji dari OOF ASLI,
        best_p2 = None                 # bukan dari OOF yang sudah diubah ambang sebelumnya
        for TH in [0.0, 0.5, 0.7, 0.9, 0.95]:                   # seberapa "ragu" baru diserahkan
            p2 = oof_base.copy()
            for k, img in enumerate(tr_df.image_id):
                if img not in pos: continue
                pr = oof_base[k]
                if {int(np.argsort(pr)[-1]), int(np.argsort(pr)[-2])} != {ia, ib}: continue
                if pr[[ia, ib]].sum() < TH: continue
                sp = s_oof[pos[img]]
                if sp.sum() == 0: continue
                tot = pr[ia] + pr[ib]
                p2[k, ia], p2[k, ib] = tot*sp[0], tot*sp[1]      # bagi ulang massa A/B
            f = popf1(p2, mask)
            print(f"  ambang {TH:.2f}: OOF {f:.4f}  ({'PAKAI' if f > bf1 + 1e-6 else 'tolak'})")
            if f > bf1 + 1e-6:
                bf1, best_p2, spec_used, SPEC_TH = f, p2, True, TH
        if spec_used:
            oof = best_p2
            for k in range(len(te_df)):                          # terapkan ke test
                pr = test_prob[k]
                if {int(np.argsort(pr)[-1]), int(np.argsort(pr)[-2])} != {ia, ib}: continue
                if pr[[ia, ib]].sum() < SPEC_TH: continue
                tot = pr[ia] + pr[ib]
                test_prob[k, ia], test_prob[k, ib] = tot*s_test[k, 0], tot*s_test[k, 1]
            print(f"\n  -> spesialis DIPAKAI (ambang {SPEC_TH:.2f}), OOF naik jadi {bf1:.4f}")
        else:
            print("\n  -> spesialis tidak menaikkan OOF, TIDAK dipakai")
    else:
        print("  semua fold spesialis kolaps, dilewati")

# ------------- 6c. AUDIT PLAFON: sisa error itu model atau LABEL? ------------
# Kalau model SANGAT yakin pada kelas lain daripada labelnya, kemungkinan besar
# labelnya yang keliru / gambarnya memang ambigu -> ini plafon data, bukan model.
fin = oof
conf = fin.max(1)/np.maximum(fin.sum(1), 1e-9)
wrong = np.where((fin.argmax(1) != ytrue) & mask)[0]
print("\n" + "="*70 + "\nSTEP 4c - AUDIT PLAFON (sisa error OOF)\n" + "="*70)
print(f"\nsisa error OOF: {len(wrong)} dari {int(mask.sum())} gambar")
if len(wrong):
    hi = wrong[conf[wrong] >= 0.90]
    print(f"  model SANGAT yakin tapi beda label (>=0.90): {len(hi)} -> kandidat label keliru/ambigu")
    print(f"  model ragu (<0.90)                        : {len(wrong)-len(hi)} -> masih bisa diperbaiki model")
    aud = pd.DataFrame({
        "image_id": tr_df.image_id.values[wrong],
        "label_data": [CLASSES[i] for i in ytrue[wrong]],
        "prediksi":   [CLASSES[i] for i in fin.argmax(1)[wrong]],
        "confidence": conf[wrong].round(4),
    }).sort_values("confidence", ascending=False)
    aud.to_csv("audit_kasus_sulit.csv", index=False)
    print("\n10 kasus paling mencurigakan (disimpan: audit_kasus_sulit.csv):")
    print(aud.head(10).to_string(index=False))
    print("\n  Periksa gambar-gambar ini manual. Kalau labelnya memang keliru,")
    print("  skor 1.00000 secara definisi tidak tercapai lewat modeling.")

# ---- Optimasi AMBANG PER-KELAS untuk macro-F1 (metode #202) ---------------
# BEDA PENTING dari kuota Hungarian: bobot ini DITALA di OOF TERTIMBANG, jadi
# angkanya punya dasar, bukan tebakan. argmax bukan keputusan optimal untuk
# macro-F1 pada kelas timpang - kelas kecil butuh ambang lebih longgar.
# Dipakai HANYA kalau menaikkan OOF tertimbang melebihi ambang derau.
print("\n" + "="*70 + "\nSTEP 4d - OPTIMASI AMBANG PER-KELAS (macro-F1)\n" + "="*70)
CW_OPT = np.ones(NC)
_base_w = popf1(oof, mask)
for _sweep in range(3):                       # coordinate ascent
    for c in range(NC):
        best_m, best_f = CW_OPT[c], popf1(oof * CW_OPT, mask)
        for m in (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.45, 1.7, 2.0):
            t = CW_OPT.copy(); t[c] = m
            f = popf1(oof * t, mask)
            if f > best_f + 1e-6: best_m, best_f = m, f
        CW_OPT[c] = best_m
_f_opt = popf1(oof * CW_OPT, mask)
print(f"  OOF tertimbang: {_base_w:.4f} -> {_f_opt:.4f}  ({_f_opt-_base_w:+.4f})")
print("  bobot per kelas: " + "  ".join(f"{c}={CW_OPT[i]:.2f}" for i, c in enumerate(CLASSES)))
USE_CW = _f_opt > _base_w + 0.002            # ambang derau; kalau tidak, jangan
print(f"  -> {'DIPAKAI' if USE_CW else 'TIDAK dipakai (kenaikan di dalam derau)'}")
if USE_CW:
    test_prob = test_prob * CW_OPT
    oof = oof * CW_OPT
    bf1 = _f_opt

# ---------------------- 7. PREDIKSI TEST & SUBMISSION ------------------------
print("\n" + "="*70 + "\nSTEP 5 - SUBMISSION\n" + "="*70)
sub = pd.DataFrame({"image_id": te_df.image_id.values,
                    "label": [CLASSES[i] for i in test_prob.argmax(1)]})
if sub_csv:
    ss = pd.read_csv(sub_csv)
    if set(ss.image_id) == set(sub.image_id): sub = ss[["image_id"]].merge(sub, on="image_id", how="left")
    else: print("PERINGATAN: sample_submission != test.csv -> pakai urutan test.csv")
if LEAK:                                   # timpa dgn label yang sudah pasti
    _before = sub.label.copy()
    sub["label"] = [LEAK.get(i, l) for i, l in zip(sub.image_id, sub.label)]
    _ch = int((_before != sub.label).sum())
    print(f"\nkebocoran diterapkan: {len(LEAK)} baris dipastikan, {_ch} di antaranya mengubah prediksi")
sub.to_csv("submission.csv", index=False)
assert len(sub) == len(te_df) and sub.label.notna().all() and sub.image_id.is_unique
print(f"\nsubmission.csv: {len(sub)} baris")
print("\nDistribusi prediksi test vs train (INI PENANDA KEBERHASILAN v6):")
pv = sub.label.value_counts(); worst = 0.0
for c in CLASSES:
    d = pv.get(c, 0)/len(sub)*100 - cnt[C2I[c]]/cnt.sum()*100
    worst = max(worst, abs(d))
    print(f"  {c:10s} test {pv.get(c,0):5d} ({pv.get(c,0)/len(sub)*100:5.2f}%)  train "
          f"{cnt[C2I[c]]/cnt.sum()*100:5.2f}%  selisih {d:+5.2f}pp {'  <-- CEK' if abs(d) > 3 else ''}")
# PLAFON: macro-F1 tertinggi yg MUNGKIN dicapai dgn jumlah prediksi ini, kalau
# prior test ~ prior train. Riwayat: v3.1 plafon 0.8181/LB 0.8073; v5 0.8445/0.8105;
# v7 0.8407/0.8342. Ketiganya mendarat tepat di bawah plafonnya sendiri.
_ceil = 0.0
for c in CLASSES:
    _t = prior[C2I[c]]*len(sub); _p = pv.get(c, 0)
    _ceil += 2*min(_t, _p)/(_p + _t) if _p else 0.0
_ceil /= NC
print(f"\nskew terbesar     = {worst:.2f}pp   (v3.1 11.48 | v5 10.00 | v7 9.59)")
print(f"PLAFON macro-F1   = {_ceil:.4f}   <- batas atas dari JUMLAH prediksi saja")
print("  CATATAN: ini diagnostik, BUKAN perintah submit/jangan-submit. Gerbang")
print("  otomatis di v6/v7 pernah memvonis 'jangan submit' pada run yang justru")
print("  memberi kenaikan LB terbesar (+0.024). Baca angkanya, jangan vonisnya.")
# ---- Decoding berbatas-hitungan -> submission KEDUA (BELUM TERUJI) ----------
# Dasar: v7 hanya 0.0065 di bawah plafonnya, jadi akurasi sudah mentok; plafon
# hanya naik kalau jumlah prediksi per kelas mendekati prior. TIDAK dapat diuji
# dari OOF (prior OOF sudah cocok), jadi ini taruhan beralasan, bukan hasil uji.
# ALPHA=0 -> tanpa perubahan; 1 -> paksa penuh ke prior train; 0.5 -> separuh.
ALPHA = 0.5
try:
    from scipy.optimize import linear_sum_assignment
    _pp = test_prob / np.maximum(test_prob.sum(1, keepdims=True), 1e-12)
    _now = np.bincount(_pp.argmax(1), minlength=NC) / len(_pp)
    _tgt = ALPHA*prior + (1-ALPHA)*_now; _tgt = _tgt/_tgt.sum()
    _q = np.floor(_tgt*len(_pp)).astype(int)
    while _q.sum() < len(_pp): _q[np.argmax(_tgt*len(_pp) - _q)] += 1
    _cols = np.concatenate([[c]*_q[c] for c in range(NC) if _q[c] > 0])
    _r, _c = linear_sum_assignment(-np.log(np.maximum(_pp, 1e-12))[:, _cols])
    _p2 = np.empty(len(_pp), int); _p2[_r] = _cols[_c]
    _s2 = pd.DataFrame({"image_id": te_df.image_id.values,
                        "label": [CLASSES[i] for i in _p2]})
    if sub_csv and set(ss.image_id) == set(_s2.image_id):
        _s2 = ss[["image_id"]].merge(_s2, on="image_id", how="left")
    if LEAK: _s2["label"] = [LEAK.get(i, l) for i, l in zip(_s2.image_id, _s2.label)]
    _s2.to_csv("submission_kuota.csv", index=False)
    _ch2 = int((_p2 != test_prob.argmax(1)).sum())
    print(f"\nsubmission_kuota.csv ditulis (ALPHA={ALPHA}): {_ch2} prediksi berubah")
    _pv2 = _s2.label.value_counts()
    _w2 = max(abs(_pv2.get(c,0)/len(_s2)*100 - prior[C2I[c]]*100) for c in CLASSES)
    print(f"  skew terbesar setelah kuota = {_w2:.2f}pp (sebelumnya {worst:.2f}pp)")
except Exception as e:
    print(f"\nsubmission_kuota.csv dilewati ({type(e).__name__}: {e})")

print(f"\n>>> F1-Score Macro OOF (ensemble {len(set(chosen))} model) = {bf1:.4f} <<<")
