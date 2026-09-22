# =============================================================================
# v14_colab.py - ENSEMBLE LINTAS-KELUARGA: SigLIP2-384 + DINOv3-B/16 @384
#                Jalan di Colab MAUPUN Kaggle. Tahan sesi putus.
# =============================================================================
# KENAPA v14 ADA, dan kenapa ia berbeda dari semua percobaan ensemble sebelumnya
# -----------------------------------------------------------------------------
# v13 mencetak LB 0.89166 (peringkat 5) dgn SigLIP2-384 TUNGGAL. Semua tambahan
# yang dicobanya ditolak gerbangnya sendiri: seed 1 (-0.0033) dan seed 2
# (+0.0005) di bawah ambang, TTA +0.0001, spesialis pegon/jawi +0.0003, bobot
# kelas +0.0001. Artinya jalur "lebih banyak dari resep yang sama" buntu.
#
# Yang TIDAK buntu, dan sekarang SUDAH TERUKUR: keluarga arsitektur lain.
# Sonde v14_sonde.py melatih DINOv3-B/16 @384 di fold 0 dan mendapat
#     DINOv3   fold0 = 0.9858
#     SigLIP2  fold0 = 0.9849 / 0.9857 / 0.9879  (3 seed, rata2 0.9862, sd 0.0013)
# Selisihnya -0.0003, yaitu 0.3x sd seed. Dua model itu TIDAK BISA DIBEDAKAN
# kualitasnya, padahal arsitektur dan pra-latihnya sama sekali berbeda
# (SigLIP2 = kontrastif gambar-teks WebLI; DINOv3 = self-supervised LVD-1689M).
#
# Itulah syarat yang GAGAL dipenuhi `submission_iw.csv` (LB 0.85695, turun
# 0.02054 dari 0.87749): ia merata-ratakan tiga model dari backbone yang SAMA,
# salah satunya 0.058 lebih lemah. Di sini keduanya setara dalam 0.0003 dan
# keluarganya berbeda. Ini pertama kalinya di proyek ini prasyarat ensemble
# benar-benar terpenuhi, bukan diasumsikan.
#
# APA YANG DIGERBANG, DAN APA YANG TIDAK
# --------------------------------------
# Satu-satunya keputusan baru di v14 adalah BOBOT CAMPUR w antara dua anggota,
# dan ia dicari di OOF tertimbang (popf1) lalu HARUS melewati ambang +0.0010 di
# atas anggota tunggal TERBAIK. Kalau tidak lewat, v14 memakai anggota tunggal
# terbaik dan mengatakannya. Gerbang ini bisa menghasilkan w=0 atau w=1, dan
# itu jawaban yang sah - persis seperti v13 yang berakhir dgn satu seed.
#
# Yang sudah terbantah TIDAK dihidupkan lagi: tanpa Otsu, tanpa LP-FT, tanpa
# pra-latih sintetis, tanpa koreksi prior EM, tanpa seed-bag, tanpa TTA
# (keempat view v13 ada DI DALAM rentang augmentasi latih, jadi tidak ada
# informasi baru - itu struktural, bukan kebetulan), tanpa spesialis biner.
# Resep latih per anggota SAMA PERSIS dgn v13: 384x384, 18 epoch, lr body 3e-5
# head 1e-3, layer_decay 0.75, micro 16 x accum 2, wd 0.05, ls 0.05, EMA 0.999,
# 5 fold StratifiedKFold(shuffle=True, random_state=42).
#
# HARGA DAN HARAPAN YANG JUJUR
# ----------------------------
# Biaya: 2 anggota x 5 fold x 0.59 jam = ~5,9 jam T4 terukur, plus cache dan
# tolok ukur. Anggaran bawaan 8 jam.
# Harapan: TIDAK DIKETAHUI. Ini ensemble pertama di proyek ini yang anggotanya
# benar-benar setara dan benar-benar beragam, jadi tidak ada preseden di data
# ini untuk dipakai meramal. Yang bisa dikatakan: gerbangnya membuat sisi
# rugi mendekati nol (terburuk = sama dgn anggota tunggal terbaik), sementara
# sisi untungnya terbuka. Kalau OOF naik >= +0.005, itu setara ~+0.011 LB
# dgn laju tukar v12->v13, yang kira-kira sebesar jarak ke peringkat 4.
#
# BERJALAN DI COLAB - DAN KENAPA ITU BUTUH PENANGANAN KHUSUS
# ----------------------------------------------------------
# Colab tidak menjamin GPU, tidak menerbitkan kuota, dan MENGHAPUS /content
# begitu sesi putus. Run 6 jam yang mati di jam ke-5 di Colab kehilangan
# SEMUANYA kalau checkpoint-nya ditaruh di disk lokal. Jadi di Colab skrip ini
# memasang Google Drive dan menaruh checkpoint + submission DI SANA. Jalankan
# ulang sel yang sama setelah putus dan ia melanjutkan dari fold terakhir yang
# selesai, bukan dari nol. Di Kaggle perilakunya tidak berubah
# (/kaggle/working sudah persisten).
#
# CARA PAKAI
#   Colab : satu sel. Ia akan minta izin memasang Drive. Data dicari di
#           /content/drive/MyDrive/infest/ (lihat DRIVE_DATA di bawah) lalu di
#           /content. Kalau data tidak ketemu ia BERHENTI dan bilang di mana
#           ia sudah mencari - ia tidak akan diam-diam melatih data yang salah.
#   Kaggle: satu sel, apa adanya, seperti v13.
#   INFEST_BUDGET_H mengatur pagar waktu (bawaan 8.0).
# =============================================================================
import os, sys, csv, glob, math, time, json, copy, random, zipfile, hashlib, inspect, warnings
from collections import Counter, OrderedDict
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# =============================================================================
# 0. KONFIGURASI
# =============================================================================
SMOKE = os.environ.get("INFEST_SMOKE", "0") == "1"
SEED  = 42

# --- ANGGARAN ----------------------------------------------------------------
# Dua anggota x 5 fold x 0.59 jam terukur = ~5,9 jam, plus cache dan tolok ukur.
# Gubernur MENGUKUR kecepatan GPU dulu; kalau tidak muat ia memangkas jumlah
# fold anggota KEDUA, bukan merusak anggota pertama yang sudah terbukti.
BUDGET = dict(
    total_h   = float(os.environ.get("INFEST_BUDGET_H", "8.0")),
    reserve_h = 0.40,
    bench     = True,
    bench_steps = 12,
    safety    = 1.15,
)

# --- DUA ANGGOTA -------------------------------------------------------------
# Urutannya penting. Anggota 0 adalah resep v13 yang SUDAH mencetak 0.89166;
# ia dilatih lebih dulu supaya kalau sesi mati di tengah, yang tertinggal
# adalah submission yang setara v13, bukan setengah eksperimen.
# `alias_h`/`alias_w` tidak ada: keduanya 384, sengaja, supaya satu-satunya
# perbedaan antar anggota adalah backbone-nya.
MEMBERS = [
    dict(mid=0, name="siglip2-384", model="vit_base_patch16_siglip_384.v2_webli",
         h=384, w=384, epochs=18,
         note="resep v13 apa adanya. OOF popf1 terukur 0.9799 -> LB 0.89166."),
    dict(mid=1, name="dinov3-b16", model="vit_base_patch16_dinov3.lvd1689m",
         h=384, w=384, epochs=18,
         note="keluarga lain. Sonde fold0 0.9858 vs SigLIP2 0.9862 (sd 0.0013)."),
]

# --- GERBANG CAMPUR (satu-satunya keputusan BARU di v14) ---------------------
# w adalah bobot anggota 1 (DINOv3): prob = (1-w)*anggota0 + w*anggota1.
# Dicari di OOF tertimbang, dipakai hanya kalau melewati anggota tunggal
# TERBAIK dgn selisih di atas ambang derau.
BLEND = dict(
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    gate = 0.0010,      # sama dgn ambang seed-bag v13
    # Campur terpisah utk populasi blok dan strip. Blok = 49,2% test tapi F1-nya
    # 0.9634 lawan strip 0.9958, jadi bobot optimalnya BOLEH berbeda. Ini
    # menambah 1 derajat kebebasan atas 3.771 baris, jadi ambangnya DINAIKKAN:
    # ia harus mengalahkan campur tunggal, bukan cuma anggota tunggal.
    per_pop = True,
    per_pop_gate = 0.0020,
)

# --- resep latih: SAMA PERSIS dgn v13, tidak satu knob pun digeser -----------
FT = dict(lr_body=3e-5, lr_head=1e-3, layer_decay=0.75)
CFG = dict(n_folds=5, batch=16, accum=2, nw=2, wd=0.05, ls=0.05, ema=0.999)
MAXSIDE   = 1600
CACHE_MAX = 1280
CACHE     = dict(enable=True)
SEV = 1.0
DEGRADE = dict(p_blur=0.70, blur=(0.6*SEV, 2.2*SEV),
               p_contrast=0.60, contrast=(0.45, 0.95),
               p_rescale=0.50, rescale=(max(0.15, 0.35/SEV), 0.80),
               p_jpeg=0.50, jpeg=(max(10, int(25/SEV)), 88))
AUGMAX = dict(enable=True, p=0.35, chains=3)
ARJIT  = dict(enable=True, lo=0.65, hi=1.55)
GATE   = dict(cw=0.002)
AR_SPLIT = 3.0
SYNMIX = dict(enable=False, frac=0.15, n_per_class=1200)   # terbantah sbg init; MATI

# --- TTA: MATI, dan itu keputusan terukur ------------------------------------
# v13 mengadu tiga himpunan view dan yang menang adalah "tanpa TTA" (+0.0001,
# ambang +0.0015). Alasannya struktural: keempat view berada DI DALAM rentang
# augmentasi latih (ARJIT 0.65-1.55, letterbox 0.92-1.08), jadi model sudah
# pernah melihat semuanya. Menyalakannya lagi cuma membakar ~5 menit per fold
# untuk informasi nol. Satu view saja.
TTA_VIEWS = [(1.00, 1.00)]

# --- ANGKA ACUAN -------------------------------------------------------------
REF = dict(oof_v12=0.9735, oof_v13=0.9799,
           lb_v11=0.85030, lb_iw=0.85695, lb_v12=0.87749, lb_v13=0.89166,
           top5=0.88976, rank4=0.90299,
           sonde_dinov3_fold0=0.9858, sonde_siglip2_fold0=[0.9849, 0.9857, 0.9879])

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
seed_all()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T_START = time.time()

def el_h():  return (time.time() - T_START) / 3600.0
def rem_h(): return BUDGET["total_h"] - el_h()
def clk():   return f"[t={el_h():5.2f}j sisa={rem_h():5.2f}j]"

print(f"device={DEV} | torch={torch.__version__}"
      + ("  [SMOKE: bobot boleh ACAK]" if SMOKE else ""))
if DEV == "cuda":
    _p = torch.cuda.get_device_properties(0)
    print(f"GPU: {_p.name} | VRAM {_p.total_memory/2**30:.1f} GB | "
          f"bf16 didukung={torch.cuda.is_bf16_supported()}")
    if not torch.cuda.is_bf16_supported():
        print("  -> Turing (T4): autocast dipaku ke float16 + GradScaler. Benar.")
else:
    print("  !! TIDAK ADA GPU. Di Colab: Runtime > Change runtime type > T4 GPU.")
print(f"anggaran: {BUDGET['total_h']:.1f} jam, cadangan {BUDGET['reserve_h']:.2f} jam")

# =============================================================================
# 0b. LINGKUNGAN: Colab vs Kaggle vs lokal
# =============================================================================
# Yang membedakan Colab dari Kaggle bukan kuotanya, tapi PENYIMPANANNYA:
# /content dihapus begitu sesi putus. Kalau checkpoint ada di sana, run 6 jam
# yang mati di jam ke-5 kehilangan semuanya. Jadi di Colab kita pasang Drive
# dan menaruh _ck14/ + submission DI DRIVE. Jalankan ulang sel ini setelah
# putus dan ia melanjutkan dari fold terakhir yang selesai.
IS_KAGGLE = os.path.isdir("/kaggle/working")
IS_COLAB  = (not IS_KAGGLE) and ("google.colab" in sys.modules or os.path.isdir("/content"))
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_DIR  = os.path.join(DRIVE_ROOT, "infest")     # tempat kerja di Drive
DRIVE_DATA = os.path.join(DRIVE_DIR, "data")        # tempat data DIHARAPKAN ada

SEARCH_EXTRA, WORK = [], "."
if IS_KAGGLE:
    print("\nlingkungan: KAGGLE. /kaggle/working sudah persisten, Drive tidak dipakai.")
    WORK = "/kaggle/working"
elif IS_COLAB:
    print("\nlingkungan: COLAB.")
    try:
        from google.colab import drive as _gdrive
        if not os.path.isdir(DRIVE_ROOT):
            print("  memasang Google Drive (akan minta izin Anda)...")
            _gdrive.mount("/content/drive")
        else:
            print("  Google Drive sudah terpasang.")
    except Exception as e:
        print(f"  !! gagal memasang Drive: {type(e).__name__}: {e}")
    if os.path.isdir(DRIVE_ROOT):
        os.makedirs(DRIVE_DIR, exist_ok=True)
        WORK = DRIVE_DIR
        SEARCH_EXTRA = [DRIVE_DATA, DRIVE_DIR]
        print(f"  checkpoint & submission -> {WORK}  (TAHAN sesi putus)")
    else:
        WORK = "/content"
        print("  !! Drive TIDAK terpasang. Checkpoint akan ditaruh di /content,")
        print("     yang DIHAPUS saat sesi putus. Kalau sesi mati, semuanya hilang.")
        print("     Sangat disarankan memasang Drive sebelum melanjutkan.")
else:
    print("\nlingkungan: lokal / lainnya. Bekerja di direktori saat ini.")

CK = os.path.join(WORK, "_ck14"); os.makedirs(CK, exist_ok=True)
print(f"direktori kerja: {WORK}")
print(f"checkpoint     : {CK}")

# =============================================================================
# 1. LOAD / DISCOVER DATA   (identik v12/v13 - JANGAN diubah, fold bergantung
#    padanya. Yang ditambahkan cuma lokasi pencarian untuk Colab/Drive.)
# =============================================================================
SEARCH = [p for p in (SEARCH_EXTRA +
                      ["/kaggle/input", "/kaggle/working", ".", "./data", "/content"])
          if os.path.isdir(p)]
_seen_s = set(); SEARCH = [p for p in SEARCH if not (p in _seen_s or _seen_s.add(p))]
print("mencari data di:", ", ".join(SEARCH) if SEARCH else "(tidak ada direktori)")

def find_csv(names):
    hits = []
    for root in SEARCH:
        for dp, _, fs in os.walk(root, followlinks=True):
            for f in fs:
                if f.lower() in names: hits.append(os.path.join(dp, f))
    return hits[0] if hits else None

train_csv = find_csv({"train.csv"})
test_csv  = find_csv({"test.csv"})
sub_csv   = find_csv({"sample_submission.csv"})
print("train.csv :", train_csv)
print("sample_sub:", sub_csv)
if not train_csv:
    print("\n" + "=" * 70)
    print("BERHENTI: train.csv tidak ketemu di satu pun direktori di atas.")
    print("=" * 70)
    if IS_COLAB:
        print("  Di Colab data lomba TIDAK ikut otomatis. Dua cara memasukkannya:")
        print(f"  (a) Salin folder datanya ke Drive di {DRIVE_DATA}")
        print("      Isinya harus: train.csv, sample_submission.csv, folder train/,")
        print("      folder test/. Ini cara paling tahan putus - sekali salin,")
        print("      semua sesi berikutnya langsung menemukannya.")
        print("  (b) Pakai Kaggle API di sel TERPISAH sebelum sel ini:")
        print("        !pip -q install kaggle")
        print("        from google.colab import files; files.upload()   # kaggle.json")
        print("        !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
        print("        !kaggle competitions download -c infest-xii-data-science -p /content/data")
        print("        !unzip -q /content/data/*.zip -d /content/data")
        print("      Cara ini harus diulang tiap kali sesi Colab baru.")
    else:
        print("  Pastikan train.csv ada di salah satu direktori yang dicari di atas.")
    print("\n  Skrip berhenti di sini dengan sengaja: melatih data yang salah")
    print("  jauh lebih mahal daripada berhenti sekarang.")
    raise SystemExit(1)
tr_df = pd.read_csv(train_csv)

# PENTING: prediksi digerakkan dari sample_submission.csv, BUKAN test.csv.
# Di ruang kerja ini `test.csv` berisi 1.118 id test LAMA yang irisannya dgn
# sample_submission (1.220 id) adalah NOL.
if sub_csv:
    sub_tpl = pd.read_csv(sub_csv)
    te_df = pd.DataFrame({"image_id": sub_tpl.image_id.tolist()})
    src_test = "sample_submission.csv"
    if test_csv:
        _t = pd.read_csv(test_csv)
        if "image_id" in _t:
            _ov = len(set(_t.image_id) & set(te_df.image_id))
            print(f"cek test.csv vs sample_submission: {len(_t)} vs {len(te_df)} baris,"
                  f" irisan id = {_ov}")
            if _ov == 0:
                print("  !! test.csv TIDAK BERIRISAN -> test set LAMA, DIABAIKAN.")
elif test_csv:
    te_df = pd.read_csv(test_csv)[["image_id"]]; src_test = "test.csv"
    print("  PERHATIAN: sample_submission.csv tidak ketemu, memakai test.csv.")
else:
    raise AssertionError("sample_submission.csv maupun test.csv tidak ketemu")
print(f"daftar test diambil dari: {src_test} ({len(te_df)} baris)")

for root in list(SEARCH):
    for z in glob.glob(os.path.join(root, "*.zip")):
        tag = os.path.splitext(os.path.basename(z))[0].lower()
        if tag in ("train", "test", "images"):
            out = os.path.join(WORK, "_imgs")
            if not os.path.isdir(os.path.join(out, tag)):
                os.makedirs(out, exist_ok=True)
                try:
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(out, members=[m for m in zf.namelist()
                                                    if not m.startswith("__MACOSX")])
                    print(f"extracted {z} -> {out}")
                except Exception as e:
                    print(f"  ! gagal ekstrak {z}: {type(e).__name__}")
if os.path.isdir(os.path.join(WORK, "_imgs")): SEARCH.append(os.path.join(WORK, "_imgs"))

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
    c = IDX.get(b) or STEM.get(os.path.splitext(b)[0])
    if not c: return None
    if len(c) == 1: return c[0]
    pref = [p for p in c if f"{os.sep}{split}" in p.lower()]
    return (pref or c)[0]

tr_df["path"] = [resolve(x, "train") for x in tr_df.image_id]
te_df["path"] = [resolve(x, "test")  for x in te_df.image_id]
print(f"train rows={len(tr_df)} resolved={tr_df.path.notna().sum()} | "
      f"test rows={len(te_df)} resolved={te_df.path.notna().sum()}")
mt, me = tr_df[tr_df.path.isna()], te_df[te_df.path.isna()]
if len(me) or len(mt) == len(tr_df):
    print("\n!! GAGAL MENEMUKAN GAMBAR")
    print(f"   train tidak ketemu : {len(mt)}/{len(tr_df)}")
    print(f"   test  tidak ketemu : {len(me)}/{len(te_df)}")
    print(f"   folder dipindai    : {SEARCH}")
    raise AssertionError("Gambar tidak ketemu.")
tr_df = tr_df[tr_df.path.notna()].reset_index(drop=True)

def _md5(path):
    try:
        with open(path, "rb") as fh: return hashlib.md5(fh.read()).hexdigest()
    except Exception: return None
_htr = {}
for _p, _l in zip(tr_df.path, tr_df.label):
    _h = _md5(_p)
    if _h: _htr.setdefault(_h, []).append(_l)
LEAK = {}
for _i, _p in zip(te_df.image_id, te_df.path):
    if not isinstance(_p, str): continue
    _h = _md5(_p)
    if _h in _htr and len(set(_htr[_h])) == 1: LEAK[_i] = _htr[_h][0]
print(f"kebocoran train<->test terdeteksi: {len(LEAK)} gambar test labelnya sudah pasti")

# =============================================================================
# 2. EDA + DUA POPULASI + BOBOT PENILAIAN
# =============================================================================
print("\n" + "=" * 70 + "\nSTEP 1 - EDA & BOBOT POPULASI\n" + "=" * 70)
CLASSES = sorted(tr_df.label.unique()); C2I = {c: i for i, c in enumerate(CLASSES)}
tr_df["y"] = tr_df.label.map(C2I); NC = len(CLASSES)
cnt = np.bincount(tr_df.y, minlength=NC); ytrue = tr_df.y.values
print("kelas:", CLASSES)
for i, c in enumerate(CLASSES):
    print(f"  {c:10s}{cnt[i]:5d}  {100*cnt[i]/cnt.sum():5.2f}%")

def _meta(paths):
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
AR_TE, BIL_TE = _meta([p for p in te_df.path.tolist() if isinstance(p, str)])
BLOK_TR = AR_TR < AR_SPLIT
# `AR_TE < AR_SPLIT` meruntuhkan NaN jadi False lebih dulu, sehingga gambar yang
# gagal dibaca akan terhitung sebagai 'strip'. W_BLOK ada di balik SETIAP gerbang
# di skrip ini, jadi NaN dibuang eksplisit.
_ok_te = ~np.isnan(AR_TE)
if _ok_te.sum() == 0: raise AssertionError("tidak ada gambar test yang bisa dibaca")
W_BLOK = float((AR_TE[_ok_te] < AR_SPLIT).mean())
print(f"\npopulasi (AR<{AR_SPLIT} = 'blok'):")
print(f"  train: blok {BLOK_TR.mean()*100:5.1f}%  strip {100-BLOK_TR.mean()*100:5.1f}%")
print(f"  test : blok {W_BLOK*100:5.1f}%  strip {100-W_BLOK*100:5.1f}%   <- bobot penilaian")
print(f"  v13 mengukur F1 blok 0.9634 vs strip 0.9958 (v12: 0.9533 / 0.9932).")
print(f"  Blok tetap populasi yang rugi, dan ia hampir separuh test - di situlah")
print(f"  sisa kesalahan terkonsentrasi.")

def popf1(p, mask, detail=False):
    """macro-F1 ditimbang ke komposisi strip/blok TEST. INILAH satu-satunya
    angka yang memutuskan apa pun di skrip ini (lihat [3] di kepala berkas)."""
    out, parts = 0.0, {}
    for nm, sub, w in (("blok", BLOK_TR, W_BLOK), ("strip", ~BLOK_TR, 1 - W_BLOK)):
        m = mask & sub
        v = f1_score(ytrue[m], p[m].argmax(1), average="macro") if m.sum() >= NC else float("nan")
        parts[nm] = (v, int(m.sum()))
        if not np.isnan(v): out += w * v
    return (out, parts) if detail else out

skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=SEED)
FOLDS = list(skf.split(tr_df, tr_df.y))
print(f"\n{CFG['n_folds']} fold dibangun dgn random_state={SEED} atas {len(tr_df)} baris.")
print("  Ini SAMA PERSIS dgn v12 dan v13, jadi popf1 di bawah sebanding")
print("  baris-per-baris dgn angka acuan:")
print(f"    v12 ft-siglip2 @224 = {REF['oof_v12']:.4f} (LB {REF['lb_v12']:.5f})")
print(f"    v13 SigLIP2   @384 = {REF['oof_v13']:.4f} (LB {REF['lb_v13']:.5f})")

# --- audit pintasan: berapa banyak skor yang bisa didapat TANPA melihat aksara
# --- sama sekali. handoff mencatat 0.7826; diulang di sini sbg pengingat, dan
# --- karena ia gratis (CPU, beberapa detik).
try:
    _X = np.c_[np.nan_to_num(AR_TR, nan=0.0), np.nan_to_num(BIL_TR, nan=0.0)]
    _rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    _oo = np.zeros(len(tr_df), int)
    for _itr, _iva in FOLDS:
        _rf.fit(_X[_itr], ytrue[_itr]); _oo[_iva] = _rf.predict(_X[_iva])
    print(f"\n[S0] PINTASAN: RandomForest pada METADATA SAJA (AR + derajat 2-aras)")
    print(f"     macro-F1 OOF = {f1_score(ytrue, _oo, average='macro'):.4f}"
          f"  (tebakan acak {1/NC:.4f})")
    print(f"     Sebagian besar skor lomba ini memang bisa diraih tanpa membaca")
    print(f"     aksara sama sekali. Sisa yang BELUM tergarap ada di bentuk huruf,")
    print(f"     dan itulah yang resolusi 384 coba jangkau.")
except Exception as e:
    print(f"\n[S0] audit pintasan gagal ({type(e).__name__}) - diagnostik saja, run lanjut.")

# =============================================================================
# 3. CACHE GAMBAR (dekode sekali saja)
# =============================================================================
# Otsu/CANON DIBUANG TOTAL dari v13 - dua uji terkontrol independen mengukurnya
# merugikan (-0.0131 di DiT, -0.0091 di SigLIP2). Yang di-cache hanyalah
# operasi DETERMINISTIK: thumbnail -> abu-abu. Augmentasi tetap dihitung saat
# latih, jadi tidak ada kebocoran.
CDIR = os.path.join(WORK, "_cache13"); os.makedirs(CDIR, exist_ok=True)

def _cache_path(path):
    return os.path.join(CDIR, hashlib.md5(path.encode()).hexdigest() + ".png")

def load_img(path):
    if CACHE["enable"]:
        cp = _cache_path(path)
        if os.path.exists(cp):
            try: return Image.open(cp).convert("L")
            except Exception: pass
    try:
        im = Image.open(path); im.draft("L", (CACHE_MAX, CACHE_MAX))
        g = im.convert("L")
    except Exception:
        return Image.new("L", (64, 64), 255)
    if max(g.size) > CACHE_MAX:
        g.thumbnail((CACHE_MAX, CACHE_MAX), Image.BILINEAR)
    if CACHE["enable"]:
        try: g.save(_cache_path(path), "PNG", optimize=False)
        except Exception: pass
    return g

def warm_cache(paths, tag):
    if not CACHE["enable"]: return
    t0 = time.time(); n = 0
    for p in paths:
        if isinstance(p, str) and not os.path.exists(_cache_path(p)):
            load_img(p); n += 1
    if n: print(f"  cache {tag}: {n} gambar didekode dalam {time.time()-t0:.0f}s")

print("\nmenyiapkan cache gambar (sekali saja, dipakai semua fold & semua seed)...")
warm_cache(tr_df.path.tolist(), "train")
warm_cache([p for p in te_df.path.tolist() if isinstance(p, str)], "test")
print(f"{clk()} cache siap.")

# =============================================================================
# 4. VIEW + AUGMENTASI   (resep v6/v7/v12 - JANGAN diubah setelannya)
# =============================================================================
MEAN_DEF, STD_DEF = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def letterbox(g, H, W, train, stretch=1.0, scale=1.0):
    """Muat gambar utuh ke kanvas HxW dgn padding putih.

    `stretch` dan `scale` hanya dipakai saat train=False: keduanya adalah versi
    DETERMINISTIK dari dua augmentasi yang sudah dipakai saat latih -
    ARJIT (regangan sumbu-x 0.65-1.55) dan jitter skala letterbox (0.92-1.08).
    Itu sebabnya view TTA di v13 dijamin berada di dalam distribusi latih:
    ia bukan transformasi baru, melainkan titik tetap dari transformasi lama."""
    w, h = g.size
    f = random.uniform(ARJIT["lo"], ARJIT["hi"]) if (train and ARJIT["enable"]) else stretch
    if abs(f - 1.0) > 1e-6:
        w = max(1, int(round(w * f))); g = g.resize((w, h), Image.BILINEAR)
    s = min(H / max(h, 1), W / max(w, 1))
    s *= random.uniform(0.92, 1.08) if train else scale
    nw, nh = max(1, min(W, int(round(w*s)))), max(1, min(H, int(round(h*s))))
    g = g.resize((nw, nh), Image.BILINEAR)
    c = Image.new("L", (W, H), 255)
    ox = random.randint(0, W-nw) if train else (W-nw)//2
    oy = random.randint(0, H-nh) if train else (H-nh)//2
    c.paste(g, (ox, oy)); return c

class AddNoise(nn.Module):
    def __init__(s, p=0.3, sd=0.04): super().__init__(); s.p, s.sd = p, sd
    def forward(s, x):
        return (x + torch.randn_like(x)*s.sd).clamp(0, 1) if random.random() < s.p else x

AUG_GEO = T.Compose([
    T.RandomApply([T.RandomRotation(4, fill=255)], p=0.5),
    T.RandomApply([T.RandomAffine(0, translate=(0.02, 0.05), shear=4, fill=255)], p=0.3),
    T.ColorJitter(brightness=0.30, contrast=0.30)])
TO_T  = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
NOISE = AddNoise()        # posisi WAJIB: setelah ToTensor, sebelum Normalize
ERASE = T.RandomErasing(p=0.25, scale=(0.01, 0.06))

def degrade(g):
    """Domain randomization. Setelan v6, BUKAN v7: bukti leaderboard v6
    (blur 0.6-2.2) = 0.83902 vs v7 (blur 0.18-0.66, 'dikalibrasi' ke statistik
    test) = 0.83422. Kalibrasi ke tingkat korupsi test justru menurunkan skor."""
    d = DEGRADE
    if random.random() < d["p_blur"]:
        g = g.filter(ImageFilter.GaussianBlur(random.uniform(*d["blur"])))
    if random.random() < d["p_contrast"]:
        g = ImageEnhance.Contrast(g).enhance(random.uniform(*d["contrast"]))
    if random.random() < d["p_rescale"]:
        w, h = g.size; f = random.uniform(*d["rescale"])
        g = g.resize((max(8, int(w*f)), max(8, int(h*f))), Image.BILINEAR) \
             .resize((w, h), Image.BILINEAR)
    if random.random() < d["p_jpeg"]:
        b = io.BytesIO(); g.convert("L").save(b, "JPEG", quality=random.randint(*d["jpeg"]))
        b.seek(0); g = Image.open(b).convert("L")
    return g

def _chain(g):
    ops = [lambda x: x.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 2.0*SEV))),
           lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.35, 1.5)),
           lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.55, 1.45)),
           lambda x: ImageEnhance.Sharpness(x).enhance(random.uniform(0.0, 2.0)),
           lambda x: x.filter(ImageFilter.SMOOTH_MORE),
           lambda x: x.rotate(random.uniform(-4, 4), fillcolor=255)]
    for f in random.sample(ops, random.randint(1, 3)):
        try: g = f(g)
        except Exception: pass
    return g

def augmax(g):
    """AugMax/AugMix (Wang dkk., NeurIPS 2021)."""
    k = AUGMAX["chains"]
    w = np.random.dirichlet([1.0]*k); m = np.random.beta(1.0, 1.0)
    a = np.asarray(g, np.float32); mix = np.zeros_like(a)
    for i in range(k): mix += w[i] * np.asarray(_chain(g.copy()), np.float32)
    return Image.fromarray(np.clip((1-m)*a + m*mix, 0, 255).astype(np.uint8), "L")

class ScriptDS(Dataset):
    """r wajib punya: h, w, mean, std. `view` hanya dipakai saat train=False
    dan menunjuk ke TTA_VIEWS."""
    def __init__(s, df, r, train, view=0, syn=None):
        s.p = df.path.tolist(); s.y = df.y.tolist() if "y" in df else [0]*len(df)
        s.r, s.t, s.v = r, train, view
        s.syn = syn or []          # [(PIL, y)] korpus sintetis, hanya kalau SYNMIX
        s.norm = T.Normalize(r.get("mean", MEAN_DEF), r.get("std", STD_DEF))
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        if s.t and s.syn and random.random() < SYNMIX["frac"]:
            j = random.randrange(len(s.syn)); g, y = s.syn[j][0].copy(), s.syn[j][1]
        else:
            g = load_img(s.p[i]) if isinstance(s.p[i], str) else Image.new("L", (64, 64), 255)
            y = s.y[i]
        if s.t:
            g = augmax(g) if (AUGMAX["enable"] and random.random() < AUGMAX["p"]) else degrade(g)
            g = letterbox(g, s.r["h"], s.r["w"], True)
        else:
            st, sc = TTA_VIEWS[s.v]
            g = letterbox(g, s.r["h"], s.r["w"], False, st, sc)
        if s.t: g = AUG_GEO(g)
        x = TO_T(g)
        if x.shape[0] == 1: x = x.repeat(3, 1, 1)
        if s.t: x = NOISE(x)
        x = s.norm(x)
        if s.t: x = ERASE(x)
        return x, y

# =============================================================================
# 5. BACKBONE: timm saja, TANPA substitusi senyap
# =============================================================================
# v10 `build()` diam-diam jatuh ke resnet34 kalau nama backbone salah, sehingga
# run yang seolah 'dinov2' sebenarnya resnet34. Di sini kegagalan DILAPORKAN
# dan skrip berhenti, karena di v13 hanya ADA satu backbone: kalau ia gagal,
# tidak ada yang bisa dikerjakan dan lebih baik tahu sekarang.
class Net(nn.Module):
    def __init__(s, body, feat, nc):
        super().__init__(); s.body = body; s.head = nn.Linear(feat, nc); s.feat = feat
    def forward(s, x): return s.head(s.body(x))

_LOADED = {}
def load_body(name, h, w, fresh=True):
    """kembalikan (body, dim, mean, std). Kunci cache MEMUAT resolusi: ViT dgn
    position embedding absolut adalah model LAIN kalau img_size-nya lain.
    fresh=True -> SALINAN. Modul bersama membuat fold 1 mulai dari bobot yang
    sudah dirusak fold 0 (gejala: fold pertama sehat, sisanya kolaps)."""
    key = (name, h, w)
    if key not in _LOADED:
        import timm
        def _mk(pre):
            try:
                return timm.create_model(name, pretrained=pre, num_classes=0, img_size=(h, w))
            except TypeError:
                return timm.create_model(name, pretrained=pre, num_classes=0)
        try:
            m = _mk(True)
        except Exception as e:
            if not SMOKE:
                print(f"  !! GAGAL MEMUAT {name}: {type(e).__name__}: {e}")
                raise
            print(f"    (SMOKE) unduhan gagal {type(e).__name__} -> bobot ACAK")
            m = _mk(False)
        mean, std = MEAN_DEF, STD_DEF
        try:
            from timm.data import resolve_data_config
            dc = resolve_data_config({}, model=m)
            mean, std = list(dc.get("mean", mean)), list(dc.get("std", std))
        except Exception: pass
        _LOADED[key] = (m, int(m.num_features), mean, std)
    body, dim, mean, std = _LOADED[key]
    return (copy.deepcopy(body) if fresh else body), dim, mean, std

# =============================================================================
# 6. OPTIMISASI: layer-wise lr decay yang BENAR-BENAR aktif
# =============================================================================
CW = torch.tensor(cnt.sum() / (NC * cnt), dtype=torch.float32, device=DEV)
_SEEN_DECAY = {}
def _depth_of(name, nmax):
    parts = name.split(".")
    for k, tok in enumerate(parts):
        if tok in ("embeddings", "patch_embed", "stem", "cls_token", "pos_embed"): return 0
        if tok in ("layer", "layers", "block", "blocks", "stage", "stages"):
            for q in parts[k+1:k+3]:
                if q.isdigit(): return 1 + int(q)
    return nmax

def param_groups(model, lr, wd, decay):
    """DUA no-op senyap yang diperbaiki di sini:
    [1] `timm.optim.param_groups_layer_decay` mengembalikan grup ber-kunci
        `lr_scale`, dan AdamW polos TIDAK MEMBACA kunci itu; OneCycleLR lalu
        menimpa `lr` tiap grup. Jadi di v10 layer-wise decay tidak pernah aktif.
        Di sini skalanya dikalikan ke `lr` secara eksplisit.
    [2] Pengelompokan dihitung dari NAMA parameter dan hasilnya DICETAK supaya
        bisa diperiksa, bukan dipercaya."""
    ps = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if decay >= 1.0 or not ps:
        return [{"params": [p for _, p in ps], "weight_decay": wd, "lr": lr}]
    nmax = 0
    for n, _ in ps:
        parts = n.split(".")
        for k, tok in enumerate(parts):
            if tok in ("layer", "layers", "block", "blocks", "stage", "stages"):
                for q in parts[k+1:k+3]:
                    if q.isdigit(): nmax = max(nmax, 1 + int(q))
    nmax += 1
    buckets = {}
    for n, p in ps:
        d  = min(_depth_of(n, nmax), nmax)
        nd = (p.ndim <= 1 or n.endswith(".bias"))
        buckets.setdefault((d, nd), []).append(p)
    return [{"params": prm, "weight_decay": (0.0 if nd else wd),
             "lr": lr * (decay ** (nmax - d))}
            for (d, nd), prm in sorted(buckets.items())]

def report_decay(model, lr, decay, tag):
    gs = param_groups(model, lr, CFG["wd"], decay)
    sc = sorted({round(g["lr"] / max(lr, 1e-12), 5) for g in gs})
    print(f"    layer-decay {decay}: {len(gs)} grup, {len(sc)} skala berbeda "
          f"({sc[0]:.4g} .. {sc[-1]:.4g})")
    if decay < 1.0 and len(sc) <= 1:
        print(f"    ! PERHATIAN: layer-decay TIDAK AKTIF untuk {tag} -> sama dgn 1.0.")

class EMA:
    """Polyak averaging - bobot rata-rata jauh lebih stabil drpd epoch terakhir."""
    def __init__(s, model, decay=0.999):
        s.decay = decay
        s.sh = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    def update(s, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                s.sh[k].mul_(s.decay).add_(v.detach().float(), alpha=1 - s.decay)
            else: s.sh[k] = v.detach().clone().float()
    def state(s, model):
        ref = model.state_dict()
        return {k: v.to(ref[k].dtype) for k, v in s.sh.items()}

def is_collapsed(pred, f1, nc):
    """Kolaps = prediksi menumpuk di ~1 kelas. Dideteksi lewat KONSENTRASI,
    bukan ambang F1 absolut, supaya model yang sekadar LEMAH tidak ikut dibuang."""
    share = np.bincount(pred, minlength=nc).max() / max(len(pred), 1)
    return bool(share > 0.90 or f1 < (0.10 if nc > 2 else 0.35))

@torch.no_grad()
def predict_views(model, df, r, views, bs):
    """kembalikan array (len(views), N, nc) berisi softmax per view.
    Disimpan TERPISAH per view, bukan langsung dirata-ratakan, supaya himpunan
    bagian TTA bisa diadu di OOF tanpa satu pun lintasan maju tambahan."""
    model.eval(); out = []
    for v in views:
        acc = []
        for x, _ in DataLoader(ScriptDS(df, r, False, v), batch_size=bs,
                               shuffle=False, num_workers=CFG["nw"]):
            x = x.to(DEV, non_blocking=True)
            with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                acc.append(torch.softmax(model(x).float(), 1).cpu().numpy())
        out.append(np.concatenate(acc))
    return np.stack(out)

# =============================================================================
# 7. PELATIHAN SATU FOLD  (+ akumulasi gradien + guard OOM + guard kolaps)
# =============================================================================
OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
def _is_oom(e):
    return isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ()))  \
           or ("out of memory" in str(e).lower())

def train_fold(r, dtr, dva, epochs, lr_body, lr_head, seed, decay, nc,
               micro, accum, syn=None, tag="?"):
    """Satu fold. Mengembalikan (model, f1_val, pred_val, r_terpakai).
    r_terpakai sudah memuat mean/std backbone sehingga pemanggil tidak perlu
    memuat ulang bobotnya.

    Pemilihan checkpoint per epoch memakai view 0 SAJA (bukan seluruh TTA).
    Alasannya biaya: dgn 4 view, validasi raw+EMA tiap epoch akan menelan
    ~40% waktu fold di resolusi 384. View 0 adalah view test v12 apa adanya,
    jadi ia sinyal relatif yang sah untuk MEMILIH epoch; TTA penuh baru
    dihitung SEKALI di akhir, pada checkpoint yang sudah terpilih."""
    seed_all(seed)
    body, dim, mean, std = load_body(r["model"], r["h"], r["w"], fresh=True)
    r = dict(r); r["mean"], r["std"] = mean, std
    model = Net(body, dim, nc)
    if r.get("grad_ckpt"):
        try:
            model.body.set_grad_checkpointing(True)
            print("    gradient checkpointing DINYALAKAN (hemat VRAM, ~30% lebih lambat)")
        except Exception:
            print("    ! backbone tidak mendukung set_grad_checkpointing")
    model = model.to(DEV)
    ld = DataLoader(ScriptDS(dtr, r, True, syn=syn), batch_size=micro, shuffle=True,
                    num_workers=CFG["nw"], drop_last=(len(dtr) >= 2*micro),
                    pin_memory=(DEV == "cuda"))
    cw = CW if nc == NC else None          # spesialis biner punya prior sendiri
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=CFG["ls"])
    pg = param_groups(model.body, lr_body, CFG["wd"], decay)
    pg = pg + [{"params": list(model.head.parameters()), "weight_decay": 0.0, "lr": lr_head}]
    opt = torch.optim.AdamW(pg, lr=lr_body, weight_decay=CFG["wd"])
    if _SEEN_DECAY.get(tag) is None:
        _SEEN_DECAY[tag] = True; report_decay(model.body, lr_body, decay, tag)
    spe   = max(1, math.ceil(len(ld) / accum))
    steps = max(10, epochs * spe)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g.get("lr", lr_body)*2 for g in opt.param_groups],
        total_steps=steps, pct_start=0.25)
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
    ema = EMA(model, CFG["ema"]); best, best_pred, best_state = -1, None, None
    ibs = max(4, micro * 2)                # batch inferensi
    for ep in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        for bi, (x, y) in enumerate(ld):
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                loss = crit(model(x), y) / accum
            scaler.scale(loss).backward()
            if (bi + 1) % accum == 0 or (bi + 1) == len(ld):
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if sch.last_epoch < steps - 1: sch.step()
                ema.update(model)
        # Gerbang checkpoint `ep >= epochs//3` diwarisi dari v6/v7: ia menghemat
        # sepertiga lintasan validasi DAN mencegah checkpoint epoch warm-up
        # terpilih. Sempat hilang di draf v11.
        if ep < epochs // 3: continue
        bak = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pr = predict_views(model, dva, r, [0], ibs)[0].argmax(1)
        f_raw = f1_score(dva.y, pr, average="macro")
        model.load_state_dict(ema.state(model))
        pe = predict_views(model, dva, r, [0], ibs)[0].argmax(1)
        f_ema = f1_score(dva.y, pe, average="macro")
        if f_ema < f_raw: model.load_state_dict(bak)
        f1 = max(f_raw, f_ema)
        if f1 > best:
            best, best_pred = f1, (pe if f_ema >= f_raw else pr)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(bak)
    if best_state is not None: model.load_state_dict(best_state)
    return model, best, best_pred, r


# =============================================================================
# 8. GUBERNUR ANGGARAN: ukur dulu, baru putuskan berapa anggota yang muat
# =============================================================================
# Kecepatan T4 bervariasi (T4 vs T4 x2, 2 vs 4 inti CPU untuk dataloader).
# Salah tebak 2x berarti entah membuang jam kuota atau terpotong di tengah.
# Tolok ukurnya puluhan detik - harga yang sangat murah.
_BENCH = {}
def bench_member(m):
    """detik per GAMBAR per langkah latih, micro-batch yang MUAT, perlu grad ckpt."""
    key = (m["model"], m["h"])
    if key in _BENCH: return _BENCH[key]
    r = dict(model=m["model"], h=m["h"], w=m["w"])
    micro, gck = CFG["batch"], False
    for attempt in range(4):
        try:
            body, dim, mean, std = load_body(r["model"], r["h"], r["w"], fresh=True)
            rr = dict(r); rr["mean"], rr["std"] = mean, std
            mm = Net(body, dim, NC)
            if gck:
                try: mm.body.set_grad_checkpointing(True)
                except Exception: pass
            mm = mm.to(DEV).train()
            opt = torch.optim.AdamW(mm.parameters(), lr=1e-6)
            sc = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
            ld = DataLoader(ScriptDS(tr_df.head(micro*BUDGET["bench_steps"]+micro), rr, True),
                            batch_size=micro, shuffle=False, num_workers=CFG["nw"])
            crit = nn.CrossEntropyLoss(weight=CW)
            it, n, t0 = iter(ld), 0, None
            for i in range(BUDGET["bench_steps"] + 3):
                try: x, y = next(it)
                except StopIteration: break
                x, y = x.to(DEV), y.to(DEV)
                with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                    loss = crit(mm(x), y)
                sc.scale(loss).backward(); sc.step(opt); sc.update()
                opt.zero_grad(set_to_none=True)
                if i == 2:
                    if DEV == "cuda": torch.cuda.synchronize()
                    t0 = time.time(); n = 0
                elif t0 is not None: n += len(y)
            if DEV == "cuda": torch.cuda.synchronize()
            dt = max(time.time() - t0, 1e-6) if t0 else 1e-6
            out = dict(sec_img=dt/max(n, 1), micro=micro, grad_ckpt=gck)
            del mm, opt, sc, ld
            if DEV == "cuda": torch.cuda.empty_cache()
            _BENCH[key] = out
            print(f"    tolok ukur {m['name']:14s} @{m['h']}: {n/dt:5.1f} gambar/detik latih,"
                  f" micro={micro}" + (" +grad_ckpt" if gck else ""))
            return out
        except Exception as e:
            if not _is_oom(e):
                print(f"    !! tolok ukur {m['name']} gagal: {type(e).__name__}: {e}"); raise
            if DEV == "cuda": torch.cuda.empty_cache()
            if micro > 4: micro //= 2
            elif not gck: gck = True
            else: raise
            print(f"    OOM -> coba lagi dgn micro={micro}" + (" +grad_ckpt" if gck else ""))
    raise RuntimeError("tidak ada konfigurasi memori yang muat")

def project_fold(m, b, n_tr, n_va, n_te, nviews):
    """proyeksi detik untuk SATU fold, dari kecepatan terukur. Inferensi
    diperkirakan ~1/3 biaya langkah latih (hanya maju, tanpa backward)."""
    si = b["sec_img"]; inf = si / 3.0; ep = m["epochs"]
    return (ep * n_tr * si
            + (ep - ep//3) * 2 * n_va * inf          # validasi raw + EMA
            + (n_va + n_te) * nviews * inf)          # prediksi akhir

def project_member(m, b, nviews=len(TTA_VIEWS)):
    n = len(tr_df); n_tr = int(n*(1-1/CFG["n_folds"])); n_va = n - n_tr
    return CFG["n_folds"] * project_fold(m, b, n_tr, n_va, len(te_df), nviews)

print("\n" + "=" * 70 + "\nSTEP 2 - GUBERNUR ANGGARAN\n" + "=" * 70)
BENCH, PLANNED = {}, []
_need_total = 0.0
for m in MEMBERS:
    try:
        b = bench_member(m)
    except Exception as e:
        print(f"  {m['name']}: tidak bisa diukur ({type(e).__name__}) -> DILEWATI")
        continue
    BENCH[m["mid"]] = b
    need = project_member(m, b) / 3600.0
    _need_total += need
    print(f"  {m['name']:14s} @{m['h']} x {m['epochs']}ep -> {need:5.2f} jam"
          f" untuk {CFG['n_folds']} fold")
    print(f"                 {m['note']}")
    PLANNED.append(m)

if not PLANNED:
    print("\n  !! tidak ada anggota yang bisa dimuat sama sekali. Berhenti.")
    raise SystemExit(1)

_safe = rem_h() - BUDGET["reserve_h"]
print(f"\n  total perlu {_need_total*BUDGET['safety']:.2f} jam (sudah x{BUDGET['safety']}),"
      f" sisa aman {_safe:.2f} jam")
if _need_total * BUDGET["safety"] > _safe:
    print("  Tidak semuanya muat. Gubernur akan memangkas fold pada anggota")
    print("  BELAKANGAN saat berjalan, bukan sekarang - checkpoint yang sudah ada")
    print("  dari sesi sebelumnya bisa membuat sisanya muat.")
    print("  Anggota 0 (resep v13 yang terbukti) dilatih LEBIH DULU, jadi kalau")
    print("  sesi mati di tengah, yang tertinggal tetap submission setara v13.")
else:
    print("  MUAT. Kedua anggota akan dilatih penuh.")
MICRO0 = {m["mid"]: BENCH[m["mid"]]["micro"] for m in PLANNED}
print(f"\n  batch efektif per anggota: "
      + ", ".join(f"{m['name']}={MICRO0[m['mid']]}x"
                  f"{max(1, int(round(CFG['batch']*CFG['accum']/MICRO0[m['mid']])))}"
                  for m in PLANNED) + "  (v13 memakai 32)")

# =============================================================================
# 9. MENJALANKAN SATU ANGGOTA (5 fold) - dgn checkpoint & resume
# =============================================================================
NV     = len(TTA_VIEWS)
N_TR   = len(tr_df); N_TE = len(te_df)
OOF_M, TE_M, COVER, FOLDF1 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

def write_sub(name, labs):
    """cek DULU, tulis kemudian: berkas dgn jumlah baris salah tidak boleh ada."""
    assert len(labs) == N_TE, f"{name}: {len(labs)} label vs {N_TE} baris test"
    assert all(l in C2I for l in labs), f"{name}: ada label di luar 7 kelas sah"
    p = os.path.join(WORK, name)
    with open(p, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["image_id", "label"])
        for iid, l in zip(te_df.image_id, labs): wr.writerow([iid, l])
    return p

def labels_from(prob):
    labs = [CLASSES[i] for i in prob.argmax(1)]
    for k, iid in enumerate(te_df.image_id):        # kebocoran byte-identik
        if iid in LEAK: labs[k] = LEAK[iid]
    return labs

def flush_submission():
    """Ditulis ULANG setiap kali satu fold selesai, ke WORK (di Colab itu
    Drive). Kalau sesi Colab putus di jam ke-5, yang tertinggal tetap berkas
    submission yang SAH dari fold yang sudah jadi - bukan tidak ada apa-apa.
    Versi ini memakai rata-rata polos anggota yang ada; gerbang campur baru
    dijalankan di akhir dan hasilnya akan MENIMPA berkas ini."""
    have = [q for q in TE_M if TE_M[q] is not None]
    if not have: return
    p = np.mean([TE_M[q][0] for q in have], 0)
    try:
        write_sub("submission.csv", labels_from(p))
    except Exception as e:
        print(f"    ! gagal menulis submission sementara: {type(e).__name__}: {e}")

def _ckpath(mid, fold): return os.path.join(CK, f"m{mid}_f{fold}.npz")

def run_member(m):
    """Satu anggota penuh atas 5 fold. True kalau setidaknya satu fold jadi."""
    mid = m["mid"]
    b   = BENCH[mid]
    micro = b["micro"]
    accum = max(1, int(round(CFG["batch"] * CFG["accum"] / micro)))
    r_mem = dict(model=m["model"], h=m["h"], w=m["w"], grad_ckpt=b["grad_ckpt"])
    oof = np.zeros((NV, N_TR, NC)); tep = np.zeros((NV, N_TE, NC))
    cov = np.zeros(N_TR, bool); f1s, used, dead = [], 0, 0
    print(f"\n  -- anggota {mid} : {m['name']} ({m['model'].split('.')[0]}) "
          f"@{m['h']} x {m['epochs']}ep --")
    for fold, (itr, iva) in enumerate(FOLDS):
        cp = _ckpath(mid, fold)
        if os.path.exists(cp):                       # RESUME
            try:
                z = np.load(cp)
                oof[:, iva] = z["va"]; tep += z["te"]; cov[iva] = True
                f1s.append(float(z["f1"])); used += 1
                print(f"    fold{fold} dimuat dari checkpoint (F1={float(z['f1']):.4f})")
                continue
            except Exception as e:
                print(f"    fold{fold} checkpoint tidak terbaca "
                      f"({type(e).__name__}: {e}) -> dilatih ulang")
        est = project_fold(m, b, len(itr), len(iva), N_TE, NV) / 3600.0
        if est * BUDGET["safety"] > rem_h() - BUDGET["reserve_h"]:
            print(f"    {clk()} fold{fold} DILEWATI: perlu ~{est:.2f} jam, "
                  f"sisa aman {rem_h()-BUDGET['reserve_h']:.2f} jam.")
            print(f"    Anggota {mid} berhenti di {used} fold. Ini keputusan gubernur,")
            print(f"    bukan kegagalan - fold yang sudah jadi tetap dipakai, dan")
            print(f"    menjalankan ulang sel ini akan melanjutkan dari sini.")
            break
        dtr = tr_df.iloc[itr].reset_index(drop=True)
        dva = tr_df.iloc[iva].reset_index(drop=True)
        lb, lh = FT["lr_body"], FT["lr_head"]        # di-RESET tiap fold
        model, f1, vp, rr, bad = None, -1.0, None, None, True
        for attempt in range(3):                      # guard KOLAPS: ulang dgn lr/3
            try:
                model, f1, vp, rr = train_fold(
                    r_mem, dtr, dva, m["epochs"], lb, lh,
                    SEED + mid*1000 + fold*10 + attempt, FT["layer_decay"], NC,
                    micro, accum, syn=None, tag=f"m{mid}")
            except Exception as e:
                if not _is_oom(e): raise
                if DEV == "cuda": torch.cuda.empty_cache()
                if micro > 4:
                    micro //= 2
                    accum = max(1, int(round(CFG["batch"]*CFG["accum"] / micro)))
                else:
                    r_mem["grad_ckpt"] = True
                print(f"    OOM di fold{fold} -> micro={micro}, accum={accum}"
                      + (" +grad_ckpt" if r_mem.get("grad_ckpt") else "") + ", ulangi fold")
                continue
            bad = is_collapsed(vp, f1, NC)
            if not bad: break
            lb /= 3; lh /= 3
            sh = np.bincount(vp, minlength=NC).max()/len(vp)
            print(f"    fold{fold} KOLAPS (F1={f1:.4f}, {sh*100:.0f}% satu kelas)"
                  f" -> ulang dgn lr={lb:.1e}")
        if model is None or bad:
            dead += 1; print(f"    fold{fold} tetap kolaps -> DIKELUARKAN")
            del model
            if DEV == "cuda": torch.cuda.empty_cache()
            continue
        ibs = max(4, micro * 2)
        va = predict_views(model, dva, rr, list(range(NV)), ibs)
        te = predict_views(model, te_df, rr, list(range(NV)), ibs)
        oof[:, iva] = va; tep += te; cov[iva] = True; used += 1; f1s.append(f1)
        try:
            np.savez_compressed(cp, va=va.astype(np.float32),
                                te=te.astype(np.float32), f1=np.float32(f1))
        except Exception as e:
            print(f"    ! gagal menyimpan checkpoint fold{fold}: "
                  f"{type(e).__name__}: {e}")
            print(f"      (di Colab ini berarti fold ini HILANG kalau sesi putus)")
        print(f"    {clk()} fold{fold} macro-F1={f1:.4f}")
        del model
        if DEV == "cuda": torch.cuda.empty_cache()
        TE_M[mid] = tep / max(used, 1); flush_submission()
    if used == 0:
        print(f"    !! anggota {mid} tidak menghasilkan satu fold pun")
        TE_M.pop(mid, None); return False
    OOF_M[mid] = oof; TE_M[mid] = tep / used; COVER[mid] = cov
    FOLDF1[mid] = (float(np.mean(f1s)), float(np.std(f1s)), dead, used)
    print(f"    anggota {mid} selesai: {used}/{CFG['n_folds']} fold, "
          f"F1 fold rata2 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}, fold mati {dead}")
    flush_submission()
    return True

print("\n" + "=" * 70 + "\nSTEP 3 - LATIH KEDUA ANGGOTA\n" + "=" * 70)
print("Anggota 0 (resep v13) lebih dulu, supaya sesi yang mati di tengah tetap")
print("meninggalkan submission setara v13. Anggota 1 adalah taruhan barunya.")
if os.path.isdir(CK) and os.listdir(CK):
    print(f"\n  checkpoint yang sudah ada di {CK}: "
          + ", ".join(sorted(os.listdir(CK))[:12]))
    print("  fold yang sudah ada TIDAK akan dilatih ulang.")
for m in PLANNED:
    run_member(m)

if not OOF_M:
    print("\n  !! tidak ada anggota yang berhasil. Tidak ada yang bisa dirakit.")
    raise SystemExit(1)

# =============================================================================
# 10. PERAKITAN: gerbang campur, lalu gerbang bobot kelas
# =============================================================================
# Urutannya sengaja: anggota tunggal dulu (supaya ada garis dasar yang
# sebanding langsung dgn v13), baru campur, baru campur per-populasi, baru
# bobot kelas. Tiap tahap mencetak angka SEBELUM dan SESUDAH, jadi kalau
# hasilnya merugikan di leaderboard nanti, sumbernya bisa ditunjuk.
print("\n" + "=" * 70 + "\nSTEP 4 - PERAKITAN & GERBANG\n" + "=" * 70)

MIDS = [q for q in OOF_M]
MASK = np.ones(N_TR, bool)
for q in MIDS: MASK &= COVER[q]
print(f"baris OOF yang diliput SEMUA anggota: {int(MASK.sum())} dari {N_TR}")
if MASK.sum() < 0.5 * N_TR:
    print("  !! liputan bersama terlalu sedikit untuk menggerbang campur dgn jujur.")

P = {q: OOF_M[q][0] for q in MIDS}          # OOF view 0 per anggota
T = {q: TE_M[q][0]  for q in MIDS}          # test view 0 per anggota

# --- [M] anggota tunggal ------------------------------------------------------
print("\n[M] Anggota tunggal, dinilai di baris yang SAMA:")
solo = {}
for q in MIDS:
    solo[q] = popf1(P[q], MASK)
    nm = next(m["name"] for m in MEMBERS if m["mid"] == q)
    print(f"    anggota {q} {nm:14s} -> {solo[q]:.4f}")
BEST_M = max(solo, key=solo.get)
BEST_V = solo[BEST_M]
print(f"    terbaik: anggota {BEST_M} ({BEST_V:.4f})")

# --- [R] perbandingan langsung dgn v13 dan v12 -------------------------------
print(f"\n[R] PERBANDINGAN LANGSUNG (fold identik, baris identik):")
print(f"    v12 ft-siglip2 @224          : {REF['oof_v12']:.4f}  -> LB {REF['lb_v12']:.5f}")
print(f"    v13 SigLIP2 @384 tunggal     : {REF['oof_v13']:.4f}  -> LB {REF['lb_v13']:.5f}")
if 0 in solo:
    d13 = solo[0] - REF["oof_v13"]
    print(f"    v14 anggota 0 (resep sama)   : {solo[0]:.4f}  ({d13:+.4f} vs v13)")
    if abs(d13) > 0.004:
        print("    !! anggota 0 memakai resep yang SAMA PERSIS dgn v13, jadi ia")
        print("       seharusnya mendarat di 0.9799 +/- derau seed (~0.0015).")
        print("       Selisih sebesar ini berarti ADA YANG BERBEDA di pipeline")
        print("       ini - data, fold, atau view. Periksa sebelum percaya")
        print("       SATU PUN angka di bawah.")
    else:
        print("    -> anggota 0 mereproduksi v13. Angka di bawah bisa dipercaya.")
if 1 in solo:
    print(f"    sonde DINOv3 fold0 dulu      : {REF['sonde_dinov3_fold0']:.4f}"
          f"  (SigLIP2 fold0 rata2 "
          f"{np.mean(REF['sonde_siglip2_fold0']):.4f})")

# --- [B] gerbang campur -------------------------------------------------------
W, BLEND_OK = 0.0, False
OOF_B, TE_B = P[BEST_M], T[BEST_M]
if len(MIDS) >= 2:
    a, b_ = MIDS[0], MIDS[1]
    print(f"\n[B] Campur anggota {a} dan {b_} (ambang {BLEND['gate']:+.4f} di atas"
          f" tunggal terbaik {BEST_V:.4f}):")
    print(f"    prob = (1-w)*anggota{a} + w*anggota{b_}")
    best_w, best_v = None, BEST_V
    for w in BLEND["grid"]:
        v = popf1((1-w)*P[a] + w*P[b_], MASK)
        mark = ""
        if v > best_v: best_w, best_v, mark = w, v, "  <-"
        print(f"    w={w:4.2f} -> {v:.4f} ({v-BEST_V:+.4f}){mark}")
    if best_w is not None and best_v > BEST_V + BLEND["gate"]:
        W, BLEND_OK = best_w, True
        OOF_B = (1-W)*P[a] + W*P[b_]
        TE_B  = (1-W)*T[a] + W*T[b_]
        print(f"    -> PAKAI w={W:.2f}, OOF {BEST_V:.4f} -> {best_v:.4f}"
              f" ({best_v-BEST_V:+.4f})")
        print(f"    Ini kenaikan dari KEBERAGAMAN arsitektur, bukan dari menambah")
        print(f"    kapasitas: kedua anggota setara sendirian.")
    else:
        gain = (best_v - BEST_V) if best_w is not None else 0.0
        print(f"    -> TOLAK (terbaik {gain:+.4f}, di bawah ambang derau).")
        print(f"    Dipakai anggota tunggal {BEST_M} saja. Ini hasil yang SAH:")
        print(f"    gerbang ini ada supaya anggota yang tidak membantu TIDAK ikut,")
        print(f"    dan itulah yang menjatuhkan submission_iw.csv ({REF['lb_iw']:.5f}).")
else:
    print("\n[B] Cuma satu anggota yang jadi -> tidak ada yang dicampur.")

# --- [B2] campur terpisah per populasi ---------------------------------------
W_BLOK_MIX, W_STRIP_MIX, PERPOP_OK = W, W, False
if BLEND["per_pop"] and len(MIDS) >= 2:
    a, b_ = MIDS[0], MIDS[1]
    base_pp = popf1(OOF_B, MASK)
    print(f"\n[B2] Campur TERPISAH blok vs strip (ambang {BLEND['per_pop_gate']:+.4f}"
          f" di atas {base_pp:.4f}):")
    print(f"     Dasarnya: blok 49,2% test tapi F1-nya jauh lebih rendah, jadi")
    print(f"     bobot optimalnya BOLEH berbeda. Ini 1 derajat kebebasan tambahan")
    print(f"     atas {int(MASK.sum())} baris, jadi ambangnya sengaja lebih tinggi.")
    def _mix_pp(wb, ws, PA, PB):
        out = np.empty_like(PA)
        out[BLOK_TR]  = (1-wb)*PA[BLOK_TR]  + wb*PB[BLOK_TR]
        out[~BLOK_TR] = (1-ws)*PA[~BLOK_TR] + ws*PB[~BLOK_TR]
        return out
    bwb, bws, bvv = W, W, base_pp
    for wb in BLEND["grid"]:
        for ws in BLEND["grid"]:
            v = popf1(_mix_pp(wb, ws, P[a], P[b_]), MASK)
            if v > bvv: bwb, bws, bvv = wb, ws, v
    print(f"     terbaik: w_blok={bwb:.2f}, w_strip={bws:.2f} -> {bvv:.4f}"
          f" ({bvv-base_pp:+.4f})")
    if bvv > base_pp + BLEND["per_pop_gate"]:
        W_BLOK_MIX, W_STRIP_MIX, PERPOP_OK = bwb, bws, True
        OOF_B = _mix_pp(bwb, bws, P[a], P[b_])
        # Untuk test, populasi ditentukan dari AR_TE. AR_TE dihitung hanya atas
        # baris yang path-nya string, jadi ia disejajarkan ulang ke N_TE dulu -
        # salah sejajar di sini berarti separuh test dicampur dgn bobot yang
        # salah, dan tidak ada satu pun angka yang akan menunjukkannya.
        _ar_full = np.full(N_TE, np.nan)
        _k = 0
        for _i, _p in enumerate(te_df.path.tolist()):
            if isinstance(_p, str) and _k < len(AR_TE):
                _ar_full[_i] = AR_TE[_k]; _k += 1
        _nan_te = int(np.isnan(_ar_full).sum())
        if _nan_te:
            print(f"     ! {_nan_te} gambar test tidak terbaca rasio aspeknya"
                  f" -> diperlakukan sbg strip.")
        _blok_te = np.nan_to_num(_ar_full, nan=AR_SPLIT + 1.0) < AR_SPLIT
        TE_B = np.empty_like(T[a])
        TE_B[_blok_te]  = (1-bwb)*T[a][_blok_te]  + bwb*T[b_][_blok_te]
        TE_B[~_blok_te] = (1-bws)*T[a][~_blok_te] + bws*T[b_][~_blok_te]
        print(f"     -> PAKAI. Test dipisah dgn ambang AR yang SAMA (AR<{AR_SPLIT}),"
              f" {int(_blok_te.sum())} blok / {int((~_blok_te).sum())} strip.")
    else:
        print(f"     -> TOLAK (di bawah ambang). Tetap pakai campur tunggal"
              f" w={W:.2f}.")

# --- [W] bobot per kelas ------------------------------------------------------
# macro-F1 TIDAK dioptimalkan oleh argmax probabilitas: kelas kecil butuh
# dorongan. Ditala HANYA di OOF, dipakai hanya kalau lewat ambang derau.
print(f"\n[W] Bobot per-kelas (ambang {GATE['cw']:+.4f}):")
CW_OPT = np.ones(NC); _bw = popf1(OOF_B, MASK)
for _ in range(3):
    for c in range(NC):
        bv, bs = CW_OPT[c], popf1(OOF_B * CW_OPT, MASK)
        for v in (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.45, 1.7):
            t = CW_OPT.copy(); t[c] = v
            s = popf1(OOF_B * t, MASK)
            if s > bs: bv, bs = v, s
        CW_OPT[c] = bv
_fw = popf1(OOF_B * CW_OPT, MASK)
USE_CW = _fw > _bw + GATE["cw"]
print(f"    {_bw:.4f} -> {_fw:.4f} ({_fw-_bw:+.4f}) -> "
      f"{'PAKAI' if USE_CW else 'tolak (di bawah ambang derau)'}")
if USE_CW: print("    " + "  ".join(f"{CLASSES[i]}={CW_OPT[i]:.2f}" for i in range(NC)))
else: CW_OPT = np.ones(NC)

OOF_F = OOF_B * CW_OPT
TE_F  = TE_B  * CW_OPT
FINAL = popf1(OOF_F, MASK)

# =============================================================================
# 11. DIAGNOSTIK
# =============================================================================
print("\n" + "=" * 70 + "\nSTEP 5 - DIAGNOSTIK\n" + "=" * 70)
_plain = f1_score(ytrue[MASK], OOF_F[MASK].argmax(1), average="macro")
_bf, _pp = popf1(OOF_F, MASK, detail=True)
print(f"F1 Macro OOF polos (campuran train)       : {_plain:.4f}   <- angka yang menipu")
print(f"  blok  (n={_pp['blok'][1]:5d}) : {_pp['blok'][0]:.4f}   (v13: 0.9634)")
print(f"  strip (n={_pp['strip'][1]:5d}) : {_pp['strip'][0]:.4f}   (v13: 0.9958)")
print(f"F1 Macro OOF TERTIMBANG ke komposisi test : {_bf:.4f}   <<< dasar pemilihan")

print(f"\n[A] Ringkasan keputusan v14:")
for m in MEMBERS:
    q = m["mid"]
    if q in FOLDF1:
        mu, sd, dd, us = FOLDF1[q]
        print(f"    anggota {q} {m['name']:14s}: {us}/{CFG['n_folds']} fold,"
              f" F1 fold {mu:.4f} +/- {sd:.4f}, fold mati {dd}"
              + ("   <- CEK, harus 0" if dd else ""))
    else:
        print(f"    anggota {q} {m['name']:14s}: TIDAK JADI")
print(f"    campur          : " + (f"w={W:.2f} (anggota {MIDS[1]})" if BLEND_OK
                                   else f"TIDAK - anggota tunggal {BEST_M}"))
if PERPOP_OK:
    print(f"    campur per-pop  : w_blok={W_BLOK_MIX:.2f}, w_strip={W_STRIP_MIX:.2f}")
print(f"    bobot kelas     : {'PAKAI' if USE_CW else 'tidak'}")
print(f"\n    OOF v14 {FINAL:.4f}  vs  v13 {REF['oof_v13']:.4f}"
      f"  ({FINAL-REF['oof_v13']:+.4f})  vs  v12 {REF['oof_v12']:.4f}"
      f"  ({FINAL-REF['oof_v12']:+.4f})")
_gain = FINAL - REF["oof_v13"]
if _gain > 0.0015:
    print(f"    Dgn laju tukar v12->v13 (dOOF +0.0064 -> dLB +0.01417, rasio ~2,2)")
    print(f"    kenaikan ini kira-kira setara +{_gain*2.2:.4f} LB, jadi perkiraan")
    print(f"    kasar LB {REF['lb_v13']+_gain*2.2:.5f}. DUA TITIK saja yang")
    print(f"    membentuk rasio itu - perlakukan sbg orde besaran, bukan ramalan.")
elif _gain > -0.0015:
    print(f"    Praktis seri dgn v13. Kalau begini, submit berkas v13 yang SUDAH")
    print(f"    terbukti 0.89166 dan pakai berkas v14 sebagai slot kedua saja.")
else:
    print(f"    !! LEBIH RENDAH dari v13. Jangan submit ini sebagai slot utama.")
    print(f"    Periksa [R]: apakah anggota 0 mereproduksi 0.9799?")

pred = OOF_F.argmax(1)
print("\n[C] Laporan per kelas:\n")
print(classification_report(ytrue[MASK], pred[MASK], target_names=CLASSES,
                            digits=4, zero_division=0))
cm = confusion_matrix(ytrue[MASK], pred[MASK], labels=list(range(NC)))
print("Confusion matrix (baris = asli, kolom = prediksi):")
print(f"{'':>10s}" + "".join(f"{c[:7]:>8s}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    rs = max(cm[i].sum(), 1)
    print(f"{c:>10s}" + "".join(f"{v:8d}" for v in cm[i]) + f"   (recall {cm[i,i]/rs:.3f})")
pf = f1_score(ytrue[MASK], pred[MASK], average=None, labels=list(range(NC)), zero_division=0)
print("\nKelas paling lemah -> terkuat (v13: pegon 0.9467, jawi 0.9726, sisanya >=0.9886):")
for i in np.argsort(pf):
    off = cm[i].copy(); off[i] = 0; j = int(off.argmax())
    print(f"  {CLASSES[i]:10s} F1={pf[i]:.4f} n={cnt[i]:4d} tertukar dgn "
          f"'{CLASSES[j]}' ({off[j]}x)")
_tot_err = int((pred[MASK] != ytrue[MASK]).sum())
print(f"\n  total kesalahan OOF: {_tot_err} dari {int(MASK.sum())} baris"
      f"   (v13: 52 dari 3771)")

# --- [E] overlap kesalahan antar anggota: inilah SUMBER kenaikan campur ------
if len(MIDS) >= 2:
    a, b_ = MIDS[0], MIDS[1]
    wa = (P[a][MASK].argmax(1) != ytrue[MASK])
    wb = (P[b_][MASK].argmax(1) != ytrue[MASK])
    both, either = int((wa & wb).sum()), int((wa | wb).sum())
    print(f"\n[E] Overlap kesalahan anggota {a} vs {b_}:")
    print(f"    anggota {a} salah {int(wa.sum())} baris, anggota {b_} salah {int(wb.sum())}")
    print(f"    salah BERDUA {both}, salah salah-satu {either}")
    if either:
        print(f"    overlap = {100*both/either:.1f}% dari gabungan kesalahan")
        print(f"    Overlap rendah = kesalahannya di baris berbeda = ada yang bisa")
        print(f"    diperbaiki campur. Overlap ~100% = campur tidak punya bahan,")
        print(f"    dan gerbang [B] di atas memang akan menolaknya.")

# =============================================================================
# 12. SUBMISSION
# =============================================================================
print("\n" + "=" * 70 + "\nSTEP 6 - SUBMISSION\n" + "=" * 70)
paths_out = OrderedDict(); _seen_labels = {}
def _emit(name, prob):
    """Varian yang labelnya PERSIS sama dgn berkas yang sudah ditulis tidak
    ditulis lagi: berkas kembar cuma memperpanjang daftar [F] tanpa memberi
    satu pun kandidat slot yang benar-benar berbeda."""
    labs = labels_from(prob); key = tuple(labs)
    if key in _seen_labels:
        print(f"  {name}: identik dgn {_seen_labels[key]} -> tidak ditulis")
        return
    _seen_labels[key] = name
    paths_out[name] = write_sub(name, labs)

_emit("submission.csv", TE_F)                       # hasil akhir bergerbang
# Pembanding TERBERSIH dgn berkas yang mencetak 0.89166: anggota 0 sendirian,
# tanpa campur, tanpa bobot kelas. Satu-satunya perbedaan dari berkas v13
# adalah derau seed. Kalau gerbang campur menolak, keduanya identik dan yang
# kedua tidak ditulis - itu benar.
if 0 in T: _emit("submission_anggota0.csv", T[0])
if 1 in T: _emit("submission_anggota1.csv", T[1])
if USE_CW: _emit("submission_tanpa_bobot.csv", TE_B)

prior = cnt / cnt.sum()
share = np.bincount([C2I[l] for l in labels_from(TE_F)], minlength=NC) / N_TE
print("\n[D] Distribusi prediksi test vs prior train:\n")
print(f"  {'kelas':10s}{'train%':>9s}{'test%':>9s}{'selisih':>10s}")
for i, c in enumerate(CLASSES):
    d = (share[i] - prior[i]) * 100
    print(f"  {c:10s}{prior[i]*100:9.2f}{share[i]*100:9.2f}{d:+10.2f}"
          + ("   <-- CEK" if abs(d) > 3 else ""))
print(f"  skew maksimum = {np.abs(share-prior).max()*100:.2f}pp")
print("  Acuan v13 (LB 0.89166): pegon 18.85%, lampung 18.93%, jawi 12.30%.")
print("  Acuan v12 (LB 0.87749): pegon 18.28%, lampung 20.08%, jawi 11.97%.")
print("  TIGA pipeline berbeda (v11, v12, v13) menghasilkan distribusi yang")
print("  mirip, jadi distribusi yang tiba-tiba berbeda adalah tanda ada yang")
print("  salah - bukan tanda model yang lebih baik.")

print("\n[F] Beda antar varian (bandingkan LABEL, bukan probabilitas):")
_n = list(paths_out)
for i in range(len(_n)):
    for j in range(i+1, len(_n)):
        a_ = pd.read_csv(paths_out[_n[i]]).label.values
        b2 = pd.read_csv(paths_out[_n[j]]).label.values
        ag = float((a_ == b2).mean())
        print(f"  {_n[i]:28s} vs {_n[j]:28s} sepakat {ag*100:5.1f}% "
              f"({int((a_ != b2).sum())} baris beda)"
              + ("   <- praktis identik, slot kedua mubazir" if ag > 0.995 else ""))

print("\nFile yang ditulis:")
for k, v in paths_out.items(): print(f"  {v}")
if IS_COLAB and WORK.startswith(DRIVE_ROOT):
    print(f"\n  Semuanya ada di Google Drive Anda ({WORK}), jadi tetap ada")
    print(f"  walaupun sesi Colab ini putus sekarang juga.")
elif IS_COLAB:
    print(f"\n  !! Ini di /content, yang HILANG saat sesi putus. UNDUH SEKARANG.")

print(f"\n{clk()}  OOF TERTIMBANG akhir = {FINAL:.4f}")
print(f"Acuan: v13 OOF {REF['oof_v13']:.4f} -> LB {REF['lb_v13']:.5f} (peringkat 5)")
print(f"       peringkat 4 = {REF['rank4']:.5f}, jadi perlu"
      f" +{REF['rank4']-REF['lb_v13']:.5f} dari v13")
if SMOKE:
    print("\n!! DIJALANKAN DENGAN INFEST_SMOKE=1: sebagian bobot mungkin ACAK.")
    print("   Semua angka di atas TIDAK BERARTI. Jangan submit hasil ini.")

print("\n" + "=" * 70)
print("YANG HARUS DIPERIKSA SEBELUM SUBMIT, berurutan:")
print("=" * 70)
print("  1. [R] apakah anggota 0 mendarat di 0.9799 +/- 0.0015. Resepnya SAMA")
print("     PERSIS dgn v13, jadi kalau meleset jauh, ada yang berbeda di")
print("     pipeline ini dan tidak ada angka lain yang boleh dipercaya.")
print("  2. 'fold mati' di [A] harus 0 untuk kedua anggota.")
print("  3. [B] apakah campur lewat gerbang, dan sebesar apa. Kalau DITOLAK,")
print("     itu jawaban yang sah - submit anggota tunggal terbaiknya.")
print("  4. [E] overlap kesalahan. Overlap rendah menjelaskan kenapa campur")
print("     menang; overlap tinggi menjelaskan kenapa ia ditolak.")
print("  5. [D] distribusi prediksi harus mirip acuan v13 dan v12.")
print("  6. [F] jangan pakai dua slot untuk dua berkas yang sepakat >99.5%.")
print(f"  7. Jumlah baris submission harus {N_TE} dan urutannya ikut sample_submission.")
print("\nDUA SLOT FINAL - rekomendasi:")
print("  slot 1: submission.csv v14, TAPI hanya kalau [A] menunjukkan OOF-nya")
print(f"          di atas v13 ({REF['oof_v13']:.4f}). Kalau seri atau lebih rendah,")
print(f"          slot 1 tetap berkas v13 yang sudah terbukti {REF['lb_v13']:.5f}.")
print(f"  slot 2: berkas v13 yang sudah terbukti {REF['lb_v13']:.5f}.")
print("          Alasannya: kedua slot dinilai di himpunan privat yang SAMA, jadi")
print("          derau samplingnya saling meniadakan dan yang tersisa hanya")
print("          selisih antar-berkas. Memasangkan v14 dgn varian v14 lain")
print("          membuang separuh nilai slot kedua karena keduanya nyaris kembar.")
print("  JANGAN biarkan Kaggle memilih otomatis: kalau tidak dicentang manual ia")
print("  mengambil dua skor publik tertinggi, dan dua skor tertinggi biasanya")
print("  berasal dari model yang nyaris identik - persis pilihan terburuk.")
