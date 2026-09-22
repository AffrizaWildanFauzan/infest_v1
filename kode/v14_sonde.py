# =============================================================================
# v14_sonde.py - SONDE SATU FOLD: apakah DINOv3 layak jadi anggota ensemble?
# =============================================================================
# Ini BUKAN skrip pencetak skor. Ia tidak menulis submission dan tidak bisa
# menaikkan peringkat sendirian. Ia menjawab SATU pertanyaan, dalam <=1 jam
# T4, supaya run berikutnya tidak menebak.
#
# PERTANYAANNYA
# -------------
# v13 (SigLIP2 @384, satu seed, satu view) mencetak LB 0.89166, peringkat 5.
# Semua gerbang v13 MENOLAK tambahan: seed-bag ditolak, TTA ditolak, spesialis
# pegon/jawi ditolak, bobot kelas ditolak. Artinya jalur "lebih banyak dari
# resep yang sama" sudah buntu, dan satu-satunya keberagaman yang belum
# pernah diuji di data ini adalah KELUARGA ARSITEKTUR LAIN.
#
# automl (probe beku, 19 kandidat) menempatkan DINOv3-B/16 @384 di 0.9361
# lawan SigLIP2-384 di 0.9439 - selisih probe -0.0078. Tapi probe bukan
# fine-tune: di data ini probe melebih-lebihkan selisih sekitar 3x (probe
# 224->384 = +0.0189, fine-tune 224->384 = +0.0064). Jadi DINOv3 yang di-
# fine-tune DIPERKIRAKAN ~0.003 di bawah SigLIP2. Itu perkiraan, bukan ukuran.
#
# Yang membedakan "anggota ensemble yang bagus" dari "racun" persis angka itu:
#   submission_iw.csv TURUN 0.02054 karena merata-ratakan pemenang dgn anggota
#   0.058 LEBIH LEMAH. Anggota yang 0.003 lebih lemah tapi dari keluarga lain
#   adalah kasus yang sama sekali berbeda - itulah yang sonde ini ukur.
#
# KENAPA SATU FOLD SUDAH CUKUP
# ----------------------------
# Fold 0 sudah dilatih TIGA KALI oleh v13 dgn seed berbeda:
#     seed 0 -> 0.9849    seed 1 -> 0.9857    seed 2 -> 0.9879
#     rata2 0.9862, sd 0.0015
# Jadi derau "fold yang sama, seed berbeda" SUDAH terukur di data ini: 0.0015.
# Selisih yang diperkirakan (~0.003) dan selisih yang berbahaya (>=0.01)
# keduanya di atas derau itu. Satu fold TIDAK cukup untuk memutuskan selisih
# 0.001; ia lebih dari cukup untuk memutuskan "sekelas atau tidak".
#
# APA YANG DILAKUKAN SETELAHNYA (aturan keputusan, ditulis SEBELUM angkanya ada)
# -----------------------------------------------------------------------------
#   F1 fold0 DINOv3 >= 0.980  -> sekelas. Ensemble lintas-keluarga layak
#                                dicoba di run berikutnya, DIGERBANG di OOF.
#   0.970 .. 0.980            -> batas. Layak hanya kalau kesalahannya
#                                BERBEDA (skrip mencetak overlap kesalahan).
#   < 0.970                   -> jangan. Itu wilayah submission_iw.csv.
# Overlap kesalahan dicetak karena anggota yang sedikit lebih lemah TAPI salah
# di baris yang berbeda tetap berguna, sedangkan anggota yang salah di baris
# yang SAMA tidak menambah apa pun berapa pun skornya.
#
# CARA PAKAI
#   satu cell Kaggle, GPU T4, ~0,6 jam. INFEST_BUDGET_H mengatur pagarnya
#   (bawaan 1.0). Kalau tolok ukur bilang tidak muat, skrip BERHENTI dan
#   mengatakannya - ia tidak akan memulai fold yang pasti terpotong.
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

# Anggaran keras. Sonde ini ada justru KARENA kuota tinggal sedikit, jadi ia
# menolak memulai fold yang tidak akan selesai, bukan memulainya dan berharap.
BUDGET = dict(
    total_h   = float(os.environ.get("INFEST_BUDGET_H", "1.0")),
    reserve_h = 0.05,          # cuma perlu menulis angka, bukan submission
    bench     = True,
    bench_steps = 12,
    safety    = 1.15,
)

# Kandidat, dicoba berurutan; yang pertama muat dijalankan. Semuanya @384 dgn
# resep v13 yang TIDAK diubah - kalau resepnya ikut berubah, angkanya tidak
# menjawab pertanyaan apa pun.
CANDS = [
    dict(name="dinov3-b16", model="vit_base_patch16_dinov3.lvd1689m", h=384, w=384,
         epochs=18, note="keluarga arsitektur BERBEDA dari SigLIP2 (DINOv3, "
                         "self-supervised). Ini pertanyaan utamanya."),
    dict(name="dinov3-cnx", model="convnext_base.dinov3_lvd1689m", h=384, w=384,
         epochs=18, note="cadangan: ConvNeXt DINOv3, probe 0.9188. Lebih jauh "
                         "lagi dari ViT, tapi probe-nya juga lebih rendah."),
    dict(name="dinov3-b16-13ep", model="vit_base_patch16_dinov3.lvd1689m", h=384, w=384,
         epochs=13, note="DARURAT kalau 18 epoch tidak muat. CATAT bahwa "
                         "perbandingannya jadi tidak adil - v13 memakai 18."),
]

FOLD_ID = 0          # fold 0 SAJA, karena fold 0 punya 3 acuan seed dari v13.

# Acuan v13 di FOLD YANG SAMA (StratifiedKFold(5, shuffle=True, random_state=42)).
# Tiga seed, jadi derau seed di fold ini bukan tebakan.
REF_FOLD0 = dict(siglip2_384=[0.9849, 0.9857, 0.9879])
# Aturan keputusan, ditulis sebelum angkanya ada (lihat kepala berkas).
DECIDE = dict(sekelas=0.980, batas=0.970)

REF = dict(oof_v12_ft_siglip2=0.9735, oof_v13_siglip2_384=0.9799,
           lb_v12=0.87749, lb_v13=0.89166, lb_v11=0.85030, lb_iw=0.85695,
           top5=0.88976, rank4=0.90299)

# --- resep v13 yang TIDAK boleh diubah di sini -------------------------------
FT = dict(lr_body=3e-5, lr_head=1e-3, layer_decay=0.75)
TTA_VIEWS = [(1.00, 1.00), (0.82, 1.00), (1.22, 1.00), (1.00, 0.94)]
SYNMIX = dict(enable=False, frac=0.15, n_per_class=1200)
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
GATE   = dict(cw=0.002, bag=0.0010)
AR_SPLIT = 3.0

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
print(f"anggaran SONDE: {BUDGET['total_h']:.2f} jam (bukan run pencetak skor)")
# =============================================================================
# 1. LOAD / DISCOVER DATA   (identik v12 - jangan diubah, fold bergantung padanya)
# =============================================================================
SEARCH = [p for p in ["/kaggle/input", "/kaggle/working", ".", "./data", "/content"]
          if os.path.isdir(p)]
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
CK   = os.path.join(WORK, "_ck13"); os.makedirs(CK, exist_ok=True)

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
assert train_csv, "train.csv tidak ketemu"
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
print(f"  v12 mengukur F1 blok 0.9533 vs strip 0.9932. Blok adalah populasi yang")
print(f"  rugi, dan blok pula yang paling dirugikan resolusi 224 - itulah dasar")
print(f"  taruhan 384 di v13.")

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
print("  Ini SAMA PERSIS dgn v12, jadi popf1 di bawah sebanding baris-per-baris")
print(f"  dgn angka acuan ft-siglip2 v12 = {REF['oof_v12_ft_siglip2']:.4f} (LB {REF['lb_v12']:.5f}).")

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
# 8. GUBERNUR ANGGARAN: ukur dulu, baru pilih rencana
# =============================================================================
# Kenapa diukur dan bukan ditebak: kecepatan T4 di Kaggle bervariasi (T4 vs
# T4 x2, CPU 2 vs 4 inti untuk dataloader, gambar yang berat didekode). Salah
# tebak 2x berarti entah membuang 5 jam kuota atau terpotong di tengah fold
# ketiga. Tolok ukurnya beberapa puluh detik; itu harga yang sangat murah.
_BENCH = {}
def bench_plan(pl):
    """kembalikan dict(sec_img, micro, grad_ckpt) - detik per GAMBAR per langkah
    latih, ukuran micro-batch yang MUAT, dan apakah perlu grad checkpointing."""
    key = (pl["model"], pl["h"])
    if key in _BENCH: return _BENCH[key]
    r = dict(model=pl["model"], h=pl["h"], w=pl["w"])
    micro, gck = CFG["batch"], False
    for attempt in range(4):
        try:
            body, dim, mean, std = load_body(r["model"], r["h"], r["w"], fresh=True)
            rr = dict(r); rr["mean"], rr["std"] = mean, std
            m = Net(body, dim, NC)
            if gck:
                try: m.body.set_grad_checkpointing(True)
                except Exception: pass
            m = m.to(DEV).train()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-6)
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
                    loss = crit(m(x), y)
                sc.scale(loss).backward(); sc.step(opt); sc.update()
                opt.zero_grad(set_to_none=True)
                if i == 2:                                  # 3 langkah pertama = pemanasan
                    if DEV == "cuda": torch.cuda.synchronize()
                    t0 = time.time(); n = 0
                elif t0 is not None: n += len(y)
            if DEV == "cuda": torch.cuda.synchronize()
            dt = max(time.time() - t0, 1e-6) if t0 else 1e-6
            out = dict(sec_img=dt/max(n, 1), micro=micro, grad_ckpt=gck)
            del m, opt, sc, ld
            if DEV == "cuda": torch.cuda.empty_cache()
            _BENCH[key] = out
            print(f"    tolok ukur {pl['model'].split('.')[0]} @{pl['h']}: "
                  f"{n/dt:.1f} gambar/detik latih, micro={micro}"
                  + (" +grad_ckpt" if gck else ""))
            return out
        except Exception as e:
            if not _is_oom(e):
                print(f"    !! tolok ukur gagal: {type(e).__name__}: {e}"); raise
            if DEV == "cuda": torch.cuda.empty_cache()
            if micro > 4: micro //= 2
            elif not gck: gck = True
            else: raise
            print(f"    OOM -> coba lagi dgn micro={micro}" + (" +grad_ckpt" if gck else ""))
    raise RuntimeError("tidak ada konfigurasi memori yang muat")

def project_fold(pl, b, n_tr, n_va, n_te, nviews):
    """proyeksi detik untuk SATU fold, dari kecepatan terukur.
    Inferensi diperkirakan ~1/3 biaya langkah latih (hanya lintasan maju,
    tanpa backward dan tanpa update optimizer)."""
    si = b["sec_img"]; inf = si / 3.0
    ep = pl["epochs"]
    t_train = ep * n_tr * si
    t_val   = (ep - ep//3) * 2 * n_va * inf          # raw + EMA, view 0 saja
    t_final = (n_va + n_te) * nviews * inf           # TTA penuh, sekali
    return t_train + t_val + t_final

def project_run(pl, b, nviews=len(TTA_VIEWS)):
    n = len(tr_df); n_tr = int(n*(1-1/CFG["n_folds"])); n_va = n - n_tr
    return CFG["n_folds"] * project_fold(pl, b, n_tr, n_va, len(te_df), nviews)

# =============================================================================
# 9. GUBERNUR + SONDE
# =============================================================================
print("\n" + "=" * 70 + "\nSTEP 2 - GUBERNUR (ukur dulu, baru putuskan)\n" + "=" * 70)
print("  Sonde ini menolak memulai fold yang tidak akan selesai. Kalau tidak")
print("  ada kandidat yang muat, ia berhenti dan bilang - itu jawaban yang sah.")

itr, iva = FOLDS[FOLD_ID]
PICK, BENCH = None, None
for cd in CANDS:
    try:
        b = bench_plan(cd)
    except Exception as e:
        print(f"  {cd['name']}: tolok ukur gagal ({type(e).__name__}) -> lewati")
        continue
    need = project_fold(cd, b, len(itr), len(iva), 0, 1) / 3600.0
    ok   = need * BUDGET["safety"] <= rem_h() - BUDGET["reserve_h"]
    print(f"  {cd['name']:16s} @{cd['h']} x {cd['epochs']}ep -> perkiraan "
          f"{need:.2f} jam untuk 1 fold  ({'MUAT' if ok else 'TIDAK MUAT'})")
    print(f"                   {cd['note']}")
    if ok: PICK, BENCH = cd, b; break

if PICK is None:
    print("\n" + "=" * 70)
    print("BERHENTI: tidak ada kandidat yang muat di sisa waktu.")
    print("  Ini bukan kegagalan. Memulai fold yang akan terpotong berarti")
    print("  membakar kuota tanpa menghasilkan satu angka pun yang bisa dibaca.")
    print(f"  Sisa aman: {rem_h()-BUDGET['reserve_h']:.2f} jam.")
    print("  Naikkan INFEST_BUDGET_H kalau kuota Anda sebenarnya lebih besar,")
    print("  atau jalankan sonde ini di sesi berikutnya.")
    print("=" * 70)
    sys.exit(0)

R_S    = dict(model=PICK["model"], h=PICK["h"], w=PICK["w"], grad_ckpt=BENCH["grad_ckpt"])
MICRO  = BENCH["micro"]
ACCUM  = max(1, int(round(CFG["batch"] * CFG["accum"] / MICRO)))
print(f"\n  TERPILIH: {PICK['name']} - {PICK['model']} @{PICK['h']}x{PICK['w']},"
      f" {PICK['epochs']} epoch, fold {FOLD_ID}")
print(f"  batch: micro={MICRO} x accum={ACCUM} = efektif {MICRO*ACCUM} (v13 memakai 32)")
if MICRO * ACCUM != CFG["batch"] * CFG["accum"]:
    print("  ! batch efektif BERBEDA dari v13. Selisih F1 di bawah tidak lagi")
    print("    murni soal backbone - catat ini saat membaca hasilnya.")

# --- jalankan satu fold ------------------------------------------------------
print("\n" + "=" * 70 + f"\nSTEP 3 - SATU FOLD ({PICK['name']}, fold {FOLD_ID})\n" + "=" * 70)
print("  Tidak ada keluaran sampai fold selesai (~30-40 menit). Itu normal.")
dtr = tr_df.iloc[itr].reset_index(drop=True)
dva = tr_df.iloc[iva].reset_index(drop=True)
lb, lh = FT["lr_body"], FT["lr_head"]
model, f1, vp, rr, bad = None, -1.0, None, None, True
for attempt in range(3):
    try:
        model, f1, vp, rr = train_fold(
            R_S, dtr, dva, PICK["epochs"], lb, lh,
            SEED + FOLD_ID*10 + attempt, FT["layer_decay"], NC,
            MICRO, ACCUM, syn=None, tag=f"sonde-{PICK['name']}")
    except Exception as e:
        if not _is_oom(e): raise
        if DEV == "cuda": torch.cuda.empty_cache()
        if MICRO > 4:
            MICRO //= 2; ACCUM = max(1, int(round(CFG["batch"]*CFG["accum"] / MICRO)))
        else:
            R_S["grad_ckpt"] = True
        print(f"    OOM -> micro={MICRO}, accum={ACCUM}"
              + (" +grad_ckpt" if R_S.get("grad_ckpt") else "") + ", ulangi fold")
        continue
    bad = is_collapsed(vp, f1, NC)
    if not bad: break
    lb /= 3; lh /= 3
    sh = np.bincount(vp, minlength=NC).max() / len(vp)
    print(f"    KOLAPS (F1={f1:.4f}, {sh*100:.0f}% satu kelas) -> ulang dgn lr={lb:.1e}")

if model is None or bad:
    print("\n" + "=" * 70)
    print(f"HASIL: {PICK['name']} KOLAPS di fold {FOLD_ID} dan tidak pulih.")
    print("  Itu informasi, bukan kegagalan skrip: backbone yang tidak bisa")
    print("  dilatih dgn resep v13 memang tidak layak jadi anggota ensemble.")
    print("  Jangan pakai kandidat ini. Coba cadangan di CANDS di sesi lain.")
    print("=" * 70)
    sys.exit(0)

va = predict_views(model, dva, rr, [0], max(4, MICRO*2))[0]
del model
if DEV == "cuda": torch.cuda.empty_cache()

# =============================================================================
# 10. BACA HASILNYA
# =============================================================================
print("\n" + "=" * 70 + "\nSTEP 4 - HASIL\n" + "=" * 70)
mref  = np.array(REF_FOLD0["siglip2_384"], float)
ref_m, ref_sd = float(mref.mean()), float(mref.std())
d     = f1 - ref_m

mask0 = np.zeros(len(tr_df), bool); mask0[iva] = True
pv    = np.zeros((len(tr_df), NC)); pv[iva] = va
pop, parts = popf1(pv, mask0, detail=True)

print(f"  {PICK['name']} fold{FOLD_ID} macro-F1 (view 0) = {f1:.4f}")
print(f"  acuan v13 SigLIP2-384 fold{FOLD_ID}, 3 seed  = "
      + ", ".join(f"{v:.4f}" for v in mref) + f"  (rata2 {ref_m:.4f}, sd {ref_sd:.4f})")
print(f"  selisih = {d:+.4f}  ({abs(d)/max(ref_sd,1e-9):.1f}x sd seed)")
print(f"\n  popf1 fold ini (ditimbang ke komposisi test) = {pop:.4f}")
for nm in ("blok", "strip"):
    v, n = parts[nm]
    print(f"    {nm:5s} (n={n:5d}) : {v:.4f}")
print("  CATATAN: popf1 di atas dihitung dari SATU fold, jadi ia tidak")
print(f"  sebanding langsung dgn OOF 5-fold v13 ({REF['oof_v13_siglip2_384']:.4f}).")
print("  Yang sebanding adalah baris macro-F1 di atasnya.")

print("\n  [E] Overlap kesalahan - inilah yang menentukan nilai ensemble:")
yv      = ytrue[iva]
wrong_d = va.argmax(1) != yv
print(f"      {PICK['name']} salah di {int(wrong_d.sum())} dari {len(yv)} baris "
      f"fold ini ({100*wrong_d.mean():.2f}%).")
print("      Untuk mengukur overlap-nya dgn SigLIP2 Anda perlu OOF v13. Kalau")
print("      _ck13/main_s0_f%d.npz masih ada di output v13, jalankan:" % FOLD_ID)
print("          z = np.load('_ck13/main_s0_f%d.npz'); vs = z['va'][0]" % FOLD_ID)
print("          ws = vs.argmax(1) != yv")
print("          print((wrong_d & ws).sum(), wrong_d.sum(), ws.sum())")
print("      Overlap RENDAH = dua model salah di baris berbeda = ensemble")
print("      punya sesuatu untuk diperbaiki. Overlap TINGGI = anggota kedua")
print("      tidak menambah apa pun, berapa pun skornya.")

print("\n" + "=" * 70 + "\nKEPUTUSAN (aturannya ditulis sebelum angkanya ada)\n" + "=" * 70)
if f1 >= DECIDE["sekelas"]:
    print(f"  {f1:.4f} >= {DECIDE['sekelas']:.3f} -> SEKELAS.")
    print(f"  {PICK['name']} layak jadi anggota ensemble lintas-keluarga.")
    print("  Run berikutnya: latih 5 fold kandidat ini, lalu gabungkan dgn")
    print("  SigLIP2 lewat gerbang OOF yang SAMA spt v13 ([G], ambang +0.0010).")
    print("  JANGAN gabungkan tanpa gerbang - itu persis yang menjatuhkan")
    print(f"  submission_iw.csv ({REF['lb_iw']:.5f} dari {REF['lb_v12']:.5f}).")
elif f1 >= DECIDE["batas"]:
    print(f"  {DECIDE['batas']:.3f} <= {f1:.4f} < {DECIDE['sekelas']:.3f} -> BATAS.")
    print("  Layak HANYA kalau kesalahannya berbeda. Hitung overlap di [E]")
    print("  dulu. Overlap < ~50% = tetap layak dicoba dgn gerbang; overlap")
    print("  tinggi = tidak ada yang bisa diperbaiki, jangan buang kuota.")
else:
    print(f"  {f1:.4f} < {DECIDE['batas']:.3f} -> JANGAN.")
    print("  Ini wilayah submission_iw.csv: anggota yang cukup lebih lemah")
    print("  sampai rata-ratanya menurunkan pemenang. Jalur ensemble lintas-")
    print("  keluarga dgn kandidat ini ditutup, dan itu hasil yang berguna -")
    print("  ia menghemat 3 jam kuota di run berikutnya.")
print(f"\n  {clk()} sonde selesai. Tidak ada submission yang ditulis, sesuai rancangan.")
print("=" * 70)
