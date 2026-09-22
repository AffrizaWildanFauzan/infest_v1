# =============================================================================
# INFEST XII 2026 - Klasifikasi Citra Aksara Tradisional Nusantara  [v13]
# Satu cell Kaggle. Dirancang untuk kuota GPU T4 12 JAM.
# Metrik: F1-Score Macro
#
# RIWAYAT LEADERBOARD NYATA
#   v3.1 0.8073 | v5 0.81054 | v6 0.83902 | v7 0.83422 | v11 0.85030
#   v12 submission.csv     0.87749  <- REKOR, peringkat 9. ft-siglip2 TUNGGAL.
#   v12 submission_iw.csv  0.85695  <- TURUN. ensemble 3 model se-backbone.
#   Pelabelan MANUAL manusia atas test = 0.71330.
#   Ambang top 5 = 0.88976. Jadi target v13 = +0.0123 di atas 0.87749.
#
# -----------------------------------------------------------------------------
# APA YANG DIAJARKAN LEADERBOARD v12 (dan karena itu membentuk v13)
#
#   [1] PEMENANGNYA MODEL TUNGGAL, BUKAN ENSEMBLE. `submission.csv` adalah
#       ft-siglip2 sendirian (OOF tertimbang 0.9735) dan mencetak 0.87749.
#       `submission_iw.csv` adalah rata-rata bobot SAMA dari ft-siglip2 +
#       lpft-siglip2 (0.9698) + siglip2-canon (0.9158), dan mencetak 0.85695.
#       Dua berkas itu cuma beda 63 dari 1220 baris (5,2%), jadi SELURUH
#       kerugian -0.02054 datang dari 63 baris itu; aliran terbesarnya
#       pegon->jawi (8) dan jawi->pegon (7). Ensemble tidak menemukan apa pun
#       yang baru, ia hanya menggoyang satu-satunya batas yang rapuh.
#       SEBABNYA: ketiganya backbone SigLIP2 yang SAMA, jadi kesalahannya
#       berkorelasi dan rata-rata tidak meredam apa pun - ia cuma menarik
#       hasil ke anggota yang lebih lemah. Dan anggota terlemahnya 0.0577 di
#       bawah yang terbaik tapi tetap diberi bobot 1/3.
#       ATURAN v13: JANGAN PERNAH merata-ratakan pemenang dengan anggota yang
#       OOF-nya >0.02 di bawahnya. Keberagaman ensemble butuh backbone yang
#       BERBEDA; tiga sudut pandang dari satu backbone bukan keberagaman.
#
#   [2] `impf1` (OOF berbobot-kepentingan) TIDAK LAYAK JADI PEMUTUS. Ia yang
#       memilih iw, dgn margin 0.0047, pada ESS cuma 36% (1.368 sampel efektif
#       dari 3.771). Derau rata-rata berskala 1/sqrt(n), jadi impf1 >=1,67x
#       lebih berisik dari OOF polos - margin 0.0047 itu LEBIH KECIL dari
#       deraunya sendiri. Di v13 impf1 dan seluruh mesin adversarialnya
#       DIBUANG; kalau mau, hitung ulang sebagai diagnostik, jangan gerbang.
#
#   [3] OOF POLOS TERNYATA BISA DIPERCAYA. Dgn titik baru (0.9735 -> 0.87749)
#       korelasi OOF-vs-LB berubah tanda:
#           4 pasang lama : Pearson +0.16, Spearman -0.32
#           5 pasang      : Pearson +0.64, Spearman +0.36
#       Empat versi lama berkerumun di OOF 0.968-0.972 - rentang terlalu
#       sempit untuk membedakan apa pun, jadi korelasi lama cuma mengukur
#       derau. Begitu ada model yang benar-benar naik di OOF, LB ikut naik
#       +0.027. Karena itu di v13 SEMUA gerbang dinilai dgn popf1 (OOF
#       tertimbang ke komposisi strip/blok test), dan hanya itu.
#
#   [4] YANG SUDAH SELESAI DIUJI DAN DIBUANG DARI v13, dgn angkanya:
#         LP-FT (Kumar dkk. ICLR 2022)  -0.0037 pada anggaran disamakan  [A5]
#         Otsu / CANON                  -0.0131 (DiT) dan -0.0091 (SigLIP2) [A3]
#         DiT (dokumen IIT-CDIP)        sumber terlemah, -0.10 dari SigLIP2
#         pra-latih aksara sbg INIT     -0.0143 uji terkontrol            [A4]
#         koreksi prior EM              -0.0410 rata-rata, menang 6/12     [P1]
#         zoo fitur beku + fusi + probe  semua di bawah fine-tune penuh
#       Membuang semuanya membebaskan kira-kira separuh anggaran GPU v12.
#       Itulah yang membiayai resolusi 384 di v13.
#
# -----------------------------------------------------------------------------
# APA YANG BARU DI v13 - dan kenapa masing-masing
#
#   (a) RESOLUSI 384, BUKAN 224.  <- ini taruhan utamanya
#       ft-siglip2 mencetak 0.87749 sambil berjalan di 224 piksel saja,
#       padahal 49,2% gambar uji adalah 'blok': halaman teks padat berisi
#       puluhan baris aksara. Di 224 piksel satu huruf tinggal beberapa
#       piksel. Buktinya ada di v12 sendiri: F1 blok 0.9533 vs strip 0.9932 -
#       kerugiannya terpusat persis di populasi yang paling dirugikan
#       resolusi rendah. `vit_base_patch16_siglip_384.v2_webli` adalah bobot
#       resmi yang SAMA keluarganya, jadi ini satu perubahan tunggal di jalur
#       yang sudah terbukti menang, bukan jalur baru.
#
#   (b) TTA yang SELURUHNYA berada di dalam distribusi latih.
#       v12 jalan dgn tta=1 - tidak ada TTA sama sekali. Tapi TTA yang naif
#       (crop kuadran, flip) justru di LUAR distribusi latih dan bisa
#       merugikan. Karena itu view TTA di sini diambil PERSIS dari ragam yang
#       sudah dilihat model saat latih: regangan sumbu-x ARJIT (0.65-1.55)
#       dan jitter skala letterbox (0.92-1.08). Tidak ada satu pun view TTA
#       yang belum pernah dilihat model.
#       Dan ia TIDAK dipercaya begitu saja: probabilitas tiap view disimpan
#       TERPISAH, lalu himpunan bagian {v0}, {v0,v1,v2}, {v0..v3} dinilai di
#       OOF dan yang menang dipakai. {v0} = resep v12 apa adanya.
#
#   (c) SPESIALIS pegon-vs-jawi.
#       Enam dari tujuh kelas sudah >=0.99 F1 di v12. Pegon sendirian di
#       0.9335, dgn 20 kekeliruan ke jawi dan jawi 18 ke pegon. Karena
#       metriknya MACRO-F1, pegon menyumbang 1/7 bobot: menaikkan pegon dari
#       0.93 ke 0.98 saja sudah +0.007 macro-F1 - lebih dari separuh jarak ke
#       top 5, dari satu kelas.
#       Kepala biner yang HANYA melihat dua kelas itu bisa belajar pemisah
#       yang benar (kepadatan diakritik: pegon rapat, jawi jarang) tanpa
#       terganggu lima kelas lain. Ia dipakai sbg PEMUTUS: massa probabilitas
#       gabungan pegon+jawi dibagi ulang menurut spesialis, total massa tidak
#       berubah, kelas lain tidak tersentuh. Digerbang di OOF.
#
#   (d) SEED-BAGGING, bukan ensemble lintas-model.
#       Ini pelajaran langsung dari [1]. Rata-rata beberapa run dgn resep
#       IDENTIK dan kualitas SETARA meredam derau seed tanpa pernah menyeret
#       hasil ke anggota yang lebih lemah. Tiap seed tambahan digerbang di
#       OOF: kalau tidak naik, ia tidak dipakai.
#
#   (e) GUBERNUR ANGGARAN 12 JAM.  <- ini yang membuat run-nya tidak sia-sia
#       Kuota GPU cuma 12 jam dan sesi Kaggle bisa mati kapan saja. Karena
#       itu: (i) kecepatan GPU DIUKUR dgn tolok ukur singkat di awal, bukan
#       ditebak, lalu rencana termahal yang MUAT dipilih otomatis;
#       (ii) tiap fold yang selesai LANGSUNG disimpan ke disk dan
#       submission.csv DITULIS ULANG, jadi mati di jam ke-11 pun tetap
#       meninggalkan berkas yang sah; (iii) kalau skrip dijalankan lagi,
#       fold yang sudah ada di disk DILEWATI (resume).
#
# -----------------------------------------------------------------------------
# JEBAKAN YANG DIWARISI - jangan dihapus
#   - `test.csv` di repo berisi 1.118 id test LAMA; irisannya dgn
#     `sample_submission.csv` (1.220 id) adalah NOL. Prediksi digerakkan dari
#     sample_submission.csv.
#   - load_body() WAJIB mengembalikan SALINAN. Modul bersama membuat fold 1
#     mulai dari bobot yang sudah dirusak fold 0.
#   - layer-wise lr decay lewat timm adalah no-op pada AdamW (`lr_scale`
#     diabaikan, lalu OneCycleLR menimpa `lr`). Dihitung sendiri dari nama
#     parameter dan DICETAK supaya bisa diperiksa.
#   - AddNoise harus ada di rantai augmentasi pada posisi yang sama spt v6/v7
#     (setelah ToTensor, sebelum Normalize). Sempat hilang di draf v11 dan
#     membuat run jangkar melatih resep yang berbeda.
#   - T4 adalah Turing: bf16 TIDAK DIDUKUNG. autocast di sini dipaku ke
#     float16 + GradScaler. Jangan ganti ke bfloat16.
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
# SMOKE=1 -> boleh jalan dgn bobot ACAK kalau unduhan diblokir. Hanya untuk
# menguji JALUR KODE di mesin tanpa internet; angkanya tidak berarti apa pun.
SMOKE = os.environ.get("INFEST_SMOKE", "0") == "1"
SEED  = 42

# --- ANGGARAN WAKTU -----------------------------------------------------------
# total_h  : kuota GPU yang tersedia (salva: 12 jam T4). Sisakan margin.
# reserve_h: waktu yang TIDAK boleh dipakai melatih - untuk prediksi test,
#            diagnostik, dan menulis submission. Kalau ini habis, run terakhir
#            terpotong dan yang tersisa cuma fold yang sudah tersimpan.
# Gubernur memakai waktu FOLD TERUKUR, bukan tebakan, begitu fold pertama
# selesai. Sebelum itu ia memakai tolok ukur `bench_steps` langkah latih.
BUDGET = dict(
    total_h   = float(os.environ.get("INFEST_BUDGET_H", "11.0")),
    reserve_h = 0.45,
    bench     = True,       # tolok ukur kecepatan GPU di awal
    bench_steps = 12,
    safety    = 1.15,       # proyeksi dikalikan ini sebelum dibandingkan sisa waktu
)

# --- RENCANA: dicoba berurutan, yang PERTAMA muat dipakai --------------------
# Ketiganya memakai backbone, lr, layer_decay, augmentasi dan fold yang sama.
# Yang berbeda hanya resolusi dan jumlah epoch, yaitu dua knob yang menentukan
# biaya. Jadi apa pun yang terpilih, ia tetap resep pemenang v12 - bukan resep
# lain - hanya dgn anggaran yang muat.
PLANS = [
    dict(name="A", model="vit_base_patch16_siglip_384.v2_webli", h=384, w=384, epochs=18,
         note="taruhan utama: resolusi 384 penuh, epoch sama spt v12"),
    dict(name="B", model="vit_base_patch16_siglip_384.v2_webli", h=384, w=384, epochs=13,
         note="384 tapi epoch dipangkas - tetap menjawab pertanyaan resolusi"),
    dict(name="C", model="vit_base_patch16_siglip_256.v2_webli", h=256, w=256, epochs=18,
         note="256: kompromi, masih 1,7x piksel v12"),
    dict(name="D", model="vit_base_patch16_siglip_224.v2_webli", h=224, w=224, epochs=18,
         note="DARURAT: resep v12 apa adanya. Kalau ini yang jalan, v13 tidak "
              "menguji apa pun yang baru soal resolusi - periksa kenapa GPU lambat."),
]

# Resep pemenang v12 yang TIDAK diubah: lr, layer_decay, epoch (di rencana A),
# CW berbobot kelas, label smoothing, EMA, augmentasi, guard kolaps.
FT = dict(lr_body=3e-5, lr_head=1e-3, layer_decay=0.75)

# --- SEED-BAGGING ------------------------------------------------------------
# Urutan run. Seed 0 adalah taruhan utama; seed berikutnya hanya dijalankan
# kalau gubernur bilang muat. Tiap tambahan digerbang di OOF.
SEEDS = [0, 1, 2]

# --- SPESIALIS pegon vs jawi -------------------------------------------------
# epochs lebih kecil karena datanya cuma ~2/7 dari train, jadi satu epoch pun
# jauh lebih murah. h/w mengikuti rencana terpilih.
SPEC = dict(enable=True, pair=("pegon", "jawi"), epochs=12,
            lr_body=3e-5, lr_head=1e-3, layer_decay=0.75,
            gate=0.0015)     # ambang derau: pakai hanya kalau popf1 naik segini

# --- TTA ---------------------------------------------------------------------
# (stretch_x, scale_letterbox). SEMUA berada di dalam rentang augmentasi latih:
# ARJIT meregangkan sumbu-x 0.65-1.55 dan letterbox menjitter skala 0.92-1.08.
# v0 IDENTIK dgn view test v12 (stretch 1.0, scale 1.0) supaya ada garis dasar
# yang benar-benar sebanding.
TTA_VIEWS = [(1.00, 1.00), (0.82, 1.00), (1.22, 1.00), (1.00, 0.94)]
# himpunan bagian yang diadu di OOF. {0} = resep v12 tanpa TTA.
TTA_SETS  = OrderedDict([("v0 (tanpa TTA)", [0]),
                         ("v0+v1+v2 (regang)", [0, 1, 2]),
                         ("semua 4 view", [0, 1, 2, 3])])
TTA_GATE  = 0.0015

# --- SYNMIX: korpus aksara sintetis sebagai AUGMENTASI, bukan inisialisasi ---
# MATI secara bawaan, dan itu keputusan sadar. Yang diketahui:
#   [S2] transfer langsung sintetis->nyata = 0.4370 lawan tebakan acak 0.1429
#        -> korpusnya MEMANG mengajarkan bentuk aksara.
#   [A4] sebagai INISIALISASI bobot ia merugikan -0.0143 (uji terkontrol)
#        -> pra-latih di domain sintetis menggeser backbone menjauh dari
#           statistik gambar nyata.
# Mencampurnya ke dalam batch nyata adalah cara ketiga yang BELUM PERNAH
# diukur: model melihat bentuk aksara bersih tanpa pernah meninggalkan
# distribusi nyata. Itu menarik, tapi ia MENGUBAH resep pemenang, dan dgn
# kuota 12 jam hanya ada satu kesempatan. Jadi: biarkan mati untuk run yang
# mengejar skor. Nyalakan hanya kalau ada kuota terpisah untuk mengujinya,
# dan bandingkan dgn seed 0 yang SYNMIX-nya mati - itu satu-satunya cara
# angkanya berarti.
SYNMIX = dict(enable=False, frac=0.15, n_per_class=1200)

# --- knob warisan yang TIDAK diubah dari v12 (resep pemenang) ----------------
CFG = dict(n_folds=5, batch=16, accum=2, nw=2, wd=0.05, ls=0.05, ema=0.999)
# batch 16 x accum 2 = batch efektif 32, sama spt v12. Dipecah karena ViT-B/16
# di 384 memakai 577 token (bukan 197) sehingga aktivasinya ~2,9x dan batch 32
# utuh tidak muat di 16 GB VRAM T4. Kalau tetap OOM, guard di train_fold akan
# memotong batch dan menyalakan gradient checkpointing lalu mengulang fold.
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

# --- ANGKA ACUAN dari v12, untuk perbandingan langsung -----------------------
# Fold di v13 dibangun dgn StratifiedKFold(5, shuffle=True, random_state=42)
# atas tr_df yang disusun dgn cara yang SAMA PERSIS spt v12, jadi popf1 di
# bawah ini sebanding baris-per-baris. Jangan ubah SEED atau urutan tr_df.
REF = dict(oof_v12_ft_siglip2=0.9735, lb_v12=0.87749, lb_v11=0.85030,
           lb_iw=0.85695, top5=0.88976)

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
print(f"anggaran: {BUDGET['total_h']:.1f} jam, cadangan {BUDGET['reserve_h']:.2f} jam")

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

print("\n" + "=" * 70 + "\nSTEP 2 - GUBERNUR ANGGARAN (ukur dulu, baru putuskan)\n" + "=" * 70)
PLAN, BENCH = None, None
for pl in PLANS:
    try:
        b = bench_plan(pl)
    except Exception as e:
        print(f"  rencana {pl['name']} tidak bisa diukur ({type(e).__name__}) -> lewati")
        continue
    need = project_run(pl, b) / 3600.0
    ok   = need * BUDGET["safety"] <= rem_h() - BUDGET["reserve_h"]
    print(f"  rencana {pl['name']}: {pl['model'].split('.')[0]} @{pl['h']} x {pl['epochs']}ep"
          f" -> perkiraan {need:.2f} jam untuk {CFG['n_folds']} fold  "
          f"({'MUAT' if ok else 'tidak muat'})")
    print(f"            {pl['note']}")
    if ok: PLAN, BENCH = pl, b; break
if PLAN is None:
    PLAN, BENCH = PLANS[-1], bench_plan(PLANS[-1])
    print(f"  !! tidak ada rencana yang muat. Dipaksa ke {PLAN['name']} dan gubernur")
    print(f"     akan memotong jumlah fold. Submission tetap ditulis dari fold yang jadi.")
R_MAIN = dict(model=PLAN["model"], h=PLAN["h"], w=PLAN["w"], grad_ckpt=BENCH["grad_ckpt"])
MICRO  = BENCH["micro"]
ACCUM  = max(1, int(round(CFG["batch"] * CFG["accum"] / MICRO)))
print(f"\n  TERPILIH: rencana {PLAN['name']} - {PLAN['model']} @{PLAN['h']}x{PLAN['w']},"
      f" {PLAN['epochs']} epoch")
print(f"  batch: micro={MICRO} x accum={ACCUM} = efektif {MICRO*ACCUM}"
      f" (v12 memakai 32)")
if PLAN["h"] == 224:
    print("  ! PERINGATAN: yang terpilih resolusi 224 = resep v12 apa adanya.")
    print("    Kenaikan skor dari v13 kalau begini hanya bisa datang dari TTA,")
    print("    spesialis pegon/jawi dan seed-bag - bukan dari resolusi.")

# =============================================================================
# 8b. LAMPIRAN - KORPUS AKSARA SINTETIS sebagai AUGMENTASI (SYNMIX, bawaan MATI)
# =============================================================================
# Kode ini diwarisi UTUH dari v12, di mana ia sudah terukur:
#   [S1] audit pintasan  0.1381  vs tebakan acak 0.1429  -> LULUS, korpusnya
#        benar-benar bebas pintasan metadata (di data NYATA angka yang sama
#        adalah 0.7826);
#   [S2] transfer langsung sintetis->nyata tanpa fine-tune = 0.4370 lawan
#        tebakan acak 0.1429 -> korpusnya MEMANG mengajarkan bentuk aksara.
# Yang GAGAL di v12 adalah cara PEMAKAIANNYA: sebagai inisialisasi bobot ia
# merugikan -0.0143 ([A4], uji terkontrol), karena pra-latih di domain sintetis
# menggeser backbone menjauh dari statistik gambar nyata.
# Di sini disediakan cara ketiga yang BELUM PERNAH diukur: mencampur citra
# sintetis LANGSUNG ke dalam batch nyata, sehingga model melihat bentuk aksara
# bersih tanpa pernah meninggalkan distribusi nyata.
# Tetap MATI secara bawaan - lihat catatan di blok SYNMIX.
import urllib.request, shutil
from PIL import ImageDraw, ImageFont, features

SYN_DIR  = os.path.join(WORK, "_syn13"); FONT_DIR = os.path.join(SYN_DIR, "fonts")
FONT_BASE = ["https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf",
             "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf"]
FONT_NAMES = ["NotoSansJavanese", "NotoSansBalinese", "NotoSansSundanese",
              "NotoSansBuginese", "NotoSansRejang", "NotoNaskhArabic"]

def _dl(url, dst, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    return os.path.getsize(dst)

def fetch_fonts():
    """URL raw.githubusercontent TERBUKTI 200; `github.com/notofonts/*/raw/`
    kena 403. Font yang gagal -> kelasnya DILEWATI, bukan diganti diam-diam."""
    os.makedirs(FONT_DIR, exist_ok=True); got = {}
    for n in FONT_NAMES:
        dst = os.path.join(FONT_DIR, f"{n}-Regular.ttf")
        if os.path.exists(dst) and os.path.getsize(dst) > 4000: got[n] = dst; continue
        last = "tidak ada mirror yang dicoba"
        for base in FONT_BASE:
            try:
                if _dl(f"{base}/{n}/{n}-Regular.ttf", dst) > 4000: got[n] = dst; break
                os.remove(dst)
            except Exception as e: last = type(e).__name__
        if n not in got: print(f"    ! font {n} gagal ({last}) -> kelasnya dilewati")
    return got

def _R(a, b): return [chr(c) for c in range(a, b + 1)]
_AR_BASE = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
# jawi vs pegon: keduanya aksara Arab dan pasangan yang paling sering tertukar.
# Pembedanya dikodekan eksplisit: inventaris huruf tambahan yang berbeda, DAN
# kepadatan harakat (pegon teks keagamaan/Jawa = rapat 0.45; jawi Melayu =
# hampir gundul 0.04). Itu isyarat visual nyata, bukan rekaan.
SYN_SRC = OrderedDict([
    ("jawa",    dict(font="NotoSansJavanese",  cons=_R(0xA98F, 0xA9B2),
                     diac=_R(0xA9B4, 0xA9BD), p_diac=0.55, rtl=False)),
    ("bali",    dict(font="NotoSansBalinese",  cons=_R(0x1B13, 0x1B33),
                     diac=_R(0x1B35, 0x1B43), p_diac=0.55, rtl=False)),
    ("sunda",   dict(font="NotoSansSundanese", cons=_R(0x1B8A, 0x1BA0),
                     diac=_R(0x1BA1, 0x1BAD), p_diac=0.45, rtl=False)),
    ("lontara", dict(font="NotoSansBuginese",  cons=_R(0x1A00, 0x1A16),
                     diac=_R(0x1A17, 0x1A1B), p_diac=0.35, rtl=False)),
    # Had Lampung / Kaganga TIDAK ada di Unicode -> tidak ada font resminya.
    # Rejang (serumpun Surat Ulu) dipakai sebagai PROKSI. Baca hasil kelas
    # lampung dgn diskon.
    ("lampung", dict(font="NotoSansRejang",    cons=_R(0xA930, 0xA946),
                     diac=_R(0xA947, 0xA951), p_diac=0.45, rtl=False)),
    ("jawi",    dict(font="NotoNaskhArabic", cons=_AR_BASE + list("ڠڤݢڽچۏڬ")*3,
                     diac=list("ًٌٍَُِّْ"), p_diac=0.04, rtl=True)),
    ("pegon",   dict(font="NotoNaskhArabic", cons=_AR_BASE + list("ڠڤڮۑڎڟٹڽ")*3,
                     diac=list("ًٌٍَُِّْ"), p_diac=0.45, rtl=True)),
])

def _syn_word(s, rng):
    out = []
    for _ in range(rng.randint(2, 7)):
        out.append(rng.choice(s["cons"]))
        if rng.random() < s["p_diac"]: out.append(rng.choice(s["diac"]))
    return "".join(out)

def syn_render(tag, rng, fonts, Wt, Ht):
    """Render satu citra aksara `tag`, dikembalikan TEPAT berukuran (Wt, Ht).
    (Wt, Ht) ditarik DARI LUAR dan independen dari tag - itulah satu-satunya
    alasan korpus ini berguna: menurut konstruksi P(kelas | W,H,AR) = P(kelas),
    jadi model yang bisa memisahkan kelas di sini HARUS membaca bentuk aksara."""
    s = SYN_SRC[tag]; fs = rng.randint(18, 46)
    font = ImageFont.truetype(fonts[s["font"]], fs)
    lh = fs * rng.uniform(1.5, 2.3); ar = Wt / max(Ht, 1)
    nl, wtxt = 1, ar * lh
    for cand in sorted(range(1, 13), key=lambda _: rng.random()):
        w_try = ar * cand * lh
        if 140 <= w_try <= 2400: nl, wtxt = cand, w_try; break
    else:
        nl = max(1, min(12, int(round(math.sqrt(max(Wt*Ht, 1)) / max(lh, 1)))))
        wtxt = max(140, min(2400, ar * nl * lh))
    probe = ImageDraw.Draw(Image.new("L", (8, 8), 255)); lines = []
    for _ in range(nl):
        t = ""
        while probe.textlength(t, font=font) < wtxt and len(t) < 420:
            t += ("" if not t else " ") + _syn_word(s, rng)
        lines.append(t or _syn_word(s, rng))
    pad = rng.randint(5, 26)
    W = int(max(probe.textlength(t, font=font) for t in lines)) + 2*pad
    H = int(nl * lh) + 2*pad
    paper = rng.randint(205, 255)
    img = Image.new("L", (max(W, 24), max(H, 24)), paper); d = ImageDraw.Draw(img)
    ink = rng.randint(0, min(150, paper - 55))
    for i, t in enumerate(lines):
        x = pad if not s["rtl"] else img.width - pad
        d.text((x, pad + i*lh), t, font=font, fill=ink,
               anchor=("la" if not s["rtl"] else "ra"),
               direction=("ltr" if not s["rtl"] else "rtl"))
    # korupsi ringan SAAT RENDER: kalau tidak, korpusnya 100% 2-aras dan
    # derajat binarisasi jadi penanda sintetis-vs-nyata.
    if rng.random() < 0.75:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.8)))
    if rng.random() < 0.55:
        a = np.asarray(img, np.float32) + np.random.normal(
            0, rng.uniform(2, 12), (img.height, img.width))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")
    if rng.random() < 0.55:
        b = io.BytesIO(); img.save(b, "JPEG", quality=rng.randint(28, 92))
        b.seek(0); img = Image.open(b).convert("L")
    return img.resize((max(Wt, 16), max(Ht, 16)), Image.BILINEAR)

def build_syn_pool(n_per_class):
    """kembalikan [(PIL 'L', y)] di MEMORI - tidak ditulis ke disk karena di
    SYNMIX ia hanya dipakai sebagai sumber augmentasi, bukan dataset sendiri."""
    fonts = fetch_fonts()
    if not features.check("raqm"):
        print("    ! Pillow TANPA raqm: aksara kompleks tidak dibentuk dgn benar")
        print("      dan Arab tidak disambung. Korpusnya jauh lebih lemah.")
    use = [t for t in SYN_SRC if SYN_SRC[t]["font"] in fonts and t in C2I]
    if len(use) < 2:
        print("    !! kurang dari 2 kelas tersedia -> korpus DIBATALKAN"); return None
    if len(use) < NC: print(f"    kelas tanpa font -> dilewati: "
                            f"{[t for t in SYN_SRC if t not in use]}")
    pool_size = []
    for q in tr_df.path.tolist():
        try:
            with Image.open(q) as im: pool_size.append(im.size)
        except Exception: pass
    if not pool_size: pool_size = [(900, 300)]
    rng = random.Random(SEED); out = []; t0 = time.time()
    for tag in use:
        for _ in range(n_per_class):
            Wt, Ht = pool_size[rng.randrange(len(pool_size))]
            sc = min(1.0, 768 / max(Wt, Ht))
            Wt, Ht = max(16, int(Wt*sc)), max(16, int(Ht*sc))
            try: out.append((syn_render(tag, rng, fonts, Wt, Ht), C2I[tag]))
            except Exception: continue
        print(f"    {tag:9s} {n_per_class:5d} citra  ({time.time()-t0:.0f}s)")
    return out or None

# =============================================================================
# 9. MENJALANKAN SATU SEED PENUH  (+ simpan tiap fold + resume)
# =============================================================================
NV     = len(TTA_VIEWS)
N_TR   = len(tr_df); N_TE = len(te_df)
OOF_S, TE_S = OrderedDict(), OrderedDict()     # seed -> (NV, N, NC)
COVER  = OrderedDict()                          # seed -> mask baris OOF terisi
FOLDF1 = OrderedDict()

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
    """Ditulis ULANG setiap kali satu fold selesai. Kalau sesi Kaggle mati di
    jam ke-11, yang tertinggal tetap berkas submission yang SAH dari fold yang
    sudah jadi - bukan tidak ada apa-apa. Versi ini sengaja memakai setelan
    paling polos (view 0, rata-rata seed yang ada): gerbang OOF baru dijalankan
    di akhir, dan hasil akhirnya akan MENIMPA berkas ini."""
    have = [s for s in TE_S if TE_S[s] is not None]
    if not have: return
    p = np.mean([TE_S[s][0] for s in have], 0)
    try:
        write_sub("submission.csv", labels_from(p))
    except Exception as e:
        print(f"    ! gagal menulis submission sementara: {type(e).__name__}")

def _ckpath(kind, sid, fold): return os.path.join(CK, f"{kind}_s{sid}_f{fold}.npz")

def run_seed(sid):
    """Satu seed penuh atas 5 fold. Mengembalikan True kalau setidaknya satu
    fold jadi. Gubernur boleh memotong di tengah; yang sudah jadi tetap dipakai."""
    global MICRO, ACCUM
    oof = np.zeros((NV, N_TR, NC)); tep = np.zeros((NV, N_TE, NC))
    cov = np.zeros(N_TR, bool); f1s, used, dead = [], 0, 0
    syn = SYN_POOL if SYNMIX["enable"] else None
    print(f"\n  -- seed {sid} : {PLAN['model'].split('.')[0]} @{PLAN['h']} "
          f"x {PLAN['epochs']}ep --")
    for fold, (itr, iva) in enumerate(FOLDS):
        cp = _ckpath("main", sid, fold)
        if os.path.exists(cp):                       # RESUME
            try:
                z = np.load(cp)
                oof[:, iva] = z["va"]; tep += z["te"]; cov[iva] = True
                f1s.append(float(z["f1"])); used += 1
                print(f"    fold{fold} dimuat dari checkpoint (F1={float(z['f1']):.4f})")
                continue
            except Exception:
                print(f"    fold{fold} checkpoint rusak -> dilatih ulang")
        # --- gubernur: masih cukup waktu untuk SATU fold lagi? ---
        est = project_fold(PLAN, BENCH, len(itr), len(iva), N_TE, NV) / 3600.0
        if est * BUDGET["safety"] > rem_h() - BUDGET["reserve_h"]:
            print(f"    {clk()} fold{fold} DILEWATI: perlu ~{est:.2f} jam, "
                  f"sisa aman {rem_h()-BUDGET['reserve_h']:.2f} jam.")
            print(f"    Seed {sid} berhenti di {used} fold. Ini keputusan gubernur,")
            print(f"    bukan kegagalan - fold yang sudah jadi tetap dipakai.")
            break
        dtr = tr_df.iloc[itr].reset_index(drop=True)
        dva = tr_df.iloc[iva].reset_index(drop=True)
        lb, lh = FT["lr_body"], FT["lr_head"]        # di-RESET tiap fold
        model, f1, vp, rr, bad = None, -1, None, None, True
        for attempt in range(3):                      # guard KOLAPS: ulang dgn lr/3
            try:
                model, f1, vp, rr = train_fold(
                    R_MAIN, dtr, dva, PLAN["epochs"], lb, lh,
                    SEED + sid*1000 + fold*10 + attempt, FT["layer_decay"], NC,
                    MICRO, ACCUM, syn=syn, tag=f"main-s{sid}")
            except Exception as e:
                if not _is_oom(e): raise
                if DEV == "cuda": torch.cuda.empty_cache()
                if MICRO > 4:
                    MICRO //= 2
                    ACCUM = max(1, int(round(CFG["batch"]*CFG["accum"] / MICRO)))
                else:
                    R_MAIN["grad_ckpt"] = True
                print(f"    OOM di fold{fold} -> micro={MICRO}, accum={ACCUM}"
                      + (" +grad_ckpt" if R_MAIN.get("grad_ckpt") else "") + ", ulangi fold")
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
        ibs = max(4, MICRO * 2)
        va = predict_views(model, dva, rr, list(range(NV)), ibs)
        te = predict_views(model, te_df, rr, list(range(NV)), ibs)
        oof[:, iva] = va; tep += te; cov[iva] = True; used += 1; f1s.append(f1)
        try:
            np.savez_compressed(cp, va=va.astype(np.float32),
                                te=te.astype(np.float32), f1=np.float32(f1))
        except Exception as e:
            print(f"    ! gagal menyimpan checkpoint fold{fold}: {type(e).__name__}")
        print(f"    {clk()} fold{fold} macro-F1(view0)={f1:.4f}")
        del model
        if DEV == "cuda": torch.cuda.empty_cache()
        TE_S[sid] = tep / max(used, 1); flush_submission()
    if used == 0:
        print(f"    !! seed {sid} tidak menghasilkan satu fold pun")
        TE_S.pop(sid, None); return False
    OOF_S[sid] = oof; TE_S[sid] = tep / used; COVER[sid] = cov
    FOLDF1[sid] = (float(np.mean(f1s)), float(np.std(f1s)), dead, used)
    print(f"    seed {sid} selesai: {used}/{CFG['n_folds']} fold, "
          f"F1 fold rata2 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}, fold mati {dead}")
    flush_submission()
    return True

# --- korpus sintetis untuk SYNMIX (mati secara bawaan, lihat catatan di atas)
SYN_POOL = None
if SYNMIX["enable"]:
    print("\n  SYNMIX menyala: merender korpus aksara sintetis...")
    print("  PERINGATAN: ini MENGUBAH resep yang mencetak 0.87749. Kalau run ini")
    print("  dipakai untuk mengejar skor dan bukan untuk mengukur SYNMIX, matikan.")
    try:
        SYN_POOL = build_syn_pool(SYNMIX["n_per_class"])      # lihat lampiran
    except Exception as e:
        print(f"  !! render sintetis gagal ({type(e).__name__}) -> SYNMIX dimatikan")
        SYNMIX["enable"] = False; SYN_POOL = None

print("\n" + "=" * 70 + "\nSTEP 3 - RUN UTAMA (seed-bagging)\n" + "=" * 70)
print("Seed 0 adalah taruhan utama. Seed berikutnya hanya dijalankan kalau")
print("gubernur bilang muat, dan hanya DIPAKAI kalau ia menaikkan OOF.")
print("Ini bentuk keberagaman yang aman: semua anggota punya resep dan kualitas")
print("yang SAMA. Yang menghancurkan submission_iw.csv adalah merata-ratakan")
print("pemenang dgn anggota 0.058 lebih lemah - itu tidak bisa terjadi di sini.")
for _sid in SEEDS:
    if rem_h() - BUDGET["reserve_h"] <= 0:
        print(f"\n  {clk()} anggaran habis -> seed {_sid} dan sesudahnya dilewati.")
        break
    if _sid != SEEDS[0]:
        _est = project_run(PLAN, BENCH) / 3600.0
        if _est * BUDGET["safety"] > rem_h() - BUDGET["reserve_h"]:
            # masih boleh dicoba: run_seed akan memotong per fold kalau perlu,
            # dan fold yang jadi tetap berguna untuk bag.
            print(f"\n  {clk()} seed {_sid}: perkiraan {_est:.2f} jam > sisa aman"
                  f" {rem_h()-BUDGET['reserve_h']:.2f} jam.")
            if rem_h() - BUDGET["reserve_h"] < _est / CFG["n_folds"] * 2:
                print("    Tidak cukup bahkan untuk 2 fold -> dilewati sama sekali.")
                break
            print("    Tetap dijalankan sebagian; gubernur memotong per fold.")
    run_seed(_sid)
if not OOF_S:
    raise RuntimeError("Tidak ada satu pun seed yang menghasilkan fold. "
                       "Periksa pesan kegagalan di atas sebelum mencoba lagi.")

# =============================================================================
# 10. SPESIALIS pegon vs jawi
# =============================================================================
# Kenapa justru pasangan ini. Laporan per kelas v12: enam dari tujuh kelas
# sudah >=0.99 F1. pegon sendirian di 0.9335, tertukar 20x ke jawi, dan jawi
# tertukar 18x ke pegon. Metriknya MACRO-F1, jadi pegon menyumbang 1/7 bobot
# apa pun ukurannya: menaikkan pegon 0.93 -> 0.98 saja sudah +0.007 macro-F1,
# lebih dari separuh jarak ke top 5, dari SATU kelas.
#
# Kenapa kepala terpisah bisa lebih baik dari model 7 kelas. Pembeda pegon-jawi
# bukan repertoar huruf (keduanya aksara Arab) melainkan KEPADATAN DIAKRITIK -
# pegon rapat harakat, jawi hampir gundul. Di model 7 kelas isyarat halus itu
# harus bersaing dgn lima batas lain yang jauh lebih mudah, dan gradiennya
# tenggelam. Model yang HANYA melihat dua kelas tidak punya saingan itu.
#
# Cara pakainya sengaja konservatif: ia TIDAK menimpa prediksi. Massa
# probabilitas gabungan pegon+jawi dibagi ULANG menurut campuran
# alpha*spesialis + (1-alpha)*model utama. Total massa tidak berubah dan lima
# kelas lain tidak tersentuh sama sekali. alpha=0 berarti spesialis diabaikan
# total, dan alpha DIPILIH DI OOF, bukan ditebak.
SPEC_OOF, SPEC_TE, SPEC_COV = None, None, None
IP = C2I.get(SPEC["pair"][0], 0)          # didefinisikan tanpa syarat: apply_spec
IJ = C2I.get(SPEC["pair"][1], 0)          # memakainya walau spesialis dimatikan
if SPEC["enable"] and SPEC["pair"][0] in C2I and SPEC["pair"][1] in C2I and IP != IJ:
    _est_spec = (CFG["n_folds"] * project_fold(
        dict(epochs=SPEC["epochs"]), BENCH,
        int(cnt[IP] + cnt[IJ]) * (CFG["n_folds"]-1) // CFG["n_folds"],
        int((cnt[IP] + cnt[IJ]) / CFG["n_folds"]),
        N_TE + N_TR // CFG["n_folds"], 1) / 3600.0)
    print("\n" + "=" * 70 + f"\nSTEP 4 - SPESIALIS {SPEC['pair'][0]} vs {SPEC['pair'][1]}\n" + "=" * 70)
    print(f"  perkiraan biaya {_est_spec:.2f} jam, sisa aman "
          f"{rem_h()-BUDGET['reserve_h']:.2f} jam")
    if _est_spec * BUDGET["safety"] > rem_h() - BUDGET["reserve_h"]:
        print("  -> DILEWATI, anggaran tidak cukup. Model utama tetap utuh.")
    else:
        sm = np.isin(ytrue, [IP, IJ])
        print(f"  {int(sm.sum())} baris train dari 2 kelas "
              f"({cnt[IP]} {CLASSES[IP]} + {cnt[IJ]} {CLASSES[IJ]})")
        so = np.zeros((N_TR, 2)); ste = np.zeros((N_TE, 2))
        sc_ = np.zeros(N_TR, bool); nused = 0
        for fold, (itr, iva) in enumerate(FOLDS):
            cp = _ckpath("spec", 0, fold)
            if os.path.exists(cp):
                try:
                    z = np.load(cp)
                    so[iva] = z["va"]; ste += z["te"]; sc_[iva] = True; nused += 1
                    print(f"    fold{fold} dimuat dari checkpoint"); continue
                except Exception: pass
            if (_est_spec/CFG["n_folds"]) * BUDGET["safety"] > rem_h() - BUDGET["reserve_h"]:
                print(f"    fold{fold} DILEWATI: anggaran habis."); break
            i_tr = itr[np.isin(ytrue[itr], [IP, IJ])]
            i_va = iva[np.isin(ytrue[iva], [IP, IJ])]
            if len(i_tr) < 40 or len(i_va) < 8:
                print(f"    fold{fold} terlalu sedikit sampel -> dilewati"); continue
            dtr = tr_df.iloc[i_tr].reset_index(drop=True)
            dva = tr_df.iloc[i_va].reset_index(drop=True)
            dtr["y"] = (dtr.y.values == IJ).astype(int)     # 0 = pair[0], 1 = pair[1]
            dva["y"] = (dva.y.values == IJ).astype(int)
            lb, lh, model, f1, vp, rr, bad = SPEC["lr_body"], SPEC["lr_head"], \
                                             None, -1, None, None, True
            for attempt in range(3):
                try:
                    model, f1, vp, rr = train_fold(
                        R_MAIN, dtr, dva, SPEC["epochs"], lb, lh,
                        SEED + 7000 + fold*10 + attempt, SPEC["layer_decay"], 2,
                        MICRO, ACCUM, tag="spec")
                except Exception as e:
                    if not _is_oom(e): raise
                    if DEV == "cuda": torch.cuda.empty_cache()
                    print(f"    OOM di spesialis fold{fold} -> dilewati"); model = None
                    break
                bad = is_collapsed(vp, f1, 2)
                if not bad: break
                lb /= 3; lh /= 3
                print(f"    fold{fold} spesialis KOLAPS (F1={f1:.4f}) -> lr={lb:.1e}")
            if model is None or bad:
                print(f"    fold{fold} spesialis DIKELUARKAN")
                del model
                if DEV == "cuda": torch.cuda.empty_cache()
                continue
            ibs = max(4, MICRO * 2)
            # prediksi atas SELURUH baris validasi fold ini (bukan cuma baris
            # pegon/jawi): gerbang OOF nanti menerapkan aturan ini ke semua
            # baris, jadi semua baris butuh keluaran spesialis. Tidak ada
            # kebocoran - model fold ini tidak pernah melihat satu pun baris iva.
            dv_all = tr_df.iloc[iva].reset_index(drop=True)
            va = predict_views(model, dv_all, rr, [0], ibs)[0]
            te = predict_views(model, te_df,  rr, [0], ibs)[0]
            so[iva] = va; ste += te; sc_[iva] = True; nused += 1
            try:
                np.savez_compressed(cp, va=va.astype(np.float32), te=te.astype(np.float32))
            except Exception: pass
            print(f"    {clk()} fold{fold} spesialis biner F1={f1:.4f}")
            del model
            if DEV == "cuda": torch.cuda.empty_cache()
        if nused:
            SPEC_OOF, SPEC_TE, SPEC_COV = so, ste / nused, sc_
            print(f"  spesialis siap dari {nused}/{CFG['n_folds']} fold")
        else:
            print("  spesialis tidak menghasilkan satu fold pun -> diabaikan")

def apply_spec(prob, spec, alpha):
    """bagi ULANG massa pegon+jawi. Total massa tiap baris TIDAK berubah dan
    lima kelas lain tidak tersentuh, jadi ini tidak bisa merusak kelas lain."""
    if spec is None or alpha <= 0: return prob
    out = prob.copy()
    m = prob[:, IP] + prob[:, IJ]
    d = np.maximum(m, 1e-12)
    base = np.stack([prob[:, IP]/d, prob[:, IJ]/d], 1)       # proporsi model utama
    mix  = alpha * spec + (1 - alpha) * base
    mix /= np.maximum(mix.sum(1, keepdims=True), 1e-12)
    out[:, IP] = m * mix[:, 0]; out[:, IJ] = m * mix[:, 1]
    return out

# =============================================================================
# 11. PERAKITAN: setiap keputusan digerbang di OOF TERTIMBANG, satu per satu
# =============================================================================
# Urutannya sengaja: TTA dulu (diputuskan pada seed 0 sendirian, supaya
# sebanding langsung dgn v12), lalu seed-bag, lalu spesialis, lalu bobot kelas.
# Tiap tahap mencetak angka SEBELUM dan SESUDAH, jadi kalau ada yang merugikan
# di leaderboard nanti, sumbernya bisa ditunjuk tanpa menebak.
print("\n" + "=" * 70 + "\nSTEP 5 - PERAKITAN & GERBANG\n" + "=" * 70)
S0 = SEEDS[0] if SEEDS[0] in OOF_S else list(OOF_S)[0]

def mean_views(views_arr, idx):
    """rata-rata probabilitas atas himpunan bagian view TTA."""
    return views_arr[idx].mean(0)

# --- [T] gerbang TTA, diputuskan pada seed 0 --------------------------------
print(f"\n[T] TTA - dinilai pada seed {S0} saja (sebanding langsung dgn v12):")
print(f"    {'himpunan view':22s}{'OOF tertimbang':>16s}{'selisih vs v0':>15s}")
_base_t, _best_t = None, None
TTA_IDX, TTA_NAME = TTA_SETS["v0 (tanpa TTA)"], "v0 (tanpa TTA)"
for nm, idx in TTA_SETS.items():
    v = popf1(mean_views(OOF_S[S0], idx), COVER[S0])
    if _base_t is None: _base_t, _best_t = v, v
    print(f"    {nm:22s}{v:16.4f}{v-_base_t:+15.4f}")
    # ambang diukur terhadap v0 (resep v12), dan hanya yang TERBAIK yang diambil
    if v > _base_t + TTA_GATE and v > _best_t:
        TTA_IDX, TTA_NAME, _best_t = idx, nm, v
print(f"    -> dipakai: {TTA_NAME}")
print(f"    Semua view di sini ada DI DALAM rentang augmentasi latih (regangan")
print(f"    sumbu-x ARJIT 0.65-1.55 dan jitter skala letterbox 0.92-1.08), jadi")
print(f"    tidak ada satu pun yang belum pernah dilihat model.")

# --- [R1] perbandingan langsung dgn v12 -------------------------------------
_s0 = popf1(mean_views(OOF_S[S0], [0]), COVER[S0])
print(f"\n[R1] PERBANDINGAN LANGSUNG dgn v12 (fold identik, baris identik):")
print(f"     v12 ft-siglip2 @224, view tunggal : {REF['oof_v12_ft_siglip2']:.4f}"
      f"   -> LB {REF['lb_v12']:.5f}")
print(f"     v13 seed {S0} @{PLAN['h']}, view tunggal : {_s0:.4f}   ({_s0-REF['oof_v12_ft_siglip2']:+.4f})")
if _s0 < REF["oof_v12_ft_siglip2"] - 0.002:
    print("     !! RESOLUSI LEBIH TINGGI TIDAK MEMBANTU di data ini. Itu temuan")
    print("        nyata, bukan kegagalan skrip. Kalau begini, submit berkas v12")
    print("        yang 0.87749 dan jangan buang slot untuk v13 - kecuali gerbang")
    print("        TTA/spesialis di bawah memberi kenaikan yang cukup besar.")
elif _s0 > REF["oof_v12_ft_siglip2"] + 0.002:
    print("     -> resolusi membantu. Korelasi OOF-LB atas 5 pasang terakhir")
    print("        adalah Pearson +0.64, jadi ini sinyal yang layak dipercaya.")
else:
    print("     -> praktis seri. Kenaikan harus datang dari TTA/spesialis/bag.")

# --- [G] seed-bagging: tambahkan seed hanya kalau OOF naik ------------------
BAG = [S0]
_cov = COVER[S0].copy()
_cur = popf1(mean_views(OOF_S[S0], TTA_IDX), _cov)
print(f"\n[G] Seed-bagging (ambang {GATE['bag']:+.4f}):")
print(f"    seed {S0} sendirian -> {_cur:.4f}")
for s in OOF_S:
    if s == S0: continue
    cov2 = _cov & COVER[s]
    if cov2.sum() < 0.5 * N_TR:
        print(f"    seed {s}: liputan fold terlalu sedikit -> dilewati"); continue
    cand = BAG + [s]
    v = popf1(np.mean([mean_views(OOF_S[q], TTA_IDX) for q in cand], 0), cov2)
    ref = popf1(np.mean([mean_views(OOF_S[q], TTA_IDX) for q in BAG], 0), cov2)
    print(f"    + seed {s} -> {v:.4f} (tanpa dia {ref:.4f}, {v-ref:+.4f})"
          f"  {'PAKAI' if v > ref + GATE['bag'] else 'tolak'}")
    if v > ref + GATE["bag"]: BAG, _cov, _cur = cand, cov2, v
print(f"    bag akhir: seed {BAG} -> OOF tertimbang {_cur:.4f}")
if len(BAG) == 1:
    print("    Satu seed saja. Itu hasil yang sah: v12 membuktikan model tunggal")
    print("    bisa mengalahkan ensemble, dan gerbang ini memang ada supaya")
    print("    anggota yang tidak membantu TIDAK ikut - persis kesalahan iw.")

MASK  = _cov
OOF_B = np.mean([mean_views(OOF_S[q], TTA_IDX) for q in BAG], 0)
TE_B  = np.mean([TE_S[q][TTA_IDX].mean(0) for q in BAG], 0)

# --- [SP] gerbang spesialis: alpha dipilih di OOF ---------------------------
ALPHA = 0.0
if SPEC_OOF is not None:
    m_sp = MASK & SPEC_COV
    print(f"\n[SP] Spesialis {CLASSES[IP]} vs {CLASSES[IJ]} - alpha dicari di OOF"
          f" (ambang {SPEC['gate']:+.4f}):")
    base = popf1(OOF_B, m_sp)
    best_a, best_v = 0.0, base
    for a in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
        v = popf1(apply_spec(OOF_B, SPEC_OOF, a), m_sp)
        print(f"     alpha={a:4.2f} -> {v:.4f} ({v-base:+.4f})")
        if v > best_v: best_a, best_v = a, v
    ALPHA = best_a if best_v > base + SPEC["gate"] else 0.0
    print(f"     -> alpha dipakai = {ALPHA:.2f}"
          + ("" if ALPHA > 0 else "  (spesialis TIDAK dipakai - di bawah ambang derau)"))
    if ALPHA > 0:
        _b = f1_score(ytrue[m_sp], OOF_B[m_sp].argmax(1), average=None,
                      labels=list(range(NC)), zero_division=0)
        _a = f1_score(ytrue[m_sp], apply_spec(OOF_B, SPEC_OOF, ALPHA)[m_sp].argmax(1),
                      average=None, labels=list(range(NC)), zero_division=0)
        print(f"     F1 {CLASSES[IP]}: {_b[IP]:.4f} -> {_a[IP]:.4f} ({_a[IP]-_b[IP]:+.4f})")
        print(f"     F1 {CLASSES[IJ]}: {_b[IJ]:.4f} -> {_a[IJ]:.4f} ({_a[IJ]-_b[IJ]:+.4f})")
OOF_A = apply_spec(OOF_B, SPEC_OOF, ALPHA)
TE_A  = apply_spec(TE_B,  SPEC_TE,  ALPHA)

# --- [W] bobot per kelas untuk macro-F1 (coordinate ascent) -----------------
# macro-F1 TIDAK dioptimalkan oleh argmax probabilitas: kelas kecil butuh
# dorongan. Ditala HANYA di OOF, dipakai hanya kalau naik melebihi ambang derau.
print(f"\n[W] Bobot per-kelas (ambang {GATE['cw']:+.4f}):")
CW_OPT = np.ones(NC); _bw = popf1(OOF_A, MASK)
for _ in range(3):
    for c in range(NC):
        bv, bs = CW_OPT[c], popf1(OOF_A * CW_OPT, MASK)
        for v in (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.45, 1.7):
            t = CW_OPT.copy(); t[c] = v
            s = popf1(OOF_A * t, MASK)
            if s > bs: bv, bs = v, s
        CW_OPT[c] = bv
_fw = popf1(OOF_A * CW_OPT, MASK)
USE_CW = _fw > _bw + GATE["cw"]
print(f"    {_bw:.4f} -> {_fw:.4f} ({_fw-_bw:+.4f}) -> "
      f"{'PAKAI' if USE_CW else 'tolak (di bawah ambang derau)'}")
if USE_CW: print("    " + "  ".join(f"{CLASSES[i]}={CW_OPT[i]:.2f}" for i in range(NC)))
else: CW_OPT = np.ones(NC)

OOF_F = OOF_A * CW_OPT
TE_F  = TE_A  * CW_OPT
FINAL = popf1(OOF_F, MASK)

# =============================================================================
# 12. DIAGNOSTIK
# =============================================================================
print("\n" + "=" * 70 + "\nSTEP 6 - DIAGNOSTIK\n" + "=" * 70)
_plain = f1_score(ytrue[MASK], OOF_F[MASK].argmax(1), average="macro")
_bf, _pp = popf1(OOF_F, MASK, detail=True)
print(f"F1 Macro OOF polos (campuran train)       : {_plain:.4f}   <- angka yang menipu")
print(f"  blok  (n={_pp['blok'][1]:5d}) : {_pp['blok'][0]:.4f}   (v12: 0.9533)")
print(f"  strip (n={_pp['strip'][1]:5d}) : {_pp['strip'][0]:.4f}   (v12: 0.9932)")
print(f"F1 Macro OOF TERTIMBANG ke komposisi test : {_bf:.4f}   <<< dasar pemilihan")
print(f"\n[A] Ringkasan keputusan v13:")
print(f"    rencana         : {PLAN['name']}  {PLAN['model']} @{PLAN['h']}x{PLAN['w']}"
      f" x {PLAN['epochs']}ep")
print(f"    TTA             : {TTA_NAME}")
print(f"    seed-bag        : {BAG}")
print(f"    spesialis alpha : {ALPHA:.2f}")
print(f"    bobot kelas     : {'PAKAI' if USE_CW else 'tidak'}")
for s in BAG:
    if s in FOLDF1:
        m_, sd_, dd_, us_ = FOLDF1[s]
        print(f"    seed {s}: {us_}/{CFG['n_folds']} fold, F1 fold {m_:.4f} +/- {sd_:.4f},"
              f" fold mati {dd_}" + ("   <- CEK, harus 0" if dd_ else ""))
print(f"\n    OOF v13 {FINAL:.4f}  vs  v12 {REF['oof_v12_ft_siglip2']:.4f}"
      f"  ({FINAL-REF['oof_v12_ft_siglip2']:+.4f})")

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
print("\nKelas paling lemah -> terkuat (v12: pegon 0.9335, sisanya >=0.99):")
for i in np.argsort(pf):
    off = cm[i].copy(); off[i] = 0; j = int(off.argmax())
    print(f"  {CLASSES[i]:10s} F1={pf[i]:.4f} n={cnt[i]:4d} tertukar dgn "
          f"'{CLASSES[j]}' ({off[j]}x)")

# =============================================================================
# 13. SUBMISSION
# =============================================================================
print("\n" + "=" * 70 + "\nSTEP 7 - SUBMISSION\n" + "=" * 70)
paths_out = OrderedDict(); _seen_labels = {}
def _emit(name, prob):
    """Varian yang labelnya PERSIS sama dgn berkas yang sudah ditulis tidak
    ditulis lagi: berkas kembar cuma membuat daftar [F] panjang tanpa memberi
    satu pun kandidat slot yang benar-benar berbeda."""
    labs = labels_from(prob); key = tuple(labs)
    if key in _seen_labels:
        print(f"  {name}: identik dgn {_seen_labels[key]} -> tidak ditulis")
        return
    _seen_labels[key] = name
    paths_out[name] = write_sub(name, labs)

_emit("submission.csv", TE_F)                                   # hasil akhir bergerbang
# Pembanding TERBERSIH dgn berkas yang mencetak 0.87749: seed pertama, view
# tunggal, tanpa spesialis, tanpa bobot kelas. Satu-satunya perbedaan dari
# berkas v12 adalah RESOLUSI. Kalau dua slot final harus dipakai, inilah
# kandidat kedua yang paling informatif - ia menjawab satu pertanyaan bersih.
_emit("submission_tunggal.csv", TE_S[S0][0])
if ALPHA > 0:   _emit("submission_tanpa_spesialis.csv", TE_B * CW_OPT)
if USE_CW:      _emit("submission_tanpa_bobot.csv", TE_A)
if len(BAG) > 1: _emit("submission_bag_polos.csv", TE_B)

prior = cnt / cnt.sum()
share = np.bincount([C2I[l] for l in labels_from(TE_F)], minlength=NC) / N_TE
print("\n[D] Distribusi prediksi test vs prior train:\n")
print(f"  {'kelas':10s}{'train%':>9s}{'test%':>9s}{'selisih':>10s}")
for i, c in enumerate(CLASSES):
    d = (share[i] - prior[i]) * 100
    print(f"  {c:10s}{prior[i]*100:9.2f}{share[i]*100:9.2f}{d:+10.2f}"
          + ("   <-- CEK" if abs(d) > 3 else ""))
print(f"  skew maksimum = {np.abs(share-prior).max()*100:.2f}pp")
print("  Acuan v12 (LB 0.87749): pegon 18.28%, lampung 20.08%, jawi 11.97%.")
print("  Kalau angka di atas jauh melenceng dari itu, periksa dulu sebelum submit -")
print("  dua pipeline berbeda di v11 dan v12 menghasilkan distribusi yang mirip,")
print("  jadi distribusi yang tiba-tiba berbeda adalah tanda ada yang salah.")

print("\n[F] Beda antar varian (bandingkan LABEL, bukan probabilitas):")
_n = list(paths_out)
for i in range(len(_n)):
    for j in range(i+1, len(_n)):
        a = pd.read_csv(paths_out[_n[i]]).label.values
        b = pd.read_csv(paths_out[_n[j]]).label.values
        ag = float((a == b).mean())
        print(f"  {_n[i]:30s} vs {_n[j]:30s} sepakat {ag*100:5.1f}% "
              f"({int((a != b).sum())} baris beda)"
              + ("   <- praktis identik, slot kedua mubazir" if ag > 0.995 else ""))

print("\nFile yang ditulis:")
for k, v in paths_out.items(): print(f"  {v}")
print(f"\n{clk()}  OOF TERTIMBANG akhir = {FINAL:.4f}")
print(f"Acuan: v12 OOF {REF['oof_v12_ft_siglip2']:.4f} -> LB {REF['lb_v12']:.5f} (peringkat 9)")
print(f"       ambang top 5 = {REF['top5']:.5f}, jadi perlu +{REF['top5']-REF['lb_v12']:.5f}")
if SMOKE:
    print("\n!! DIJALANKAN DENGAN INFEST_SMOKE=1: sebagian bobot mungkin ACAK.")
    print("   Semua angka di atas TIDAK BERARTI. Jangan submit hasil ini.")

print("\n" + "=" * 70)
print("YANG HARUS DIPERIKSA SEBELUM SUBMIT, berurutan:")
print("=" * 70)
print("  1. [R1] OOF view-tunggal v13 vs v12 (0.9735). Inilah satu-satunya uji")
print("     bersih apakah resolusi membantu - fold dan barisnya identik.")
print("  2. 'fold mati' di [A] harus 0. Kalau tidak, ada fold yang kolaps dan")
print("     rata-ratanya dihitung dari lebih sedikit model.")
print("  3. [T] apakah TTA menang, dan sebesar apa. Kalau 'v0 (tanpa TTA)' yang")
print("     terpilih, itu jawaban yang sah, bukan kegagalan.")
print("  4. [SP] apakah spesialis pegon/jawi menaikkan F1 pegon. Ini satu-satunya")
print("     kelas yang masih di bawah 0.99, dan 1/7 macro-F1 ada di sana.")
print("  5. [D] distribusi prediksi harus mirip acuan v12. Distribusi yang")
print("     tiba-tiba berbeda = ada yang salah, bukan model yang lebih baik.")
print("  6. [F] jangan pakai dua slot untuk dua berkas yang sepakat >99.5%.")
print(f"  7. Jumlah baris submission harus {N_TE} dan urutannya ikut sample_submission.")
print("\nDUA SLOT FINAL - rekomendasi:")
print("  slot 1: submission.csv  (hasil v13 bergerbang penuh)")
print("  slot 2: berkas v12 yang SUDAH terbukti 0.87749, BUKAN varian v13 lain.")
print("          Alasannya: kedua slot dinilai di himpunan privat yang SAMA, jadi")
print("          derau samplingnya saling meniadakan dan yang tersisa hanya")
print("          selisih antar-berkas. Memasangkan v13 dgn v13 lain membuang")
print("          separuh nilai slot kedua karena keduanya nyaris kembar;")
print("          memasangkan dgn 0.87749 memberi jaring pengaman yang nyata.")
print("  JANGAN biarkan Kaggle memilih otomatis: kalau tidak dicentang manual ia")
print("  mengambil dua skor publik tertinggi, dan dua skor tertinggi biasanya")
print("  berasal dari model yang nyaris identik - persis pilihan terburuk.")
