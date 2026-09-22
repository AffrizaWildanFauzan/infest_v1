# =============================================================================
# INFEST XII 2026 - AutoML: cari backbone terbaik  [automl v1]
# Satu cell Kaggle. Dirancang untuk kuota GPU T4, bawaan 2 JAM.
#
# APA YANG DICARI SKRIP INI, DAN KENAPA BENTUKNYA BEGINI
#
# AutoML yang "benar" untuk lomba citra adalah mencari arsitektur DAN
# hiperparameter dgn cara melatih tiap kandidat sampai selesai. Di sini itu
# TIDAK MUNGKIN, dan angkanya jelas: satu fine-tune penuh ViT-B di 384 makan
# ~2,7 jam di T4. Dgn kuota 12 jam itu berarti 4 kandidat, dan 4 titik tidak
# cukup untuk menyimpulkan apa pun.
#
# Jadi skrip ini memakai pengganti yang MURAH dan - ini bagian pentingnya -
# sudah TERBUKTI di data ini: peringkat PROBE BEKU. Ambil backbone, bekukan,
# satu lintasan maju atas 3.771 + 1.220 gambar, lalu regresi logistik 5-fold
# di atas embedding-nya. Biayanya 20-90 detik per kandidat, bukan 2,7 jam.
#
# Kenapa peringkat murah itu boleh dipercaya di sini (bukan klaim umum):
#   v11 mengukur probe beku dan menempatkan SigLIP2 di puncak (0.9158) serta
#   DiT di dasar (0.8122). v12 lalu fine-tune penuh keduanya dan urutannya
#   BERTAHAN: ft-siglip2 0.9735 mengalahkan ft-v7/convnext 0.9667, dan DiT
#   yang dibuang memang tidak pernah kompetitif. ft-siglip2 itulah yang
#   akhirnya mencetak LB 0.87749.
#   Itu SATU konfirmasi, bukan bukti. Karena itu STEP 5 menawarkan fine-tune
#   pendek untuk menguji apakah peringkatnya benar-benar berpindah - dan
#   kalau anggaran tidak cukup, skrip mengatakannya apa adanya, bukan
#   berpura-pura sudah membuktikan.
#
# EMPAT TAHAP
#   STEP 2  sweep kandidat: backbone x resolusi -> probe beku -> popf1
#   STEP 3  hiperparameter probe di fitur yang sudah di-cache (CPU, detik)
#   STEP 4  pencarian FUSI: gabungkan embedding beberapa backbone lalu satu
#           probe. Inilah "menggabungkan beberapa model" dalam bentuk
#           termurah - tidak ada backbone yang dilatih sama sekali.
#   STEP 5  (opsional, digerbang anggaran) fine-tune PENDEK untuk juara baru
#           DAN untuk siglip2-224 sebagai kontrol, anggaran disamakan.
#
# YANG DIHASILKAN
#   - tabel peringkat lengkap, disimpan ke automl_hasil.csv
#   - baris konfigurasi SIAP TEMPEL untuk PLANS di kode/v13.py
#
# HUBUNGANNYA DGN v13
#   Skrip ini TIDAK menghasilkan submission dan tidak menggantikan v13. Ia
#   menjawab satu pertanyaan saja: "adakah backbone yang lebih baik dari
#   SigLIP2 untuk data ini?". Kalau jawabannya tidak, v13 jalan apa adanya
#   dan Anda sudah tahu itu bukan batu yang belum dibalik. Kalau ada, ganti
#   satu baris di PLANS v13.
#
# ANGKA ACUAN YANG SUDAH DIUKUR DI DATA INI (probe beku, popf1):
#   siglip2 @224 tanpa Otsu  0.9250   <- juara lama, ini yang harus dikalahkan
#   fuse-umum (3 backbone)   0.9373
#   mae-mss @224             0.8904
#   cnx-mss @224             0.8832
#   dit-* (dokumen)          0.8122 - 0.8574   <- hipotesis domain dokumen GAGAL
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

SMOKE = os.environ.get("INFEST_SMOKE", "0") == "1"
SEED  = 42

BUDGET = dict(
    total_h   = float(os.environ.get("INFEST_BUDGET_H", "2.0")),
    reserve_h = 0.15,
    safety    = 1.20,
)

# --- KANDIDAT ----------------------------------------------------------------
# Disusun berdasarkan apa yang SUDAH diukur di data ini, bukan daftar model
# populer. Tiga kelompok:
#
#  (1) SAPUAN RESOLUSI pada juara yang sudah diketahui. Ini kelompok paling
#      penting, karena v12 mengukur F1 blok 0.9533 vs strip 0.9932 dan test
#      49,2% blok - dugaan terkuat adalah resolusi, bukan arsitektur. Kalau
#      512 jauh mengalahkan 384, itu mengubah rencana v13.
#
#  (2) BACKBONE YANG BELUM PERNAH DIUJI di data ini. DINOv3 (Meta, 2025) dan
#      EVA-02 keduanya belum pernah dicoba; v11 hanya sempat menguji DINOv2.
#      ConvNeXt-V2, Hiera, BEiTv2 dan SwinV2 mewakili keluarga arsitektur yang
#      berbeda, dan keberagaman ARSITEKTUR adalah satu-satunya keberagaman yang
#      berguna untuk ensemble - pelajaran dari submission_iw.csv yang turun
#      karena menggabungkan tiga model se-backbone.
#
#  (3) JANGKAR. convnext_tiny.in12k_ft_in1k adalah backbone v6/v7 yang
#      skor LB-nya diketahui, dan siglip2-224 adalah juara lama. Keduanya ada
#      di sini supaya seluruh tabel punya titik nol yang bisa dipercaya -
#      tanpa jangkar, angka probe di run ini tidak bisa dibandingkan dgn
#      angka probe run sebelumnya.
#
# `prio` = urutan kerja. Gubernur mengerjakan prio kecil lebih dulu, jadi
# kalau anggaran habis yang hilang adalah kandidat spekulatif, bukan jangkar.
CANDIDATES = [
    # ---- (3) JANGKAR: wajib, supaya tabel ini sebanding dgn riwayat --------
    dict(tag="siglip2-224",  src="timm", prio=0, h=224, w=224,
         name="vit_base_patch16_siglip_224.v2_webli",
         note="JANGKAR - juara lama, probe v12 0.9250, fine-tune 0.9735 -> LB 0.87749"),
    dict(tag="cnx-v7-384",   src="timm", prio=0, h=384, w=384,
         name="convnext_tiny.in12k_ft_in1k",
         note="JANGKAR - backbone v6/v7, LB 0.839 diketahui"),
    # ---- (1) SAPUAN RESOLUSI pada juara -----------------------------------
    dict(tag="siglip2-256",  src="timm", prio=1, h=256, w=256,
         name="vit_base_patch16_siglip_256.v2_webli", note="sapuan resolusi"),
    dict(tag="siglip2-384",  src="timm", prio=1, h=384, w=384,
         name="vit_base_patch16_siglip_384.v2_webli",
         note="sapuan resolusi - INI yang dipakai v13, jadi angkanya langsung berguna"),
    dict(tag="siglip2-512",  src="timm", prio=1, h=512, w=512,
         name="vit_base_patch16_siglip_512.v2_webli",
         note="sapuan resolusi - 1025 token, terlalu mahal utk fine-tune tapi "
              "MURAH sebagai probe, jadi pertanyaan resolusi tetap terjawab"),
    dict(tag="siglip2L-256", src="timm", prio=3, h=256, w=256,
         name="vit_large_patch16_siglip_256.v2_webli",
         note="apakah UKURAN model membantu, terpisah dari resolusi"),
    dict(tag="siglip2L-384", src="timm", prio=4, h=384, w=384,
         name="vit_large_patch16_siglip_384.v2_webli",
         note="large + resolusi tinggi; kalau menang jauh, pertimbangkan sewa GPU lain"),
    # ---- (2) BELUM PERNAH DIUJI -------------------------------------------
    dict(tag="dinov3-b16",   src="timm", prio=1, h=384, w=384,
         name="vit_base_patch16_dinov3.lvd1689m",
         note="BARU - DINOv3 (Meta 2025), self-supervised 1,7 miliar citra. "
              "v11 hanya sempat menguji DINOv2. Fitur padat/rapat, cocok utk teks"),
    dict(tag="dinov3-cnx-b", src="timm", prio=2, h=384, w=384,
         name="convnext_base.dinov3_lvd1689m",
         note="BARU - DINOv3 dgn tulang ConvNeXt: konvolusional, bebas resolusi"),
    dict(tag="eva02-b448",   src="timm", prio=2, h=448, w=448,
         name="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
         note="BARU - EVA-02 base 448, salah satu ViT klasifikasi terkuat. "
              "patch 14 -> 448 habis dibagi 14, jangan diganti sembarangan"),
    dict(tag="cnxv2-b384",   src="timm", prio=2, h=384, w=384,
         name="convnextv2_base.fcmae_ft_in22k_in1k_384",
         note="BARU - ConvNeXt-V2 base, pra-latih FCMAE lalu IN22k"),
    dict(tag="dinov2-b14",   src="timm", prio=3, h=224, w=224,
         name="vit_base_patch14_reg4_dinov2.lvd142m",
         note="pembanding DINOv2 (v11 sudah mengukurnya). JEBAKAN: resolusi asli "
              "518 dan patch 14, jadi 224=16x14 masih habis dibagi. Jangan ganti "
              "ke angka yang tidak habis dibagi 14"),
    dict(tag="beitv2-b224",  src="timm", prio=3, h=224, w=224,
         name="beitv2_base_patch16_224.in1k_ft_in22k",
         note="BARU - BEiTv2. DiT adalah BEiT yang dilatih di dokumen dan ia "
              "GAGAL; entri ini memisahkan 'BEiT jelek' dari 'domain dokumen jelek'"),
    dict(tag="swinv2-b384",  src="timm", prio=3, h=384, w=384,
         name="swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
         note="BARU - jendela hierarkis; induktif bias berbeda dari ViT polos"),
    dict(tag="cnx-clip-384", src="timm", prio=3, h=384, w=384,
         name="convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384",
         note="BARU - ConvNeXt dilatih CLIP di LAION-2B lalu IN12k"),
    dict(tag="hiera-b224",   src="timm", prio=4, h=224, w=224,
         name="hiera_base_224.mae_in1k_ft_in1k",
         note="BARU - Hiera (Meta), MAE hierarkis tanpa komponen mahal"),
    # ---- HuggingFace: domain manuskrip ------------------------------------
    # Hipotesis domain dokumen sudah GAGAL sekali (DiT terlemah), tapi
    # mae-mss 0.8904 dan cnx-mss 0.8832 jauh lebih baik dari DiT, jadi
    # "domain manuskrip" belum tentu sama dgn "domain dokumen kantor".
    dict(tag="mae-mss",      src="hf",   prio=2, h=224, w=224,
         name="davanstrien/vit-manuscripts",
         note="ViT-MAE di manuskrip IIIF - v11 mengukur 0.8904"),
    dict(tag="vit-iiif",     src="hf",   prio=3, h=224, w=224,
         name="davanstrien/iiif_manuscript_vit",
         note="BARU - ViT lain dari koleksi IIIF yang sama, belum pernah diuji"),
    dict(tag="arab-mss",     src="hf",   prio=2, h=224, w=224,
         name="Ik45/arabic-manuscript-classifier",
         note="BARU - klasifikator manuskrip ARAB. Relevan khusus: jawi dan pegon "
              "keduanya aksara Arab dan itulah satu-satunya pasangan yang masih "
              "sering tertukar (pegon F1 0.9335). Backbone yang sudah melihat "
              "banyak manuskrip Arab mungkin memisahkannya lebih baik"),
]

PROBE   = dict(C=1.0, max_iter=3000)
PROBE_C = (0.05, 0.25, 1.0, 4.0, 16.0)     # STEP 3
FUSE    = dict(enable=True, max_k=4, l2norm=True)
# STEP 5: fine-tune pendek. Anggarannya SENGAJA kecil - tujuannya bukan
# mengalahkan v13, melainkan menguji apakah PERINGKAT probe berpindah ke
# fine-tune. Karena itu juara baru dan siglip2-224 dapat anggaran yang SAMA.
CONFIRM = dict(enable=True, epochs=6, folds=2, lr_body=3e-5, lr_head=1e-3,
               layer_decay=0.75)

CFG = dict(n_folds=5, batch=32, nw=2, wd=0.05, ls=0.05)
MAXSIDE, CACHE_MAX = 1600, 1280
CACHE  = dict(enable=True)
AR_SPLIT = 3.0
REF = dict(probe_siglip2=0.9250, probe_fuse=0.9373, probe_dit=0.8122,
           ft_siglip2=0.9735, lb_v12=0.87749, top5=0.88976)

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
    print(f"GPU: {_p.name} | VRAM {_p.total_memory/2**30:.1f} GB")
    # T4 = Turing: bf16 tidak didukung. Semua autocast di sini float16.
print(f"anggaran: {BUDGET['total_h']:.1f} jam")

# =============================================================================
# 1. DATA  (identik dgn v12/v13 - fold WAJIB sama supaya angkanya sebanding)
# =============================================================================
SEARCH = [p for p in ["/kaggle/input", "/kaggle/working", ".", "./data", "/content"]
          if os.path.isdir(p)]
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
FDIR = os.path.join(WORK, "_featauto"); os.makedirs(FDIR, exist_ok=True)

def find_csv(names):
    for root in SEARCH:
        for dp, _, fs in os.walk(root, followlinks=True):
            for f in fs:
                if f.lower() in names: return os.path.join(dp, f)
    return None

train_csv = find_csv({"train.csv"}); sub_csv = find_csv({"sample_submission.csv"})
test_csv  = find_csv({"test.csv"})
assert train_csv, "train.csv tidak ketemu"
tr_df = pd.read_csv(train_csv)
# `test.csv` di repo ini berisi 1.118 id test LAMA yang irisannya dgn
# sample_submission (1.220 id) NOL. Daftar test SELALU dari sample_submission.
if sub_csv:
    te_df = pd.DataFrame({"image_id": pd.read_csv(sub_csv).image_id.tolist()})
elif test_csv:
    te_df = pd.read_csv(test_csv)[["image_id"]]
    print("  PERHATIAN: sample_submission.csv tidak ketemu, memakai test.csv.")
else:
    raise AssertionError("sample_submission.csv maupun test.csv tidak ketemu")

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
                except Exception as e: print(f"  ! gagal ekstrak: {type(e).__name__}")
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
if te_df.path.isna().any() or tr_df.path.isna().all():
    raise AssertionError(f"gambar tidak ketemu (train {tr_df.path.isna().sum()} / "
                         f"test {te_df.path.isna().sum()}); folder: {SEARCH}")
tr_df = tr_df[tr_df.path.notna()].reset_index(drop=True)

CLASSES = sorted(tr_df.label.unique()); C2I = {c: i for i, c in enumerate(CLASSES)}
tr_df["y"] = tr_df.label.map(C2I); NC = len(CLASSES)
cnt = np.bincount(tr_df.y, minlength=NC); ytrue = tr_df.y.values
print(f"train {len(tr_df)} baris | test {len(te_df)} baris | {NC} kelas: {CLASSES}")

def _ar(paths):
    out = []
    for q in paths:
        try:
            with Image.open(q) as im: out.append(im.size[0] / max(im.size[1], 1))
        except Exception: out.append(np.nan)
    return np.array(out, float)

AR_TR = _ar(tr_df.path.tolist())
AR_TE = _ar([p for p in te_df.path.tolist() if isinstance(p, str)])
BLOK_TR = AR_TR < AR_SPLIT
_ok = ~np.isnan(AR_TE)
# perbandingan dgn NaN meruntuhkannya jadi False lebih dulu, sehingga gambar
# yang gagal dibaca akan terhitung 'strip'. W_BLOK ada di balik SETIAP angka
# di skrip ini, jadi NaN dibuang eksplisit.
W_BLOK = float((AR_TE[_ok] < AR_SPLIT).mean())
print(f"populasi test: blok {W_BLOK*100:.1f}%  strip {(1-W_BLOK)*100:.1f}%  <- bobot penilaian")

def popf1(p, mask=None):
    """macro-F1 ditimbang ke komposisi strip/blok TEST. Satu-satunya angka yang
    memutuskan apa pun di sini - OOF polos-lah yang dulu membuat v3.1 terlihat
    0.9716 padahal LB-nya 0.8073."""
    m0 = np.ones(len(ytrue), bool) if mask is None else mask
    out = 0.0
    for sub, w in ((BLOK_TR, W_BLOK), (~BLOK_TR, 1 - W_BLOK)):
        m = m0 & sub
        if m.sum() >= NC:
            out += w * f1_score(ytrue[m], p[m].argmax(1), average="macro")
    return out

FOLDS = list(StratifiedKFold(CFG["n_folds"], shuffle=True,
                             random_state=SEED).split(tr_df, tr_df.y))
print(f"{CFG['n_folds']} fold, random_state={SEED} - SAMA dgn v12/v13, jadi angka")
print(f"probe di bawah sebanding langsung dgn acuan siglip2 {REF['probe_siglip2']:.4f}.")

# --- cache gambar: dekode sekali, dipakai semua kandidat ---------------------
# Otsu/CANON tidak ada di sini sama sekali: dua uji terkontrol independen
# mengukurnya merugikan (-0.0131 di DiT, -0.0091 di SigLIP2).
CDIR = os.path.join(WORK, "_cache13"); os.makedirs(CDIR, exist_ok=True)
def _cp(path): return os.path.join(CDIR, hashlib.md5(path.encode()).hexdigest() + ".png")
def load_img(path):
    if CACHE["enable"]:
        c = _cp(path)
        if os.path.exists(c):
            try: return Image.open(c).convert("L")
            except Exception: pass
    try:
        im = Image.open(path); im.draft("L", (CACHE_MAX, CACHE_MAX)); g = im.convert("L")
    except Exception:
        return Image.new("L", (64, 64), 255)
    if max(g.size) > CACHE_MAX: g.thumbnail((CACHE_MAX, CACHE_MAX), Image.BILINEAR)
    if CACHE["enable"]:
        try: g.save(_cp(path), "PNG", optimize=False)
        except Exception: pass
    return g

print("\nmenyiapkan cache gambar (dipakai ulang oleh SEMUA kandidat)...")
_t0, _n = time.time(), 0
for _p in tr_df.path.tolist() + [p for p in te_df.path.tolist() if isinstance(p, str)]:
    if not os.path.exists(_cp(_p)): load_img(_p); _n += 1
print(f"  {_n} gambar didekode dalam {time.time()-_t0:.0f}s | {clk()}")

# =============================================================================
# 2. VIEW + DATASET  (hanya inferensi: tanpa augmentasi, deterministik)
# =============================================================================
MEAN_DEF, STD_DEF = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def letterbox(g, H, W):
    w, h = g.size
    s = min(H / max(h, 1), W / max(w, 1))
    nw, nh = max(1, min(W, int(round(w*s)))), max(1, min(H, int(round(h*s))))
    g = g.resize((nw, nh), Image.BILINEAR)
    c = Image.new("L", (W, H), 255); c.paste(g, ((W-nw)//2, (H-nh)//2)); return c
TO_T = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

class EvalDS(Dataset):
    def __init__(s, df, r):
        s.p = df.path.tolist(); s.r = r
        s.norm = T.Normalize(r.get("mean", MEAN_DEF), r.get("std", STD_DEF))
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        g = load_img(s.p[i]) if isinstance(s.p[i], str) else Image.new("L", (64, 64), 255)
        x = TO_T(letterbox(g, s.r["h"], s.r["w"]))
        if x.shape[0] == 1: x = x.repeat(3, 1, 1)
        return s.norm(x), 0

# =============================================================================
# 3. BACKBONE: timm + HuggingFace, TANPA substitusi senyap
# =============================================================================
# v10 `build()` diam-diam jatuh ke resnet34 kalau nama backbone salah, sehingga
# run yang seolah 'dinov2' sebenarnya resnet34 - dan handoff mencatat jebakan
# itu pernah kejadian. Di sini kandidat yang gagal DILEWATI dgn pesan jelas dan
# dicatat di tabel sebagai gagal. Lebih baik kehilangan satu baris daripada
# mengukur model yang salah dan mempercayainya.
def hf_pool(out):
    po = getattr(out, "pooler_output", None)
    if po is not None and getattr(po, "ndim", 0) == 2: return po
    h = out.last_hidden_state
    if h.ndim == 4: return h.mean((2, 3))          # ConvNeXt: B,C,H,W
    if h.shape[1] > 1: return h[:, 1:].mean(1)     # ViT/BEiT: buang CLS
    return h.mean(1)

class HFBody(nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
        try: s.ipe = "interpolate_pos_encoding" in inspect.signature(m.forward).parameters
        except Exception: s.ipe = False
    def forward(s, x):
        return hf_pool(s.m(pixel_values=x, interpolate_pos_encoding=True)) if s.ipe \
               else hf_pool(s.m(pixel_values=x))

def load_body(c):
    """kembalikan (body, dim, mean, std) atau None kalau kandidat gagal."""
    try:
        if c["src"] == "timm":
            import timm
            def _mk(pre):
                # Banyak ViT punya resolusi asli TETAP (dinov2 518, siglip 224,
                # eva02 448). Tanpa img_size timm melempar "Input height doesn't
                # match model"; dengan img_size ia menginterpolasi pos-embed.
                # Backbone konvolusional menolak argumennya -> fallback.
                try:
                    return timm.create_model(c["name"], pretrained=pre, num_classes=0,
                                             img_size=(c["h"], c["w"]))
                except TypeError:
                    return timm.create_model(c["name"], pretrained=pre, num_classes=0)
            try: m = _mk(True)
            except Exception as e:
                if not SMOKE: raise
                print(f"      (SMOKE) unduhan gagal {type(e).__name__} -> bobot ACAK")
                m = _mk(False)
            mean, std = MEAN_DEF, STD_DEF
            try:
                from timm.data import resolve_data_config
                dc = resolve_data_config({}, model=m)
                mean, std = list(dc.get("mean", mean)), list(dc.get("std", std))
            except Exception: pass
            return m, int(m.num_features), mean, std
        from transformers import AutoModel, AutoConfig
        kw = {}
        cf = AutoConfig.from_pretrained(c["name"])
        # ViT-MAE default-nya menutup 75% patch. mask_ratio=0 menyimpan SEMUA
        # patch; urutannya diacak tapi pos-embed sudah ditambahkan sebelum
        # pengacakan dan kita mean-pool, jadi hasilnya tetap deterministik.
        if getattr(cf, "model_type", "") == "vit_mae": kw["mask_ratio"] = 0.0
        m = AutoModel.from_pretrained(c["name"], **kw)
        mean, std = MEAN_DEF, STD_DEF
        try:
            from transformers import AutoImageProcessor
            ip = AutoImageProcessor.from_pretrained(c["name"])
            if getattr(ip, "image_mean", None): mean = list(ip.image_mean)
            if getattr(ip, "image_std", None):  std = list(ip.image_std)
        except Exception: pass
        dim = int(getattr(m.config, "hidden_size", 0) or
                  (m.config.hidden_sizes[-1] if getattr(m.config, "hidden_sizes", None) else 0))
        if dim <= 0: raise RuntimeError("dimensi fitur tidak terbaca dari config")
        return HFBody(m), dim, mean, std
    except Exception as e:
        print(f"      !! DILEWATI: {c['src']}:{c['name']} -> {type(e).__name__}: {e}")
        return None

# =============================================================================
# 4. EKSTRAKSI FITUR BEKU  (+ guard OOM + cache ke disk)
# =============================================================================
def _is_oom(e):
    return isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) \
           or ("out of memory" in str(e).lower())

@torch.no_grad()
def extract(body, df, r, bs):
    out = []
    for x, _ in DataLoader(EvalDS(df, r), batch_size=bs, shuffle=False,
                           num_workers=CFG["nw"]):
        x = x.to(DEV, non_blocking=True)
        with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
            out.append(body(x).float().cpu().numpy())
    return np.concatenate(out)

def feats_for(c):
    """(Ftr, Fte) untuk satu kandidat, dari cache kalau ada.
    Cache membuat STEP 3 dan STEP 4 gratis, dan membuat run yang diulang
    tidak membayar ulang lintasan majunya."""
    fp = os.path.join(FDIR, f"{c['tag']}_{c['h']}x{c['w']}.npz")
    if os.path.exists(fp):
        try:
            z = np.load(fp)
            if len(z["tr"]) == len(tr_df) and len(z["te"]) == len(te_df):
                return z["tr"], z["te"], True
            print(f"      cache ukurannya tidak cocok -> diekstrak ulang")
        except Exception as e:
            print(f"      cache rusak ({type(e).__name__}) -> diekstrak ulang")
    got = load_body(c)
    if got is None: return None, None, False
    body, dim, mean, std = got
    r = dict(h=c["h"], w=c["w"], mean=mean, std=std)
    body = body.to(DEV).eval()
    bs = CFG["batch"]
    while True:
        try:
            ftr = extract(body, tr_df, r, bs); fte = extract(body, te_df, r, bs)
            break
        except Exception as e:
            if not _is_oom(e) or bs <= 2:
                del body
                if DEV == "cuda": torch.cuda.empty_cache()
                print(f"      !! ekstraksi gagal: {type(e).__name__}: {e}")
                return None, None, False
            if DEV == "cuda": torch.cuda.empty_cache()
            bs //= 2; print(f"      OOM -> batch {bs}")
    del body
    if DEV == "cuda": torch.cuda.empty_cache()
    try: np.savez_compressed(fp, tr=ftr.astype(np.float32), te=fte.astype(np.float32))
    except Exception as e: print(f"      ! gagal cache fitur: {type(e).__name__}")
    return ftr, fte, False

def l2(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)

def probe(ftr, C=PROBE["C"], norm=True):
    """regresi logistik 5-fold di atas fitur beku -> (oof_prob, popf1).
    `class_weight='balanced'` dipakai karena metriknya macro-F1: tanpa itu
    kelas kecil dikorbankan dan angkanya tidak menjawab pertanyaan lomba."""
    X = l2(ftr) if norm else ftr
    oof = np.zeros((len(X), NC))
    for itr, iva in FOLDS:
        lr = LogisticRegression(C=C, max_iter=PROBE["max_iter"],
                                class_weight="balanced", n_jobs=-1)
        lr.fit(X[itr], ytrue[itr])
        p = lr.predict_proba(X[iva])
        # kelas yang absen di fold latih tidak muncul di predict_proba
        for j, cl in enumerate(lr.classes_): oof[iva, cl] = p[:, j]
    return oof, popf1(oof)

# =============================================================================
# STEP 2 - SWEEP KANDIDAT
# =============================================================================
print("\n" + "=" * 74)
print("STEP 2 - SWEEP KANDIDAT (probe beku)")
print("=" * 74)
print("Tiap baris = satu lintasan maju atas 3.771 + 1.220 gambar lalu regresi")
print("logistik 5-fold. Tidak ada backbone yang dilatih. Urutan kerja mengikuti")
print("`prio`: jangkar dulu, kandidat spekulatif belakangan, supaya kalau")
print("anggaran habis yang hilang adalah bagian yang paling tidak penting.\n")

RES, FEAT = OrderedDict(), OrderedDict()
_rate = None          # detik per (gambar x megapiksel), diukur, bukan ditebak

def est_sec(c):
    n = len(tr_df) + len(te_df)
    mp = (c["h"] * c["w"]) / (224.0 * 224.0)
    mult = 3.2 if ("large" in c["name"] or "_large" in c["tag"].lower()
                   or "L-" in c["tag"]) else 1.0
    base = _rate if _rate is not None else 0.004      # tebakan awal utk T4
    return n * mp * base * mult + 45.0                # +45s jatah unduh bobot

for c in sorted(CANDIDATES, key=lambda d: (d["prio"], d["tag"])):
    need = est_sec(c) / 3600.0
    if need * BUDGET["safety"] > rem_h() - BUDGET["reserve_h"]:
        print(f"  {c['tag']:14s} DILEWATI - perkiraan {need*60:.0f} menit, sisa aman "
              f"{(rem_h()-BUDGET['reserve_h'])*60:.0f} menit")
        RES[c["tag"]] = dict(tag=c["tag"], name=c["name"], h=c["h"], w=c["w"],
                             popf1=float("nan"), dim=0, sec=0.0,
                             status="dilewati (anggaran)", note=c["note"])
        continue
    print(f"  {c['tag']:14s} {c['name'][:52]:52s} @{c['h']}")
    t0 = time.time()
    ftr, fte, cached = feats_for(c)
    if ftr is None:
        RES[c["tag"]] = dict(tag=c["tag"], name=c["name"], h=c["h"], w=c["w"],
                             popf1=float("nan"), dim=0, sec=time.time()-t0,
                             status="GAGAL dimuat", note=c["note"])
        continue
    dt = time.time() - t0
    if not cached and dt > 5:
        mp = (c["h"]*c["w"])/(224.0*224.0)
        mult = 3.2 if ("large" in c["name"] or "L-" in c["tag"]) else 1.0
        r_new = max(dt - 45.0, 1.0) / ((len(tr_df)+len(te_df)) * mp * mult)
        _rate = r_new if _rate is None else 0.5*(_rate + r_new)
    oof, sc = probe(ftr)
    FEAT[c["tag"]] = (ftr, fte)
    RES[c["tag"]] = dict(tag=c["tag"], name=c["name"], h=c["h"], w=c["w"],
                         popf1=float(sc), dim=int(ftr.shape[1]),
                         sec=float(dt), status="ok" + (" (cache)" if cached else ""),
                         note=c["note"])
    flag = ""
    if sc > REF["probe_siglip2"] + 0.002: flag = "  <<< MENGALAHKAN juara lama"
    elif sc < REF["probe_siglip2"] - 0.02: flag = "  (jauh di bawah)"
    print(f"      popf1 = {sc:.4f}   dim={ftr.shape[1]:4d}  {dt:5.0f}s{flag}")
    # ditulis tiap baris: sesi yang mati tetap meninggalkan tabel yang berguna
    try:
        pd.DataFrame(list(RES.values())).sort_values(
            "popf1", ascending=False).to_csv(os.path.join(WORK, "automl_hasil.csv"),
                                             index=False)
    except Exception: pass
    print(f"      {clk()}")

OK = {t: r for t, r in RES.items() if r["status"].startswith("ok")}
if not OK:
    raise RuntimeError("Tidak ada satu kandidat pun yang berhasil. "
                       "Periksa koneksi ke Hugging Face dan pesan kegagalan di atas.")

# =============================================================================
# STEP 3 - HIPERPARAMETER PROBE  (CPU, di atas fitur yang sudah di-cache)
# =============================================================================
# Gratis dibanding STEP 2: tidak ada satu pun lintasan maju. Yang dicari cuma
# dua knob yang benar-benar berpengaruh pada probe linier - kekuatan
# regularisasi dan normalisasi L2. Ini juga pemeriksaan kewarasan: kalau
# peringkat kandidat BERUBAH total saat C diubah, berarti peringkat STEP 2
# adalah artefak setelan probe dan tidak boleh dipercaya.
print("\n" + "=" * 74)
print("STEP 3 - HIPERPARAMETER PROBE (di fitur cache, tanpa lintasan maju baru)")
print("=" * 74)
TOPK = [t for t, _ in sorted(OK.items(), key=lambda kv: -kv[1]["popf1"])][:5]
print(f"  dicari untuk 5 kandidat teratas: {TOPK}\n")
print(f"  {'kandidat':14s}{'C=0.05':>9s}{'C=0.25':>9s}{'C=1':>9s}{'C=4':>9s}"
      f"{'C=16':>9s}{'tanpa L2':>10s}{'terbaik':>10s}")
BEST_HP = {}
for t in TOPK:
    if t not in FEAT: continue
    ftr = FEAT[t][0]; row, best = [], (PROBE["C"], True, -1)
    for C in PROBE_C:
        _, s = probe(ftr, C=C, norm=True); row.append(s)
        if s > best[2]: best = (C, True, s)
    _, s_raw = probe(ftr, C=best[0], norm=False)
    if s_raw > best[2]: best = (best[0], False, s_raw)
    BEST_HP[t] = best
    print(f"  {t:14s}" + "".join(f"{v:9.4f}" for v in row)
          + f"{s_raw:10.4f}{best[2]:10.4f}")
    OK[t]["popf1_tuned"] = float(best[2]); OK[t]["C"] = best[0]; OK[t]["l2"] = best[1]
_r2 = [t for t, _ in sorted(((t, OK[t].get("popf1_tuned", OK[t]["popf1"])) for t in TOPK),
                            key=lambda kv: -kv[1])]
print(f"\n  urutan sebelum tuning : {TOPK}")
print(f"  urutan setelah tuning : {_r2}")
if _r2[0] != TOPK[0]:
    print("  !! JUARANYA BERUBAH setelah tuning. Artinya peringkat STEP 2 sebagian")
    print("     adalah artefak setelan probe. Pakai urutan setelah tuning, dan")
    print("     baca selisih kecil di tabel mana pun dgn curiga.")
else:
    print("  -> juaranya sama. Peringkat STEP 2 tahan terhadap setelan probe.")

# =============================================================================
# STEP 4 - PENCARIAN FUSI  (menggabungkan beberapa model, versi termurah)
# =============================================================================
# Konkatenasi embedding beberapa backbone lalu SATU probe. Tiap blok
# di-L2-normalkan dulu supaya model dgn norma fitur besar tidak otomatis
# mendominasi hanya karena skalanya.
#
# PELAJARAN dari submission_iw.csv yang TURUN 0.02054: menggabungkan model
# hanya berguna kalau kesalahan mereka TIDAK berkorelasi. Tiga model dgn
# backbone SigLIP2 yang sama bukan keberagaman. Karena itu pencarian di sini
# greedy dan DIGERBANG - anggota hanya masuk kalau ia benar-benar menaikkan
# popf1 - dan tabel di bawah mencetak berapa banyak backbone yang BERBEDA
# KELUARGA ada di dalam pilihan akhir.
FUSED = None
if FUSE["enable"] and len(FEAT) >= 2:
    print("\n" + "=" * 74)
    print("STEP 4 - PENCARIAN FUSI (greedy, digerbang)")
    print("=" * 74)
    pool = [t for t, _ in sorted(OK.items(), key=lambda kv: -kv[1]["popf1"])
            if t in FEAT][:8]
    print(f"  kolam kandidat: {pool}")
    cur, best = [], -1
    while len(cur) < FUSE["max_k"]:
        cand, cs = None, best
        for t in pool:
            if t in cur: continue
            X = np.hstack([l2(FEAT[q][0]) for q in cur + [t]])
            _, s = probe(X, norm=False)
            if s > cs: cand, cs = t, s
        if cand is None: break
        cur.append(cand); best = cs
        print(f"    + {cand:14s} -> popf1 {best:.4f}  ({len(cur)} backbone)")
    if len(cur) < 2:
        # Greedy berhenti di satu anggota: tidak ada backbone kedua yang
        # MENAMBAH apa pun. Itu jawaban yang sah dan sejalan dgn v12 -
        # model tunggal mengalahkan ensemble di leaderboard.
        print(f"\n  greedy berhenti di SATU anggota ({cur or 'tidak ada'}).")
        print("  Tidak ada backbone kedua yang menambah informasi. Jangan paksakan")
        print("  ensemble: itu persis kesalahan submission_iw.csv.")
        FUSED = None
    else:
        FUSED = (cur, best)
        fam = {t.split("-")[0].rstrip("0123456789L") for t in cur}
        print(f"\n  fusi terpilih: {cur}")
        print(f"  popf1 = {best:.4f}   (acuan fuse-umum v11 = {REF['probe_fuse']:.4f},")
        print(f"                       backbone tunggal terbaik di run ini = "
              f"{max(r['popf1'] for r in OK.values()):.4f})")
        print(f"  keluarga arsitektur berbeda di dalamnya: {len(fam)} ({sorted(fam)})")
        if len(fam) < 2:
            print("  !! SEMUANYA satu keluarga. Itu persis bentuk ensemble yang")
            print("     menjatuhkan submission_iw.csv. Jangan pakai fusi ini.")

# =============================================================================
# STEP 5 - KONFIRMASI: apakah peringkat probe BERPINDAH ke fine-tune?
# =============================================================================
# Ini bagian yang membuat seluruh skrip jujur. Probe beku mengukur "seberapa
# terpisah kelas-kelas ini secara LINIER di ruang embedding backbone" -
# berkorelasi dgn hasil fine-tune, tapi bukan hal yang sama.
#
# Karena itu: juara baru DAN siglip2-224 dilatih dgn anggaran yang SAMA PERSIS
# (epoch, fold, lr, layer-decay, augmentasi sama). Selisihnya adalah jawaban.
# Kalau anggaran tidak cukup, skrip mengatakannya - bukan diam-diam melewati.
CW = torch.tensor(cnt.sum() / (NC * cnt), dtype=torch.float32, device=DEV)
SEV = 1.0
DEGRADE = dict(p_blur=0.70, blur=(0.6, 2.2), p_contrast=0.60, contrast=(0.45, 0.95),
               p_rescale=0.50, rescale=(0.35, 0.80), p_jpeg=0.50, jpeg=(25, 88))
ARJIT = dict(lo=0.65, hi=1.55)
AUG_GEO = T.Compose([
    T.RandomApply([T.RandomRotation(4, fill=255)], p=0.5),
    T.RandomApply([T.RandomAffine(0, translate=(0.02, 0.05), shear=4, fill=255)], p=0.3),
    T.ColorJitter(brightness=0.30, contrast=0.30)])
ERASE = T.RandomErasing(p=0.25, scale=(0.01, 0.06))
class AddNoise(nn.Module):
    def __init__(s, p=0.3, sd=0.04): super().__init__(); s.p, s.sd = p, sd
    def forward(s, x):
        return (x + torch.randn_like(x)*s.sd).clamp(0, 1) if random.random() < s.p else x
NOISE = AddNoise()

def degrade(g):
    d = DEGRADE
    if random.random() < d["p_blur"]:
        g = g.filter(ImageFilter.GaussianBlur(random.uniform(*d["blur"])))
    if random.random() < d["p_contrast"]:
        g = ImageEnhance.Contrast(g).enhance(random.uniform(*d["contrast"]))
    if random.random() < d["p_rescale"]:
        w, h = g.size; f = random.uniform(*d["rescale"])
        g = g.resize((max(8, int(w*f)), max(8, int(h*f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if random.random() < d["p_jpeg"]:
        b = io.BytesIO(); g.convert("L").save(b, "JPEG", quality=random.randint(*d["jpeg"]))
        b.seek(0); g = Image.open(b).convert("L")
    return g

def lb_train(g, H, W):
    w, h = g.size
    f = random.uniform(ARJIT["lo"], ARJIT["hi"])
    w = max(1, int(round(w*f))); g = g.resize((w, h), Image.BILINEAR)
    s = min(H/max(h, 1), W/max(w, 1)) * random.uniform(0.92, 1.08)
    nw, nh = max(1, min(W, int(round(w*s)))), max(1, min(H, int(round(h*s))))
    g = g.resize((nw, nh), Image.BILINEAR)
    c = Image.new("L", (W, H), 255)
    c.paste(g, (random.randint(0, W-nw), random.randint(0, H-nh))); return c

class TrainDS(Dataset):
    def __init__(s, df, r, train):
        s.p = df.path.tolist(); s.y = df.y.tolist(); s.r, s.t = r, train
        s.norm = T.Normalize(r.get("mean", MEAN_DEF), r.get("std", STD_DEF))
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        g = load_img(s.p[i]) if isinstance(s.p[i], str) else Image.new("L", (64, 64), 255)
        g = lb_train(degrade(g), s.r["h"], s.r["w"]) if s.t else letterbox(g, s.r["h"], s.r["w"])
        if s.t: g = AUG_GEO(g)
        x = TO_T(g)
        if x.shape[0] == 1: x = x.repeat(3, 1, 1)
        if s.t: x = NOISE(x)
        x = s.norm(x)
        if s.t: x = ERASE(x)
        return x, s.y[i]

class Net(nn.Module):
    def __init__(s, body, feat, nc):
        super().__init__(); s.body = body; s.head = nn.Linear(feat, nc)
    def forward(s, x): return s.head(s.body(x))

def _depth_of(name, nmax):
    parts = name.split(".")
    for k, tok in enumerate(parts):
        if tok in ("embeddings", "patch_embed", "stem", "cls_token", "pos_embed"): return 0
        if tok in ("layer", "layers", "block", "blocks", "stage", "stages"):
            for q in parts[k+1:k+3]:
                if q.isdigit(): return 1 + int(q)
    return nmax

def param_groups(model, lr, wd, decay):
    """AdamW TIDAK membaca `lr_scale` yang dikembalikan helper timm, dan
    OneCycleLR lalu menimpa `lr` tiap grup - jadi layer-wise decay lewat timm
    adalah no-op. Di sini skalanya dikalikan ke `lr` secara eksplisit."""
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
    b = {}
    for n, p in ps:
        b.setdefault((min(_depth_of(n, nmax), nmax), (p.ndim <= 1 or n.endswith(".bias"))), []).append(p)
    return [{"params": pr, "weight_decay": (0.0 if nd else wd), "lr": lr*(decay**(nmax-d))}
            for (d, nd), pr in sorted(b.items())]

def short_ft(c, micro):
    """fine-tune PENDEK, anggaran identik untuk semua kandidat yang diuji."""
    accum = max(1, 32 // micro)
    f1s = []
    for fold in range(CONFIRM["folds"]):
        itr, iva = FOLDS[fold]
        dtr = tr_df.iloc[itr].reset_index(drop=True)
        dva = tr_df.iloc[iva].reset_index(drop=True)
        got = load_body(c)
        if got is None: return None
        body, dim, mean, std = got
        r = dict(h=c["h"], w=c["w"], mean=mean, std=std)
        seed_all(SEED + fold)
        model = Net(body, dim, NC).to(DEV)
        ld = DataLoader(TrainDS(dtr, r, True), batch_size=micro, shuffle=True,
                        num_workers=CFG["nw"], drop_last=True, pin_memory=(DEV == "cuda"))
        crit = nn.CrossEntropyLoss(weight=CW, label_smoothing=CFG["ls"])
        pg = param_groups(model.body, CONFIRM["lr_body"], CFG["wd"], CONFIRM["layer_decay"])
        pg += [{"params": list(model.head.parameters()), "weight_decay": 0.0,
                "lr": CONFIRM["lr_head"]}]
        opt = torch.optim.AdamW(pg, lr=CONFIRM["lr_body"], weight_decay=CFG["wd"])
        steps = max(10, CONFIRM["epochs"] * max(1, math.ceil(len(ld)/accum)))
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"]*2 for g in opt.param_groups], total_steps=steps, pct_start=0.25)
        sc = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
        try:
            for ep in range(CONFIRM["epochs"]):
                model.train(); opt.zero_grad(set_to_none=True)
                for bi, (x, y) in enumerate(ld):
                    x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
                    with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                        loss = crit(model(x), y) / accum
                    sc.scale(loss).backward()
                    if (bi+1) % accum == 0 or (bi+1) == len(ld):
                        sc.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        sc.step(opt); sc.update(); opt.zero_grad(set_to_none=True)
                        if sch.last_epoch < steps-1: sch.step()
        except Exception as e:
            del model
            if DEV == "cuda": torch.cuda.empty_cache()
            print(f"      !! latih gagal: {type(e).__name__}: {e}")
            return None
        model.eval(); pr = []
        with torch.no_grad():
            for x, _ in DataLoader(TrainDS(dva, r, False), batch_size=micro*2,
                                   shuffle=False, num_workers=CFG["nw"]):
                with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                    pr.append(model(x.to(DEV)).float().argmax(1).cpu().numpy())
        f1s.append(f1_score(dva.y, np.concatenate(pr), average="macro"))
        del model
        if DEV == "cuda": torch.cuda.empty_cache()
        print(f"      fold{fold} macro-F1 = {f1s[-1]:.4f}")
    return float(np.mean(f1s))

print("\n" + "=" * 74)
print("STEP 5 - KONFIRMASI: apakah peringkat probe berpindah ke fine-tune?")
print("=" * 74)
RANK = [t for t, _ in sorted(OK.items(),
        key=lambda kv: -(kv[1].get("popf1_tuned", kv[1]["popf1"])))]
CHAMP = RANK[0]
CONF = {}
if not CONFIRM["enable"]:
    print("  dimatikan lewat CONFIRM['enable'].")
elif CHAMP == "siglip2-224":
    print("  Juara probe adalah siglip2-224, yaitu backbone yang SUDAH")
    print("  di-fine-tune di v12 dan skor leaderboard-nya diketahui (0.87749).")
    print("  Tidak ada yang perlu dikonfirmasi: pakai v13 apa adanya.")
else:
    pick = [c for c in CANDIDATES if c["tag"] in (CHAMP, "siglip2-224")]
    _cost = 0.0
    for c in pick:
        mp = (c["h"]*c["w"])/(224.0*224.0)
        mult = 3.2 if ("large" in c["name"] or "L-" in c["tag"]) else 1.0
        _cost += CONFIRM["folds"] * CONFIRM["epochs"] * len(tr_df)*0.8 * \
                 (_rate or 0.004) * 3.0 * mp * mult
    print(f"  perkiraan biaya {_cost/3600:.2f} jam untuk 2 kandidat x "
          f"{CONFIRM['folds']} fold x {CONFIRM['epochs']} epoch")
    if _cost/3600 * BUDGET["safety"] > rem_h() - BUDGET["reserve_h"]:
        print(f"  -> DILEWATI, sisa aman cuma {(rem_h()-BUDGET['reserve_h']):.2f} jam.")
        print("     ARTINYA: peringkat di bawah BELUM diuji pada fine-tune. Ia")
        print("     tetap sinyal terbaik yang ada - dan sinyal yang sama pernah")
        print("     benar sekali di data ini - tapi jangan sebut ia terbukti.")
        print("     Untuk mengujinya, jalankan ulang skrip ini dgn "
              "INFEST_BUDGET_H lebih besar;")
        print("     fitur sudah di-cache jadi STEP 2 tidak dibayar dua kali.")
    else:
        _micro = 16 if max(c["h"] for c in pick) >= 384 else 32
        for c in pick:
            print(f"\n    fine-tune pendek: {c['tag']} @{c['h']}")
            v = short_ft(c, _micro)
            if v is not None:
                CONF[c["tag"]] = v
                print(f"      rata-rata {CONFIRM['folds']} fold = {v:.4f}")
        if len(CONF) == 2:
            a, b = CONF.get(CHAMP), CONF.get("siglip2-224")
            print(f"\n  [K] {CHAMP} {a:.4f}  vs  siglip2-224 {b:.4f}  ({a-b:+.4f})")
            pa = OK[CHAMP].get("popf1_tuned", OK[CHAMP]["popf1"])
            pb = OK["siglip2-224"].get("popf1_tuned", OK["siglip2-224"]["popf1"])
            print(f"      probe memprediksi selisih {pa-pb:+.4f}")
            if (a-b) * (pa-pb) > 0:
                print("      -> TANDANYA SAMA: peringkat probe berpindah ke fine-tune.")
                print("         Ganti backbone di PLANS v13.")
            else:
                print("      !! TANDANYA BERLAWANAN: peringkat probe TIDAK berpindah.")
                print("         Jangan ganti backbone v13 berdasarkan tabel probe saja.")

# =============================================================================
# STEP 6 - TABEL AKHIR + KONFIGURASI SIAP TEMPEL
# =============================================================================
print("\n" + "=" * 74)
print("STEP 6 - HASIL")
print("=" * 74)
rows = sorted(RES.values(), key=lambda r: (-(r.get("popf1_tuned", r["popf1"])
              if not np.isnan(r["popf1"]) else -9), r["tag"]))
print(f"\n{'kandidat':15s}{'res':>6s}{'dim':>6s}{'popf1':>9s}{'+tuned':>9s}"
      f"{'detik':>8s}  status")
print("-" * 74)
for r in rows:
    p  = "   nan" if np.isnan(r["popf1"]) else f"{r['popf1']:9.4f}"
    pt = f"{r['popf1_tuned']:9.4f}" if r.get("popf1_tuned") else "        -"
    print(f"{r['tag']:15s}{r['h']:>6d}{r['dim']:>6d}{p}{pt}{r['sec']:8.0f}  {r['status']}")
print("-" * 74)
print(f"acuan terukur: siglip2 @224 probe {REF['probe_siglip2']:.4f} -> fine-tune "
      f"{REF['ft_siglip2']:.4f} -> LB {REF['lb_v12']:.5f}")
print(f"               fusi 3-backbone v11 {REF['probe_fuse']:.4f} | "
      f"DiT (domain dokumen) {REF['probe_dit']:.4f}")

best_tag = RANK[0]
best_sc  = OK[best_tag].get("popf1_tuned", OK[best_tag]["popf1"])
anchor   = OK.get("siglip2-224", {}).get("popf1_tuned",
           OK.get("siglip2-224", {}).get("popf1", REF["probe_siglip2"]))
print(f"\n[J] JUARA: {best_tag}  popf1 {best_sc:.4f}")
print(f"    jangkar siglip2-224 di run INI: {anchor:.4f}  (selisih {best_sc-anchor:+.4f})")
print(f"    jangkar yang sama di v12       : {REF['probe_siglip2']:.4f}")
if abs(anchor - REF["probe_siglip2"]) > 0.02:
    print("    !! jangkar bergeser >0.02 dari angka v12. Ada yang berbeda di")
    print("       pipeline ini (data, fold, atau view) - periksa sebelum percaya")
    print("       SATU PUN baris di tabel di atas.")

# --- catatan resolusi: pertanyaan yang paling menentukan v13 ----------------
sg = {t: OK[t]["popf1"] for t in ("siglip2-224", "siglip2-256", "siglip2-384",
                                  "siglip2-512") if t in OK}
if len(sg) >= 2:
    print(f"\n[R] SAPUAN RESOLUSI pada SigLIP2 - ini yang langsung mengubah v13:")
    for t in ("siglip2-224", "siglip2-256", "siglip2-384", "siglip2-512"):
        if t in sg: print(f"      {t:14s} {sg[t]:.4f}")
    bt = max(sg, key=sg.get)
    print(f"    terbaik: {bt}")
    if bt == "siglip2-224":
        print("    !! resolusi lebih tinggi TIDAK membantu, bahkan sebagai probe.")
        print("       Taruhan utama v13 lemah. Pertimbangkan menurunkan v13 ke 224")
        print("       dan mengandalkan TTA + spesialis pegon/jawi saja.")
    elif bt == "siglip2-512":
        print("    512 menang sebagai probe. Fine-tune di 512 butuh 1025 token dan")
        print("       terlalu mahal untuk T4 12 jam, tapi ini menguatkan arah 384.")
    else:
        print(f"    -> menguatkan arah v13. Selisih dari 224 = "
              f"{sg[bt]-sg.get('siglip2-224', float('nan')):+.4f}")

print("\n" + "=" * 74)
print("KONFIGURASI SIAP TEMPEL untuk kode/v13.py")
print("=" * 74)
bc = next((c for c in CANDIDATES if c["tag"] == best_tag), None)
if bc is None:
    print("  (juara tidak ditemukan di daftar kandidat - tidak ada yang bisa ditempel)")
elif best_tag == "siglip2-384":
    print("  Juaranya PERSIS yang sudah dipakai v13. Tidak ada yang perlu diubah.")
elif best_sc <= anchor + 0.002:
    print(f"  {best_tag} hanya unggul {best_sc-anchor:+.4f} dari jangkar - di dalam")
    print("  derau. JANGAN ganti backbone v13 untuk selisih sebesar ini. Biarkan")
    print("  v13 apa adanya; kenaikan harus dicari dari tempat lain.")
else:
    print("  Ganti baris pertama PLANS di kode/v13.py dengan:\n")
    print(f'    dict(name="A", model="{bc["name"]}",')
    print(f'         h={bc["h"]}, w={bc["w"]}, epochs=18,')
    print(f'         note="juara AutoML: probe {best_sc:.4f} vs siglip2-224 {anchor:.4f}"),')
    print("\n  Lalu periksa dua hal SEBELUM menjalankan v13 dgn backbone ini:")
    print("    1. Gubernur v13 harus tetap memilih rencana A. Backbone yang lebih")
    print("       besar bisa membuat 5 fold tidak muat di 12 jam, dan rencana B")
    print("       (epoch dipangkas) menjawab pertanyaan yang berbeda.")
    print("    2. Kalau STEP 5 tidak sempat jalan, Anda mengganti backbone yang")
    print("       sudah TERBUKTI 0.87749 dgn yang belum pernah di-fine-tune")
    print("       sekali pun. Itu taruhan, bukan peningkatan.")

if FUSED:
    print(f"\n  Fusi terbaik: {FUSED[0]} -> popf1 {FUSED[1]:.4f}")
    print("  CATATAN: fusi fitur beku BUKAN jalur submission. Di v11/v12 probe")
    print("  dan fusi selalu KALAH dari fine-tune penuh (0.9373 vs 0.9735).")
    print("  Gunanya di sini cuma satu: menunjukkan backbone mana yang saling")
    print("  MELENGKAPI. Kalau fusi yang menang berisi dua keluarga arsitektur")
    print("  yang benar-benar berbeda, itu kandidat ensemble yang sah untuk")
    print("  dicoba di v13 - tidak seperti tiga model se-backbone yang")
    print("  menjatuhkan submission_iw.csv.")

try:
    pd.DataFrame(rows).to_csv(os.path.join(WORK, "automl_hasil.csv"), index=False)
    print(f"\ntabel lengkap -> {os.path.join(WORK, 'automl_hasil.csv')}")
except Exception as e:
    print(f"\n! gagal menulis automl_hasil.csv: {type(e).__name__}")
print(f"fitur ter-cache di {FDIR} - jalankan ulang skrip ini dan STEP 2 hampir gratis.")
print(f"\n{clk()} selesai.")
if SMOKE:
    print("\n!! DIJALANKAN DGN INFEST_SMOKE=1: sebagian bobot ACAK. Angka di atas")
    print("   TIDAK BERARTI APA PUN. Ini hanya menguji jalur kode.")
print("\nCARA MEMBACA TABEL INI:")
print("  - popf1 di sini adalah PROBE BEKU, bukan skor fine-tune dan bukan LB.")
print("    Skalanya berbeda: siglip2 probe 0.9250 menjadi 0.9735 setelah")
print("    fine-tune. Yang berguna adalah URUTANNYA, bukan angka mutlaknya.")
print("  - selisih di bawah ~0.005 antar kandidat jangan dianggap nyata.")
print("  - satu kandidat yang GAGAL dimuat bukan berarti ia jelek; cek pesannya.")
print("  - skrip ini tidak menghasilkan submission. Ia hanya memberi tahu")
print("    backbone mana yang pantas dibayar 2,7 jam fine-tune di v13.")
