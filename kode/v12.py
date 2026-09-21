# =============================================================================
# INFEST XII 2026 - Klasifikasi Citra Aksara Tradisional Nusantara  [v12]
# Satu cell: load -> EDA -> cache -> KORPUS AKSARA SINTETIS -> pra-latih ->
#            fitur beku -> probe -> LP-FT -> FT -> ensemble -> koreksi prior
# Metrik: F1-Score Macro
#
# RIWAYAT LB: v3.1 0.8073 | v5 0.81054 | v6 0.83902 | v7 0.83422 | v11 0.85030
#   Pelabelan MANUAL manusia atas test = 0.71330.
#
# -----------------------------------------------------------------------------
# APA YANG DIUKUR v11 DI DATA LOMBA (dan karena itu membentuk v12)
#
#   [1] Hipotesis "domain dokumen" GAGAL. DiT (BEiT di 42 juta citra dokumen
#       IIT-CDIP) justru sumber TERLEMAH: probe OOF tertimbang 0.8122, kalah
#       ~0.10 dari SigLIP2 umum (0.9158). Semua entri DiT dibuang di v12.
#   [2] Otsu (CANON) MERUGIKAN. Uji terkontrol pertama di tingkat model:
#       dit-sq (Otsu) 0.8122 vs dit-sq-raw (tanpa Otsu) 0.8253 = -0.0131.
#       v10 memaksa Otsu global. Di v12 canon=False di SEMUA sumber, dgn satu
#       pasangan pembanding di SigLIP2 supaya temuan ini diuji ulang di
#       backbone terbaik, bukan cuma di backbone terburuk.
#   [3] Fine-tune BIASA mengalahkan LP-FT. ft-v7 0.9667 vs lpft-siglip2 0.9577.
#       TAPI perbandingannya TIDAK bersih: ft-v7 jalan 18 epoch di 384 dgn
#       canon=False, lpft-siglip2 10 epoch di 224 dgn canon=True (dan canon
#       sekarang terukur merugikan). v12 menyamakan anggarannya.
#   [4] PLAFON DISTRIBUSI TERBANTAHKAN. [E] menghitung plafon 0.8264, LB nyata
#       0.85030 - skornya MELAMPAUI plafonnya sendiri. Satu-satunya asumsi
#       plafon itu adalah prior_test ~ prior_train, jadi asumsi itulah yang
#       salah: komposisi kelas di test memang BERBEDA dari train. Prediksi
#       pegon 2,3x lipat prior train ternyata BENAR, bukan bug.
#       Konsekuensinya besar: gerbang bobot per-kelas di v11 dinilai pada OOF
#       (yang memakai prior TRAIN) lalu ditolak (+0.0019). Gerbang itu mengukur
#       hal yang salah. v12 menggantinya dgn estimasi prior test lewat EM.
#
# -----------------------------------------------------------------------------
# TIGA HAL BARU DI v12
#
#   (a) KORPUS AKSARA SINTETIS + PRA-LATIH ANTARA  <- ini ide utamanya
#       Pencarian di Hugging Face: TIDAK ADA backbone citra yang dilatih pada
#       aksara Nusantara. Yang ada cuma klasifikator per-karakter kecil (keras
#       lontara, ResNet18 aksara jawa) dan korpus PixelGPT aksara (izzako/*)
#       yang berisi render teks, bukan bobot yang bisa dipakai.
#       Jadi backbone-nya kita BUAT: render teks acak 7 aksara memakai font
#       Noto resmi (Javanese, Balinese, Sundanese, Buginese utk lontara,
#       Rejang sbg proksi Lampung, Naskh Arabic utk jawi & pegon), lalu
#       pra-latih ConvNeXt di korpus itu sebelum fine-tune ke data lomba.
#       Kenapa ini menyerang masalah yang benar: di korpus sintetis, UKURAN,
#       ASPECT RATIO dan tingkat korupsi ditarik dari distribusi yang SAMA
#       untuk ke-7 kelas, jadi metadata tidak membawa informasi kelas sama
#       sekali. Model yang bisa memisahkan 7 kelas di sana HARUS membaca
#       bentuk aksara. Itu persis kemampuan yang hilang menurut temuan
#       pintasan handoff 1g (RandomForest metadata saja = 0.7826).
#       Klaim ini DIUJI, bukan diasumsikan: [S1] mengaudit korpus dgn
#       RandomForest metadata (target ~0.143 = tebakan acak) dan [S2] menguji
#       model sintetis LANGSUNG ke data train nyata tanpa fine-tune apa pun.
#       jawi vs pegon dibedakan lewat inventaris huruf tambahan yang berbeda
#       DAN kepadatan harakat (pegon rapat, jawi jarang) - persis pasangan
#       yang paling sering tertukar di v11 (pegon F1 0.9393, 13x -> jawi).
#       CATATAN JUJUR: Lampung (Had Lampung / Kaganga) TIDAK ada di Unicode,
#       jadi tidak ada font resminya. Dipakai Rejang, aksara serumpun Surat
#       Ulu yang bentuknya paling dekat. Ini PROKSI, dan hasilnya untuk kelas
#       lampung harus dibaca dgn diskon.
#
#   (b) FT-SYN sbg uji TERKONTROL. ft-v7 dan ft-syn memakai backbone, resolusi,
#       jumlah epoch, lr, augmentasi dan fold yang SAMA PERSIS. Satu-satunya
#       perbedaan: ft-syn mulai dari bobot hasil pra-latih aksara, ft-v7 dari
#       ImageNet. Selisihnya adalah nilai pra-latih aksara, bukan tebakan.
#
#   (c) KOREKSI PRIOR EM (Saerens, Latinne & Decaestecker, Neural Computation
#       2002). Karena [4] membuktikan prior test != prior train, posterior
#       model bias ke prior train. EM menaksir prior test dari probabilitas
#       test itu sendiri lalu membobot ulang posterior. Diverifikasi dulu di
#       OOF dgn pergeseran prior BUATAN ([P1]) sebelum dipakai, dan ditulis
#       sbg file submission TERPISAH supaya bisa diuji di LB tanpa
#       mengorbankan file utama.
#
# -----------------------------------------------------------------------------
# JEBAKAN YANG DIWARISI (jangan dihapus)
#   - `test.csv` di repo berisi 1.118 id test LAMA; irisannya dgn
#     `sample_submission.csv` (1.220 id) adalah NOL. Prediksi digerakkan dari
#     sample_submission.csv.
#   - v10 `build()` diam-diam jatuh ke resnet34 kalau nama backbone salah.
#     Di sini sumber yang gagal DILEWATI dgn pesan jelas.
#   - load_body() mengembalikan SALINAN; modul bersama akan membuat fold 1
#     mulai dari bobot yang sudah dirusak fold 0.
#   - layer-wise lr decay lewat timm adalah no-op pada AdamW; dihitung sendiri
#     dari nama parameter dan dilaporkan di report_decay().
# =============================================================================
import os, sys, csv, glob, math, time, json, copy, random, zipfile, hashlib, inspect, warnings
import urllib.request, shutil
from collections import Counter, OrderedDict
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image, ImageFile, ImageFilter, ImageEnhance, ImageDraw, ImageFont, features
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# 0. KONFIGURASI
# =============================================================================
# SMOKE=1 -> boleh jalan dgn bobot ACAK kalau unduhan diblokir. Hanya untuk
# menguji JALUR KODE di mesin tanpa internet; angkanya tidak berarti apa pun.
SMOKE = os.environ.get("INFEST_SMOKE", "0") == "1"
SEED = 42

# --- ZOO: sumber fitur beku. Tiap entri = satu backbone x satu view x satu
# --- setelan canon. Biayanya SATU lintasan ekstraksi, bukan pelatihan CNN.
# --- src="hf"   -> transformers AutoModel (bisa arsitektur apa pun)
# --- src="timm" -> timm.create_model
# --- PERUBAHAN v12 vs v11: ketiga entri DiT dibuang (terukur paling lemah:
# --- 0.8122 / 0.8253 / 0.8574 vs SigLIP2 0.9158), dan canon dimatikan di
# --- semua sumber karena uji terkontrol [A3] v11 menunjukkan Otsu -0.0131.
ZOO = [
    # -- fondasi UMUM kuat. SigLIP2 adalah sumber beku TERBAIK yang terukur di
    # -- v11 (0.9158) - jadi di v12 ia yang dapat perlakuan paling serius.
    dict(tag="siglip2",   src="timm", name="vit_base_patch16_siglip_224.v2_webli",
         view="square", h=224, w=224, canon=False,
         note="SigLIP2 webli - sumber beku terbaik v11 (0.9158), kini tanpa Otsu"),
    dict(tag="siglip2-canon", src="timm", name="vit_base_patch16_siglip_224.v2_webli",
         view="square", h=224, w=224, canon=True,
         note="pasangan pembanding Otsu di backbone TERBAIK (v11 hanya menguji "
              "di backbone terburuk); satu lintasan ekstraksi, ~40 detik"),
    # JEBAKAN: resolusi ASLI DINOv2 adalah 518, bukan 224, dan patch-nya 14.
    # Tanpa img_size timm menolak input 224. 224 = 16x14 jadi tetap habis
    # dibagi 14; jangan ganti ke angka yang tidak habis dibagi 14.
    dict(tag="dino2",     src="timm", name="vit_base_patch14_reg4_dinov2.lvd142m",
         view="square", h=224, w=224, canon=False,
         note="DINOv2 reg4 - asli 518, dipakai di 224 (16x14) lewat img_size"),
    # -- domain MANUSKRIP: halaman manuskrip IIIF sungguhan. Bertahan di v12
    # -- meski hipotesis domain dokumen gagal di DiT, karena mae-mss (0.8904)
    # -- dan cnx-mss (0.8832) jauh lebih baik drpd DiT dan menambah keberagaman.
    dict(tag="mae-mss",   src="hf",   name="davanstrien/vit-manuscripts",
         view="square", h=224, w=224, canon=False,
         note="ViT-MAE self-supervised di manuskrip IIIF"),
    dict(tag="cnx-mss",   src="hf",   name="davanstrien/convnext_manuscript_iiif",
         view="square", h=224, w=224, canon=False,
         note="ConvNeXt-base di halaman manuskrip IIIF"),
    # -- tulang punggung v6/v7 yang TERBUKTI 0.839, sebagai fitur beku.
    dict(tag="cnx-sq",    src="timm", name="convnext_tiny.in12k_ft_in1k",
         view="square", h=384, w=384, canon=False,
         note="backbone v6/v7 apa adanya - jangkar ke angka nyata"),
    dict(tag="cnx-wide",  src="timm", name="convnext_tiny.in12k_ft_in1k",
         view="wide",   h=160, w=640, canon=False,
         note="view strip lebar - populasi A"),
    dict(tag="cnx-patch", src="timm", name="convnext_tiny.in12k_ft_in1k",
         view="patch",  h=224, w=224, canon=False,
         note="view patch: W/H/AR tidak pernah sampai ke jaringan"),
]

# --- Fusi fitur: konkatenasi embedding beberapa backbone lalu SATU probe.
# --- Inilah "menggabungkan beberapa model HuggingFace" dalam bentuk paling
# --- murah: tidak ada pelatihan backbone, cuma satu regresi logistik di atas
# --- fitur gabungan. Tiap blok di-L2-normalkan dulu supaya model dgn norma
# --- fitur besar tidak otomatis mendominasi.
# --- v11 mengukur fuse-umum 0.9338 > fuse-semua 0.9211 > fuse-dok 0.8872,
# --- jadi set 'dok' dibuang dan set umum dipertahankan.
FUSE = dict(enable=True, l2norm=True, sets=OrderedDict([
    ("fuse-umum",  ["dino2", "siglip2", "cnx-sq"]),     # pemenang v11 (0.9338)
    ("fuse-mss",   ["siglip2", "mae-mss", "cnx-mss"]),  # umum + manuskrip
    ("fuse-semua", None),                               # None = semua yg berhasil
]))

# --- LP-FT (Kumar dkk. ICLR 2022): probe dulu, fine-tune belakangan, dgn kepala
# --- hasil probe sebagai inisialisasi. `k` = berapa sumber terbaik (menurut OOF
# --- TERTIMBANG probe) yang dilanjutkan ke tahap fine-tune.
# --- v12 menaikkan epochs 10 -> 18 supaya perbandingan LP-FT vs FT biasa
# --- BERSIH: di v11 ft-v7 menang (0.9667 vs 0.9577) tapi ia dapat 18 epoch
# --- lawan 10, dan canon=False lawan canon=True. Dua beda sekaligus berarti
# --- angka itu tidak menjawab pertanyaan "apakah LP-FT lebih baik".
LPFT = dict(enable=True, k=1, epochs=18, lr_body=2e-5, lr_head=1e-3, layer_decay=0.75)

# =============================================================================
# SYN: korpus aksara SINTETIS - inti dari v12
# =============================================================================
# Kenapa dibuat sendiri dan tidak diunduh: pencarian Hugging Face untuk
# "aksara", "lontara", "jawi", "balinese", "lampung" tidak menghasilkan SATU
# pun backbone citra yang dilatih di aksara Nusantara. Yang ada:
#   - klasifikator per-KARAKTER kecil (minggo-commits/aksara-lontara-classifier
#     berbasis keras, shivaaugusta98/Aksara-Jawa-ResNet18) - bukan backbone;
#   - dataset karakter tunggal (RayFox52/Aksara-jawa-carakan-murda 6,6rb citra
#     27 kelas; Muhammad-Ikhwan-Fathulloh/aksara-sunda-classification 10,8rb
#     citra 30 kelas) - hanya 2 dari 7 aksara, dan per karakter, bukan halaman;
#   - korpus PixelGPT aksara (izzako/javanese|sundanese|balinese|lampung-pixelgpt)
#     berisi render teks jadi - menutup 4 dari 7 kelas, tapi tidak ada jawi,
#     pegon, atau lontara.
# Tidak satu pun menutup ke-7 kelas, jadi korpusnya dirender sendiri dari font
# Noto resmi. Unduhannya kecil (~0,7 MB total) dan dari raw.githubusercontent.
#
# Yang membuat korpus ini berguna BUKAN kemiripannya dgn foto nyata (ia jelas
# tidak mirip), melainkan bahwa di dalamnya metadata tidak membawa informasi
# kelas SAMA SEKALI: aspect ratio target ditarik dari distribusi train NYATA
# secara acak dan INDEPENDEN dari kelas, lalu jumlah baris dan panjang baris
# disesuaikan supaya render mencapai AR itu. Audit [S1] membuktikannya.
SYN = dict(
    enable=True,
    n_per_class=2000,        # 7 x 2000 = 14.000 citra; render ~5 menit di CPU
    epochs=5,
    # Pra-latih jalan di resolusi LEBIH KECIL dari fine-tune (256 vs 384).
    # ConvNeXt sepenuhnya konvolusional jadi bobotnya berpindah resolusi tanpa
    # masalah, dan biayanya turun ~2,2x: 70rb sampel jadi ~10 menit, bukan ~25.
    # KALAU backbone ft-syn diganti ke ViT, samakan pre_h/pre_w dgn h/w -
    # position embedding ViT berukuran tetap dan tensornya akan dilewati.
    pre_h=256, pre_w=256,
    lr=2e-4,
    val_frac=0.08,           # holdout sintetis utk memantau pra-latih
    maxside=768,             # render dipotong di sini; view terbesar cuma 384
    audit=True,              # [S1] RandomForest metadata saja -> harus ~0.143
    audit_n=4000,            # audit di sampel; membaca 14rb berkas makan ~4 menit
    # font: nama -> (berkas, dipakai oleh kelas apa)
    font_base=[
        "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf",
        "https://cdn.jsdelivr.net/gh/googlefonts/noto-fonts@main/hinted/ttf",
    ],
    fonts=["NotoSansJavanese", "NotoSansBalinese", "NotoSansSundanese",
           "NotoSansBuginese", "NotoSansRejang", "NotoNaskhArabic"],
)

# --- Run fine-tuning LANGSUNG. ft-v7 adalah JANGKAR (resep v6/v7, LB 0.839
# --- diketahui). ft-syn identik dalam SEGALA hal kecuali titik awal bobotnya,
# --- sehingga selisih keduanya adalah nilai pra-latih aksara dan bukan yang
# --- lain. ft-siglip2 ada karena SigLIP2 adalah sumber beku terbaik v11 dan
# --- belum pernah sekalipun di-fine-tune penuh.
FT_RUNS = [
    # layer_decay=1.0 DISENGAJA pada jangkar: di v10 knob layer-wise decay tidak
    # pernah aktif (AdamW mengabaikan `lr_scale`), jadi jangkar harus meniru
    # perilaku EFEKTIF v6/v7 - kalau tidak, ia berhenti menjadi jangkar.
    dict(tag="ft-v7", src="timm", name="convnext_tiny.in12k_ft_in1k",
         view="square", h=384, w=384, canon=False, lr=1e-4, epochs=18, tta=1,
         layer_decay=1.0, syn_init=False,
         note="resep v6/v7 apa adanya - jangkar ke LB 0.839 DAN kontrol utk ft-syn"),
    dict(tag="ft-syn", src="timm", name="convnext_tiny.in12k_ft_in1k",
         view="square", h=384, w=384, canon=False, lr=1e-4, epochs=18, tta=1,
         layer_decay=1.0, syn_init=True,
         note="IDENTIK dgn ft-v7 kecuali mulai dari bobot pra-latih aksara sintetis"),
    dict(tag="ft-siglip2", src="timm", name="vit_base_patch16_siglip_224.v2_webli",
         view="square", h=224, w=224, canon=False, lr=3e-5, epochs=18, tta=1,
         layer_decay=0.75, syn_init=False,
         note="SigLIP2 di-fine-tune penuh - sumber beku terbaik v11, belum pernah dicoba"),
]

# --- Koreksi prior EM (Saerens dkk. 2002). Dipakai HANYA kalau [P1] - uji di
# --- OOF dgn pergeseran prior BUATAN - menunjukkan ia memang memperbaiki
# --- macro-F1 melebihi ambang derau. Hasilnya selalu ditulis sbg file
# --- submission TERPISAH, tidak pernah menimpa file utama.
SLD = dict(enable=True, iters=400, tol=1e-7, gate=0.003, trials=12)

# --- view PATCH: skala dinormalkan lewat sisi TERPENDEK, lalu ambil n potongan
# --- P x P pada grid. n TETAP untuk semua gambar, supaya JUMLAH patch pun tidak
# --- membocorkan aspect ratio.
PATCH = dict(size=160, n=9, minside=176, scales=(1.0, 0.72, 1.4))

MAXSIDE   = 1600   # gambar terbesar 5664x4248 -> perkecil dulu
CACHE_MAX = 1280   # sisi maks di cache; semua view <= 640 px jadi ini berlebih
CACHE     = dict(enable=True)
# CATATAN: di v6-v10 ada knob `GRAYSCALE`. Sejak v11 knob itu DIHAPUS, bukan
# dibiarkan sebagai tombol mati: seluruh jalur gambar bekerja di mode 'L' (cache,
# degrade, view, augmentasi), jadi `GRAYSCALE=False` tidak akan mengembalikan
# sumbu warna - ia hanya akan terlihat seperti pilihan yang ada padahal tidak.
CANON_BLUR_PRE = 0.5
SEV = 1.0
DEGRADE = dict(p_blur=0.70, blur=(0.6*SEV, 2.2*SEV),
               p_contrast=0.60, contrast=(0.45, 0.95),
               p_rescale=0.50, rescale=(max(0.15, 0.35/SEV), 0.80),
               p_jpeg=0.50, jpeg=(max(10, int(25/SEV)), 88))
AUGMAX = dict(enable=True, p=0.35, chains=3)
ARJIT  = dict(enable=True, lo=0.65, hi=1.55)
PROBE  = dict(C=1.0, max_iter=3000)
GATE   = dict(ens=0.002, cw=0.002)               # ambang derau utk tiap gerbang
CFG = dict(n_folds=5, batch=32, nw=2, wd=0.05, ls=0.05, ema=0.999)
AR_SPLIT = 3.0

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
seed_all()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEV} | torch={torch.__version__}" + ("  [SMOKE: bobot boleh ACAK]" if SMOKE else ""))

# =============================================================================
# 1. LOAD / DISCOVER DATA
# =============================================================================
SEARCH = [p for p in ["/kaggle/input", "/kaggle/working", ".", "./data", "/content"]
          if os.path.isdir(p)]
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

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
print("test.csv  :", test_csv)
print("sample_sub:", sub_csv)
assert train_csv, "train.csv tidak ketemu"
tr_df = pd.read_csv(train_csv)

# --- PENTING: prediksi digerakkan dari sample_submission.csv, BUKAN test.csv.
# --- Panitia pernah mengganti dataset; di ruang kerja ini `test.csv` berisi
# --- 1.118 id test LAMA yang irisannya dgn sample_submission (1.220 id) NOL.
# --- Kalau keduanya ada dan tidak cocok, sample_submission yang menang karena
# --- dialah format yang dinilai.
if sub_csv:
    sub_tpl = pd.read_csv(sub_csv)
    te_df = pd.DataFrame({"image_id": sub_tpl.image_id.tolist()})
    src_test = "sample_submission.csv"
    if test_csv:
        _t = pd.read_csv(test_csv)
        if "image_id" in _t:
            _ov = len(set(_t.image_id) & set(te_df.image_id))
            print(f"cek test.csv vs sample_submission: {len(_t)} vs {len(te_df)} baris, "
                  f"irisan id = {_ov}")
            if _ov == 0:
                print("  !! test.csv TIDAK BERIRISAN dgn sample_submission -> test.csv"
                      " adalah test set LAMA dan DIABAIKAN. Ini jebakan yang membuat"
                      " satu diagnostik sebelumnya salah total (handoff 1g).")
            elif _ov < len(te_df):
                print(f"  !! hanya {_ov}/{len(te_df)} id yang sama - periksa manual.")
elif test_csv:
    te_df = pd.read_csv(test_csv)[["image_id"]]
    src_test = "test.csv"
    print("  PERHATIAN: sample_submission.csv tidak ketemu, memakai test.csv."
          " Verifikasi jumlah barisnya sebelum submit.")
else:
    raise AssertionError("sample_submission.csv maupun test.csv tidak ketemu")
print(f"daftar test diambil dari: {src_test} ({len(te_df)} baris)")

# gambar bisa datang sebagai folder ATAU .zip -> ekstrak sekali kalau perlu
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
    c = IDX.get(b) or STEM.get(os.path.splitext(b)[0])   # cocokkan tanpa ekstensi
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
    print(f"   train tidak ketemu : {len(mt)}/{len(tr_df)}  contoh: {mt.image_id.head(5).tolist()}")
    print(f"   test  tidak ketemu : {len(me)}/{len(te_df)}  contoh: {me.image_id.head(5).tolist()}")
    print(f"   file gambar terindeks: {sum(len(v) for v in IDX.values())}")
    print(f"   ekstensi di disk     : {dict(Counter(os.path.splitext(f)[1].lower() for f in IDX))}")
    print(f"   folder dipindai      : {SEARCH}")
    raise AssertionError("Gambar tidak ketemu - lihat diagnosis di atas.")
tr_df = tr_df[tr_df.path.notna()].reset_index(drop=True)   # baris test TIDAK pernah dibuang

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
for _k, _v in list(LEAK.items())[:8]: print(f"   {_k} -> {_v}")

# =============================================================================
# 2. EDA + DUA POPULASI + BOBOT PENILAIAN
# =============================================================================
print("\n" + "=" * 70 + "\nSTEP 1 - EDA\n" + "=" * 70)
CLASSES = sorted(tr_df.label.unique()); C2I = {c: i for i, c in enumerate(CLASSES)}
tr_df["y"] = tr_df.label.map(C2I); NC = len(CLASSES)
cnt = np.bincount(tr_df.y, minlength=NC); ytrue = tr_df.y.values
print("kelas:", CLASSES)
print("distribusi train (macro-F1 -> kelas terkecil sama pentingnya):")
for i, c in enumerate(CLASSES):
    print(f"  {c:10s}{cnt[i]:5d}  {100*cnt[i]/cnt.sum():5.2f}%")
print(f"  imbalance = {cnt.max()/cnt.min():.2f}x")

def _meta(paths):
    """satu lintasan: aspect ratio + 'bil' = porsi piksel yang sudah 2-aras."""
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
# `np.nanmean(AR_TE < AR_SPLIT)` TIDAK bisa melewati NaN: perbandingannya sudah
# meruntuhkan NaN jadi False lebih dulu, sehingga gambar yang gagal dibaca ikut
# terhitung sebagai 'strip'. W_BLOK adalah bobot di balik SETIAP gerbang di
# skrip ini, jadi NaN dibuang secara eksplisit.
_ok_te = ~np.isnan(AR_TE)
if _ok_te.sum() == 0: raise AssertionError("tidak ada gambar test yang bisa dibaca")
if (~_ok_te).sum():
    print(f"  catatan: {(~_ok_te).sum()} gambar test gagal dibaca saat mengukur AR,"
          f" dikeluarkan dari perhitungan bobot populasi")
W_BLOK  = float((AR_TE[_ok_te] < AR_SPLIT).mean())   # diukur dari TEST, bukan train
BIN_TR  = np.nan_to_num(BIL_TR, nan=0.0) > 0.90
print(f"\npintasan binarisasi: train {100*BIN_TR.mean():.1f}% gambar sudah 2-aras, "
      f"test {100*np.nanmean(np.nan_to_num(BIL_TE, nan=0.0) > 0.90):.1f}%")
print("  per kelas (kalau angkanya jauh berbeda, ini PINTASAN, bukan aksara):")
for _c in CLASSES:
    _m = (tr_df.y.values == C2I[_c])
    print(f"    {_c:10s} 2-aras {100*BIN_TR[_m].mean():5.1f}%   AR p50 {np.nanmedian(AR_TR[_m]):6.2f}")
print(f"\npopulasi (AR<{AR_SPLIT} = 'blok'):")
print(f"  train: blok {BLOK_TR.mean()*100:5.1f}%  strip {100-BLOK_TR.mean()*100:5.1f}%")
print(f"  test : blok {W_BLOK*100:5.1f}%  strip {100-W_BLOK*100:5.1f}%   <- bobot penilaian")
if abs(BLOK_TR.mean() - W_BLOK) > 0.05:
    print("  PERHATIAN: komposisi train != test. OOF polos AKAN terlalu optimistis.")

def popf1(p, mask, detail=False):
    """macro-F1 ditimbang ke komposisi strip/blok TEST, bukan train.
    INILAH angka yang dipakai untuk setiap keputusan di skrip ini. OOF polos
    yang tidak ditimbang adalah angka yang membuat v3.1 terlihat 0.9716
    padahal LB-nya 0.8073."""
    out, parts = 0.0, {}
    for nm, sub, w in (("blok", BLOK_TR, W_BLOK), ("strip", ~BLOK_TR, 1 - W_BLOK)):
        m = mask & sub
        v = f1_score(ytrue[m], p[m].argmax(1), average="macro") if m.sum() >= NC else float("nan")
        parts[nm] = (v, int(m.sum()))
        if not np.isnan(v): out += w * v
    return (out, parts) if detail else out

# =============================================================================
# 3. CACHE GAMBAR (dekode + Otsu sekali saja)
# =============================================================================
# Jalur fitur beku melewati gambar yang SAMA berkali-kali (9 sumber x n view).
# Dekode PNG/JPEG besar berulang kali adalah biaya terbesar yang bisa dihindari.
# Yang di-cache hanyalah operasi DETERMINISTIK: thumbnail -> abu-abu -> Otsu.
# Augmentasi/degradasi tetap dihitung saat latih, jadi tidak ada kebocoran.
#
# CATATAN JUJUR soal urutan operasi: v6/v7 menjalankan degrade() di RGB lalu
# baru convert("L"). Karena cache menyimpan versi abu-abu, di v11 urutannya
# terbalik (abu-abu dulu, degrade sesudahnya). Bedanya hanya pada subsampling
# chroma JPEG di dalam degrade(), yang pada citra abu-abu praktis tidak berarti.
def otsu_thr(a):
    """ambang Otsu tanpa cv2/skimage (keduanya belum tentu ada di runtime lomba)"""
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

def canon_l(g):
    """Kanonikalisasi Otsu pada citra mode 'L'. TERUKUR (handoff 1h): pintasan
    fotometrik 0.6477 -> 0.2176 (tebak acak 0.143) dan AUC pembeda domain
    train-vs-test 0.9110 -> 0.5955, yaitu dua domain jadi nyaris tak
    terbedakan. TIDAK menyentuh pintasan geometri - itu tugas view patch."""
    if CANON_BLUR_PRE > 0: g = g.filter(ImageFilter.GaussianBlur(CANON_BLUR_PRE))
    a = np.asarray(g, np.float32)
    if a.size == 0: return g
    return Image.fromarray((a > otsu_thr(a)).astype(np.uint8) * 255, "L")

CACHE_DIR = os.path.join(WORK, "_cache")
_CSIG = f"m{CACHE_MAX}_b{CANON_BLUR_PRE}"
_cache_fail = {"n": 0}

def _cache_path(path, canon):
    key = hashlib.md5(f"{os.path.abspath(path)}|{_CSIG}|{int(canon)}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, ("canon" if canon else "raw"), key[:2], key + ".png")

def load_img(path, canon):
    """kembalikan citra mode 'L' hasil thumbnail (+Otsu kalau canon)."""
    if CACHE["enable"] and _cache_fail["n"] < 20:
        cp = _cache_path(path, canon)
        if os.path.exists(cp):
            try:
                with Image.open(cp) as im: return im.convert("L")
            except Exception: pass
    try:
        with Image.open(path) as im:
            if max(im.size) > MAXSIDE: im.thumbnail((MAXSIDE, MAXSIDE))
            g = im.convert("L")
    except Exception:
        return Image.new("L", (64, 64), 255)
    if max(g.size) > CACHE_MAX: g.thumbnail((CACHE_MAX, CACHE_MAX))
    if canon: g = canon_l(g)
    if CACHE["enable"] and _cache_fail["n"] < 20:
        try:
            cp = _cache_path(path, canon)
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            g.save(cp, "PNG", optimize=False)
        except Exception as e:
            _cache_fail["n"] += 1
            if _cache_fail["n"] == 1:
                print(f"  ! cache gambar gagal ditulis ({type(e).__name__}) -> lanjut tanpa cache")
            if _cache_fail["n"] == 20:
                print("  ! cache dimatikan setelah 20 kegagalan (disk penuh?)")
    return g

def warm_cache(paths, canon, tag):
    if not CACHE["enable"]: return
    t0 = time.time(); todo = [p for p in paths if isinstance(p, str)]
    for k, p in enumerate(todo):
        load_img(p, canon)
        if k and k % 1500 == 0: print(f"    cache {tag} {k}/{len(todo)} ({time.time()-t0:.0f}s)")
    print(f"    cache {tag} siap: {len(todo)} gambar, {time.time()-t0:.0f}s")

# =============================================================================
# 3b. KORPUS AKSARA SINTETIS  (baru di v12 - inti pendekatannya)
# =============================================================================
# Tidak ada backbone citra di Hugging Face yang pernah melihat aksara Nusantara
# (lihat catatan di blok SYN pada bagian konfigurasi), jadi korpusnya dirender
# sendiri dari font Noto resmi dan dipakai sebagai TUGAS ANTARA sebelum
# fine-tune ke data lomba.
#
# SATU-SATUNYA alasan korpus ini berguna: di dalamnya metadata TIDAK membawa
# informasi kelas. Lebarnya, tingginya dan aspect ratio-nya ditarik dari
# distribusi ukuran train NYATA secara acak dan independen dari kelas, lalu
# jumlah baris dan panjang tiap baris DISESUAIKAN supaya render mencapai AR
# itu, dan terakhir citranya diubah ukurannya persis ke (W,H) yang ditarik tadi.
# Jadi menurut konstruksi, P(kelas | W, H, AR) = P(kelas). Tingkat tinta,
# tingkat kertas, blur dan kualitas JPEG juga ditarik dari distribusi yang sama
# untuk ke-7 kelas. Yang tersisa untuk dipelajari hanyalah BENTUK AKSARA.
# Ini diaudit, bukan diasumsikan: [S1] menjalankan RandomForest yang HANYA
# melihat metadata; kalau korpusnya benar ia harus mendarat di ~1/7 = 0.143.
#
# Keterbatasan yang harus dibaca apa adanya:
#   - Lampung (Had Lampung / Kaganga) TIDAK ada di Unicode, jadi tidak ada font
#     resminya. Dipakai Rejang, aksara serumpun Surat Ulu. Ini PROKSI.
#   - Teksnya acak, bukan bahasa sungguhan: statistik n-gram-nya salah. Yang
#     dipelajari adalah repertoar dan bentuk goresan, bukan kata.
#   - Render bersih != foto manuskrip. Karena itu ia dipakai sbg INISIALISASI
#     yang lalu di-fine-tune ke data nyata, bukan sebagai model akhir.
SYN_DIR  = os.path.join(WORK, "_syn")
FONT_DIR = os.path.join(SYN_DIR, "fonts")
syn_df, SYN_OK, SYN_NOTE = None, False, ""

def _dl(url, dst, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    return os.path.getsize(dst)

def fetch_fonts():
    """unduh font Noto. Gagal sebagian -> kelas itu dilewati, bukan diganti
    diam-diam dgn font lain (pelajaran dari substitusi senyap resnet34 di v10)."""
    os.makedirs(FONT_DIR, exist_ok=True)
    got = {}
    for n in SYN["fonts"]:
        dst = os.path.join(FONT_DIR, f"{n}-Regular.ttf")
        if os.path.exists(dst) and os.path.getsize(dst) > 4000:
            got[n] = dst; continue
        last = "tidak ada mirror yang dicoba"
        for base in SYN["font_base"]:
            try:
                sz = _dl(f"{base}/{n}/{n}-Regular.ttf", dst)
                if sz > 4000:
                    got[n] = dst; break
                os.remove(dst)
            except Exception as e:
                last = f"{type(e).__name__}"
        if n not in got:
            print(f"    ! font {n} gagal diunduh ({last}) -> kelas yang memakainya dilewati")
    return got

def _R(a, b): return [chr(c) for c in range(a, b + 1)]

# Inventaris per aksara. Untuk aksara Brahmi (jawa/bali/sunda/lontara/rejang)
# dipisah konsonan dasar vs tanda vokal (sandhangan/pangangge) supaya
# rendernya punya tumpukan diakritik seperti tulisan sungguhan.
# jawi vs pegon: KEDUANYA aksara Arab, dan di v11 inilah pasangan paling
# sering tertukar (pegon F1 0.9393, tertukar 13x jadi jawi). Yang membedakan:
#   - huruf tambahan yang berbeda (jawi: ڠ ڤ ݢ ڽ چ ۏ | pegon: ڠ ڤ ڮ ۑ ڎ ڟ ٹ);
#   - kepadatan harakat. Pegon (teks keagamaan/Jawa) rapat harakat, jawi
#     (Melayu) hampir selalu gundul. Itu isyarat visual yang nyata, bukan
#     rekaan, dan dikodekan lewat p_diac 0.45 vs 0.04.
_AR_BASE = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
SYN_SRC = OrderedDict([
    ("jawa",    dict(font="NotoSansJavanese",  cons=_R(0xA98F, 0xA9B2),
                     diac=_R(0xA9B4, 0xA9BD), p_diac=0.55, rtl=False)),
    ("bali",    dict(font="NotoSansBalinese",  cons=_R(0x1B13, 0x1B33),
                     diac=_R(0x1B35, 0x1B43), p_diac=0.55, rtl=False)),
    ("sunda",   dict(font="NotoSansSundanese", cons=_R(0x1B8A, 0x1BA0),
                     diac=_R(0x1BA1, 0x1BAD), p_diac=0.45, rtl=False)),
    ("lontara", dict(font="NotoSansBuginese",  cons=_R(0x1A00, 0x1A16),
                     diac=_R(0x1A17, 0x1A1B), p_diac=0.35, rtl=False)),
    ("lampung", dict(font="NotoSansRejang",    cons=_R(0xA930, 0xA946),
                     diac=_R(0xA947, 0xA951), p_diac=0.45, rtl=False,
                     proxy="Rejang (serumpun Surat Ulu) - Lampung tidak ada di Unicode")),
    ("jawi",    dict(font="NotoNaskhArabic",
                     cons=_AR_BASE + list("ڠڤݢڽچۏڬ")*3,
                     diac=list("ًٌٍَُِّْ"),
                     p_diac=0.04, rtl=True)),
    ("pegon",   dict(font="NotoNaskhArabic",
                     cons=_AR_BASE + list("ڠڤڮۑڎڟٹڽ")*3,
                     diac=list("ًٌٍَُِّْ"),
                     p_diac=0.45, rtl=True)),
])

def _syn_word(s, rng):
    out = []
    for _ in range(rng.randint(2, 7)):
        out.append(rng.choice(s["cons"]))
        if rng.random() < s["p_diac"]: out.append(rng.choice(s["diac"]))
    return "".join(out)

def syn_render(tag, rng, fonts, Wt, Ht):
    """Render satu citra aksara `tag` lalu kembalikan tepat berukuran (Wt, Ht).
    (Wt, Ht) DITARIK DARI LUAR, independen dari tag - itulah yang membuat
    metadata tidak membawa informasi kelas."""
    s = SYN_SRC[tag]
    f_path = fonts[s["font"]]
    fs = rng.randint(18, 46)
    font = ImageFont.truetype(f_path, fs)
    lh = fs * rng.uniform(1.5, 2.3)
    ar = Wt / max(Ht, 1)
    # pilih jumlah baris yang membuat lebar teks masuk akal untuk AR ini
    nl, wtxt = 1, ar * lh
    for cand in sorted(range(1, 13), key=lambda _: rng.random()):
        w_try = ar * cand * lh
        if 140 <= w_try <= 2400: nl, wtxt = cand, w_try; break
    else:
        nl = max(1, min(12, int(round(math.sqrt(max(Wt*Ht, 1)) / max(lh, 1)))))
        wtxt = max(140, min(2400, ar * nl * lh))
    probe = ImageDraw.Draw(Image.new("L", (8, 8), 255))
    lines = []
    for _ in range(nl):
        t = ""
        while probe.textlength(t, font=font) < wtxt and len(t) < 420:
            t += ("" if not t else " ") + _syn_word(s, rng)
        lines.append(t if t else _syn_word(s, rng))
    pad = rng.randint(5, 26)
    W = int(max(probe.textlength(t, font=font) for t in lines)) + 2*pad
    H = int(nl * lh) + 2*pad
    paper = rng.randint(205, 255)
    img = Image.new("L", (max(W, 24), max(H, 24)), paper)
    d = ImageDraw.Draw(img)
    ink = rng.randint(0, min(150, paper - 55))
    for i, t in enumerate(lines):
        x = pad if not s["rtl"] else img.width - pad
        d.text((x, pad + i*lh), t, font=font, fill=ink,
               anchor=("la" if not s["rtl"] else "ra"),
               direction=("ltr" if not s["rtl"] else "rtl"))
    # korupsi ringan SAAT RENDER supaya derajat binarisasi ikut beragam - kalau
    # tidak, korpusnya 100% 2-aras dan 'bil' jadi penanda sintetis-vs-nyata.
    if rng.random() < 0.75:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.8)))
    if rng.random() < 0.55:
        a = np.asarray(img, np.float32) + np.random.normal(0, rng.uniform(2, 12), (img.height, img.width))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")
    if rng.random() < 0.55:
        b = io.BytesIO(); img.save(b, "JPEG", quality=rng.randint(28, 92))
        b.seek(0); img = Image.open(b).convert("L")
    return img.resize((max(Wt, 16), max(Ht, 16)), Image.BILINEAR)

def build_syn_corpus():
    """render korpus ke disk lalu kembalikan DataFrame(path, y) yang bentuknya
    sama dgn tr_df, supaya bisa langsung masuk ScriptDS tanpa kelas baru."""
    fonts = fetch_fonts()
    if not features.check("raqm"):
        print("    ! Pillow TANPA raqm: aksara kompleks (jawa/bali) tidak dibentuk")
        print("      dgn benar dan Arab tidak disambung. Korpusnya jadi jauh lebih")
        print("      lemah, tapi tetap dijalankan - lihat [S2] untuk dampaknya.")
    use = [t for t in SYN_SRC if SYN_SRC[t]["font"] in fonts and t in C2I]
    miss = [t for t in SYN_SRC if t not in use]
    if miss: print(f"    kelas tanpa font -> dilewati: {miss}")
    if len(use) < 2:
        print("    !! kurang dari 2 kelas tersedia -> korpus sintetis DIBATALKAN")
        return None, ""
    # kolam ukuran NYATA dari train; ditarik acak dan INDEPENDEN dari kelas
    pool = []
    for q in tr_df.path.tolist():
        try:
            with Image.open(q) as im: pool.append(im.size)
        except Exception: pass
    if not pool: pool = [(900, 300)]
    ms = SYN["maxside"]
    rng = random.Random(SEED)
    rows, t0 = [], time.time()
    for tag in use:
        outd = os.path.join(SYN_DIR, "img", tag)
        os.makedirs(outd, exist_ok=True)
        for i in range(SYN["n_per_class"]):
            dst = os.path.join(outd, f"{i:05d}.jpg")
            if not os.path.exists(dst):
                Wt, Ht = pool[rng.randrange(len(pool))]
                sc = min(1.0, ms / max(Wt, Ht))
                Wt, Ht = max(16, int(Wt*sc)), max(16, int(Ht*sc))
                try:
                    syn_render(tag, rng, fonts, Wt, Ht).save(dst, "JPEG", quality=92)
                except Exception as e:
                    print(f"    ! render {tag} #{i} gagal: {type(e).__name__}: {e}")
                    continue
            rows.append((dst, C2I[tag]))
        print(f"    {tag:9s} {SYN['n_per_class']:5d} citra  ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows, columns=["path", "y"])
    note = f"{len(use)} kelas ({', '.join(use)})"
    if "lampung" in use: note += " | lampung = PROKSI Rejang"
    return df, note

def audit_syn(df_all):
    """[S1] RandomForest yang HANYA melihat metadata. Kalau korpusnya benar ia
    harus mendarat di sekitar tebakan acak. Ini uji yang sama persis yang di
    data NYATA menghasilkan 0.7826 (handoff 1g) - angka yang membuktikan
    sebagian besar skor lomba ini adalah pintasan."""
    # sampel berimbang per kelas. TIDAK memakai groupby().apply(): di pandas
    # baru kolom kunci grup ikut hilang dari hasilnya dan `df.y` lenyap.
    if len(df_all) > SYN["audit_n"]:
        per = max(NC, SYN["audit_n"] // max(int(df_all.y.nunique()), 1))
        rs_ = np.random.RandomState(SEED); keep = []
        for v in sorted(df_all.y.unique()):
            ix = np.where(df_all.y.values == v)[0]
            keep.append(rs_.choice(ix, size=min(len(ix), per), replace=False))
        df = df_all.iloc[np.concatenate(keep)].reset_index(drop=True)
    else:
        df = df_all
    ar, bil = _meta(df.path.tolist())
    wh = []
    for q in df.path.tolist():
        try:
            with Image.open(q) as im: wh.append(im.size)
        except Exception: wh.append((0, 0))
    wh = np.array(wh, float)
    X = np.column_stack([wh[:, 0], wh[:, 1], np.nan_to_num(ar, nan=1.0),
                         np.nan_to_num(bil, nan=0.0)])
    y = df.y.values
    n = len(df); idx = np.random.RandomState(SEED).permutation(n); cut = int(n*0.7)
    rf = RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=SEED)
    rf.fit(X[idx[:cut]], y[idx[:cut]])
    f = f1_score(y[idx[cut:]], rf.predict(X[idx[cut:]]), average="macro")
    chance = 1.0 / len(np.unique(y))
    print(f"\n[S1] AUDIT PINTASAN korpus sintetis (RandomForest, metadata saja)")
    print(f"     macro-F1 = {f:.4f}   |   tebakan acak = {chance:.4f}")
    print(f"     pembanding di data NYATA (handoff 1g)  = 0.7826")
    if f > chance + 0.10:
        print("     !! GAGAL: metadata masih membocorkan kelas. Korpus ini akan")
        print("        mengajarkan pintasan yang sama, bukan bentuk aksara.")
    else:
        print("     LULUS: ukuran/AR/binarisasi tidak membawa informasi kelas, jadi")
        print("        apa pun yang dipelajari di korpus ini adalah BENTUK AKSARA.")
    return float(f)

if SYN["enable"]:
    print("\n" + "=" * 70 + "\nSTEP 2b - KORPUS AKSARA SINTETIS\n" + "=" * 70)
    try:
        syn_df, SYN_NOTE = build_syn_corpus()
    except Exception as e:
        syn_df, SYN_NOTE = None, ""
        print(f"  !! korpus sintetis gagal dibuat ({type(e).__name__}: {e})")
        print("     -> v12 lanjut TANPA pra-latih aksara; ft-syn akan identik ft-v7.")
    if syn_df is not None and len(syn_df) > 0:
        SYN_OK = True
        print(f"  korpus siap: {len(syn_df)} citra, {SYN_NOTE}")
        if SYN["audit"]:
            try:
                audit_syn(syn_df)
            except Exception as e:
                # audit itu DIAGNOSTIK. Ia tidak boleh membatalkan run 5 jam.
                print(f"  ! audit [S1] gagal ({type(e).__name__}: {e}) -> dilewati;"
                      f" korpus tetap dipakai tapi klaim 'bebas pintasan' BELUM diuji.")

# =============================================================================
# 4. VIEW: letterbox / wide / patch
# =============================================================================
MEAN_DEF, STD_DEF = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def letterbox(g, H, W, train):
    w, h = g.size
    if train and ARJIT["enable"]:
        # [v10] regangkan HANYA sumbu-x -> aspect ratio berhenti jadi info kelas.
        f = random.uniform(ARJIT["lo"], ARJIT["hi"])
        w = max(1, int(round(w * f))); g = g.resize((w, h), Image.BILINEAR)
    s = min(H / max(h, 1), W / max(w, 1))
    if train: s *= random.uniform(0.92, 1.08)
    nw, nh = max(1, min(W, int(round(w*s)))), max(1, min(H, int(round(h*s))))
    g = g.resize((nw, nh), Image.BILINEAR)
    c = Image.new("L", (W, H), 255)
    ox = random.randint(0, W-nw) if train else (W-nw)//2
    oy = random.randint(0, H-nh) if train else (H-nh)//2
    c.paste(g, (ox, oy)); return c

def patchgrid(g, train, view=0, nview=1):
    """Gomez dkk. (Pattern Recognition 2017): normalkan SKALA lewat sisi
    terpendek, lalu ambil potongan berukuran TETAP P x P.
    Konsekuensi yang kita cari: lebar, tinggi, dan aspect ratio gambar asli
    TIDAK IKUT masuk ke jaringan - hanya coretan pada skala tetap.

    KOREKSI dari draf pertama. Dulu posisi dihitung dgn step tetap P//2, dan itu
    DEGENERASI pada gambar 'blok': diukur pada 120 gambar test sungguhan, 23 dari
    57 gambar blok hanya menghasilkan SATU posisi unik untuk 9 view, jadi 9
    lintasan maju menghitung patch yang sama persis. Lebih buruk lagi, banyaknya
    patch unik itu sendiri berkorelasi dgn AR - pintasan geometri yang mau
    dibuang justru masuk lewat pintu belakang.

    Sekarang keberagaman dipasok dua sumbu sekaligus, seperti Gomez dkk. yang
    juga memakai DUA skala (32 dan 40 px), bukan satu:
      - SKALA: tiap view memakai satu dari PATCH['scales'], jadi gambar nyaris
        persegi tetap melihat isi yang berbeda walau posisinya berhimpit.
      - POSISI: disebar merata (linspace) atas rentang yang benar-benar tersedia,
        bukan dgn step tetap, jadi rentang sekecil apa pun tetap terbagi n.
    Jumlah lintasan maju tetap n untuk SEMUA gambar."""
    P, MS = PATCH["size"], PATCH["minside"]
    scales = PATCH["scales"]
    w, h = g.size
    if train:
        sc = random.choice(scales) * random.uniform(0.9, 1.12)   # jitter skala
    else:
        sc = scales[view % len(scales)]                          # skala per view
    s = (MS / max(1, min(w, h))) * sc
    nw, nh = max(1, int(round(w*s))), max(1, int(round(h*s)))
    cap = 24 * P                                             # batasi kerja utk strip ekstrem
    if max(nw, nh) > cap:
        f = cap / max(nw, nh); nw, nh = max(1, int(nw*f)), max(1, int(nh*f))
    g = g.resize((nw, nh), Image.BILINEAR)
    if nw < P or nh < P:                                     # pad putih kalau kekecilan
        c = Image.new("L", (max(P, nw), max(P, nh)), 255); c.paste(g, (0, 0)); g = c
        nw, nh = g.size
    rx, ry = nw - P, nh - P                                  # rentang geser yang tersedia
    if train:
        x = random.randint(0, rx) if rx > 0 else 0
        y = random.randint(0, ry) if ry > 0 else 0
    else:
        # posisi ke-`slot` dari npos slot, disebar merata di rentang yg tersedia.
        npos = max(1, int(math.ceil(nview / len(scales))))
        slot = view // len(scales)
        if npos == 1: fx = fy = 0.5
        else:
            # sebar sepanjang sumbu yang rentangnya lebih besar (sumbu panjang);
            # sumbu pendek ikut tapi lebih rapat.
            t = slot / (npos - 1)
            if rx >= ry: fx, fy = t, (0.5 if ry == 0 else ((slot % 2) * 0.9 + 0.05))
            else:        fy, fx = t, (0.5 if rx == 0 else ((slot % 2) * 0.9 + 0.05))
        x = int(round(fx * rx)) if rx > 0 else 0
        y = int(round(fy * ry)) if ry > 0 else 0
    return g.crop((x, y, x + P, y + P))

def make_view(g, r, train, view=0, nview=1):
    v = r["view"]
    if v == "patch":
        return patchgrid(g, train, view, nview).resize((r["w"], r["h"]), Image.BILINEAR)
    return letterbox(g, r["h"], r["w"], train)               # 'square' & 'wide' sama fungsinya

# ---- augmentasi (diwarisi dari v6/v7/v10 tanpa perubahan setelan) ----
class AddNoise(nn.Module):
    def __init__(s, p=0.3, sd=0.04): super().__init__(); s.p, s.sd = p, sd
    def forward(s, x):
        return (x + torch.randn_like(x)*s.sd).clamp(0, 1) if random.random() < s.p else x

AUG_GEO = T.Compose([
    T.RandomApply([T.RandomRotation(4, fill=255)], p=0.5),
    T.RandomApply([T.RandomAffine(0, translate=(0.02, 0.05), shear=4, fill=255)], p=0.3),
    T.ColorJitter(brightness=0.30, contrast=0.30)])
TO_T = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
# AddNoise ada di rantai augmentasi v6/v7/v10. Sempat hilang di draf v11, yang
# membuat run JANGKAR ft-v7 melatih resep yang BERBEDA dari v7 dan karena itu
# tidak menjangkar apa pun. Dikembalikan pada posisi yang sama: setelah ToTensor,
# sebelum Normalize.
NOISE = AddNoise()
ERASE = T.RandomErasing(p=0.25, scale=(0.01, 0.06))

def degrade(g):
    """Domain randomization: rusak gambar TRAIN supaya menyerupai statistik TEST.
    Setelan dikembalikan ke v6 karena BUKTI LEADERBOARD: v6 (blur 0.6-2.2) =
    0.83902 sementara v7 (blur 0.18-0.66, 'dikalibrasi' ke statistik test) =
    0.83422. Kalibrasi ke tingkat korupsi test justru MENURUNKAN skor 0.0048."""
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
    """AugMax/AugMix (Wang dkk., NeurIPS 2021): campur BEBERAPA rantai augmentasi
    dgn bobot Dirichlet - keberagaman DAN kekerasan, bukan salah satu."""
    k = AUGMAX["chains"]
    w = np.random.dirichlet([1.0]*k); m = np.random.beta(1.0, 1.0)
    a = np.asarray(g, np.float32); mix = np.zeros_like(a)
    for i in range(k): mix += w[i] * np.asarray(_chain(g.copy()), np.float32)
    return Image.fromarray(np.clip((1-m)*a + m*mix, 0, 255).astype(np.uint8), "L")

class ScriptDS(Dataset):
    """r wajib punya: view, h, w, canon, mean, std."""
    def __init__(s, df, r, train, view=0, nview=1):
        s.p = df.path.tolist(); s.y = df.y.tolist() if "y" in df else [0]*len(df)
        s.r, s.t, s.v, s.n = r, train, view, nview
        s.norm = T.Normalize(r.get("mean", MEAN_DEF), r.get("std", STD_DEF))
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        g = load_img(s.p[i], s.r["canon"]) if isinstance(s.p[i], str) \
            else Image.new("L", (64, 64), 255)
        if s.t:
            g = augmax(g) if (AUGMAX["enable"] and random.random() < AUGMAX["p"]) else degrade(g)
        g = make_view(g, s.r, s.t, s.v, s.n)
        if s.t: g = AUG_GEO(g)
        x = TO_T(g)
        if x.shape[0] == 1: x = x.repeat(3, 1, 1)             # abu-abu -> 3 kanal
        if s.t: x = NOISE(x)                                  # urutan sama spt v6/v7
        x = s.norm(x)
        if s.t: x = ERASE(x)
        return x, s.y[i]

# =============================================================================
# 5. BACKBONE: timm + HuggingFace, TANPA substitusi senyap
# =============================================================================
# v10 `build()` diam-diam jatuh ke resnet34 kalau nama backbone salah, jadi run
# yang seolah 'dinov2' sebenarnya resnet34 - dan handoff mencatat jebakan itu
# pernah kejadian (`siglip_base_patch16_224` yang tidak eksis). Di v11 sumber
# yang gagal DILEWATI dan alasannya dicetak. Lebih baik kehilangan satu sumber
# daripada mengukur model yang salah.
def hf_pool(out):
    po = getattr(out, "pooler_output", None)
    if po is not None and getattr(po, "ndim", 0) == 2: return po
    h = out.last_hidden_state
    if h.ndim == 4: return h.mean((2, 3))                    # ConvNext: B,C,H,W
    if h.shape[1] > 1: return h[:, 1:].mean(1)               # ViT/BEiT: buang CLS
    return h.mean(1)

class HFBody(nn.Module):
    """bungkus AutoModel jadi ekstraktor fitur B x D.
    `interpolate_pos_encoding` dipakai kalau modelnya mendukung, supaya resolusi
    selain resolusi asli tidak langsung melempar error. ConvNext tidak menerima
    argumen itu, jadi dukungannya diperiksa SEKALI di __init__ dan bukan
    try/except tiap batch."""
    def __init__(s, m):
        super().__init__(); s.m = m
        try:
            s.ipe = "interpolate_pos_encoding" in inspect.signature(m.forward).parameters
        except Exception:
            s.ipe = False
    def forward(s, x):
        if s.ipe: return hf_pool(s.m(pixel_values=x, interpolate_pos_encoding=True))
        return hf_pool(s.m(pixel_values=x))

class Net(nn.Module):
    def __init__(s, body, feat, nc):
        super().__init__(); s.body = body; s.head = nn.Linear(feat, nc); s.feat = feat
    def forward(s, x): return s.head(s.body(x))

# Cache dipakai supaya bobot tidak diunduh 5x (sekali per fold), TAPI yang
# dikembalikan selalu SALINAN. Ini bug yang ditemukan lewat smoke test: kalau
# modul yang sama dikembalikan berulang kali, fine-tuning fold 0 akan MENGUBAH
# bobot yang dipakai fold 1, dan tahap LP-FT berhenti berangkat dari bobot
# pra-latih. Gejalanya di smoke test: fold pertama sehat, fold berikutnya
# kolaps, dan hasilnya berubah tiap kali dijalankan.
_LOADED = {}     # (src,name,h,w) -> (body_pristine, dim, mean, std) | None

def load_body(src, name, h=None, w=None, fresh=True):
    """kembalikan (body, dim, mean, std) atau None.
    `fresh=True` -> salinan baru yang aman untuk dilatih.
    Kunci cache MEMUAT resolusi: ViT dgn position embedding absolut adalah model
    lain kalau img_size-nya lain."""
    key = (src, name, h, w)
    if key in _LOADED:
        got = _LOADED[key]
        if got is None: return None
        body, dim, mean, std = got
        return (copy.deepcopy(body) if fresh else body), dim, mean, std
    res = None
    try:
        if src == "timm":
            import timm
            def _mk(pre):
                # Banyak ViT punya resolusi asli TETAP (dinov2 reg4 = 518,
                # siglip = 224). Tanpa img_size, timm melempar
                # "Input height doesn't match model" dan sumbernya hilang.
                # Dengan img_size, timm menginterpolasi position embedding.
                # ConvNeXt & EfficientNet tidak menerima img_size -> fallback.
                if h and w:
                    try:
                        return timm.create_model(name, pretrained=pre, num_classes=0,
                                                 img_size=(h, w))
                    except TypeError:
                        pass                       # backbone konvolusional: bebas ukuran
                return timm.create_model(name, pretrained=pre, num_classes=0)
            try:
                m = _mk(True)
            except Exception as e:
                if not SMOKE: raise
                print(f"    (SMOKE) unduhan gagal {type(e).__name__} -> bobot ACAK")
                m = _mk(False)
            mean, std = MEAN_DEF, STD_DEF
            try:
                from timm.data import resolve_data_config
                dc = resolve_data_config({}, model=m)
                mean, std = list(dc.get("mean", mean)), list(dc.get("std", std))
            except Exception: pass
            res = (m, int(m.num_features), mean, std)
        elif src == "hf":
            from transformers import AutoModel, AutoConfig
            kw = {}
            try:
                cf = AutoConfig.from_pretrained(name)
                # ViT-MAE default-nya menutup 75% patch. mask_ratio=0 menyimpan
                # SEMUA patch; urutannya diacak tapi position embedding sudah
                # ditambahkan sebelum pengacakan, dan kita mean-pool, jadi
                # hasilnya tetap deterministik (dicek di smoke test).
                if getattr(cf, "model_type", "") == "vit_mae": kw["mask_ratio"] = 0.0
                native = getattr(cf, "image_size", None)
                if native and h and (h != native or w != native):
                    print(f"    catatan: resolusi asli {name} = {native}, diminta {h}x{w}"
                          f" -> position embedding akan diinterpolasi kalau didukung")
            except Exception as e:
                if not SMOKE: raise
                print(f"    (SMOKE) config gagal {type(e).__name__} -> sumber dilewati")
                _LOADED[key] = None; return None
            m = AutoModel.from_pretrained(name, **kw)
            mean, std = MEAN_DEF, STD_DEF
            try:
                from transformers import AutoImageProcessor
                ip = AutoImageProcessor.from_pretrained(name)
                if getattr(ip, "image_mean", None): mean = list(ip.image_mean)
                if getattr(ip, "image_std", None):  std = list(ip.image_std)
            except Exception: pass
            dim = int(getattr(m.config, "hidden_size", 0) or
                      (m.config.hidden_sizes[-1] if getattr(m.config, "hidden_sizes", None) else 0))
            if dim <= 0: raise RuntimeError("dimensi fitur tidak terbaca dari config")
            res = (HFBody(m), dim, mean, std)
        else:
            raise ValueError(f"src tidak dikenal: {src}")
    except Exception as e:
        print(f"    !! SUMBER DILEWATI: {src}:{name} -> {type(e).__name__}: {e}")
        print("       (tidak ada penggantian backbone - lebih baik kurang satu sumber"
              " daripada mengukur model yang salah, yang persis terjadi di v10)")
        _LOADED[key] = None; return None
    _LOADED[key] = res
    body, dim, mean, std = res
    return (copy.deepcopy(body) if fresh else body), dim, mean, std

# =============================================================================
# 6. EKSTRAKSI FITUR BEKU
# =============================================================================
FEAT_DIR = os.path.join(WORK, "_feat"); os.makedirs(FEAT_DIR, exist_ok=True)

@torch.no_grad()
def extract(body, df, r, nview):
    """rata-rata fitur antar nview (utk 'patch' = rata-rata antar posisi patch)."""
    body.eval().to(DEV)
    acc = None
    for k in range(nview):
        out = []
        for x, _ in DataLoader(ScriptDS(df, r, False, k, nview), batch_size=CFG["batch"]*2,
                               shuffle=False, num_workers=CFG["nw"]):
            x = x.to(DEV, non_blocking=True)
            with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                out.append(body(x).float().cpu().numpy())
        f = np.concatenate(out) if out else np.zeros((len(df), 1), np.float32)
        acc = f if acc is None else acc + f
    return (acc / nview).astype(np.float32)

# --- diagnostik view patch: apakah n view benar-benar melihat isi yang BERBEDA?
# --- Kalau tidak, n lintasan maju terbuang dan - lebih buruk - banyaknya patch
# --- unik akan berkorelasi dgn AR, yaitu pintasan yang justru mau dibuang.
if any(z["view"] == "patch" for z in ZOO):
    _n = PATCH["n"]; _u = {"blok": [], "strip": []}
    _rs = np.random.RandomState(SEED)
    _pt = [p for p in te_df.path.tolist() if isinstance(p, str)]
    _ix = _rs.choice(len(_pt), min(120, len(_pt)), replace=False)
    for _i in _ix:
        try:
            with Image.open(_pt[_i]) as _im:
                _g = _im.convert("L"); _g.thumbnail((CACHE_MAX, CACHE_MAX))
                _ar = _im.size[0] / max(_im.size[1], 1)
        except Exception: continue
        _s = {np.asarray(patchgrid(_g, False, k, _n)).tobytes() for k in range(_n)}
        _u["blok" if _ar < AR_SPLIT else "strip"].append(len(_s))
    print(f"\ndiagnostik view patch (n={_n} view per gambar, contoh {len(_ix)} gambar test):")
    for _k, _v in _u.items():
        if _v: print(f"  {_k:5s} n={len(_v):3d}  patch unik: min {min(_v)}  "
                     f"median {int(np.median(_v))}  maks {max(_v)}")
    _allu = _u["blok"] + _u["strip"]
    if _allu and min(_allu) < 2:
        print("  ! ada gambar yang semua view-nya identik -> keberagaman patch gagal"
              " di gambar itu; naikkan keragaman PATCH['scales'].")
    elif _allu:
        print("  OK: setiap gambar melihat >=2 potongan berbeda di semua view.")

print("\n" + "=" * 70 + "\nSTEP 2 - EKSTRAKSI FITUR BEKU (backbone TIDAK dilatih)\n" + "=" * 70)
print(f"{len(ZOO)} sumber. Biaya tiap sumber = satu lintasan inferensi atas "
      f"{len(tr_df)}+{len(te_df)} gambar, bukan pelatihan CNN.")
FEAT = OrderedDict()      # tag -> dict(tr=..., te=..., dim=..., note=...)
for z in ZOO:
    nview = PATCH["n"] if z["view"] == "patch" else 1
    # Nama cache memuat TANDA TANGAN semua hal yang mengubah isi fitur. Kalau
    # cache hanya diberi nama tag, mengubah `canon`/`view`/resolusi sebuah entri
    # akan diam-diam memakai matriks fitur LAMA - dan itu justru akan merusak
    # gerbang CANON di tabel [A3], yaitu uji terkontrol utama skrip ini.
    _sig = hashlib.md5(json.dumps(
        [z["src"], z["name"], z["view"], z["h"], z["w"], bool(z["canon"]), nview,
         PATCH["size"], PATCH["minside"], list(PATCH["scales"]), MAXSIDE, CACHE_MAX,
         CANON_BLUR_PRE, len(tr_df), len(te_df)], sort_keys=True).encode()).hexdigest()[:10]
    fp = os.path.join(FEAT_DIR, f"{z['tag']}_{_sig}.npz")
    if os.path.exists(fp):
        try:
            d = np.load(fp)
            if len(d["tr"]) == len(tr_df) and len(d["te"]) == len(te_df):
                FEAT[z["tag"]] = dict(tr=d["tr"], te=d["te"], dim=d["tr"].shape[1], z=z)
                print(f"  {z['tag']:11s} dipakai dari cache fitur ({d['tr'].shape[1]}-dim, "
                      f"sig {_sig})")
                continue
        except Exception: pass
    print(f"\n  {z['tag']:11s} {z['src']}:{z['name']}")
    print(f"              view={z['view']} {z['h']}x{z['w']} canon={z['canon']} nview={nview}")
    print(f"              {z['note']}")
    got = load_body(z["src"], z["name"], z["h"], z["w"], fresh=False)   # beku: tak dilatih
    if got is None: continue
    body, dim, mean, std = got
    r = dict(z); r["mean"], r["std"] = mean, std
    warm_cache(tr_df.path.tolist() + te_df.path.tolist(), z["canon"], z["tag"])
    t0 = time.time()
    try:
        ftr = extract(body, tr_df, r, nview)
        fte = extract(body, te_df, r, nview)
    except Exception as e:
        print(f"    !! ekstraksi gagal: {type(e).__name__}: {e} -> sumber dilewati")
        body.to("cpu"); del body
        if DEV == "cuda": torch.cuda.empty_cache()
        continue
    body.to("cpu"); del body
    if DEV == "cuda": torch.cuda.empty_cache()
    FEAT[z["tag"]] = dict(tr=ftr, te=fte, dim=ftr.shape[1], z=z)
    print(f"    OK {ftr.shape[1]}-dim, {time.time()-t0:.0f}s")
    try: np.savez_compressed(fp, tr=ftr, te=fte)
    except Exception as e: print(f"    ! cache fitur tak tersimpan ({type(e).__name__})")

assert FEAT, ("Semua sumber fitur gagal. Cek: internet Kaggle aktif? "
              "`transformers`/`timm` terpasang? Jalankan dgn INFEST_SMOKE=1 "
              "hanya untuk menguji jalur kode, BUKAN untuk submit.")
print(f"\n{len(FEAT)}/{len(ZOO)} sumber fitur berhasil: {list(FEAT)}")


# =============================================================================
# 6b. VALIDASI ADVERSARIAL + OOF BERBOBOT-KEPENTINGAN   (baru di v12)
# =============================================================================
# Kenapa bagian ini ada, dan kenapa ia lebih penting daripada model mana pun
# di skrip ini.
#
# Empat pasang (OOF tertimbang, LB) yang kita punya:
#     v3.1  0.9716 / 0.8073
#     v5    0.9685 / 0.81054
#     v7    0.9685 / 0.83422
#     v11   0.9715 / 0.85030
# Korelasi Pearson +0.16, korelasi PERINGKAT (Spearman) -0.20. Simpangan baku
# OOF cuma 0.0018 sementara simpangan baku LB 0.0204 - sebelas kali lipat.
# Artinya: OOF tertimbang SUDAH JENUH dan hampir tidak membawa informasi
# tentang urutan. v3.1 punya OOF TERTINGGI dan LB TERENDAH.
# Empat titik memang sedikit, jadi ini indikasi, bukan bukti. Tapi arahnya
# jelas, dan konsekuensinya serius: SETIAP gerbang di skrip ini - pemilihan
# ensemble greedy, gerbang bobot per-kelas, perbandingan LP-FT vs FT - dinilai
# memakai angka itu.
#
# Perbaikannya yang standar adalah validasi adversarial: latih pengklasifikasi
# untuk membedakan gambar TRAIN dari gambar TEST. Kalau ia berhasil, kedua
# himpunan memang tidak sedistribusi, dan skornya memberi bobot kepentingan
#     w(x) = p_test(x) / (1 - p_test(x))
# yang membuat OOF dinilai ulang seolah-olah train berkomposisi seperti test.
# Ini generalisasi dari popf1(): popf1 hanya menimbang ulang SATU sumbu
# (strip vs blok), w(x) menimbang ulang seluruh tampilan gambar.
#
# Yang dilaporkan: AUC adversarial, sebaran bobot, dan kolom `imp` di tabel [A]
# di samping `tertimbang`. Kalau keduanya mengurutkan sumber secara BERBEDA,
# itu sendiri temuan - dan submission_iw.csv adalah taruhan atas urutan `imp`.
IWCFG = dict(clip_q=0.95, min_ess=0.25, min_auc=0.60)
ADV_AUC, IW, USE_IW = float("nan"), np.ones(len(tr_df)), False

def build_iw():
    """kembalikan (auc, bobot) dari fitur beku yang sudah terlanjur dihitung."""
    pref = [t for t in ("siglip2", "dino2", "cnx-sq", "mae-mss") if t in FEAT]
    src = pref[0] if pref else next((t for t in FEAT if FEAT[t]["z"]["view"] != "patch"),
                                    next(iter(FEAT)))
    a = FEAT[src]["tr"].astype(np.float32); b = FEAT[src]["te"].astype(np.float32)
    X = np.vstack([a, b])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    lab = np.r_[np.zeros(len(a)), np.ones(len(b))]
    p = np.zeros(len(X))
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, lab):
        m = LogisticRegression(C=0.1, max_iter=2000)
        m.fit(X[itr], lab[itr])
        p[iva] = m.predict_proba(X[iva])[:, 1]
    auc = roc_auc_score(lab, p)
    ptr = np.clip(p[:len(a)], 1e-3, 1 - 1e-3)
    raw = ptr / (1 - ptr)
    # Bobot p/(1-p) mentah rentangnya ekstrem: sebagian gambar dapat bobot ~0
    # (hilang sama sekali dari metrik) dan sedikit gambar mendominasi, sehingga
    # UKURAN SAMPEL EFEKTIF runtuh dan metriknya jadi lebih berisik daripada
    # yang diperbaikinya. Karena itu bobot dilunakkan, w^alpha, dgn alpha
    # TERBESAR yang masih menyisakan sampel efektif di atas lantai. alpha=1
    # adalah bobot kepentingan yang benar secara teori; alpha<1 menukar sedikit
    # bias demi ragam yang jauh lebih kecil - dan alpha yang terpakai dicetak,
    # bukan disembunyikan.
    best = None
    for al in (1.0, 0.7, 0.5, 0.35, 0.25, 0.15):
        w = raw ** al
        w = np.clip(w, 0, np.quantile(w, IWCFG["clip_q"]))
        w = w / max(w.mean(), 1e-9)
        e = float(w.sum()**2 / max((w**2).sum(), 1e-9)) / len(w)
        if best is None: best = (al, w, e)
        if e >= IWCFG["min_ess"]: best = (al, w, e); break
        best = (al, w, e)
    al, w, e = best
    return float(auc), w, src, al

print("\n" + "=" * 70 + "\nSTEP 2c - VALIDASI ADVERSARIAL (train vs test)\n" + "=" * 70)
try:
    ADV_AUC, IW, _advsrc, _alpha = build_iw()
    ess = float(IW.sum()**2 / max((IW**2).sum(), 1e-9))    # ukuran sampel efektif
    print(f"[V1] AUC memisahkan train dari test = {ADV_AUC:.4f}  (fitur: {_advsrc})")
    print(f"     0.5 = tidak terbedakan | 1.0 = dua himpunan yang sama sekali lain")
    print(f"     pelunakan bobot alpha = {_alpha:.2f} (1.00 = bobot kepentingan penuh)")
    print(f"     bobot kepentingan: min {IW.min():.3f}  p50 {np.median(IW):.3f}  "
          f"maks {IW.max():.3f}")
    print(f"     ukuran sampel efektif {ess:.0f} dari {len(IW)} "
          f"({100*ess/len(IW):.0f}%)")
    USE_IW = ADV_AUC > IWCFG["min_auc"] and ess > IWCFG["min_ess"] * len(IW)
    if ADV_AUC < IWCFG["min_auc"]:
        print("     -> train dan test praktis SEDISTRIBUSI. Kalau begitu jurang")
        print("        CV-LB 0.12 itu BUKAN pergeseran domain, dan penyebabnya")
        print("        harus dicari di tempat lain (pintasan, prior, atau bocor).")
    elif not USE_IW:
        print("     -> bobotnya terlalu timpang (sampel efektif kecil); dilaporkan")
        print("        saja, tidak dipakai sebagai gerbang.")
    else:
        print("     -> train dan test JELAS berbeda. Kolom `imp` di tabel [A] adalah")
        print("        OOF yang dinilai ulang seolah train berkomposisi spt test.")
except Exception as e:
    print(f"  ! validasi adversarial gagal ({type(e).__name__}: {e}) -> dilewati")

def impf1(p, mask):
    """macro-F1 dgn bobot kepentingan adversarial per sampel."""
    if mask.sum() < NC: return float("nan")
    return f1_score(ytrue[mask], p[mask].argmax(1), average="macro",
                    sample_weight=IW[mask], zero_division=0)

# =============================================================================
# 7. LINEAR PROBE PER FOLD  (tahap 1 dari LP-FT)
# =============================================================================
print("\n" + "=" * 70 + "\nSTEP 3 - LINEAR PROBE (backbone beku)\n" + "=" * 70)
print("Kumar dkk. ICLR 2022: di bawah pergeseran distribusi BESAR dgn fitur")
print("pra-latih bagus, linear probe MENGALAHKAN fine-tuning penuh out-of-")
print("distribution (66.2% vs 59.3%). Sepuluh versi pipeline ini belum pernah")
print("mengujinya, padahal gejalanya sudah terlihat: model lebih kuat -> OOF")
print("naik, LB diam (v5 OOF 0.9795 vs v3.1 0.9702, LB praktis sama).")

skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=SEED)
FOLDS = list(skf.split(tr_df, tr_df.y))

def probe_source(ftr, fte, tag=""):
    """regresi logistik per fold di atas fitur beku -> (oof, test_prob, head_per_fold).
    `head_per_fold` dipakai ulang sebagai inisialisasi kepala di tahap LP-FT."""
    oof = np.zeros((len(ftr), NC)); tp = np.zeros((len(fte), NC)); heads = []
    for itr, iva in FOLDS:
        mu, sd = ftr[itr].mean(0), ftr[itr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=PROBE["max_iter"], C=PROBE["C"],
                                 class_weight="balanced")   # class_weight -> macro-F1
        clf.fit((ftr[itr] - mu) / sd, ytrue[itr])
        def _full(F):
            q = clf.predict_proba((F - mu) / sd)
            o = np.zeros((len(F), NC))
            for j, c in enumerate(clf.classes_): o[:, int(c)] = q[:, j]
            return o
        oof[iva] = _full(ftr[iva]); tp += _full(fte)
        # Lipat standardisasi ke dalam bobot linear supaya kepala hasil probe bisa
        # dipasang langsung ke nn.Linear tanpa lapisan normalisasi tambahan:
        #   W((f-mu)/sd) + b  =  (W/sd) f  +  (b - W.(mu/sd))
        W = np.zeros((NC, ftr.shape[1])); b = np.zeros(NC)
        if clf.coef_.shape[0] == 1 and len(clf.classes_) == 2:
            # sklearn memberi SATU baris koefisien untuk soal biner (arah kelas
            # ke-1). Menyalinnya ke kedua kelas akan membalik tandanya untuk
            # kelas ke-0. Tidak terjadi pada 7 kelas, tapi jangan sampai salah
            # diam-diam kalau satu fold kehilangan kelas.
            c0, c1 = int(clf.classes_[0]), int(clf.classes_[1])
            W[c1], b[c1] = clf.coef_[0], clf.intercept_[0]
            W[c0], b[c0] = -clf.coef_[0], -clf.intercept_[0]
        else:
            for j, c in enumerate(clf.classes_):
                W[int(c)], b[int(c)] = clf.coef_[j], clf.intercept_[j]
        heads.append((W / sd, b - (W @ (mu / sd))))
    return oof, tp / len(FOLDS), heads

oof_by, test_by, HEADS, REPORT = OrderedDict(), OrderedDict(), {}, []
allmask = np.ones(len(tr_df), bool)
for tag, d in FEAT.items():
    t0 = time.time()
    o, t, hd = probe_source(d["tr"], d["te"], tag)
    oof_by[tag], test_by[tag], HEADS[tag] = o, t, hd
    wt, pp = popf1(o, allmask, detail=True)
    pl = f1_score(ytrue, o.argmax(1), average="macro")
    REPORT.append(dict(sumber=tag, jenis="probe", dim=d["dim"], view=d["z"]["view"],
                       canon=d["z"]["canon"], blok=pp["blok"][0], strip=pp["strip"][0],
                       polos=pl, tertimbang=wt))
    print(f"  {tag:11s} probe: blok {pp['blok'][0]:.4f}  strip {pp['strip'][0]:.4f}  "
          f"polos {pl:.4f}  TERTIMBANG {wt:.4f}   ({time.time()-t0:.0f}s)")

# ---- fusi fitur: konkatenasi beberapa backbone -> satu probe ----------------
if FUSE["enable"] and len(FEAT) >= 2:
    print("\n  -- fusi fitur (konkatenasi embedding, satu probe di atasnya) --")
    def _blocks(tags):
        tags = [t for t in tags if t in FEAT]
        if len(tags) < 2: return None
        A, B = [], []
        for t in tags:
            a, b = FEAT[t]["tr"].astype(np.float32), FEAT[t]["te"].astype(np.float32)
            if FUSE["l2norm"]:
                # L2-normalkan per blok, kalau tidak backbone dgn norma fitur
                # besar otomatis mendominasi konkatenasi tanpa alasan apa pun.
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
                b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
            A.append(a); B.append(b)
        return np.concatenate(A, 1), np.concatenate(B, 1), tags
    for fname, tags in FUSE["sets"].items():
        got = _blocks(list(FEAT) if tags is None else tags)
        if got is None:
            print(f"  {fname:11s} dilewati (butuh >=2 sumber yang berhasil)"); continue
        FA, FB, used = got
        o, t, hd = probe_source(FA, FB, fname)
        wt, pp = popf1(o, allmask, detail=True)
        oof_by[fname], test_by[fname] = o, t
        REPORT.append(dict(sumber=fname, jenis="fusi", dim=FA.shape[1], view="-",
                           canon="-", blok=pp["blok"][0], strip=pp["strip"][0],
                           polos=f1_score(ytrue, o.argmax(1), average="macro"),
                           tertimbang=wt))
        print(f"  {fname:11s} {FA.shape[1]:5d}-dim dari {len(used)} sumber -> "
              f"TERTIMBANG {wt:.4f}   ({', '.join(used)})")

# =============================================================================
# 8. LP-FT: fine-tune dgn kepala hasil probe sebagai inisialisasi
# =============================================================================
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

def is_collapsed(pred, f1):
    """Model kolaps = menumpuk prediksi di ~1 kelas (macro-F1 ~0.044). Dideteksi
    lewat KONSENTRASI, bukan ambang F1 absolut, supaya model yang sekadar LEMAH
    tidak ikut terbuang. Diuji di v2: 'semua 1 kelas' BUANG, 'lemah tapi sehat'
    (F1 0.378) PAKAI."""
    share = np.bincount(pred, minlength=NC).max() / max(len(pred), 1)
    return bool(share > 0.90 or f1 < 0.10)

CW = torch.tensor(cnt.sum() / (NC * cnt), dtype=torch.float32, device=DEV)
_SEEN_DECAY = {}
_SEEN_SYNINIT = {}

_DEPTH_TOK = ("layer", "layers", "block", "blocks", "stage", "stages", "stem",
              "embeddings", "patch_embed")

def _depth_of(name, nmax):
    """perkirakan kedalaman parameter dari NAMANYA. 0 = paling dekat input."""
    parts = name.split(".")
    for k, tok in enumerate(parts):
        if tok in ("embeddings", "patch_embed", "stem", "cls_token", "pos_embed"):
            return 0
        if tok in ("layer", "layers", "block", "blocks", "stage", "stages"):
            for q in parts[k+1:k+3]:
                if q.isdigit(): return 1 + int(q)
    return nmax          # head / norm akhir / tak dikenali -> LR penuh

def param_groups(model, lr, wd, decay):
    """Layer-wise LR decay (BEiT/ELECTRA): layer awal LR kecil, layer akhir penuh.

    DUA no-op senyap yang diperbaiki di sini:

    [1] `timm.optim.param_groups_layer_decay` mengembalikan grup ber-kunci
        `lr_scale`, dan AdamW polos TIDAK MEMBACA kunci itu; OneCycleLR lalu
        menimpa `lr` tiap grup. Jadi di v10 layer-wise decay tidak pernah
        benar-benar aktif. Di sini skalanya dikalikan ke `lr` secara eksplisit.

    [2] Untuk backbone HuggingFace, helper timm itu bahkan tidak bisa membaca
        strukturnya: HFBody tidak punya `group_matcher`, sehingga timm jatuh ke
        `auto_group_layers`, mendapat num_layers=1, dan mengembalikan lr_scale
        1.0 untuk SEMUA parameter - persis bug [1] dalam bentuk lain, dan justru
        pada sumber `dit-*` yang paling mungkin masuk tahap LP-FT. Karena itu
        pengelompokan di sini dihitung dari NAMA parameter, satu jalur untuk
        timm maupun HF, dan hasilnya DICETAK supaya bisa diperiksa, bukan
        dipercaya.

    Run JANGKAR (ft-v7) sengaja memakai decay=1.0: ia harus mereproduksi
    PERILAKU EFEKTIF v6/v7, bukan versi yang sudah diperbaiki - kalau tidak, ia
    berhenti menjadi jangkar.
    """
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
        d = min(_depth_of(n, nmax), nmax)
        nd = (p.ndim <= 1 or n.endswith(".bias"))       # norm/bias: tanpa weight decay
        buckets.setdefault((d, nd), []).append(p)
    gs = []
    for (d, nd), prm in sorted(buckets.items()):
        gs.append({"params": prm, "weight_decay": (0.0 if nd else wd),
                   "lr": lr * (decay ** (nmax - d))})
    return gs

def report_decay(model, lr, decay, tag):
    gs = param_groups(model, lr, CFG["wd"], decay)
    scales = sorted({round(g["lr"] / max(lr, 1e-12), 5) for g in gs})
    print(f"    layer-decay {decay}: {len(gs)} grup, {len(scales)} skala berbeda "
          f"({scales[0]:.4g} .. {scales[-1]:.4g})")
    if decay < 1.0 and len(scales) <= 1:
        print(f"    ! PERHATIAN: layer-decay TIDAK AKTIF untuk {tag} (nama parameter"
              f" tidak terbaca). Efeknya sama dgn decay=1.0.")

@torch.no_grad()
def predict_net(model, df, r, nview):
    acc = np.zeros((len(df), NC)); model.eval()
    for k in range(nview):
        out = []
        for x, _ in DataLoader(ScriptDS(df, r, False, k, nview), batch_size=CFG["batch"]*2,
                               shuffle=False, num_workers=CFG["nw"]):
            x = x.to(DEV, non_blocking=True)
            with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                out.append(torch.softmax(model(x).float(), 1).cpu().numpy())
        acc += np.concatenate(out)
    return acc / nview

def train_fold(r, dtr, dva, epochs, lr_body, lr_head, head_init, nview, seed, decay,
               body_init=None):
    """satu fold. head_init=(W,b) -> LP-FT; None -> fine-tune biasa (kepala acak).
    body_init = state_dict hasil pra-latih aksara sintetis; None -> bobot asli
    backbone (ImageNet/webli). Mengembalikan (model, f1, pred_val, r_terpakai);
    r_terpakai sudah memuat mean/std backbone sehingga pemanggil tidak perlu
    memuat ulang bobotnya."""
    seed_all(seed)
    got = load_body(r["src"], r["name"], r["h"], r["w"], fresh=True)   # WAJIB salinan
    if got is None: return None, -1, None, None
    body, dim, mean, std = got
    r = dict(r); r["mean"], r["std"] = mean, std
    if body_init is not None:
        # Salin state_dict-nya, JANGAN pasang objek bersama: sama seperti
        # load_body(fresh=True), fold berikutnya harus mulai dari bobot yang
        # sama persis, bukan dari bobot yang sudah dilatih fold sebelumnya.
        # Tensor disaring menurut BENTUK lebih dulu: pra-latih berjalan di
        # resolusi lain, dan `strict=False` TIDAK memaafkan beda ukuran - ia
        # tetap melempar RuntimeError. Yang bentuknya beda dilewati, bukan
        # membatalkan seluruh run.
        tgt = body.state_dict()
        okd = {k: v.clone() for k, v in body_init.items()
               if k in tgt and tuple(tgt[k].shape) == tuple(v.shape)}
        skip = len(body_init) - len(okd)
        body.load_state_dict(okd, strict=False)
        if _SEEN_SYNINIT.get(r["tag"]) is None:
            _SEEN_SYNINIT[r["tag"]] = True
            print(f"    init dari pra-latih aksara: {len(okd)}/{len(body_init)} tensor"
                  f" dipasang, {skip} dilewati karena bentuknya beda")
            if len(okd) < 0.5 * len(tgt):
                print("    ! PERHATIAN: kurang dari separuh tensor terpasang ->")
                print("      pra-latih praktis TIDAK aktif dan [A4] tidak mengukur apa pun.")
    model = Net(body, dim, NC)
    if head_init is not None:
        W, b = head_init
        with torch.no_grad():
            model.head.weight.copy_(torch.tensor(W, dtype=torch.float32))
            model.head.bias.copy_(torch.tensor(b, dtype=torch.float32))
    model = model.to(DEV)
    ld = DataLoader(ScriptDS(dtr, r, True), batch_size=CFG["batch"], shuffle=True,
                    num_workers=CFG["nw"], drop_last=(len(dtr) >= 2*CFG["batch"]),
                    pin_memory=(DEV == "cuda"))
    crit = nn.CrossEntropyLoss(weight=CW, label_smoothing=CFG["ls"])
    pg = param_groups(model.body, lr_body, CFG["wd"], decay)
    pg = pg + [{"params": list(model.head.parameters()), "weight_decay": 0.0, "lr": lr_head}]
    opt = torch.optim.AdamW(pg, lr=lr_body, weight_decay=CFG["wd"])
    if _SEEN_DECAY.get(r["tag"]) is None:          # cetak sekali per run, bukan per fold
        _SEEN_DECAY[r["tag"]] = True
        report_decay(model.body, lr_body, decay, r["tag"])
    steps = max(10, epochs * max(1, len(ld)))      # OneCycleLR pecah kalau kekecilan
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g.get("lr", lr_body)*2 for g in opt.param_groups],
        total_steps=steps, pct_start=0.25)
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
    ema = EMA(model, CFG["ema"]); best, best_pred, best_state = -1, None, None
    for ep in range(epochs):
        model.train()
        for x, y in ld:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); ema.update(model)
        # bandingkan bobot mentah vs EMA, ambil yang macro-F1-nya menang.
        # `nview` DIPAKAI di sini, bukan dipaku 1. Untuk sumber 'patch',
        # validasi 1-crop mengukur hal yang berbeda dari OOF/test yang
        # merata-ratakan n crop, sehingga fold sehat bisa dinilai kolaps lalu
        # dibuang. Gerbang checkpoint `ep >= epochs//3` (ada di v6/v7/v10, sempat
        # hilang di draf v11) juga menghemat sepertiga lintasan validasi dan
        # mencegah checkpoint epoch warm-up terpilih.
        if ep < epochs // 3: continue
        bak = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pr = predict_net(model, dva, r, nview).argmax(1)
        f_raw = f1_score(dva.y, pr, average="macro")
        model.load_state_dict(ema.state(model))
        pe = predict_net(model, dva, r, nview).argmax(1)
        f_ema = f1_score(dva.y, pe, average="macro")
        if f_ema < f_raw: model.load_state_dict(bak)
        f1 = max(f_raw, f_ema)
        if f1 > best:
            best, best_pred = f1, (pe if f_ema >= f_raw else pr)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(bak)
    if best_state is not None: model.load_state_dict(best_state)
    return model, best, best_pred, r

def run_ft(r, tag, head_init_per_fold, epochs, lr_body, lr_head, nview, decay,
           body_init=None):
    oof = np.zeros((len(tr_df), NC)); tp = np.zeros((len(te_df), NC))
    f1s, dead, used = [], 0, 0
    for fold, (itr, iva) in enumerate(FOLDS):
        dtr = tr_df.iloc[itr].reset_index(drop=True)
        dva = tr_df.iloc[iva].reset_index(drop=True)
        hi = head_init_per_fold[fold] if head_init_per_fold else None
        # lb/lh di-RESET tiap fold. Kalau tidak, penurunan lr akibat kolaps di
        # fold 0 akan ikut ke semua fold sesudahnya tanpa alasan.
        lb, lh, model, f1, bad, rr = lr_body, lr_head, None, -1, True, r
        for attempt in range(3):                   # GUARD kolaps: ulang dgn lr/3
            model, f1, vp, rr = train_fold(r, dtr, dva, epochs, lb, lh, hi, nview,
                                           SEED + fold*10 + attempt, decay, body_init)
            if model is None: return None
            bad = is_collapsed(vp, f1)
            if not bad: break
            lb /= 3; lh /= 3
            sh = np.bincount(vp, minlength=NC).max()/len(vp)
            print(f"    fold{fold} KOLAPS (F1={f1:.4f}, {sh*100:.0f}% prediksi 1 kelas)"
                  f" -> ulang dgn lr={lb:.1e}")
        if bad:
            dead += 1; print(f"    fold{fold} tetap kolaps -> DIKELUARKAN dari ensemble")
            del model
            if DEV == "cuda": torch.cuda.empty_cache()
            continue
        oof[iva] = predict_net(model, dva, rr, nview)
        tp += predict_net(model, te_df, rr, nview); used += 1
        f1s.append(f1); print(f"    fold{fold} macro-F1={f1:.4f}")
        del model
        if DEV == "cuda": torch.cuda.empty_cache()
    if used == 0:
        print(f"    !! {tag} gagal total, dilewati"); return None
    tp /= used                                     # rata-rata HANYA dari fold sehat
    return oof, tp, np.mean(f1s), np.std(f1s), dead, (oof.sum(1) > 0)

# ---- tahap 2 LP-FT untuk k sumber terbaik menurut OOF TERTIMBANG probe ------
if LPFT["enable"] and LPFT["k"] > 0:
    print("\n" + "=" * 70 + "\nSTEP 4 - LP-FT tahap 2 (fine-tune dari kepala probe)\n" + "="*70)
    cands = sorted([t for t in FEAT], key=lambda t: -popf1(oof_by[t], allmask))
    pick = cands[:LPFT["k"]]
    print(f"  {LPFT['k']} sumber terbaik menurut OOF tertimbang probe: {pick}")
    print(f"  Ini bagian TERMAHAL skrip. Turunkan LPFT['k'] kalau kena batas waktu GPU.")
    for tag in pick:
        z = FEAT[tag]["z"]; nview = PATCH["n"] if z["view"] == "patch" else 1
        nt = f"lpft-{tag}"
        print(f"\n  -- {nt}: {z['src']}:{z['name']} {z['view']} {z['h']}x{z['w']} "
              f"canon={z['canon']} --")
        heads = HEADS[tag]
        if z["view"] == "patch":
            # Seluruh premis LP-FT adalah kepala yang SELARAS dgn fitur yang akan
            # dilihat saat fine-tune, supaya gradien awal kecil dan fitur
            # pra-latih tidak dirusak. Kepala di HEADS[tag] dipasang pada fitur
            # RATA-RATA n patch, sedangkan latihnya memakai SATU patch acak per
            # langkah - dua distribusi yang berbeda, jadi premisnya batal persis
            # di sumber patch. Kepala dipasang ulang di fitur 1-patch. Biayanya
            # satu lintasan ekstraksi, dan hanya untuk sumber yang terpilih.
            print("    view=patch -> kepala probe dipasang ulang pada fitur 1-patch")
            got = load_body(z["src"], z["name"], z["h"], z["w"], fresh=False)
            if got is None: continue
            b1, _d1, m1, s1 = got
            r1 = dict(z); r1["mean"], r1["std"] = m1, s1
            try:
                f1tr, f1te = extract(b1, tr_df, r1, 1), extract(b1, te_df, r1, 1)
                _o1, _t1, heads = probe_source(f1tr, f1te, tag + "-1p")
                print(f"    probe 1-patch OOF tertimbang {popf1(_o1, allmask):.4f} "
                      f"(vs {popf1(oof_by[tag], allmask):.4f} pada rata-rata {nview} patch)")
            except Exception as e:
                print(f"    ! gagal memasang ulang kepala ({type(e).__name__}) -> "
                      f"pakai kepala rata-rata, premis LP-FT melemah")
            b1.to("cpu")
            if DEV == "cuda": torch.cuda.empty_cache()
        res = run_ft(z, nt, heads, LPFT["epochs"], LPFT["lr_body"], LPFT["lr_head"],
                     nview, LPFT["layer_decay"])
        if res is None: continue
        o, t, fm, fs, dead, ok = res
        oof_by[nt], test_by[nt] = o, t
        wt, pp = popf1(o, ok, detail=True)
        REPORT.append(dict(sumber=nt, jenis="LP-FT", dim=FEAT[tag]["dim"], view=z["view"],
                           canon=z["canon"], blok=pp["blok"][0], strip=pp["strip"][0],
                           polos=f1_score(ytrue[ok], o[ok].argmax(1), average="macro"),
                           tertimbang=wt))
        base = popf1(oof_by[tag], allmask)
        print(f"    fold mean={fm:.4f} sd={fs:.4f} | fold mati={dead}")
        print(f"    probe {base:.4f} -> LP-FT {wt:.4f} ({wt-base:+.4f})  "
              f"<- inilah uji langsung klaim Kumar dkk. di data kita")

# ---- PRA-LATIH ANTARA di korpus aksara sintetis (baru di v12) ---------------
SYN_STATE, SYN_ZERO = None, None

def pretrain_syn(r):
    """Latih backbone pada korpus aksara sintetis, lalu kembalikan state_dict
    BODY-nya saja (kepala dibuang; fine-tune nyata memasang kepalanya sendiri).

    Ini 'pra-latih tugas antara': alih-alih mencari backbone HuggingFace yang
    sudah melihat aksara Nusantara - yang setelah dicari memang TIDAK ADA -
    tugas antaranya dibuat sendiri. Biayanya kecil: 14rb citra x 4 epoch kira2
    seperlima satu run fine-tuning 5-fold."""
    global SYN_ZERO
    seed_all(SEED)
    ph, pw = SYN.get("pre_h", r["h"]), SYN.get("pre_w", r["w"])
    got = load_body(r["src"], r["name"], ph, pw, fresh=True)
    if got is None:
        print("    ! backbone tidak bisa dimuat -> pra-latih dibatalkan"); return None
    body, dim, mean, std = got
    rr = dict(r); rr["h"], rr["w"] = ph, pw
    rr["mean"], rr["std"] = mean, std
    model = Net(body, dim, NC).to(DEV)
    n = len(syn_df)
    idx = np.random.RandomState(SEED).permutation(n)
    nv = max(NC, int(n * SYN["val_frac"]))
    dva = syn_df.iloc[idx[:nv]].reset_index(drop=True)
    dtr = syn_df.iloc[idx[nv:]].reset_index(drop=True)
    ld = DataLoader(ScriptDS(dtr, rr, True), batch_size=CFG["batch"], shuffle=True,
                    num_workers=CFG["nw"], drop_last=(len(dtr) >= 2*CFG["batch"]),
                    pin_memory=(DEV == "cuda"))
    # korpus sintetis SEIMBANG menurut konstruksi, jadi tidak pakai CW (bobot
    # kelas yang dihitung dari cnt train nyata) - itu akan memiringkannya lagi.
    crit = nn.CrossEntropyLoss(label_smoothing=CFG["ls"])
    pg = param_groups(model.body, SYN["lr"], CFG["wd"], 1.0)
    pg = pg + [{"params": list(model.head.parameters()), "weight_decay": 0.0,
                "lr": SYN["lr"]*10}]
    opt = torch.optim.AdamW(pg, lr=SYN["lr"], weight_decay=CFG["wd"])
    steps = max(10, SYN["epochs"] * max(1, len(ld)))
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g.get("lr", SYN["lr"])*2 for g in opt.param_groups],
        total_steps=steps, pct_start=0.25)
    scaler = torch.amp.GradScaler(DEV, enabled=(DEV == "cuda"))
    t0 = time.time()
    for ep in range(SYN["epochs"]):
        model.train()
        for x, y in ld:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        pv = predict_net(model, dva, rr, 1).argmax(1)
        print(f"    epoch {ep+1}/{SYN['epochs']}  holdout sintetis macro-F1="
              f"{f1_score(dva.y, pv, average='macro'):.4f}  ({time.time()-t0:.0f}s)")

    # ---- [S2] uji transfer TANPA fine-tune apa pun ----
    present = sorted(syn_df.y.unique().tolist())
    pr = predict_net(model, tr_df, rr, 1).argmax(1)
    m = np.isin(tr_df.y.values, present)
    zero = f1_score(tr_df.y.values[m], pr[m], average="macro",
                    labels=present, zero_division=0)
    SYN_ZERO = float(zero)
    chance = 1.0 / len(present)
    print(f"\n[S2] TRANSFER LANGSUNG sintetis -> data NYATA (tanpa fine-tune):")
    print(f"     macro-F1 = {zero:.4f} pada {len(present)} kelas  |  tebakan acak"
          f" = {chance:.4f}  (n={int(m.sum())})")
    print("     Angka ini belum pernah ada di proyek ini. Ia mengukur satu hal")
    print("     yang tidak bisa diukur oleh OOF mana pun: seberapa jauh BENTUK")
    print("     AKSARA saja - tanpa ukuran, AR, atau derajat binarisasi - sudah")
    print("     cukup untuk memisahkan ketujuh kelas.")
    if zero < chance + 0.05:
        print("     -> praktis setara tebakan acak. Pra-latih ini kemungkinan besar")
        print("        TIDAK akan membantu; lihat selisih ft-syn vs ft-v7 di [A].")
    elif zero > 0.45:
        print("     -> jauh di atas acak meski jurang domain sintetis->foto lebar.")
        print("        Bentuk aksara memang bisa dibaca; pra-latih ini punya isi.")
    st = {k: v.detach().cpu().clone() for k, v in model.body.state_dict().items()}
    del model
    if DEV == "cuda": torch.cuda.empty_cache()
    return st

if SYN_OK and any(r.get("syn_init") for r in FT_RUNS):
    _rsyn = next(r for r in FT_RUNS if r.get("syn_init"))
    print("\n" + "=" * 70 + "\nSTEP 4b - PRA-LATIH ANTARA DI AKSARA SINTETIS\n" + "=" * 70)
    print(f"  backbone {_rsyn['name']} | pra-latih di "
          f"{SYN.get('pre_h', _rsyn['h'])}x{SYN.get('pre_w', _rsyn['w'])}, "
          f"fine-tune nanti di {_rsyn['h']}x{_rsyn['w']}")
    print(f"  {len(syn_df)} citra | {SYN['epochs']} epoch")
    try:
        SYN_STATE = pretrain_syn(_rsyn)
    except Exception as e:
        print(f"  !! pra-latih gagal ({type(e).__name__}: {e}) -> ft-syn akan"
              f" berjalan dari bobot biasa dan MENJADI DUPLIKAT ft-v7")
        SYN_STATE = None

# ---- jangkar: resep v6/v7 apa adanya (fine-tune biasa, kepala acak) --------
if FT_RUNS:
    print("\n" + "=" * 70 + "\nSTEP 5 - JANGKAR: resep v6/v7 (fine-tune biasa)\n" + "=" * 70)
    print("Ini satu-satunya sumber yang skor leaderboard-nya DIKETAHUI (0.839).")
    print("Tanpa jangkar, seluruh ensemble berdiri di atas angka yang belum diuji.")
    for r in FT_RUNS:
        if r.get("syn_init") and SYN_STATE is None:
            print(f"\n  -- {r['tag']} DILEWATI: pra-latih aksara tidak tersedia, jadi")
            print(f"     run ini akan identik dgn ft-v7 dan cuma membuang waktu GPU.")
            continue
        nview = PATCH["n"] if r["view"] == "patch" else r.get("tta", 1)
        print(f"\n  -- {r['tag']}: {r['name']} {r['view']} {r['h']}x{r['w']} "
              f"canon={r['canon']} --\n     {r['note']}")
        res = run_ft(r, r["tag"], None, r["epochs"], r["lr"], r["lr"], nview,
                     r.get("layer_decay", 1.0),
                     body_init=(SYN_STATE if r.get("syn_init") else None))
        if res is None: continue
        o, t, fm, fs, dead, ok = res
        oof_by[r["tag"]], test_by[r["tag"]] = o, t
        wt, pp = popf1(o, ok, detail=True)
        REPORT.append(dict(sumber=r["tag"], jenis="FT", dim=0, view=r["view"],
                           canon=r["canon"], blok=pp["blok"][0], strip=pp["strip"][0],
                           polos=f1_score(ytrue[ok], o[ok].argmax(1), average="macro"),
                           tertimbang=wt))
        print(f"    fold mean={fm:.4f} sd={fs:.4f} | fold mati={dead} | TERTIMBANG {wt:.4f}")

# =============================================================================
# 9. TABEL PERBANDINGAN + ENSEMBLE
# =============================================================================
print("\n" + "=" * 70 + "\nSTEP 6 - PERBANDINGAN SEMUA SUMBER\n" + "=" * 70)
assert REPORT, "Tidak ada sumber yang berhasil sama sekali."
rep = pd.DataFrame(REPORT)
# `imp` = OOF yang sama, tapi dinilai dgn bobot kepentingan adversarial. Kalau
# urutan menurut `tertimbang` dan menurut `imp` berbeda, dua metrik itu tidak
# setuju soal sumber mana yang terbaik - dan hanya satu dari keduanya yang
# bisa benar tentang leaderboard.
rep["imp"] = [impf1(oof_by[t], allmask & (oof_by[t].sum(1) > 0)) if t in oof_by
              else float("nan") for t in rep.sumber]
rep = rep.sort_values("tertimbang", ascending=False)
print("\n[A] Urut dari terbaik menurut OOF TERTIMBANG (bukan OOF polos):\n")
print(rep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n  'polos' = OOF di komposisi TRAIN; 'tertimbang' = ditimbang ke komposisi")
print("  strip/blok TEST. Selisih keduanya adalah besar penipuan yang dulu membuat")
print("  v3.1 terlihat 0.9716 padahal LB 0.8073.")
if "jenis" in rep:
    print("\n[A2] Rata-rata OOF tertimbang per jenis pendekatan:")
    for j, g in rep.groupby("jenis"):
        print(f"      {j:8s} n={len(g)}  terbaik {g.tertimbang.max():.4f}  "
              f"median {g.tertimbang.median():.4f}")

# --- gerbang CANON, kini di backbone TERBAIK. v11 sudah menguji ini di DiT
# --- (Otsu 0.8122 vs tanpa Otsu 0.8253 = -0.0131) tapi DiT adalah sumber
# --- TERLEMAH, jadi temuannya bisa saja khas DiT. Pasangan siglip2 di sini
# --- mengujinya di sumber terkuat, dgn biaya satu lintasan ekstraksi.
if "siglip2" in oof_by and "siglip2-canon" in oof_by:
    _b, _a = popf1(oof_by["siglip2"], allmask), popf1(oof_by["siglip2-canon"], allmask)
    print(f"\n[A3] Gerbang CANON pada backbone yang sama (SigLIP2, sumber terbaik):")
    print(f"      dengan Otsu {_a:.4f}  vs  tanpa Otsu {_b:.4f}  ({_a-_b:+.4f})")
    print(f"      Pembanding v11 di DiT: -0.0131 (merugikan). Kalau tandanya sama")
    print(f"      di sini, Otsu memang merugikan dan bukan gejala khas DiT.")

# --- [A4] UJI TERKONTROL: apakah pra-latih aksara sintetis berguna?
# --- ft-v7 dan ft-syn memakai backbone, resolusi, epoch, lr, augmentasi, fold
# --- dan seed yang SAMA PERSIS. Satu-satunya beda adalah titik awal bobot.
if "ft-v7" in oof_by and "ft-syn" in oof_by:
    _v, _y = popf1(oof_by["ft-v7"], allmask), popf1(oof_by["ft-syn"], allmask)
    print(f"\n[A4] NILAI PRA-LATIH AKSARA SINTETIS (uji terkontrol):")
    print(f"      ft-v7 (init ImageNet) {_v:.4f}  vs  ft-syn (init aksara) {_y:.4f}"
          f"  ({_y-_v:+.4f})")
    if SYN_ZERO is not None:
        print(f"      transfer langsung sintetis->nyata [S2] = {SYN_ZERO:.4f}")
    print("      Kedua run identik kecuali bobot awalnya, jadi selisih ini ADALAH")
    print("      nilai pra-latihnya - bukan taksiran. Kalau negatif, korpus")
    print("      sintetis merusak fitur dan harus dimatikan lewat SYN['enable'].")
elif "ft-syn" not in oof_by:
    print("\n[A4] ft-syn tidak berjalan -> nilai pra-latih aksara TIDAK terukur"
          " di run ini.")

# --- [A5] LP-FT vs FINE-TUNE BIASA, anggaran disamakan.
# --- Di v11 ft-v7 (0.9667) menang atas lpft-siglip2 (0.9577), tapi ft-v7 dapat
# --- 18 epoch di 384 tanpa Otsu sementara lpft-siglip2 dapat 10 epoch di 224
# --- DENGAN Otsu. Tiga beda sekaligus, jadi angka itu tidak menjawab apa pun.
# --- Di sini lpft-siglip2 dan ft-siglip2 memakai backbone, resolusi, epoch dan
# --- canon yang SAMA; bedanya cuma kepala probe sbg inisialisasi.
for _base in ("siglip2",):
    _l, _f = f"lpft-{_base}", f"ft-{_base}"
    if _l in oof_by and _f in oof_by:
        _a, _b = popf1(oof_by[_l], allmask), popf1(oof_by[_f], allmask)
        print(f"\n[A5] LP-FT vs fine-tune biasa pada {_base}, anggaran SAMA:")
        print(f"      LP-FT {_a:.4f}  vs  FT biasa {_b:.4f}  ({_a-_b:+.4f})")
        print("      Inilah uji bersih klaim Kumar dkk. (ICLR 2022) di data ini;")
        print("      perbandingan v11 (+/-) tercemar tiga perbedaan sekaligus.")

# --- greedy ensemble (Caruana dkk. 2004), dinilai di OOF TERTIMBANG ----------
tags = list(oof_by)
mask = np.ones(len(tr_df), bool)
for t in tags: mask &= oof_by[t].sum(1) > 0
cur, chosen, best = np.zeros((len(tr_df), NC)), [], -1
for _ in range(12):                            # mengulang model = memberi bobot
    cand = max(tags, key=lambda t: popf1((cur*len(chosen) + oof_by[t])/(len(chosen)+1), mask))
    sc = popf1((cur*len(chosen) + oof_by[cand])/(len(chosen)+1), mask)
    if sc <= best + 1e-6: break
    cur = (cur*len(chosen) + oof_by[cand])/(len(chosen)+1); chosen.append(cand); best = sc
    print(f"  greedy += {cand:13s} -> OOF tertimbang {sc:.4f}")
single = max(tags, key=lambda t: popf1(oof_by[t], mask))
if popf1(oof_by[single], mask) >= best: chosen, best = [single], popf1(oof_by[single], mask)
# Di v6/v7/v8 greedy SELALU berhenti di 1 model -> 4 dari 5 run terbuang. Paksa
# uji rata-rata semua sumber; pakai kalau tidak lebih buruk dari ambang derau.
if len(set(chosen)) == 1 and len(tags) > 1:
    _all = np.mean([oof_by[t] for t in tags], 0); _fa = popf1(_all, mask)
    print(f"  cek rata-rata SEMUA {len(tags)} sumber -> {_fa:.4f} (tunggal {best:.4f})")
    if _fa >= best - GATE["ens"]:
        chosen, best = list(tags), _fa
        print("  -> pakai ENSEMBLE penuh (lebih tahan pergeseran domain)")
w = pd.Series(Counter(chosen)) / len(chosen)
print(f"\n[B] Bobot ensemble terpilih (OOF tertimbang={best:.4f}):")
for t, v in w.sort_values(ascending=False).items(): print(f"      {t:13s} {v*100:5.1f}%")
print(f"    sumber tunggal terbaik: {single} ({popf1(oof_by[single], mask):.4f})")
oof = np.mean([oof_by[t] for t in chosen], 0)
test_prob = np.mean([test_by[t] for t in chosen], 0)

# --- greedy KEDUA, dinilai dgn bobot kepentingan adversarial -----------------
# Bukan pengganti, melainkan TARUHAN KEDUA. popf1 dan impf1 menimbang ulang hal
# yang berbeda: popf1 satu sumbu (strip vs blok) yang kita pilih sendiri, impf1
# seluruh tampilan gambar menurut apa yang benar-benar membedakan train dari
# test. Kalau keduanya memilih ensemble yang sama, tidak ada yang hilang. Kalau
# berbeda, perbedaannya ditulis sbg submission tersendiri - dan leaderboard yang
# memutuskan, bukan kita.
test_prob_iw, chosen_iw = None, []
if USE_IW and not np.isnan(ADV_AUC):
    cur2, best2 = np.zeros((len(tr_df), NC)), -1
    for _ in range(12):
        c2 = max(tags, key=lambda t: impf1((cur2*len(chosen_iw) + oof_by[t])/(len(chosen_iw)+1), mask))
        s2 = impf1((cur2*len(chosen_iw) + oof_by[c2])/(len(chosen_iw)+1), mask)
        if s2 <= best2 + 1e-6: break
        cur2 = (cur2*len(chosen_iw) + oof_by[c2])/(len(chosen_iw)+1)
        chosen_iw.append(c2); best2 = s2
    if chosen_iw:
        test_prob_iw = np.mean([test_by[t] for t in chosen_iw], 0)
        w2 = pd.Series(Counter(chosen_iw)) / len(chosen_iw)
        print(f"\n[V2] Ensemble menurut OOF BERBOBOT-KEPENTINGAN (imp={best2:.4f}):")
        for t, v in w2.sort_values(ascending=False).items(): print(f"      {t:13s} {v*100:5.1f}%")
        if sorted(set(chosen_iw)) == sorted(set(chosen)):
            print("      Sama persis dgn pilihan [B] -> kedua metrik sepakat, tidak")
            print("      ada slot submission tambahan yang perlu dibelanjakan.")
        else:
            print("      BERBEDA dari pilihan [B]. Dua metrik validasi tidak sepakat;")
            print("      submission_iw.csv adalah taruhan atas metrik adversarial.")

# --- optimasi bobot per-kelas untuk macro-F1 (coordinate ascent) -------------
# macro-F1 tidak dioptimalkan oleh argmax probabilitas: kelas kecil butuh dorongan.
# Ditala HANYA di OOF tertimbang, dipakai hanya kalau naik melebihi ambang derau.
print("\n" + "=" * 70 + "\nSTEP 7 - BOBOT PER-KELAS UNTUK MACRO-F1\n" + "=" * 70)
CW_OPT = np.ones(NC); _base_w = popf1(oof, mask)
for _ in range(3):
    for c in range(NC):
        bv, bs = CW_OPT[c], popf1(oof * CW_OPT, mask)
        for v in (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.45, 1.7):
            t = CW_OPT.copy(); t[c] = v
            s = popf1(oof * t, mask)
            if s > bs: bv, bs = v, s
        CW_OPT[c] = bv
_f_opt = popf1(oof * CW_OPT, mask)
USE_CW = _f_opt > _base_w + GATE["cw"]
print(f"  OOF tertimbang {_base_w:.4f} -> {_f_opt:.4f} ({_f_opt-_base_w:+.4f}) -> "
      f"{'PAKAI' if USE_CW else 'tolak (di bawah ambang derau)'}")
if USE_CW:
    print("  bobot: " + "  ".join(f"{CLASSES[i]}={CW_OPT[i]:.2f}" for i in range(NC)))
else:
    CW_OPT = np.ones(NC)

prior = cnt / cnt.sum()   # dipakai [P1]/[P2] dan [D]/[E]

# =============================================================================
# STEP 7b - KOREKSI PRIOR EM  (baru di v12)
# =============================================================================
# Kenapa ini ada. Diagnostik [E] menghitung PLAFON macro-F1 dari jumlah
# prediksi per kelas, dgn satu asumsi: prior_test ~ prior_train. Di v11 plafon
# itu 0.8264 sementara LB nyata 0.85030 - skornya MELAMPAUI plafonnya sendiri,
# yang hanya mungkin kalau asumsinya salah. Jadi komposisi kelas di test memang
# BERBEDA dari train, dan posterior model - yang dilatih pada prior train -
# bias ke prior yang salah.
#
# Akibat langsung: gerbang bobot per-kelas di v11 dinilai pada OOF, dan OOF
# memakai prior TRAIN. Gerbang itu mengukur hal yang salah, lalu menolak
# koreksi (+0.0019) yang mungkin justru dibutuhkan.
#
# Saerens, Latinne & Decaestecker (Neural Computation 2002) menaksir prior
# baru lewat EM dari probabilitas test itu sendiri, tanpa label:
#   E: q_ik  ∝ P(k|x_i) * (p_k / prior_train_k), dinormalkan per baris
#   M: p_k   = rata-rata q_ik
# Ini TIDAK dipercaya begitu saja: [P1] menguji dulu di OOF dgn pergeseran
# prior BUATAN, dan hasilnya selalu ditulis sbg file submission TERPISAH.
def sld_em(prob, prior_src, iters=400, tol=1e-7):
    ps = np.maximum(np.asarray(prior_src, float), 1e-12)
    p = ps.copy()
    for _ in range(iters):
        q = prob * (p / ps)
        q /= np.maximum(q.sum(1, keepdims=True), 1e-12)
        pn = q.mean(0)
        if np.abs(pn - p).max() < tol: p = pn; break
        p = pn
    q = prob * (p / ps)
    q /= np.maximum(q.sum(1, keepdims=True), 1e-12)
    return p, q

USE_SLD, SLD_PRIOR, SLD_TEST = False, None, None
if SLD["enable"]:
    print("\n" + "=" * 70 + "\nSTEP 7b - KOREKSI PRIOR EM (Saerens dkk. 2002)\n" + "=" * 70)
    _pm = mask & (oof.sum(1) > 0)
    _P, _Y = oof[_pm], ytrue[_pm]
    # Prior SUMBER bukan prior train mentah. Semua run di skrip ini dilatih dgn
    # CrossEntropyLoss ber-bobot kelas (CW), jadi posterior yang keluar sudah
    # sebagian menghapus ketidakseimbangan train; memakai cnt/cnt.sum() sebagai
    # prior sumber akan mengoreksi dua kali. Taksiran yang tepat adalah prior
    # yang SEBENARNYA tersirat di posterior itu, yaitu rata-ratanya di OOF.
    prior_src = _P.mean(0); prior_src = prior_src / prior_src.sum()
    print(f"  prior sumber (rata2 posterior OOF) vs prior train mentah:")
    for i, c in enumerate(CLASSES):
        print(f"    {c:10s} sumber {prior_src[i]*100:6.2f}%   train {prior[i]*100:6.2f}%")
    rs = np.random.RandomState(SEED)
    gains, errs, errs0 = [], [], []
    for _t in range(SLD["trials"]):
        # prior test BUATAN, lalu resample OOF ke prior itu
        tgt = rs.dirichlet(np.ones(NC) * 1.2)
        wsel = tgt[_Y] / np.maximum(prior[_Y], 1e-12)
        wsel = wsel / wsel.sum()
        sel = rs.choice(len(_Y), size=len(_Y), replace=True, p=wsel)
        Ps, Ys = _P[sel], _Y[sel]
        emp = np.bincount(Ys, minlength=NC) / len(Ys)
        f_raw = f1_score(Ys, Ps.argmax(1), average="macro", zero_division=0)
        p_hat, q = sld_em(Ps, prior_src, SLD["iters"], SLD["tol"])
        f_sld = f1_score(Ys, q.argmax(1), average="macro", zero_division=0)
        gains.append(f_sld - f_raw)
        errs.append(float(np.abs(p_hat - emp).sum()))
        errs0.append(float(np.abs(prior_src - emp).sum()))
    g = float(np.mean(gains))
    print(f"\n[P1] Uji di OOF dgn {SLD['trials']} pergeseran prior BUATAN:")
    print(f"     macro-F1 rata-rata {g:+.4f}  (menang {sum(1 for v in gains if v>0)}"
          f"/{len(gains)} percobaan, terburuk {min(gains):+.4f})")
    print(f"     galat prior L1: tanpa koreksi {np.mean(errs0):.4f} -> EM"
          f" {np.mean(errs):.4f}")
    USE_SLD = g > SLD["gate"]
    print(f"     -> {'PAKAI' if USE_SLD else 'tolak (di bawah ambang derau)'}"
          f" (ambang {SLD['gate']:+.4f})")
    print("     Catatan: uji ini mensimulasikan pergeseran prior MURNI. Kalau di")
    print("     test yang bergeser bukan cuma prior melainkan juga tampilan")
    print("     gambarnya, asumsi Saerens dkk. (P(x|k) tetap) ikut goyah dan")
    print("     hasilnya bisa berbeda. Karena itu EM tidak pernah menimpa file")
    print("     submission utama - ia ditulis sebagai file sendiri.")

    _pte = test_prob * CW_OPT
    _pte = _pte / np.maximum(_pte.sum(1, keepdims=True), 1e-12)   # harus posterior sah
    SLD_PRIOR, _q_te = sld_em(_pte, prior_src, SLD["iters"], SLD["tol"])
    SLD_TEST = _q_te
    print(f"\n[P2] Prior test taksiran EM vs prior sumber:\n")
    print(f"  {'kelas':10s}{'sumber%':>10s}{'EM%':>9s}{'selisih':>10s}")
    for i, c in enumerate(CLASSES):
        d = (SLD_PRIOR[i] - prior_src[i]) * 100
        print(f"  {c:10s}{prior_src[i]*100:10.2f}{SLD_PRIOR[i]*100:9.2f}{d:+10.2f}")
    print(f"  jarak L1 dari prior sumber = {np.abs(SLD_PRIOR-prior_src).sum():.4f}")
    print("  (kalau angka ini ~0, test memang sepriors train dan [E] v11 yang")
    print("   terbantahkan butuh penjelasan LAIN selain pergeseran prior)")

# =============================================================================
# 10. DIAGNOSTIK
# =============================================================================
oof_f = oof * CW_OPT
base = f1_score(ytrue[mask], oof_f[mask].argmax(1), average="macro")
bf1, _pp = popf1(oof_f, mask, detail=True)
print(f"\nF1 Macro OOF polos (campuran train)       : {base:.4f}   <- angka yang menipu")
print(f"  blok  (n={_pp['blok'][1]:5d}) : {_pp['blok'][0]:.4f}")
print(f"  strip (n={_pp['strip'][1]:5d}) : {_pp['strip'][0]:.4f}")
print(f"F1 Macro OOF TERTIMBANG ke komposisi test : {bf1:.4f}   <<< dasar pemilihan")
if base - bf1 > 0.005:
    print(f"  -> OOF polos melebihkan {base-bf1:+.4f}; selisih seperti inilah yang"
          f" dulu muncul sebagai jurang CV-LB.")

pred = oof_f.argmax(1)
print("\n[C] Laporan per kelas (ensemble akhir):\n")
print(classification_report(ytrue[mask], pred[mask], target_names=CLASSES,
                            digits=4, zero_division=0))
cm = confusion_matrix(ytrue[mask], pred[mask], labels=list(range(NC)))
print("Confusion matrix (baris = asli, kolom = prediksi):")
print(f"{'':>10s}" + "".join(f"{c[:7]:>8s}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    rs = max(cm[i].sum(), 1)
    print(f"{c:>10s}" + "".join(f"{v:8d}" for v in cm[i]) + f"   (recall {cm[i,i]/rs:.3f})")
pf = f1_score(ytrue[mask], pred[mask], average=None, labels=list(range(NC)), zero_division=0)
print("\nKelas paling lemah -> terkuat:")
for i in np.argsort(pf):
    off = cm[i].copy(); off[i] = 0; j = int(off.argmax())
    print(f"  {CLASSES[i]:10s} F1={pf[i]:.4f} n={cnt[i]:4d} tertukar dgn "
          f"'{CLASSES[j]}' ({off[j]}x)")

# =============================================================================
# 11. PREDIKSI TEST + SUBMISSION
# =============================================================================
print("\n" + "=" * 70 + "\nSTEP 8 - PREDIKSI TEST & SUBMISSION\n" + "=" * 70)
tp_f = test_prob * CW_OPT
te_pred = tp_f.argmax(1)
labels = [CLASSES[i] for i in te_pred]
# kebocoran: gambar test yang IDENTIK byte dgn gambar train labelnya sudah pasti
n_leak = 0
for k, iid in enumerate(te_df.image_id):
    if iid in LEAK and labels[k] != LEAK[iid]:
        labels[k] = LEAK[iid]; n_leak += 1
if LEAK: print(f"  {n_leak} label ditimpa dari kebocoran byte-identik "
               f"({len(LEAK)} terdeteksi)")

share = np.bincount([C2I[l] for l in labels], minlength=NC) / len(labels)
print("\n[D] Distribusi prediksi test vs prior train:\n")
print(f"  {'kelas':10s}{'train%':>9s}{'test%':>9s}{'selisih':>10s}")
for i, c in enumerate(CLASSES):
    d = (share[i] - prior[i]) * 100
    print(f"  {c:10s}{prior[i]*100:9.2f}{share[i]*100:9.2f}{d:+10.2f}"
          + ("   <-- CEK" if abs(d) > 3 else ""))
skew = float(np.abs(share - prior).max() * 100)
print(f"  skew maksimum = {skew:.2f}pp")

# PLAFON dari jumlah prediksi per kelas saja.
# RIWAYAT DAN PEMBATALANNYA - baca ini sebelum memakai angkanya:
#   v3.1 plafon 0.8181 / LB 0.8073 | v5 0.8445 / 0.8105 | v7 0.8407 / 0.8342.
#   Ketiganya mendarat TEPAT di bawah plafonnya, jadi alatnya tampak kokoh.
#   Lalu v11: plafon 0.8264, LB 0.85030 - MELAMPAUI plafonnya sendiri.
#   Plafon ini hanya punya SATU asumsi, prior_test ~ prior_train, jadi v11
#   membuktikan asumsi itu salah: komposisi kelas test memang berbeda.
#   Karena itu di v12 plafonnya dihitung DUA KALI - sekali dgn prior train
#   (versi lama, dipertahankan agar sebanding dgn riwayat) dan sekali dgn
#   prior taksiran EM. Versi EM-lah yang lebih layak dipercaya.
def _ceiling(pri, sh, n):
    out = []
    for i in range(NC):
        tp_i = min(pri[i]*n, sh[i]*n)                 # recall=1.0 terbaik yg mungkin
        pr_ = tp_i / max(sh[i]*n, 1e-9); rc_ = tp_i / max(pri[i]*n, 1e-9)
        out.append(0.0 if pr_+rc_ == 0 else 2*pr_*rc_/(pr_+rc_))
    return float(np.mean(out))

ntest = len(labels)
ceil_train = _ceiling(prior, share, ntest)
print(f"\n[E] PLAFON macro-F1 dari jumlah prediksi saja")
print(f"    dgn prior TRAIN (versi lama, sebanding dgn riwayat) : {ceil_train:.4f}")
if SLD_PRIOR is not None:
    ceil_em = _ceiling(SLD_PRIOR, share, ntest)
    print(f"    dgn prior taksiran EM (versi yang lebih layak)      : {ceil_em:.4f}")
else:
    ceil_em = ceil_train
    print("    (EM dimatikan -> hanya versi prior train yang tersedia)")
ceil_f1 = [ceil_em]
print(f"    Riwayat: v3.1 0.8181/LB 0.8073 | v5 0.8445/0.8105 | v7 0.8407/0.8342")
print(f"             | v11 0.8264/0.85030  <- MELAMPAUI plafon, asumsinya batal.")
print("\n    PERINGATAN: JANGAN memakai angka ini sebagai gerbang submit. Gerbang")
print("    otomatis serupa di v6/v7 memvonis 'JANGAN submit' pada run v7 yang")
print("    justru memberi kenaikan LB terbesar (+0.024), dan di v11 plafonnya")
print("    sendiri terbukti salah. Angkanya informatif, vonisnya tidak.")

def write_sub(name, labs):
    # cek DULU, tulis kemudian: file dgn jumlah baris salah tidak boleh sampai ada
    assert len(labs) == len(te_df), f"{name}: {len(labs)} label vs {len(te_df)} baris test"
    assert all(l in C2I for l in labs), f"{name}: ada label di luar 7 kelas yang sah"
    p = os.path.join(WORK, name)
    with open(p, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["image_id", "label"])
        for iid, l in zip(te_df.image_id, labs): wr.writerow([iid, l])
    return p

paths_out = {"submission.csv": write_sub("submission.csv", labels)}

# varian TANPA bobot per-kelas: kalau gerbang bobot menyala, simpan juga versi
# polosnya supaya ada dua kandidat yang benar-benar berbeda untuk 2 slot final.
if USE_CW:
    lab2 = [CLASSES[i] for i in test_prob.argmax(1)]
    for k, iid in enumerate(te_df.image_id):
        if iid in LEAK: lab2[k] = LEAK[iid]
    paths_out["submission_tanpa_bobot.csv"] = write_sub("submission_tanpa_bobot.csv", lab2)

def _variant(name, probs):
    labs = [CLASSES[i] for i in probs.argmax(1)]
    for k, iid in enumerate(te_df.image_id):
        if iid in LEAK: labs[k] = LEAK[iid]
    paths_out[name] = write_sub(name, labs)

# varian jangkar: HANYA resep v6/v7, satu-satunya sumber ber-skor diketahui.
if "ft-v7" in test_by: _variant("submission_jangkar_v7.csv", test_by["ft-v7"])

# varian KOREKSI PRIOR EM: ditulis apa pun vonis [P1], karena vonis itu diambil
# dari simulasi di OOF dan yang benar-benar menjawab hanyalah leaderboard.
# Kalau [P1] menolak, file ini jangan dipakai untuk slot final - kecuali kamu
# memang mau membelanjakan satu submission untuk mengujinya.
if SLD_TEST is not None: _variant("submission_prior_em.csv", SLD_TEST)

# varian PENDEKATAN BARU sendirian: ft-syn tanpa dicampur apa pun. Ini satu-
# satunya cara mengukur pra-latih aksara di LEADERBOARD, bukan cuma di OOF.
if "ft-syn" in test_by: _variant("submission_syn.csv", test_by["ft-syn"])

# varian dari ensemble yang dipilih metrik ADVERSARIAL, kalau memang beda.
if test_prob_iw is not None and sorted(set(chosen_iw)) != sorted(set(chosen)):
    _variant("submission_iw.csv", test_prob_iw)

# ---- sub-diff: apakah varian ini benar-benar BEDA, atau buang slot? ---------
print("\n[F] Beda antar varian submission (bandingkan LABEL, bukan probabilitas):")
_names = list(paths_out)
for i in range(len(_names)):
    for j in range(i+1, len(_names)):
        a = pd.read_csv(paths_out[_names[i]]).label.values
        b = pd.read_csv(paths_out[_names[j]]).label.values
        agree = float((a == b).mean())
        print(f"  {_names[i]:28s} vs {_names[j]:28s} sepakat {agree*100:5.1f}% "
              f"({int((a != b).sum())} baris beda)"
              + ("   <- praktis identik, slot kedua mubazir" if agree > 0.995 else ""))

print("\nFile yang ditulis:")
for k, v in paths_out.items(): print(f"  {v}")
print(f"\nOOF TERTIMBANG akhir = {bf1:.4f} | plafon distribusi = {np.mean(ceil_f1):.4f}"
      f" | skew maks = {skew:.2f}pp")
if SMOKE:
    print("\n!! DIJALANKAN DENGAN INFEST_SMOKE=1: sebagian bobot mungkin ACAK.")
    print("   Semua angka di atas TIDAK BERARTI. Jangan submit hasil ini.")
print("\nYang harus diperiksa sebelum submit, berurutan:")
print("  1. [S1] audit pintasan korpus sintetis harus LULUS (~0.14, bukan ~0.78).")
print("     Kalau gagal, korpusnya mengajarkan pintasan dan [A4] tidak berarti.")
print("  2. [S2] transfer sintetis->nyata: angka baru, ini yang mengukur seberapa")
print("     jauh bentuk aksara saja sudah memisahkan kelas.")
print("  3. [A4] ft-syn vs ft-v7: uji TERKONTROL nilai pra-latih aksara.")
print("     [A5] LP-FT vs FT biasa pada backbone yang sama, anggaran disamakan.")
print("  4. [A3] gerbang Otsu di SigLIP2 - apakah temuan -0.0131 dari DiT bertahan.")
print("  5. [P1] apakah koreksi prior EM menang di simulasi, dan [P2] seberapa")
print("     jauh prior taksirannya dari prior train.")
print("  6. tabel [A]: 'fold mati' harus 0 pada baris LP-FT/FT.")
print("  7. [F] sub-diff: jangan pakai dua slot untuk dua file yang sepakat >99.5%.")
print("  8. Jumlah baris submission harus", len(te_df), "dan urutannya ikut sample_submission.")
print("\nKalau cuma ada 2 slot final: submission.csv (ensemble) plus SATU dari")
print("submission_prior_em.csv (kalau [P1] menang) atau submission_syn.csv")
print("(kalau [A4] positif) - pilih yang [F] tunjukkan paling BEDA dari yang utama.")
