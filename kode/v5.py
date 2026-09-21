# =============================================================================
# INFEST XII 2026 - Klasifikasi Citra Aksara Tradisional Nusantara
# Satu cell: load data -> EDA -> preprocessing -> training -> validasi -> submission
# Metrik: F1-Score Macro
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

# ---- Daftar model yang diadu. Tiap baris dilatih 5-fold & dilaporkan metriknya,
# ---- jadi kelihatan mana yang lemah. Kurangi isinya kalau waktu GPU terbatas.
RUNS = [
    # Dataset ini punya DUA populasi: strip teks (AR>3) dan blok/foto (AR<=3).
    # Di test porsinya 50/50, jadi ensemble harus melihat keduanya.
    #   heightnorm -> untuk strip teks (tinggi disamakan, jendela di-crop)
    #   letterbox persegi -> untuk blok/foto (tabel aksara, foto manuskrip)
    dict(tag="cnx-hn64",  backbone="convnext_tiny.in12k_ft_in1k",     view="heightnorm",
         h=64,  w=256, tta=5, lr=1e-4),
    dict(tag="cnx-sq384", backbone="convnext_tiny.in12k_ft_in1k",     view="letterbox",
         h=384, w=384, tta=1, lr=1e-4),
    dict(tag="eff-hn96",  backbone="tf_efficientnet_b0.ns_jft_in1k",  view="heightnorm",
         h=96,  w=384, tta=5, lr=3e-4),
    dict(tag="eff-sq320", backbone="tf_efficientnet_b0.ns_jft_in1k",  view="letterbox",
         h=320, w=320, tta=1, lr=3e-4),
    dict(tag="cnx-lb128", backbone="convnext_tiny.in12k_ft_in1k",     view="letterbox",
         h=128, w=512, tta=1, lr=1e-4),
]
SPEC = dict(enable=True, backbone="convnext_tiny.in12k_ft_in1k", view="letterbox",
            h=384, w=384, tta=1, lr=1e-4, epochs=22)   # spesialis pasangan tersulit
MAXSIDE = 1600      # gambar terbesar 5664x4248 -> perkecil dulu biar loading tak lambat
JPEG_AUG = dict(p=0.35, q=(30, 92))   # test 21% JPEG vs train 4.6% -> tiru artefaknya
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
def _ar(paths):
    out = []
    for q in paths:
        try:
            with Image.open(q) as im: out.append(im.size[0] / max(im.size[1], 1))
        except Exception: out.append(np.nan)
    return np.array(out, float)
BLOK_TR = _ar(tr_df.path.tolist()) < AR_SPLIT
W_BLOK  = float(np.nanmean(_ar(te_df.path.tolist()) < AR_SPLIT))   # diukur dari TEST
print(f"\npopulasi (AR<{AR_SPLIT} = 'blok'):")
print(f"  train: blok {BLOK_TR.mean()*100:5.1f}%  strip {100-BLOK_TR.mean()*100:5.1f}%")
print(f"  test : blok {W_BLOK*100:5.1f}%  strip {100-W_BLOK*100:5.1f}%   <- bobot penilaian")
if abs(BLOK_TR.mean() - W_BLOK) > 0.05:
    print("  PERHATIAN: komposisi train != test. OOF polos AKAN terlalu optimistis.")


# ------------------- 3. PREPROCESSING & AUGMENTASI ---------------------------
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def letterbox(im, H, W, train):
    w, h = im.size; s = min(H / h, W / w)
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

import io
def rand_jpeg(im):
    """Test berisi 21% JPEG vs train 4.6%. Latih model supaya kebal artefak JPEG."""
    if random.random() >= JPEG_AUG["p"]: return im
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=random.randint(*JPEG_AUG["q"])); buf.seek(0)
    return Image.open(buf).convert("RGB")

class ScriptDS(Dataset):
    def __init__(s, df, r, train, view=0, nview=1):
        s.p = df.path.tolist(); s.y = df.y.tolist() if "y" in df else [0]*len(df)
        s.r, s.t, s.v, s.n = r, train, view, nview
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        try:
            with Image.open(s.p[i]) as im:
                if max(im.size) > MAXSIDE: im.thumbnail((MAXSIDE, MAXSIDE))  # 5664x4248 -> cepat
                im = im.convert("RGB")
        except Exception: im = Image.new("RGB", (s.r["w"], s.r["h"]), (255, 255, 255))
        if s.t: im = rand_jpeg(im)
        return (AUG if s.t else PLAIN)(prep(im, s.r, s.t, s.v, s.n)), s.y[i]

# ----------------------------- 4. MODEL --------------------------------------
_OK = {}                       # cache: jangan coba unduh ulang tiap fold kalau sudah gagal
def build(name, nc):
    """Coba timm (pretrained) -> torchvision -> random init. Kembalikan (model, pretrained?)."""
    import torchvision.models as tvm
    if _OK.get(name, True):
        try:
            import timm
            m = timm.create_model(name, pretrained=True, num_classes=nc)
            _OK[name] = True; return m, True
        except Exception as e:
            _OK[name] = False
            print(f"    ! {name} tak bisa diunduh ({type(e).__name__}) -> fallback resnet34")
    try:
        m = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1); pre = True
    except Exception:
        m = tvm.resnet34(weights=None); pre = False
    m.fc = nn.Linear(m.fc.in_features, nc); return m, pre

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
def predict(model, df, r, tta=None):
    nv = tta or r["tta"]; acc = np.zeros((len(df), NC)); model.eval()
    for k in range(nv):
        out = []
        for x, _ in DataLoader(ScriptDS(df, r, False, k, nv), batch_size=CFG["batch"]*2,
                               shuffle=False, num_workers=CFG["nw"]):
            x = x.to(DEV, non_blocking=True)
            with torch.autocast(DEV, torch.float16, enabled=(DEV == "cuda")):
                out.append(torch.softmax(model(x).float(), 1).cpu().numpy())
        acc += np.concatenate(out)
    return acc / nv

def train_fold(r, dtr, dva, lr, epochs, seed):
    seed_all(seed)
    # drop_last hanya kalau datanya cukup, kalau tidak len(ld) bisa 0 -> scheduler pecah
    ld = DataLoader(ScriptDS(dtr, r, True), batch_size=CFG["batch"], shuffle=True,
                    num_workers=CFG["nw"], drop_last=(len(dtr) >= 2*CFG["batch"]),
                    pin_memory=(DEV == "cuda"))
    model, _ = build(r["backbone"], NC); model = model.to(DEV)
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
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # cegah divergensi
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
    f1s, dead, used = [], 0, 0
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
        oof[iva] = predict(model, dva, r); tp += predict(model, te_df, r); used += 1
        f1s.append(f1); print(f"  fold{fold} macro-F1={f1:.4f}")
        del model
        if DEV == "cuda": torch.cuda.empty_cache()
    if used == 0:
        print(f"  !! {r['tag']} gagal total, dilewati"); continue
    tp /= used                                                          # rata-rata HANYA fold sehat
    ok = oof.sum(1) > 0                                                 # baris tanpa model sehat
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
w = pd.Series(Counter(chosen)) / len(chosen)      # tampilkan sbg BOBOT, bukan daftar berulang
print(f"\n[C] Bobot ensemble terpilih (OOF={best:.4f}):")
for t, v in w.sort_values(ascending=False).items(): print(f"      {t:14s} {v*100:5.1f}%")
print(f"    model tunggal terbaik: {single} (OOF-tertimbang={popf1(oof_by[single], mask):.4f})")
oof = np.mean([oof_by[t] for t in chosen], 0); test_prob = np.mean([test_by[t] for t in chosen], 0)

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
print("\nDistribusi prediksi test vs train (selisih besar = tanda bahaya):")
pv = sub.label.value_counts()
for c in CLASSES:
    d = pv.get(c, 0)/len(sub)*100 - cnt[C2I[c]]/cnt.sum()*100
    print(f"  {c:10s} test {pv.get(c,0):5d} ({pv.get(c,0)/len(sub)*100:5.2f}%)  train "
          f"{cnt[C2I[c]]/cnt.sum()*100:5.2f}%  selisih {d:+5.2f}pp {'  <-- CEK' if abs(d) > 3 else ''}")
print(f"\n>>> F1-Score Macro OOF (ensemble {len(set(chosen))} model) = {bf1:.4f} <<<")
