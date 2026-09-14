"""Uji kebocoran sumber scan: bisakah fitur NON-PIKSEL memprediksi kelas aksara?

Jika bisa, model CNN apa pun yang dilatih pada dataset ini bisa mencapai skor
tinggi dengan menghafal "gambar ini dari scan buku mana", bukan dengan
mengenali bentuk aksara.

Label acuan = prediksi visual independen (submission_claude.csv, ~0.91 akurat).
Itu memberi plafon: seandainya metadata memprediksi label SEBENARNYA dengan
sempurna, kecocokan terukurnya tetap ~0.91.
"""
import csv, os, collections, numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.dummy import DummyClassifier

rows = list(csv.DictReader(open('submission_claude.csv')))
recs = []
for r in rows:
    p = 'data/test/' + r['image_id']
    im = Image.open(p); w, h = im.size
    g = np.array(im.convert('L')).astype(np.float32)
    bg = g[g >= 128]
    recs.append(dict(label=r['label'], w=w, h=h, aspect=w/h, npix=w*h,
                     fsize=os.path.getsize(p), bpp=os.path.getsize(p)/(w*h),
                     levels=len(np.unique(g.astype(np.uint8))),
                     bg_mean=float(bg.mean()) if bg.size else 255.0,
                     bg_std=float(bg.std()) if bg.size else 0.0,
                     g_mean=float(g.mean()), g_std=float(g.std())))

y = np.array([r['label'] for r in recs])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
def run(cols, est=None):
    X = np.array([[r[c] for c in cols] for r in recs], dtype=float)
    est = est or RandomForestClassifier(400, random_state=0, n_jobs=-1)
    p = cross_val_predict(est, X, y, cv=cv)
    return accuracy_score(y, p), f1_score(y, p, average='macro')

CONTAINER = ['w', 'h', 'aspect', 'npix', 'fsize']          # nol informasi bentuk huruf
ALL = CONTAINER + ['bpp', 'levels', 'bg_mean', 'bg_std', 'g_mean', 'g_std']
for name, cols, est in [
    ("chance", ALL, DummyClassifier(strategy='stratified', random_state=0)),
    ("w,h saja", ['w', 'h'], None),
    ("fsize,bpp saja", ['fsize', 'bpp'], None),
    ("metadata kontainer murni", CONTAINER, None),
    ("semua metadata non-piksel", ALL, None),
]:
    a, f = run(cols, est)
    print(f"{name:30} acc={a:.3f}  macroF1={f:.3f}")
