# Kode pipeline INFEST XII 2026

Tiap file adalah **satu cell utuh** (load data -> EDA -> training -> submission).
Jalankan apa adanya di Kaggle; tidak ada dependensi antar file.

## Riwayat versi dan skor

| file | LB | OOF tertimbang | perubahan utama |
|---|---|---|---|
| (v3.1, kode tim) | 0.8073 | 0.9716 | dasar; punya bug `tau` dan spesialis berlapis |
| `v5.py` | 0.81054 | 0.9685 | `tau` dicabut, pemilihan ditimbang populasi strip/blok, tabel `[B2]`, bug spesialis diperbaiki |
| `v6.py` | tidak disubmit | — | `GRAYSCALE` + `degrade()`; parameter degradasi masih tebakan (3.3x terlalu kuat) |
| `v7.py` | **0.83422** | 0.9685 | `DEGRADE` ditera ke statistik test terukur; + blok DAE (opsional, off) |
| `v8.py` | belum dijalankan | — | blok DAE dibuang, gerbang vonis diganti diagnostik plafon, + `submission_kuota.csv` |

## Yang dipakai sekarang

`v8.py`. Menghasilkan DUA file:
- `submission.csv` — sama seperti v7 (terbukti 0.83422)
- `submission_kuota.csv` — decoding berbatas-hitungan, `ALPHA=0.5`. **Belum teruji**,
  taruhan pada satu-satunya tuas yang menyentuh plafon.

Runtime ~3,5 jam (T4) / ~4,6 jam (P100). Pemangkas tercepat kalau kena batas
waktu: `SPEC["enable"]=False` (-32 mnt) lalu buang run `cnx-sq384` (-88 mnt).

## Peringatan penting

**Jangan percaya `F1 Macro OOF polos`.** Angka itu yang membuat v3.1 terlihat
0.9716 padahal LB 0.8073. Yang relevan adalah `OOF TERTIMBANG` dan tabel `[B2]`.

**Gerbang otomatis di v6/v7 memberi saran yang MERUGIKAN** dan sudah dihapus di
v8. Ia memvonis "JANGAN submit" pada run v7 yang justru memberi kenaikan LB
terbesar (+0.024). Kalau menjalankan v6/v7, abaikan vonisnya.

Latar belakang lengkap, termasuk metode yang sudah diuji dan gugur, ada di
`../handoff.md`.


## v9 — perombakan (BELUM diuji di LB)

`v9.py`. Perubahan dari v8, semuanya berdasar bukti leaderboard/pengukuran:

1. **`heightnorm` dibuang total.** Pada populasi `blok` (AR<3, ~49% test)
   letterbox F1 0.94 vs heightnorm 0.82; greedy tidak pernah memilih heightnorm.
2. **`DEGRADE` dikembalikan ke setelan v6.** v6 = 0.83902, v7 (kalibrasi ke
   statistik test) = 0.83422. Kalibrasi menurunkan skor 0.0048. Knob `SEV`
   ditambahkan untuk menaikkan/menurunkan severitas serempak.
3. **AugMax** (Wang dkk., NeurIPS 2021): campuran 3 rantai augmentasi acak
   dengan bobot Dirichlet, p=0.35.
4. **Ensemble penuh dipaksa diuji** saat greedy runtuh ke satu model (terjadi di
   v6/v7/v8, membuang 4 dari 5 run).
5. **Optimasi bobot per-kelas untuk macro-F1** (coordinate ascent), ditala di
   OOF tertimbang, dipakai hanya kalau naik > 0.002.

Status: hanya lolos smoke test (data sintetis, CPU). Belum pernah dijalankan di
data kompetisi. Perubahan dengan keyakinan tertinggi adalah nomor 2 (senilai
+0.0048 terukur); sisanya belum terukur.

## v10 — menyerang PINTASAN (BELUM diuji di LB)

`v10.py`. Dasar pengukuran (lihat `../handoff.md` bagian 1g/1h): RandomForest
yang hanya melihat metadata — tanpa melihat bentuk aksara — mencapai macro-F1
0.7826 di CV train, sementara model terbaik kita 0.839 di LB. AUC pembeda
domain train-vs-test dari metadata yang sama 0.9169.

Tambahan v10 terhadap v9:

1. **CANON** — binarisasi Otsu ke train DAN test. Terukur di tingkat metadata:
   pintasan fotometrik 0.6477 → 0.2176 (tebak acak 0.143), AUC domain
   0.9110 → 0.5955.
2. **ARJIT** — regangan sumbu-x acak 0.65–1.55× saat latih. Terukur: pintasan
   metadata 0.7833 → 0.6932. **Jauh lebih lemah dari CANON.** Memperlebar
   jitter nyaris tidak menolong (0.2–5.0× hanya sampai 0.6670), karena tinggi
   dan skala masih membocorkan geometri.
3. **DFR** [#5] — latih ulang HANYA kepala linear pada fitur group-balanced
   (grup = kelas × binarisasi). CNN dibekukan, jadi biayanya satu lintasan
   ekstraksi fitur per fold. Digerbangi OOF tertimbang per run.
4. **CLEAN** [#50] — confident learning membuang sampel yang kepala aslinya pun
   memberi probabilitas < 0.02 pada labelnya. Catatan jujur: probabilitas
   IN-SAMPLE, jadi lebih lemah dari cleanlab sungguhan.
5. **SAM** [#99] — tersedia, `enable=False` secara default.

TIDAK diimplementasikan, dengan alasan: JTT (menggandakan waktu latih, sudah
~6 jam), Group DRO (butuh id grup di dalam loop latih; risiko tinggi, manfaat
belum terukur), DANN (rewel dan bisa merusak).

Status: lolos smoke test (data sintetis, CPU). `canon()` diverifikasi di gambar
kompetisi sungguhan: train 79.1% → 100% dua-aras, test 33.9% → 100%. Kedua
cabang gerbang DFR (PAKAI dan tolak) sudah dieksekusi.
