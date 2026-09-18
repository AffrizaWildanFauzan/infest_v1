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
