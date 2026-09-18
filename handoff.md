# HANDOFF — INFEST XII 2026: Klasifikasi Citra Aksara Tradisional Nusantara

Dokumen ini ditujukan untuk **AI/reviewer lain yang TIDAK memiliki file dataset**.
Semua angka di bawah adalah hasil pengukuran langsung, bukan perkiraan, kecuali
yang ditandai eksplisit. Tujuannya: siapa pun bisa meng-crosscheck keputusan
desain tanpa perlu mengakses datanya.

---

## 1. Tugas & metrik

Klasifikasi citra aksara tradisional Nusantara ke **7 kelas asal daerah**:
`bali, jawa, jawi, lampung, lontara, pegon, sunda`.

**Metrik: F1-Score Macro** — tiap kelas berbobot sama tanpa memandang jumlah
sampel. Konsekuensi: kelas terkecil sama pentingnya dengan kelas terbesar, dan
model yang hanya jago di kelas mayoritas tidak diuntungkan.

Submission: CSV dua kolom `image_id,label`, jumlah baris = jumlah gambar test.

---
---

## 1b. HASIL v3.1 DI DATASET BARU — temuan terpenting sejauh ini

v3.1 dijalankan tim pada dataset baru dan menghasilkan:

| | nilai |
|---|---|
| OOF macro-F1 (dilaporkan pipeline) | **0.9716** |
| Skor Kaggle (private/public LB) | **0.8073** |
| Selisih CV−LB | **0.164** |

Sebagai pembanding, di dataset LAMA selisih CV−LB hanya 0.026. Jadi ini bukan
overfitting biasa; ada yang salah secara struktural.

### Penyebab: model salah MENGHITUNG, bukan salah MEMBACA

Dari jumlah prediksi tiap kelas di submission v3.1 saja — tanpa tahu label test —
batas atas macro-F1 bisa dihitung, dengan asumsi prior kelas test ≈ prior train:

```
kelas       true~   pred   F1 maks (recall=1.0)
bali          128    102        0.887
jawa          227    234        0.985
jawi          253    173        0.812
lampung       163    234        0.821
lontara       131     97        0.851
pegon         100    240        0.588   <-- diprediksi 2.4x lipat
sunda         218    140        0.782
----------------------------------------------
plafon macro-F1                 0.8181
```

**Plafon 0.8181 vs skor nyata 0.8073 — selisih hanya 0.011.** Artinya model
nyaris tidak kehilangan apa pun dari salah mengenali gambar; hampir SELURUH
kerugian 0.19 berasal dari proporsi prediksi per kelas yang salah.

Konsekuensi langsung: memperbaiki backbone/augmentasi/resolusi tidak akan banyak
menolong. Yang harus diperbaiki adalah lapisan kalibrasi prior.

Catatan asumsi: plafon di atas bergantung pada prior test ≈ prior train, yang
TIDAK bisa diverifikasi (label test tidak tersedia). Kedekatan 0.8181 vs 0.8073
adalah bukti tidak langsung yang kuat, bukan bukti langsung.

### Bug: `tau` dipilih atas dasar gain NOL

Output v3.1 mencatat:

```
F1 Macro OOF (argmax)     : 0.9702
F1 Macro OOF (tau=0.20)   : 0.9702  (+0.0000)
```

Kode memakai perbandingan `if f > bf1` sehingga `tau=0.20` terpilih pada selisih
di bawah 0.00005 (tingkat derau pembulatan). `tau` lalu DITERAPKAN ke test.
Karena `prob / prior**tau` menaikkan kelas langka, dan pegon adalah kelas
terlangka (8.19%), tau kemungkinan besar MEMPERBESAR over-prediksi pegon yang
jadi sumber kerugian utama — semuanya tanpa bukti manfaat sama sekali.
**Perbaikan: butuh margin minimum + validasi yang tahan pergeseran, atau cabut.**

### Bug: spesialis diterapkan berlapis

Di loop pemilihan ambang spesialis, `p2 = oof.copy()` membaca `oof` yang SUDAH
dimodifikasi oleh ambang sebelumnya yang diterima. Akibatnya ambang-ambang
berikutnya dievaluasi di atas probabilitas yang sudah disesuaikan, dan OOF yang
dilaporkan sedikit lebih optimistis dari semestinya. Sudah diperbaiki (snapshot
`oof_base` di luar loop, penugasan sekali di akhir).

### Kenapa OOF 0.9716 menyesatkan

OOF dihitung pada train (33% gambar 'blok'), sedangkan test 50% 'blok'. Kelas
pegon sendiri 79.6% blok di train. Jadi OOF mengukur performa pada komposisi
populasi yang berbeda dari yang dinilai Kaggle. Angkanya benar, tapi untuk soal
yang berbeda. **Perbaikan: laporkan macro-F1 terpisah strip vs blok, dan bobot
ulang OOF ke komposisi test sebelum memilih model/ensemble/ambang/tau.**

---

## 1c. KOREKSI: saran tim soal "noise" BENAR; pengukuran pertama saya salah

PENTING untuk reviewer: bagian di bawah (ditandai [DIBANTAH]) adalah kesimpulan
LAMA yang TERBUKTI KELIRU. Ringkasan yang benar ada di bagian 1d.

Singkatnya: saya memakai estimator noise Immerkaer, yang mengukur derau
frekuensi-TINGGI. Korupsi utama di test ternyata BLUR, yang justru MENURUNKAN
nilai metrik itu. Jadi alat ukurnya buta terhadap jenis kerusakan yang ada, dan
hasil "test 4x lebih bersih" adalah artefak pemilihan metrik, bukan fakta.

### [DIBANTAH] kesimpulan lama

Tim mengusulkan denoising (mis. variational autoencoder) dengan alasan gambar
test banyak berderau. Ini diukur langsung, bukan diperdebatkan.

Estimator noise Immerkaer (konvolusi Laplacian 3x3), jendela 64x64 pada
**resolusi asli tanpa resize** (agar perbedaan ukuran gambar tidak menjadi
penjelasan alternatif), 632 train vs 787 test:

```
sigma noise      p10    p25    p50    p75    p90
TRAIN           2.79   6.26   9.01  12.56  18.60
TEST            0.46   0.97   2.27   5.45  10.00
median train = 9.01 | median test = 2.27 | rasio 0.25x
Mann-Whitney p = 5.29e-89
```

Gambar test ~4x LEBIH BERSIH daripada train. Pada subpopulasi gambar besar saja
(jendela 192x192, n=45/165) selisihnya tidak signifikan (p=0.42). Kesimpulan yang
didukung data: **test tidak lebih berderau di kondisi mana pun.**

Bukti pendukung dari eksperimen v3.1 sendiri: model terbaik adalah `cnext-lb`
(letterbox 128x512, resolusi tertinggi, paling mempertahankan detail; OOF 0.9702)
dan greedy memilihnya SENDIRIAN; dua model terburuk adalah heightnorm 64x256
(0.9616 dan 0.9520). Kalau masalahnya derau, arahnya harus terbalik.

"Noise" yang dirasakan tim sebenarnya adalah **heterogenitas domain** (foto
halaman, screenshot, tabel vs potongan teks bersih) — itu pergeseran distribusi,
bukan derau piksel, dan denoising tidak menanganinya.

---

---

## 1d. PENYEBAB SEBENARNYA: pintasan palsu berbasis WARNA

Setelah v5 (yang memperbaiki pembobotan populasi strip/blok) hanya naik dari
LB 0.8073 ke 0.81054 (+0.003), hipotesis populasi juga gugur. Pemeriksaan
LANGSUNG pada gambar test (bukan statistik agregat) membuka penyebabnya.

### Ukuran jarak domain (train vs test)

```
                                       TRAIN     TEST
citra abu-abu murni (grayscale)        95.3%    36.8%    <- 63% test BERWARNA
citra biner (>90% piksel hitam/putih)  45.2%    14.8%
ketajaman (var Laplacian, median)      13101     2464    <- test 5.3x lebih buram
```

n=600 vs 600 (grayscale), n=495 vs 496 (ketajaman).

### Mekanisme

Porsi gambar berwarna per kelas DI TRAIN: pegon 11.3% (tertinggi), jawa 9.0%,
bali 4.8%, lampung 4.4%, lontara 3.7%, jawi 2.2%, sunda 0.9% (terendah).

Model belajar pintasan "berwarna -> pegon". Test 63% berwarna, jadi pintasan
menyala terus. Korelasi %berwarna-train vs skew prediksi test:

```
v5   : Spearman rho = +0.893  (p=0.0068)
v3.1 : Spearman rho = +0.857  (p=0.0137)     <- dua submission INDEPENDEN
```

Ini menjelaskan semua kegagalan sebelumnya sekaligus: model lebih kuat justru
mempelajari pintasan LEBIH BAIK (v5 OOF 0.9795 vs v3.1 0.9702, LB praktis sama,
kesepakatan prediksi 89.6%); OOF tidak melihatnya karena train 95% abu-abu;
koreksi prior tidak bisa memperbaiki fitur palsu.

### Uji terkontrol (latih pada train abu-abu, uji pada 177 train berwarna)

```
                              skew pegon      macro-F1 (n=177, sd derau 0.036)
A  RGB apa adanya               -18.1pp          0.1305
B  dipaksa abu-abu               -6.2pp          0.1798
C  abu-abu + degradasi            0.0pp          0.1572
```

KESIMPULAN YANG SAH: intervensi menormalkan SEBARAN prediksi (efek besar,
monoton, 1 gambar = 0.56pp jadi jauh di atas derau).
KESIMPULAN YANG TIDAK SAH: bahwa akurasinya naik. Selisih macro-F1 di antara
A/B/C berada DI DALAM derau (sd 0.036 pada n=177, sunda hanya n=6).

### Kenapa sebaran itu yang menentukan skor

Plafon macro-F1 yang dihitung HANYA dari jumlah prediksi per kelas:
v3.1 plafon 0.8181 vs LB 0.8073; v5 plafon 0.8445 vs LB 0.8105. Keduanya
mendarat tepat di bawah plafonnya sendiri. Jadi menormalkan sebaran menaikkan
plafonnya, bukan sekadar kosmetik. (Asumsi: prior test ~ prior train; tidak
dapat diverifikasi.)

### Perubahan di v6

1. `GRAYSCALE=True` - train DAN test dipaksa abu-abu (matikan sumbu warna)
2. `degrade()` - blur/kontras/rescale/JPEG pada train, ditera ke statistik test
3. `rand_jpeg()` lama diserap ke `degrade()` (JPEG saja tidak cukup; blur terbesar)

Cara verifikasi TANPA submit: jalankan v6, lihat blok distribusi di akhir.
Kalau skew pegon turun dari +10.00pp mendekati nol, mekanismenya bekerja.
Kalau tetap +10pp, hipotesis warna juga gugur - JANGAN submit, laporkan saja.

---

---

## 1e. HASIL v6 (LB 0.81054) DAN KALIBRASI ULANG -> v7

v6 belum disubmit; v5 memberi LB 0.81054 (dari 0.8073). Perbaikan pembobotan
populasi hanya menyumbang +0.003, jadi hipotesis populasi gugur (lihat 1d).

### Bangku uji ketahanan korupsi (n=943)

Dibangun dari 25% train yang DIRUSAK dgn korupsi ditera ke statistik test.
Daya statistik jauh lebih baik dari bangku 177-gambar (sd 0.036 -> ~0.015).

PENTING - kesalahan kalibrasi yang ditemukan dan diperbaiki:
- `DEGRADE` v6 ternyata 3.3x TERLALU KUAT (median ketajaman train+degrade 762
  vs test 2549). Model akan belajar membaca citra lebih buruk dari yang akan
  ditemuinya.
- Bangku uji versi pertama juga 4.5x terlalu rusak. Hasil spektakuler
  0.157 -> 0.883 yang sempat dilaporkan TIDAK BERLAKU untuk masalah nyata.

Setelah KEDUANYA ditera ulang (bangku rasio 1.08x vs test asli):

```
[A] augmentasi v5 (RGB)                macro-F1 0.4306  skew 34.68pp
[B] grayscale + degrade k=0.30         macro-F1 0.9032  skew  1.59pp
[C] B + DAE (penormal masukan)         macro-F1 0.9119  skew  1.70pp
```

KESIMPULAN: grayscale + degradasi terkalibrasi = +0.47 macro-F1, jauh di atas
derau. DAE = +0.0087, DI BAWAH derau (sd 0.015) dengan biaya +1.5-2.5 jam GPU
(dilatih 25x: 5 model x 5 fold) -> TIDAK DIPAKAI, USE_DAE=False.

### Setelan v7 (perubahan dari v6)

```python
DEGRADE = dict(p_blur=0.70, blur=(0.18, 0.66),        # dulu (0.6, 2.2)
               p_contrast=0.60, contrast=(0.45, 0.95),
               p_rescale=0.50, rescale=(0.55, 0.90),  # dulu (0.35, 0.80)
               p_jpeg=0.50, jpeg=(40, 90))            # dulu (25, 88)
USE_DAE = False
```

### Ide lain yang DIUJI DAN GUGUR (jangan diulang)

- Retrieval duplikat test<->train: hanya 0.6% gambar test punya kembaran dekat
  (kemiripan >=0.95); 3.6% pada ambang longgar 0.80. Potensi terlalu kecil.
- Tiling resolusi asli untuk gambar blok: coretan di test blok tetap 7.4 px
  setelah letterbox 384 (train 8.8 px). Tidak ada detail yang hilang.
- AugMix: 0.7513 vs degradasi bertarget 0.9032 (bangku salah tera) - kalah.
- Pseudo-label naif: merugikan (-0.017, -0.025 direplikasi 2x).
- tau / EM-SLD / temperature / prior oracle: semua merugikan (lihat 1d).

### Keterbatasan yang tetap ada

Korupsi di bangku uji adalah MODEL saya tentang korupsi test (blur, kontras,
tint, oklusi, JPEG), ditera pada ketajaman dan porsi berwarna. Kalau test punya
jenis kerusakan yang tidak dimodelkan, manfaatnya bisa berbeda. Semua angka
diukur pada CNN kecil 48x192, bukan convnext 384.

### Nama model fondasi (TERVERIFIKASI di HF Hub, untuk dicoba tim)

- `vit_base_patch14_reg4_dinov2.lvd142m`  (patch 14 -> resolusi HARUS kelipatan 14)
- `vit_base_patch16_clip_224.openai`
- `eva02_base_patch14_224.mim_in22k`       (patch 14)
- `vit_base_patch16_siglip_224.v2_webli`
JEBAKAN: `siglip_base_patch16_224` TIDAK ADA; nama salah -> build() diam-diam
fallback ke resnet34 tanpa error. Dan 384 TIDAK habis dibagi 14 -> pakai 378.
Model-model ini 86M param (vs convnext_tiny 28.6M); pakai 224 agar runtime wajar.

---

---

## 1f. HASIL v7: LB 0.83422 — dan penemuan bahwa AKURASI SUDAH MENTOK

v7 (grayscale + degrade terkalibrasi) memberi LB 0.83422, naik +0.0237 dari v5.
Kenaikan terbesar sejak dataset diganti.

### Plafon: v7 praktis sudah maksimal

Plafon macro-F1 dihitung HANYA dari jumlah prediksi per kelas (asumsi prior
test ~ prior train):

```
versi   plafon    LB nyata   selisih   skew maks
v3.1    0.8181    0.8073     -0.0108   11.48pp
v5      0.8445    0.8105     -0.0339   10.00pp
v7      0.8407    0.8342     -0.0065    9.59pp
```

v7 hanya 0.0065 di bawah plafonnya. Kenaikan +0.024 datang dari MENUTUP JARAK
ke plafon (-0.034 -> -0.0065), BUKAN dari menaikkan plafon (0.8445 -> 0.8407,
malah sedikit turun). Artinya: tidak ada lagi ruang dari perbaikan akurasi.
Untuk melewati ~0.84, HARUS mengubah jumlah prediksi per kelas.

### DUA kesalahan yang perlu diketahui reviewer

1. GERBANG OTOMATIS SAYA SALAH DAN MERUGIKAN. Kode v6/v7 mencetak vonis
   "skew >= 5pp -> hipotesis gugur, JANGAN submit". v7 skew-nya 9.59pp, jadi
   divonis gagal. Tim submit tetap, dan skornya naik paling banyak. Kalau
   gerbang itu dituruti, kenaikan +0.024 hilang. Kesalahan desain: menjadikan
   satu indikator (skew) sebagai proksi untuk hal lain (skor), berdasar
   hipotesis yang belum teruji. ABAIKAN vonisnya; angkanya tetap berguna.

2. MEKANISME "PINTASAN WARNA" TIDAK TERBUKTI. Grayscale dimatikan, skew pegon
   hanya bergerak 10.00 -> 9.59pp. Nyaris nol. Jadi warna BUKAN penyebab skew.
   Intervensinya menolong lewat jalur lain (ketahanan blur/degradasi).
   Korelasi rho=0.89 ternyata korelasi tanpa kausalitas.

### Metode tambahan yang DIUJI SETELAH v7 dan tidak terbukti

```
oklusi poligon dalam degrade()   0.9060 vs 0.9084 tanpa  -> -0.0024 (derau)
decoding berbatas (kuota train)  0.9066 vs 0.9084        -> TIDAK BERMAKNA:
decoding berbatas (kuota oracle) 0.9074 vs 0.9084           bangku uji skew-nya
                                                            cuma 2pp, tak ada
                                                            yang perlu dibetulkan
```

Oklusi buatan MEMANG ada di test (18.3% gambar, luas median 9.8%, abu-abu
110-196, bentuk poligon) dan TIDAK ada di degrade(). Tapi melatihnya tidak
menolong di bangku uji yang justru mengandung oklusi di sisi pengujian.

Decoding berbatas TIDAK BISA diuji dari bangku ini (bangku tidak mereproduksi
skew 9.59pp). Statusnya: belum teruji, bukan gagal.

### Mutu data train

Hanya 4 dari 3771 gambar bermasalah (1 hitam total, 3 nyaris tanpa tinta).
0.11% - tidak signifikan, tidak perlu ditangani.

### Rekomendasi

Slot 1: v7 apa adanya (0.8342, angka nyata).
Slot 2 (kalau ada): submission_kuota.csv dgn ALPHA=0.5 - taruhan pada
satu-satunya tuas tersisa yang menyentuh plafon.
Jangan tambah metode akurasi lagi: 7 metode diuji, 0 terbukti, dan sisa ruang
hanya 0.0065.

---

## 2. PENTING: dataset pernah diganti panitia

Ada **dua dataset berbeda** dalam riwayat pengerjaan ini. Jangan tertukar.

| | dataset LAMA | dataset BARU (aktif) |
|---|---|---|
| train | 4.468 | **3.771** |
| test | 1.118 | **1.220** |
| format | PNG semua | **PNG, JPG, WEBP, BMP** |
| gambar corrupt | 1 (`mPmaODVU.png`) | **0** |
| isi | 100% potongan baris teks | **2 populasi campur** (lihat §4) |
| kelas terkecil | bali (367, 8,21%) | **pegon (309, 8,19%)** |
| imbalance | 2,24x | **2,53x** |

Skor leaderboard yang pernah dicapai (**pada dataset LAMA**, tidak bisa
dibandingkan langsung dengan dataset baru):

- v1: OOF 0,9916 → **LB 0,96586**
- v2: OOF 0,9953 → **LB 0,99592**

---

## 3. Distribusi kelas (dataset BARU, train)

```
jawi      781  20.71%
jawa      703  18.64%
sunda     673  17.85%
lampung   505  13.39%
lontara   404  10.71%
bali      396  10.50%
pegon     309   8.19%   <- TERKECIL, dan paling berisiko (lihat §4)
imbalance = 2.53x
```

Perubahan penting dari dataset lama: `pegon` dulu kelas ke-3 **terbesar**
(771, 17,3%), sekarang **terkecil**. Karena metriknya macro-F1, kelas ini jadi
titik paling menentukan.

---

## 4. Temuan utama: dataset berisi DUA populasi gambar yang berbeda total

Ini fakta terpenting di dokumen ini dan dasar dari sebagian besar keputusan desain.

```
split  fmt       n       %   W p50   H p50  AR p50   KB p50
train  PNG    3584  95.04%     486      62    7.10      3.7
train  JPEG    173   4.59%     738     570    1.33     72.1
train  WEBP     14   0.37%     720     432    1.78     19.0
test   PNG     944  77.38%     383      66    4.44     17.9
test   JPEG    256  20.98%     768     598    1.33     75.7
test   WEBP     18   1.48%    1200     675    1.56     29.5
test   BMP       2   0.16%     483     460    0.90    857.4
```

- **Populasi A — "strip teks"** (AR > 3, mayoritas PNG): potongan satu baris
  aksara, tinggi ~64px, lebar ratusan px. Ini satu-satunya jenis di dataset lama.
- **Populasi B — "blok/foto"** (AR ≤ 3, mayoritas JPEG/WEBP): hasil scraping web.
  Inspeksi visual menemukan: foto manuskrip lontar, foto papan nama jalan, tabel
  aksara (Hanacaraka dsb), screenshot aplikasi Android, foto orang menulis,
  sampul buku 3120×4160, tabel ter-scan 1654×2340.

**Pergeseran distribusi train → test (ini risiko utama):**

```
JPEG          train  4.6%  ->  test 21.0%   (4,5x lebih banyak)
blok (AR<=3)  train 33.0%  ->  test 49.7%
```

Separuh test adalah populasi B, sementara di train hanya sepertiga. Model yang
hanya dituning untuk strip teks akan terlihat bagus di OOF tapi jatuh di LB.

**Komposisi per kelas (train):**

```
kelas         n  strip%   blok%  AR p50  %JPEG
bali        396   72.7%   27.3%   19.19   6.3%
jawa        703   87.1%   12.9%    9.53   5.4%
jawi        781   66.2%   33.8%    6.05   2.8%
lampung     505   62.0%   38.0%    3.34   5.7%
lontara     404   93.3%    6.7%   16.88   4.0%
pegon       309   20.4%   79.6%    1.16  10.4%   <- terkecil DAN 80% blok
sunda       673   53.2%   46.8%    3.51   1.6%
```

`pegon` menumpuk dua faktor risiko: kelas terkecil **dan** hampir seluruhnya
populasi B. Untuk macro-F1 ini titik rawan nomor satu.

---

## 5. Rentang ukuran (dasar pemilihan resolusi)

```
q0.01   W=   32  H=   31  AR= 0.70
q0.25   W=  149  H=   51  AR= 1.77
q0.50   W=  541  H=   66  AR= 5.08
q0.75   W=  864  H=   85  AR=11.64
q0.95   W= 1372  H=  632  AR=20.83
q0.99   W= 1546  H= 1280  AR=25.20
maks    5664x4248        terkecil 14x15
gambar dgn sisi terpendek <= 64px: 47.6%
```

Implikasi: resize paksa ke persegi merusak strip teks (AR sampai 30), tapi
`heightnorm` tinggi 64 merusak blok/foto — tabel aksara 768×1024 jadi 48×64.
**Tidak ada satu skema preprocessing yang benar untuk keduanya**; harus dua-duanya.

---

## 6. Kebocoran train ↔ test

Terdeteksi lewat MD5: **2 gambar test identik byte-per-byte dengan gambar train**,
jadi labelnya sudah pasti:

```
test ipYw1jbF.jpg == train zasgaCVk.jpg -> pegon
test czonWT2R.png == train HrgEozJY.png -> lampung
```

Dengan perceptual hash (16×16 grayscale, bit di atas median) ditemukan 5 grup
near-duplicate, total ~6 gambar test. Nilainya kecil (~0,005 macro-F1) tapi gratis.

Higienis: **0 duplikat di dalam train**, **0 konflik label**, **0 duplikat di dalam
test**. Validasi silang aman.

---

## 7. Riwayat kegagalan yang sudah diperbaiki (jangan diulang)

### 7a. Fold kolaps mencemari prediksi test (penyebab jurang CV–LB v1)

Di v1, log training menunjukkan 3 dari 10 model punya `macro-F1 ≈ 0.044`. Angka
itu persis skor "selalu tebak satu kelas" (dihitung: `jawi` → 0,0440,
`lampung` → 0,0444). Model-model itu divergen dan runtuh.

**Kenapa OOF tidak menangkapnya:** tiap baris OOF hanya dinilai model fold-nya
sendiri, dan tiap fold yang mati di satu view masih tertolong view lain yang
sehat. OOF tetap 0,9916.

**Kenapa test tercemar:** `test_prob` dirata-rata dari SEMUA fold termasuk yang
mati, jadi probabilitas konstan model rusak ikut masuk. Simulasi numerik:
0/5 fold mati → porsi prediksi `jawi` 20,57%; 1/5 → 21,91%; 2/5 → 25,04%.
Submission v1 memang over-predict `jawi` +2,98pp di atas distribusi train.

**Akar masalah:** `OneCycleLR max_lr = 3e-4 × 3 = 9e-4` untuk ConvNeXt — terlalu
besar (fine-tuning ConvNeXt biasanya ≤1e-4), tanpa gradient clipping.

**Perbaikan (v2, LB 0,96586 → 0,99592):** lr per-backbone, gradient clipping 1.0,
warmup 25%, EMA, layer-wise LR decay, deteksi kolaps + retry lr/3, dan yang
terpenting — **rata-rata test HANYA dari fold sehat**.

### 7b. Deteksi kolaps harus berbasis konsentrasi, bukan ambang F1

Versi pertama memakai `F1 < 0.50` dan salah membuang model yang hanya *lemah*
(F1 0,38) padahal prediksinya masih beragam. Sekarang:

```python
def is_collapsed(pred, f1):
    share = np.bincount(pred, minlength=NC).max() / max(len(pred), 1)
    return bool(share > 0.90 or f1 < 0.10)
```

Diuji: "semua 1 kelas" (F1 0,044) → BUANG; "95% 1 kelas" (0,06) → BUANG;
"lemah tapi sehat" (0,378) → PAKAI.

### 7c. Bug penemuan file (menyebabkan AssertionError di dataset baru)

1. `.webp` tidak ada di daftar ekstensi → semua file webp tak ketemu.
2. Ekstensi di CSV tidak selalu sama dengan file asli → perlu pencocokan
   berdasarkan nama tanpa ekstensi (*stem*).
3. `if p and ...` pada nilai `NaN` → `TypeError`, karena `bool(nan) is True`.
4. `OneCycleLR total_steps` terlalu kecil → `ZeroDivisionError` kalau satu fold
   datanya sedikit (`drop_last=True` bisa membuat `len(loader) == 0`).
5. Saat OOF sempurna, `argmax` matriks confusion nol memberi `A == B`, sehingga
   spesialis dilatih untuk "bali vs bali".

Semua sudah ditutup. Ekstensi yang didukung sekarang:
`.png .jpg .jpeg .webp .bmp .gif .tif .tiff`.

---

## 8. Desain saat ini (v4)

### Preprocessing — aspect ratio SELALU dipertahankan
- `heightnorm(H, W)`: tinggi dinormalkan ke H, jendela selebar W di-crop
  (pad putih kalau lebih pendek). Untuk **populasi A**. TTA = beberapa jendela
  sepanjang lebar, softmax dirata-rata.
- `letterbox(H, W)`: diskalakan agar muat lalu pad putih. Dengan `H == W` ini
  menjadi view persegi untuk **populasi B**.

### Augmentasi
Rotasi ±4°, translate/shear kecil, brightness/contrast 0.30, noise gaussian
ringan, RandomErasing, dan **kompresi JPEG acak** (p=0.35, quality 30–92) untuk
menirukan profil test yang 21% JPEG.

**TANPA FLIP** — aksara punya orientasi bermakna; `jawi`/`pegon` turunan Arab
dibaca kanan-ke-kiri, sehingga pencerminan menghasilkan bentuk huruf tidak valid.

### Training
5-fold StratifiedKFold (seed 42), class-weighted CE (inverse-frequency) +
label smoothing 0.05, AdamW + layer-wise LR decay 0.75, OneCycleLR warmup 25%,
gradient clipping 1.0, EMA decay 0.999 (tiap epoch dibandingkan bobot raw vs EMA,
diambil yang macro-F1-nya menang), checkpoint dipilih berdasarkan **macro-F1**
bukan akurasi.

### Seleksi ensemble
Greedy selection (Caruana et al. 2004) dengan pengulangan = pembobotan. Model
hanya ditambahkan **kalau OOF macro-F1 naik**. Kalau model tunggal lebih baik
dari blend, model tunggal yang dipakai.

### Kalibrasi macro-F1
Logit adjustment: `prob / prior^tau`, `tau` dicari **hanya di OOF** pada rentang
0–1 langkah 0,05. Menaikkan recall kelas minoritas yang dihargai penuh macro-F1.

### Spesialis pasangan tersulit (cascade)
Pasangan kelas dengan error off-diagonal terbanyak dideteksi otomatis dari
confusion matrix, lalu dilatih classifier **biner** resolusi tinggi khusus untuk
pasangan itu. Hasilnya dipakai untuk membagi ulang massa probabilitas antara dua
kelas tersebut, **hanya** bila top-2 ensemble memang pasangan itu. Ambang
diserahkan dicari di OOF; spesialis dipakai **hanya kalau OOF naik**.

Di dataset lama pasangan tersulit adalah `jawi` ↔ `pegon` (17 dari 22 sisa error).

### Guard yang wajib dipertahankan
- Fold kolaps → retry dengan lr/3 (maks 3x) → kalau tetap, **dikeluarkan dari
  ensemble**; rata-rata test hanya dari fold sehat.
- Gambar raksasa (maks 5664×4248) di-`thumbnail` ke 1600px dulu.
- Baris test **tidak pernah** dibuang; baris train yang corrupt/hilang dibuang.
- Peringatan otomatis kalau distribusi prediksi test menyimpang >3pp dari train.

---

## 9. Status verifikasi — apa yang SUDAH dan BELUM diuji

**Sudah diverifikasi** (dijalankan end-to-end):
- Penemuan file: 3771/3771 train dan 1220/1220 test ter-resolve di dataset baru.
- Deteksi kebocoran otomatis menemukan 2 gambar.
- Guard kolaps menyala, retry, dan mengeluarkan fold mati tanpa crash.
- Logika redistribusi probabilitas spesialis (uji unit: massa kekal, kelas lain
  tak tersentuh, ambang dihormati).
- Penulisan submission: jumlah baris benar, urutan mengikuti `sample_submission`,
  tidak ada NaN, `image_id` unik.

**BELUM bisa diverifikasi di lingkungan pengembangan:**
- **Kualitas model.** Lingkungan ini memblokir unduhan bobot pretrained
  (`huggingface.co` dan `download.pytorch.org` → 403), sehingga semua uji memakai
  bobot acak. Angka OOF sesungguhnya baru terlihat saat dijalankan di Kaggle
  dengan `timm` pretrained.
- Apakah view persegi benar-benar menaikkan skor pada populasi B — ini hipotesis
  berdasar EDA, dan **greedy ensemble akan menolaknya sendiri kalau OOF tidak naik**.

---

## 10. Risiko terbuka & yang perlu dicek reviewer

1. **Pergeseran train→test adalah risiko terbesar.** Test 50% populasi B vs train
   33%. OOF dihitung di train, jadi OOF akan optimistis. Pertimbangkan validasi
   berbobot: ukur macro-F1 terpisah untuk subset strip dan subset blok.
2. **`pegon`**: kelas terkecil (309) sekaligus 80% populasi B. Periksa F1 per
   kelas untuk `pegon` secara khusus; ini kandidat kuat titik terlemah.
3. **Plafon data.** Di dataset lama ditemukan gambar berlabel `jawi` yang isinya
   hanya angka Arab (`٩٦`, `٨٩`) — nol informasi pembeda antara Jawi dan Pegon.
   26,3% gambar `jawi` hanya berisi ≤3 komponen tersambung. Kalau pola ini
   berulang di dataset baru, skor 1,00000 tidak tercapai lewat modeling.
   Pipeline menulis `audit_kasus_sulit.csv` berisi kandidat label keliru.
4. **Jangan percaya OOF sempurna.** Kalau OOF = 1,0000, curigai kebocoran atau
   bug, bukan keberhasilan.
5. **Selalu cek `dead = 0`** di tabel ringkasan per model. Kalau ada fold kolaps,
   turunkan `lr` di `RUNS`.

---

## 11. Cara menjalankan

Satu cell Python. Menemukan `train.csv`/`test.csv`/`sample_submission.csv` dan
folder gambar secara otomatis (mendukung struktur `images/train`, `images/test`,
maupun `.zip` yang diekstrak sendiri). Output: `submission.csv` dan
`audit_kasus_sulit.csv`.

Yang harus diperhatikan di output:
- `train rows=... resolved=...` harus sama besar; test **wajib** 100%.
- Blok `[A]` — kolom `dead` harus 0 di semua baris.
- Blok `[B]` — F1 per kelas per model; kolom terendah = titik lemah.
- Blok `[C]` — bobot ensemble hasil greedy.
- `STEP 4b` — spesialis dipakai atau ditolak (keduanya sah).
- `STEP 5` — distribusi prediksi test vs train; tanda `<-- CEK` berarti
  menyimpang >3pp dan perlu diselidiki.

Kalau waktu GPU terbatas, kurangi isi `RUNS` — buang entri dengan OOF terendah,
**jangan** membuang guard kolaps atau pemilihan berbasis OOF.
