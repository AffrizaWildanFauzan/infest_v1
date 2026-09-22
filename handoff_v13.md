# HANDOFF — INFEST XII 2026, klasifikasi aksara Nusantara

**Status per 22 September 2026: LB publik 0.89166, peringkat 5.**

Berkas ini menggantikan `handoff.md` untuk SEMUA hal setelah v10. `handoff.md`
yang lama masih berguna sebagai arkeologi v1–v10 (terutama audit pintasan dan
sejarah Otsu), tapi angka dan kesimpulannya sudah dilampaui di banyak tempat.
Kalau keduanya bertentangan, **berkas ini yang benar.**

Ditulis untuk diberikan ke sesi Claude lain yang melanjutkan pekerjaan ini.
Bacalah sampai bagian "Langkah berikutnya" sebelum menulis satu baris kode pun.

---

## 1. Ringkasan 30 detik

Lomba klasifikasi citra 7 kelas aksara Nusantara (bali, jawa, jawi, lampung,
lontara, pegon, sunda). Train 3.771 citra, test 1.220. Metrik **macro-F1**.

Pipeline pemenang saat ini (`kode/v13.py`) adalah satu hal yang sangat
sederhana: **SigLIP2 ViT-B/16 @384, fine-tune penuh, 5 fold, satu seed, satu
view, tanpa TTA, tanpa ensemble, tanpa spesialis, tanpa bobot kelas.**
Semua tambahan yang dicoba v13 DITOLAK oleh gerbangnya sendiri karena tidak
melewati ambang derau. Itu bukan kegagalan — itu hasilnya.

Sisa kesalahan terkonsentrasi di satu tempat: **pegon ditukar dgn jawi**,
31 dari 52 total kesalahan OOF (60%).

---

## 2. Data, dan tiga jebakan yang harus diketahui lebih dulu

Repo: `https://github.com/AffrizaWildanFauzan/infest_v1`

| | |
|---|---|
| train | 3.771 citra + `train.csv` (`image_id`, `label`) |
| test | folder `test/`, 1.220 citra |
| submission | `image_id,label`, 1.220 baris, urutan ikut `sample_submission.csv` |

**Jebakan 1 — ada DUA versi test set.** Folder `test/` (1.220 citra) cocok dgn
`sample_submission.csv`. Berkas `test.csv` berisi 1.118 id test LAMA yang
irisannya dgn `sample_submission.csv` adalah **NOL**. Selalu ambil daftar test
dari `sample_submission.csv`, jangan pernah dari `test.csv`. v10 melakukan ini
dan hasilnya tidak berarti apa-apa.

**Jebakan 2 — dua populasi citra, dan test-nya tidak seperti train.**
Dgn ambang aspect ratio 3.0:

| | blok (AR<3) | strip (AR>=3) |
|---|---|---|
| train | 32,9% | 67,1% |
| **test** | **49,2%** | **50,8%** |

Blok adalah populasi yang jauh lebih sulit (v13: F1 blok 0.9634 vs strip
0.9958) DAN porsinya di test hampir 1,5x porsinya di train. Akibatnya OOF
polos selalu terlalu optimistis. Lihat bagian 3.

**Jebakan 3 — sebagian besar skor lomba ini bisa diraih TANPA membaca
aksara.** RandomForest pada metadata citra saja (ukuran, aspect ratio,
saturasi) mencapai macro-F1 **0.7826** di `handoff.md`; v13 mengukur ulang
versi yang lebih ketat (AR + derajat 2-aras) dan dapat **0.6016**. Tebakan
acak 0.1429. Artinya: model Anda sebagian besar sedang belajar pintasan
geometris, dan kenaikan skor yang nyata harus datang dari bentuk huruf.

Ada juga **kebocoran kecil**: 2 citra test byte-identik dgn citra train,
jadi labelnya pasti. v13 sudah menerapkannya (`LEAK` di `labels_from`).
Gratis, sudah diambil, jangan dihitung dua kali.

---

## 3. Metrik keputusan: `popf1`, dan kenapa bukan yang lain

**Satu-satunya angka yang boleh memutuskan apa pun di proyek ini adalah
`popf1`**: macro-F1 OOF yang ditimbang ulang ke komposisi strip/blok TEST
(W_BLOK = 0.492), bukan komposisi train.

```python
popf1 = W_BLOK * macroF1(baris blok) + (1 - W_BLOK) * macroF1(baris strip)
```

Kenapa bukan yang lain:

- **OOF polos menipu.** v13: OOF polos 0.9846, `popf1` 0.9799. Selisihnya
  bukan derau, itu bias komposisi yang sistematis.
- **`impf1` (importance-weighted) sudah dicoba dan GAGAL.** v12 memakainya
  untuk memilih `submission_iw.csv`, dgn margin 0.0047 pada effective sample
  size cuma 36%. Berkas itu **TURUN 0.02054 di LB**. Pada ESS 36% angka itu
  >=1,67x lebih berderau daripada OOF polos — marginnya lebih kecil dari
  deraunya sendiri. Jangan pakai `impf1` untuk memutuskan apa pun.
- **`popf1` melacak LB.** Atas 5 pasang (OOF, LB) terakhir: Pearson +0.64,
  Spearman +0.36. Itu bukan korelasi sempurna, tapi cukup untuk dipercaya
  ketika selisihnya nyata (>= 0.005). Angka lama "+0.16 / -0.32" di catatan
  sebelumnya dihitung dari 4 pasang yang OOF-nya berdempetan; abaikan.

**Ambang derau yang dipakai sebagai gerbang** (jangan dilonggarkan tanpa
alasan terukur):

| keputusan | ambang |
|---|---|
| tambah anggota seed-bag | +0.0010 |
| pakai TTA | +0.0015 |
| pakai spesialis pegon/jawi | +0.0015 |
| pakai bobot per-kelas | +0.0020 |

Derau seed di fold yang sama sudah terukur: fold 0 dilatih 3x dgn seed
berbeda menghasilkan 0.9849 / 0.9857 / 0.9879, **sd = 0.0015**. Apa pun yang
di bawah itu bukan perbaikan.

---

## 4. Riwayat skor — yang benar-benar terjadi

| versi | resep | OOF (`popf1`) | **LB publik** |
|---|---|---|---|
| v3.1 | — | tidak diukur sbg `popf1` | 0.8073 |
| v5 | — | tidak diukur sbg `popf1` | 0.81054 |
| v6 | — | tidak diukur sbg `popf1` | 0.83902 |
| v7 / v8 | — | tidak diukur sbg `popf1` | 0.83422 |
| v11 | ensemble ft-v7 + lpft-siglip2 | tidak dicatat | 0.85030 |
| v12 `submission_iw.csv` | ensemble 3x SigLIP2 sekeluarga | `impf1` 0.9526 | **0.85695 (TURUN)** |
| v12 `submission.csv` | ft-siglip2 @224 tunggal | 0.9735 | 0.87749 (peringkat 9) |
| **v13 `submission.csv`** | **SigLIP2 @384 tunggal, 1 seed, 1 view** | **0.9799** | **0.89166 (peringkat 5)** |

`popf1` baru dipakai konsisten sejak v11/v12, jadi baris v3.1–v8 hanya punya
skor LB. Angka OOF lama yang beredar di `handoff.md` (mis. v5 0.9795) adalah
OOF POLOS, bukan `popf1` — jangan dibandingkan dgn kolom ini.

Papan peringkat per 22 Sep 2026:
1. black&yellow 0.92091 · 2. DoaAyahIbu 0.91820 · 3. CH Tembok Ratapan UTY
0.91482 · 4. Lucy Supremacy 0.90299 · **5. Atrahasis777 0.89166**

**Laju tukar OOF→LB, dari dua titik terakhir:** ΔOOF +0.0064 → ΔLB +0.01417,
rasio ≈ 2,2. **Dua titik tidak membentuk garis dgn galat**, jadi pakai angka
ini sebagai orde besaran saja, bukan ramalan. Dgn rasio itu, naik ke
peringkat 4 (0.90299) butuh **ΔLB +0.01133 ≈ ΔOOF +0.0051**, yaitu `popf1`
sekitar **0.9850**. Untuk perspektif: seluruh lompatan v12→v13 cuma +0.0064.

---

## 5. Yang SUDAH TERBUKTI (jangan diuji ulang)

Semua angka di bawah OOF `popf1`, 5 fold, `random_state=42`, baris-per-baris
sebanding karena fold-nya identik di v12 dan v13.

1. **Resolusi membantu.** SigLIP2 224 → 384 memberi **+0.0064** OOF dan
   **+0.01417** LB. Ini kenaikan tunggal terbesar yang pernah diukur di
   proyek ini. Fold dan baris identik, jadi ini uji bersih.
2. **SigLIP2 adalah backbone terbaik.** `vit_base_patch16_siglip_*.v2_webli`.
   Sapuan 19 backbone (`kode/automl.py`, probe beku dgn C disetel) menempatkan
   siglip2-512 0.9472, siglip2-384 0.9439, dinov3-b16 0.9361, siglip2L-384
   0.9271, dinov3-cnx 0.9188; ekornya (swinv2, eva02, dinov2, convnext,
   beitv2, hiera, ViT-IIIF, manuskrip-Arab) semua di bawah.
3. **Probe beku memprediksi peringkat fine-tune di data ini.** v11
   memeringkat lewat probe (SigLIP2 0.9158 > DiT 0.8122); v12 fine-tune penuh
   mempertahankan urutan itu. Tapi probe **melebih-lebihkan selisihnya ~3x**:
   probe Δ(224→384) = +0.0189 sedangkan fine-tune Δ = +0.0064. Pakai probe
   untuk MEMERINGKAT, jangan untuk memperkirakan besar kenaikan.
4. **Layer-wise LR decay 0.75 aktif dan benar** di v11–v13 (28 grup, 14 skala
   berbeda). Di v10 knob ini diam-diam mati — lihat bagian 9.
5. **Distribusi prediksi test stabil lintas pipeline.** v11, v12 dan v13
   semuanya memprediksi pegon ~18–19%, lampung ~19–20%, jawi ~12% padahal
   prior train-nya pegon 8,19%, lampung 13,39%, jawi 20,71%. Tiga keluarga
   model yang berbeda setuju, jadi prior test memang bergeser ke arah itu.
   **Gunakan ini sebagai uji kewarasan**: submission baru yang distribusinya
   tiba-tiba berbeda hampir pasti rusak, bukan lebih baik.

---

## 6. Yang SUDAH TERBANTAH (jangan diulangi — ini bagian paling mahal dari catatan ini)

| ide | hasil terukur | uji |
|---|---|---|
| **Biner Otsu** | **-0.0091** di SigLIP2, **-0.0131** di DiT | dua konfirmasi independen. TUTUP. |
| **Pra-latih aksara sintetis sbg inisialisasi bobot** | **-0.0143** | uji terkontrol, identik kecuali bobot awal (ft-v7 0.9667 vs ft-syn 0.9524) |
| **LP-FT (Kumar dkk. ICLR 2022)** | **-0.0037** | anggaran disamakan, backbone/resolusi/epoch/canon sama. Klaimnya tidak berlaku di sini. |
| **Koreksi prior EM** | rata-rata **-0.0410**, menang 6/12 simulasi, terburuk -0.1419 | EM menaksir prior dgn sangat baik (galat L1 0.6821→0.0733) tapi membobot ulang posterior tetap merusak macro-F1 |
| **Ensemble model sekeluarga** | **-0.02054 di LB** (`submission_iw.csv`) | merata-ratakan pemenang dgn anggota 0.058 lebih lemah. Lihat kotak di bawah. |
| **Seed-bagging** | seed 1: -0.0033, seed 2: +0.0005 (ambang +0.0010) | keduanya DITOLAK gerbang v13. Tiga seed penuh dilatih, tidak satu pun membantu. |
| **TTA** | +0.0001 terbaik (ambang +0.0015) | DITOLAK. Alasannya struktural, lihat di bawah. |
| **Spesialis biner pegon vs jawi** | +0.0003 terbaik pada alpha 0.20–0.35 (ambang +0.0015) | DITOLAK meski spesialisnya sendiri F1 0.944–0.972 |
| **Bobot per-kelas** | +0.0001 (ambang +0.0020) | DITOLAK |
| **`impf1` sbg metrik pemilih** | memilih berkas yang turun 0.02054 | lihat bagian 3 |

> ### Kotak: kenapa `submission_iw.csv` turun
> Ia merata-ratakan tiga model **dari backbone yang sama** (ft-siglip2,
> lpft-siglip2, siglip2-canon) dgn bobot sama. Salah satunya dibangun di atas
> Otsu dan OOF-nya 0.9158 — **0.058 lebih lemah** dari pemenang. Hanya 63 dari
> 1220 baris yang berubah, dan seluruh -0.02054 berasal dari 63 baris itu;
> aliran terbesarnya pegon→jawi (8) dan jawi→pegon (7).
> **Pelajarannya bukan "jangan ensemble".** Pelajarannya: (a) anggota ensemble
> harus setara kualitasnya, (b) keberagaman harus datang dari keluarga
> arsitektur yang berbeda, bukan seed atau pra-proses yang berbeda, dan
> (c) setiap anggota harus lewat gerbang OOF sebelum masuk.

**Kenapa TTA mati secara struktural.** Keempat view TTA v13 — regangan sumbu-x
0.82/1.00/1.22 dan jitter skala letterbox 0.94 — semuanya berada DI DALAM
rentang augmentasi latih (ARJIT meregangkan 0.65–1.55, letterbox menjitter
0.92–1.08). Model sudah pernah melihat semuanya, jadi tidak ada informasi
baru. TTA yang punya peluang harus berada DI LUAR distribusi augmentasi latih
(misalnya warna padding letterbox yang berbeda, atau multi-crop) — dan itu
**belum pernah dicoba**.

---

## 7. Keadaan v13 secara rinci

Konfigurasi yang benar-benar terpakai:

```
backbone : vit_base_patch16_siglip_384.v2_webli @384x384
epoch    : 18, 5 fold, StratifiedKFold(shuffle=True, random_state=42)
lr       : body 3e-5, head 1e-3, layer_decay 0.75
batch    : micro 16 x accum 2 = efektif 32
lainnya  : wd 0.05, label smoothing 0.05, EMA 0.999, CW berbobot kelas
TTA      : tidak    seed-bag: [0] saja    spesialis: alpha 0    bobot kelas: tidak
OOF popf1: 0.9799   (blok 0.9634, strip 0.9958; OOF polos 0.9846)
biaya    : 34,4 citra/detik latih di T4; 0,59 jam/fold; 2,97 jam per seed
```

Laporan per kelas (OOF, 3.771 baris), dari terlemah:

| kelas | F1 | n | tertukar dgn |
|---|---|---|---|
| **pegon** | **0.9467** | 309 | jawi (15x) |
| **jawi** | **0.9726** | 781 | pegon (16x) |
| bali | 0.9886 | 396 | jawi (4x) |
| jawa | 0.9943 | 703 | bali (2x) |
| sunda | 0.9948 | 673 | jawi (2x) |
| lontara | 0.9975 | 404 | jawi (2x) |
| lampung | 0.9980 | 505 | sunda (1x) |

**Total kesalahan OOF: 52 dari 3.771. Pegon↔jawi: 31, yaitu 60%.**
Ini masuk akal secara linguistik: jawi dan pegon sama-sama aksara Arab yang
diadaptasi (jawi untuk Melayu, pegon untuk Jawa/Sunda), jadi bentuk hurufnya
memang hampir identik. Pegon juga kelas terkecil (309 baris, 8,19%).

Distribusi prediksi test v13: bali 8,69% · jawa 19,02% · jawi 12,30% ·
lampung 18,93% · lontara 8,77% · pegon 18,85% · sunda 13,44%. Konsisten dgn
v11 dan v12 (lihat poin 5 di bagian 5).

`submission.csv` v13 vs v12: **sepakat 92,70%** (89 dari 1220 baris berbeda).
Aliran terbesar pegon→jawi (14) dan jawi→pegon (9).

---

## 8. Letak kode

Semuanya di branch `claude/project-thread-owrttc`. Branch `main` cuma berisi
data, tanpa kode.

| berkas | isi |
|---|---|
| `kode/v13.py` | **pipeline pemenang saat ini.** 1.601 baris, satu cell Kaggle. Mulai dari sini. |
| `kode/v14_sonde.py` | sonde 1 fold: apakah DINOv3 layak jadi anggota ensemble? ~0,6 jam T4, tidak menulis submission |
| `kode/automl.py` | sapuan 19 backbone lewat probe beku; menghasilkan `automl_hasil.csv` (tabel peringkat, **bukan** submission) |
| `kode/v12.py` | korpus aksara sintetis, pra-latih antara, koreksi prior EM — semuanya terbantah, tapi kodenya berguna |
| `kode/v11.py` | fitur beku domain dokumen, LP-FT, view patch |
| `kode/v5.py`–`v10.py` | arsip, jangan dipakai |
| `handoff.md` | catatan lama v1–v10 |
| `submisi/` | submission lama |

**Cara menjalankan:** tiap `vN.py` adalah SATU cell Kaggle, dijalankan apa
adanya dgn GPU menyala. Tidak ada `requirements.txt`, tidak ada CI, tidak ada
test otomatis. Verifikasinya adalah blok diagnostik yang dicetak skrip itu
sendiri. Kontainer pengembangan tidak punya GPU, jadi run sungguhan selalu di
Kaggle.

**Anggaran:** `INFEST_BUDGET_H` mengatur pagar waktu. v13 memakai gubernur
anggaran yang MENGUKUR kecepatan GPU dgn tolok ukur 12 langkah lalu memilih
rencana yang muat, dan menyimpan checkpoint per fold (`_ck13/*.npz`) supaya
sesi yang mati bisa dilanjutkan. Proyeksinya terukur 17% optimistis di run
nyata (2,54 jam diproyeksikan vs 2,97 jam aktual per seed) — tertutup oleh
faktor keamanan 1,15, tapi ketahuilah angkanya.

---

## 9. Jebakan kode yang berulang di proyek ini

Semuanya **kegagalan senyap**: skrip tetap jalan dan mencetak angka yang
masuk akal.

- **Substitusi backbone diam-diam.** `build()` di v5–v10 jatuh ke resnet34
  kalau nama timm gagal. Run berlabel `dinov2` bisa sebenarnya resnet34.
  v11+ melewati sumber itu dan mencetak alasannya.
- **ViT beresolusi tetap.** `vit_base_patch14_reg4_dinov2.lvd142m` img_size
  aslinya 518, bukan 224. Cache loader harus di-key berdasarkan resolusi,
  karena ViT di img_size lain punya pos-embed yang berbeda.
- **Mengembalikan modul ter-cache, bukan salinannya.** Kalau loader
  menyerahkan instance yang sama ke tiap fold, fine-tune fold 0 mengubah
  bobot awal fold 1. Gejalanya: fold pertama sehat, fold berikutnya kolaps.
- **Layer-wise LR decay yang tidak melakukan apa-apa.**
  `timm.optim.param_groups_layer_decay` mengembalikan grup ber-key `lr_scale`,
  yang diabaikan AdamW biasa lalu ditimpa OneCycleLR. Knob ini MATI di v10.
  v11+ menghitung kedalaman dari nama parameter dan mencetak berapa skala
  berbeda yang dihasilkan — kalau cetakannya "1 skala", ia mati lagi.
- **Menggerakkan prediksi dari `test.csv`.** Lihat jebakan 1 di bagian 2.
- **Grid patch berlangkah tetap merosot.** Dgn step = P//2 pada citra yang
  dinormalkan sisi-pendek, citra blok yang hampir persegi menghasilkan satu
  posisi untuk semua view (terukur: 23 dari 57 citra blok test nyata).
- **v13 hanya menyimpan PROBABILITAS, bukan bobot model.** `_ck13/*.npz`
  berisi `va`, `te`, `f1` saja. Artinya eksplorasi TTA atau ensemble
  pasca-run **mengharuskan latih ulang**. Kalau Anda menulis v14, simpan
  `state_dict` — biayanya ruang disk, hematnya berjam-jam GPU.

---

## 10. Langkah berikutnya, diperingkat, dgn biaya dan ekspektasi jujur

**Kenyataan yang harus diterima lebih dulu:** sumbu resolusi hampir habis,
dan tidak ada lagi tombol murah. Naik ke peringkat 4 butuh `popf1` ~0.9850
dari 0.9799 sekarang, sementara total kesalahan OOF cuma 52 baris. Itu
berarti memperbaiki sekitar 15–20 baris, 60% di antaranya pegon↔jawi.
Itu lompatan kualitatif, bukan penyetelan.

### Peringkat 1 — sonde DINOv3 (0,6 jam T4) → `kode/v14_sonde.py`

Satu-satunya sumbu keberagaman yang belum pernah diuji di data ini adalah
**keluarga arsitektur lain**. DINOv3-B/16 @384 (`vit_base_patch16_dinov3.lvd1689m`)
adalah satu-satunya kandidat yang lolos dari sapuan 19 backbone: probe 0.9361
lawan SigLIP2-384 0.9439, selisih -0.0078 probe ≈ **-0.003 fine-tune** setelah
dikoreksi faktor 3x dari poin 3 bagian 5.

Sonde melatih SATU fold (fold 0) dan membandingkannya dgn tiga acuan seed v13
di fold yang sama (0.9849 / 0.9857 / 0.9879, sd 0.0015). Aturan keputusannya
ditulis di dalam skrip SEBELUM angkanya ada:

| F1 fold 0 | artinya |
|---|---|
| >= 0.980 | sekelas → ensemble lintas-keluarga layak, DIGERBANG di OOF |
| 0.970–0.980 | batas → layak hanya kalau overlap kesalahannya rendah |
| < 0.970 | jangan → ini wilayah `submission_iw.csv` |

Skrip juga mencetak jumlah baris yang salah, supaya overlap kesalahan dgn
SigLIP2 bisa dihitung kalau `_ck13/main_s0_f0.npz` dari run v13 masih ada.
Itu penting: anggota yang sedikit lebih lemah tapi salah di baris BERBEDA
tetap berguna; anggota yang salah di baris yang SAMA tidak menambah apa pun.

Sonde menolak memulai kalau tolok ukurnya bilang tidak muat, dan tidak pernah
menulis submission. Itu memang rancangannya.

### Peringkat 2 — serang pegon↔jawi (3–4 jam T4, nilai tidak diketahui)

60% kesalahan ada di sini dan pegon adalah kelas terkecil (309 baris). Yang
SUDAH dicoba dan gagal: spesialis biner (+0.0003), bobot per-kelas (+0.0001),
pra-latih sintetis sbg inisialisasi (-0.0143). Yang BELUM pernah dicoba:

**`SYNMIX`** — mencampur citra aksara sintetis ke dalam batch nyata sebagai
augmentasi, bukan sebagai inisialisasi. Sudah dikodekan penuh di `kode/v13.py`
(`SYNMIX = dict(enable=False, ...)`) tapi **belum pernah dijalankan sekali pun**.
Dasarnya: `[S2]` transfer langsung sintetis→nyata mencapai 0.4370 lawan
tebakan acak 0.1429, jadi korpusnya memang mengajarkan bentuk huruf; yang
gagal cuma cara memakainya sebagai titik awal. Mencampurnya ke batch nyata
adalah cara ketiga yang belum terukur. Render 14.000 citra ~15 menit CPU.

**Wajib:** jalankan sebagai uji terkontrol lawan seed 0 yang SYNMIX-nya mati.
Tanpa pembanding itu angkanya tidak berarti apa-apa.

### Peringkat 3 — resolusi 512 (5,3 jam T4, ekspektasi +0.001 OOF)

Probe menempatkan siglip2-512 di 0.9472 lawan siglip2-384 di 0.9439, selisih
+0.0033. Dikoreksi faktor 3x itu jadi **≈ +0.0011 fine-tune** — persis di
ambang gerbang seed-bag, dan menerjemah ke ~+0.002 LB. Biayanya 1.025 token
per citra lawan 577, jadi ~1,78x lebih lambat: 5 fold ≈ 5,3 jam.

**Nilai per jam terendah dari ketiganya.** Masukkan hanya kalau kuota
berlimpah dan dua peringkat di atas sudah dijawab.

### Peringkat 4 — TTA di luar distribusi augmentasi latih (murah, TAPI butuh bobot)

Lihat penjelasan struktural di bagian 6. Idenya sah, tapi v13 tidak menyimpan
bobot model, jadi mencobanya sekarang berarti melatih ulang 3 jam untuk
eksperimen inferensi 10 menit. **Simpan `state_dict` di v14**, lalu ini jadi
murah.

### Yang TIDAK perlu dicoba lagi

Seluruh isi tabel di bagian 6. Khususnya: jangan tambah seed, jangan ulang
spesialis biner dgn resep yang sama, jangan sentuh Otsu, jangan pakai
`impf1`, dan jangan gabungkan dua model SigLIP2.

---

## 11. Aturan kerja yang dipakai proyek ini

1. **Satu gerbang per keputusan, dinilai di `popf1` OOF, dgn ambang derau
   yang ditulis sebelum angkanya ada.** Inilah alasan v13 naik sementara v12
   `submission_iw.csv` turun. Jangan longgarkan gerbang karena hasilnya
   mengecewakan — gerbang yang menolak semua tambahan sudah bekerja benar.
2. **Ukur, jangan tebak.** Gubernur anggaran menjalankan tolok ukur nyata
   sebelum memilih rencana. Kecepatan T4 di Kaggle bervariasi 2x.
3. **Checkpoint tiap fold.** Sesi Kaggle bisa mati di jam ke-11.
4. **Tulis submission sesering mungkin.** v13 menulis ulang `submission.csv`
   tiap fold selesai, jadi sesi yang mati tetap meninggalkan berkas sah.
5. **Periksa distribusi prediksi sebelum submit.** Lihat poin 5 bagian 5.
6. **Dua slot final: JANGAN biarkan Kaggle memilih otomatis.** Kalau tidak
   dicentang manual, sistem mengambil dua skor publik tertinggi — dan dua
   skor tertinggi biasanya berasal dari model yang nyaris identik, yang
   persis pilihan terburuk. Kedua slot dinilai di himpunan privat yang SAMA,
   jadi derau samplingnya saling meniadakan dan yang tersisa hanya selisih
   antar-berkas: dua berkas kembar membuang separuh nilai slot kedua.
   **Rekomendasi saat ini: slot 1 = `submission.csv` v13 (0.89166), slot 2 =
   `submission.csv` v12 (0.87749).** Keduanya cuma sepakat 92,70%, jadi
   slot kedua benar-benar membeli sesuatu.

---

## 12. Konteks orang & bahasa

Pemilik: **salva** (GitHub `AffrizaWildanFauzan`). Berbahasa Indonesia —
balas dalam bahasa Indonesia. Dia menjalankan semua run di Kaggle dgn GPU T4
dan kuotanya terbatas (mingguan), jadi setiap usulan harus menyertakan
**biaya jam GPU-nya**, dan harus jujur kalau sesuatu tidak muat.

Dia mengirimkan log run lengkap dan berkas CSV untuk diperiksa; membaca log
itu baris demi baris adalah bagian dari pekerjaan, bukan formalitas.
