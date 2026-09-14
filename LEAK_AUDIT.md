# Audit kebocoran: apakah model belajar aksara, atau belajar sumber scan?

## Pertanyaan
Bisakah kelas aksara diprediksi dari fitur yang **tidak memuat informasi bentuk
huruf sama sekali** — dimensi gambar, ukuran file, tingkat noise scan?

Kalau bisa, maka CNN yang dilatih pada dataset ini dapat mencapai skor tinggi
dengan menghafal "potongan ini berasal dari scan buku mana", bukan dengan
mengenali aksaranya.

## Metode
Label acuan memakai prediksi visual independen (`submission_claude.csv`,
akurasi ~0.91 menurut leaderboard). Itu menetapkan **plafon**: seandainya
metadata memprediksi label sebenarnya dengan sempurna, kecocokan terukurnya
tetap hanya ~0.91.

RandomForest, 5-fold stratified CV, 1118 gambar test.

## Hasil

| fitur yang dipakai | acc | macro F1 |
|---|---:|---:|
| chance (stratified dummy) | 0.157 | 0.145 |
| **hanya `w,h`** (dua bilangan bulat) | 0.615 | 0.615 |
| hanya `fsize,bpp` | 0.542 | 0.530 |
| **metadata kontainer murni** (`w,h,aspect,npix,fsize`) | **0.713** | **0.705** |
| semua metadata non-piksel | 0.809 | 0.801 |
| *plafon (akurasi label acuan)* | *~0.91* | — |

**Lima bilangan yang tidak bisa "melihat" aksara sama sekali memulihkan ~78%
dari sinyal yang mungkin dipulihkan.** Lebar dan tinggi gambar saja sudah 4x
lipat di atas chance.

## Sidik jari sumber per kelas

| label | tinggi px (p10/p50/p90) | bg_std (noise) | bpp (kompleksitas scan) |
|---|---|---:|---:|
| pegon | 69 / 102 / 154 | 8.0 | 0.293 |
| lampung | 35 / 42 / 81 | 14.9 | 0.366 |
| jawi | 40 / 63 / 87 | 6.0 | 0.226 |
| jawa | 56 / 72 / 95 | **18.1** | **0.658** |
| sunda | 41 / 50 / 79 | 7.0 | 0.246 |
| lontara | 40 / 53 / 71 | 5.9 | **0.158** |
| bali | 56 / 63 / 73 | 7.1 | 0.263 |

Kemurnian kelas hanya dari tinggi gambar:

| pita tinggi | n | dominan |
|---|---:|---|
| 100–130 px | 61 | pegon **63.9%** |
| 130–160 px | 63 | pegon **88.9%** |
| >160 px | 15 | pegon **100%** |

Klaster KMeans pada metadata saja (k=30) menghasilkan kemurnian kelas
rata-rata **0.653** — acak murni hanya 0.19.

## Tafsiran
Tiap kelas dipotong dari sekumpulan kecil dokumen sumber, dan tiap dokumen
punya resolusi scan, tingkat noise, serta ukuran baris yang khas. Akibatnya
label kelas hampir setara dengan identitas dokumen sumber.

Konsekuensinya, skor tinggi pada lomba ini **tidak membuktikan model mengenali
aksara**. Model bisa saja hanya mengenali buku asalnya — dan itu cukup untuk
skor mendekati sempurna selama train dan test dipotong dari kumpulan scan
yang sama.

## Batas analisis ini
Label acuan berasal dari inspeksi visual saya sendiri, dan dalam menilai saya
memang sebagian memakai gaya goresan/scan (mis. "bali = goresan tipis terukir",
"pegon = blok kecil padat"). Itu menaikkan sebagian angka di tabel.

Namun `w`, `h`, `aspect`, `npix`, `fsize` **tidak pernah saya lihat** saat
melabeli — saya hanya melihat gambarnya. Jadi baris **0.713 / 0.705** bersih
dari sirkularitas, dan itulah angka intinya.

## Cara memverifikasi pada notebook Anda
1. Hitung `w, h, fsize` untuk tiap gambar **train**, lalu latih RandomForest
   pada tiga fitur itu saja terhadap label train yang asli. Kalau akurasinya
   jauh di atas chance (~0.15), kebocoran terkonfirmasi dengan label sebenarnya.
2. Klaster gambar train pada metadata itu menjadi ~30 grup (proxy "dokumen sumber").
3. Ganti validasi Anda ke `GroupKFold` dengan grup tersebut, sehingga tidak ada
   dokumen sumber yang muncul di train dan validasi sekaligus.
4. Bandingkan skornya dengan CV acak Anda sekarang. Selisihnya adalah besarnya
   bagian skor Anda yang sebenarnya berasal dari menghafal sumber.

Kalau langkah 4 menunjukkan penurunan besar, skor 100% itu nyata untuk
leaderboard ini tetapi tidak akan bertahan pada scan naskah baru.

Reproduksi: `python leak_probe.py` (perlu `data/test/` hasil ekstrak `test.zip`).
