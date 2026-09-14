# Klasifikasi aksara pada data test (pembanding independen)

## Ringkasan
`submission_claude.csv` berisi prediksi label untuk 1118 gambar di `test.zip`,
urutan baris identik dengan `sample_submission.csv`.

## Catatan penting: gambar train TIDAK tersedia
Repo ini hanya memuat `test.csv` dan `test.zip`. Tidak ada gambar train di
branch manapun maupun di release. `train.csv` hanya berisi pasangan
`image_id,label` (4468 baris) tanpa file gambarnya.

Konsekuensi: tidak mungkin melatih model terawasi. Klasifikasi ini dibuat
lewat inspeksi visual tiap gambar, dikalibrasi terhadap render referensi
tiap aksara (Noto Sans Javanese/Balinese/Sundanese/Buginese + Arabic).
Jadi ini pembanding independen, bukan hasil model.

## Aturan pembeda yang dipakai
- **lontara**  - chevron/zigzag (∧ ∨) beraturan dengan titik; sangat khas.
- **lampung**  - tulisan tangan, garis tipis menyudut, bentuk ᴨ/ᴜ/ᴎ + diakritik kecil.
- **sunda**    - blok menyudut miring seperti jajaran genjang (tampak "italic").
- **jawa**     - lengkung membulat, puncak rata dan lebar, sering ada tanda ulang "2".
- **bali**     - goresan tipis tinggi, banyak kait di atas, pemisah carik ᭞.
- **pegon**    - Arab dengan harakat padat, gaya kitab, sering ada gloss Jawa miring
                 di antar baris; kata Jawa (iku, wong, kang).
- **jawi**     - Arab tanpa harakat, naskh tipis; kata Melayu (dan, pada, yang, daripada).

## Validasi
Distribusi prediksi vs distribusi train (test tidak berlabel, jadi ini satu-satunya
pemeriksaan global yang tersedia):

| label   | prediksi | prediksi % | train % |
|---------|---------:|-----------:|--------:|
| lampung |      209 |      18.7% |   18.4% |
| pegon   |      213 |      19.1% |   17.3% |
| jawi    |      184 |      16.5% |   18.2% |
| jawa    |      175 |      15.7% |   14.6% |
| sunda   |      163 |      14.6% |   14.7% |
| lontara |       95 |       8.5% |    8.5% |
| bali    |       79 |       7.1% |    8.2% |

## Batas keandalan (harap dibaca sebelum dipakai membandingkan)
Tidak ada ground truth, jadi akurasi sebenarnya tidak terukur. Sumber galat
yang diperkirakan, dari yang terbesar:

1. **pegon vs jawi** - keduanya aksara Arab. ~15 potongan yang isinya hanya
   nomor halaman (angka Arab, mis. ٤٩ ١٢٥ ٣٠٨) tidak punya petunjuk bahasa
   sama sekali; setelah dinormalkan tingginya, bobot hurufnya pun seragam.
   Potongan itu diberi label `jawi` karena termasuk tebakan paling lemah dan
   prior train menunjukkan jawi kurang terwakili - bukan karena ada bukti visual.
2. **jawa vs bali** - pasangan paling mirip secara struktur. Dipisah lewat
   gaya goresan (bali: tipis-tinggi-berkait; jawa: bulat-lebar-rata).
3. Potongan sangat kecil (satu huruf Latin/Arab, coretan) yang praktis tak
   punya informasi aksara.

Bagian 1 dan 3 saja sudah mencakup ~5% test set.

## Reproduksi
- `montage.py` menyusun 1118 gambar jadi 70 lembar berlabel indeks.
- `sheet_index.json` memetakan lembar -> nama file (urut).
- `labels.txt` label mentah per lembar, 16 per baris, urut sesuai indeks lembar.
