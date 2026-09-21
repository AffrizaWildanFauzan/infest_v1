# Submission yang pernah dikirim ke kompetisi

File di folder ini adalah keluaran model kami sendiri (hanya `image_id,label`),
bukan data kompetisi.

## Skor yang DIKONFIRMASI oleh tim

| file | baris | skor LB | catatan |
|---|---|---|---|
| `submission_infest_v6.csv` | 1220 | **0.83902** | skor terbaik kami sampai sekarang |
| `submission_infest_v7.csv` | 1220 | 0.83422 | |
| `submission_infest_v8.csv` | 1220 | 0.83422 | **identik byte-for-byte dengan v7** — lihat peringatan di bawah |
| `pelabelan_manual_manusia.csv` | 1220 | 0.71330 | pelabelan manual manusia atas test, dipakai sebagai patokan bawah |
| `pembanding_claude.csv` | 1220 | 0.5763 | pembanding yang dibuat Claude, jauh di bawah model |

## Skor BELUM dikonfirmasi

| file | baris | catatan |
|---|---|---|
| `submission_infest.csv` | 1220 | versi awal; pemetaan ke nomor versi kode belum diverifikasi |
| `submission_infest_v2.csv` | 1220 | idem |

Riwayat LB yang disebut di `../handoff.md` (v3.1 = 0.8073, v5 = 0.81054) belum
bisa dipetakan ke dua file ini secara pasti, jadi namanya dibiarkan apa adanya.

## PERINGATAN: v8 tidak pernah benar-benar diukur

`submission_infest_v7.csv` dan `submission_infest_v8.csv` **identik byte-for-byte**
(md5 `cd5af34e`). Karena itu skor 0.83422 yang tercatat untuk "v8" sebenarnya
skor file v7 yang dikirim ulang. Semua perbandingan yang memperlakukan v8
sebagai hasil terukur tidak sah — perubahan kode v8 belum pernah diuji di
leaderboard.

## PERINGATAN: `pelabelan_manual_manusia.csv` punya label non-kelas

Isinya memuat `??` (1 baris) dan `bali mungkin` (1 baris), yang bukan nama kelas
yang sah. Dua baris itu dihitung salah oleh penilai, jadi 0.71330 adalah batas
bawah dari pelabelan manual itu, bukan angka bersihnya.

## `dataset-lain/`

`submission-6.csv` dan `submission_inf_1.csv` masing-masing 1118 baris, dan
`image_id`-nya **tidak beririsan sama sekali** dengan test kompetisi (1220
baris). Keduanya milik dataset lain yang sempat masuk ke ruang kerja, bukan
submission untuk lomba ini. Disimpan terpisah supaya tidak tertukar lagi —
kekeliruan itu pernah membuat satu diagnostik menghasilkan kesimpulan yang
salah total (lihat `../handoff.md` bagian 1g).
