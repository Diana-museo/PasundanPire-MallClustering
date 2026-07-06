<!-- Tampilkan di bagian paling atas README -->
<div align="center">
  <a href="https://dianamuseo.pythonanywhere.com/">
    <img src="https://img.shields.io/badge/🚀_Akses_Versi_Live_Web_kami_di_sini!-brightgreen?style=for-the-badge&logo=render" alt="Live Demo">
  </a>
</div>

# 🛒 Mall Experience & Resource Optimization – Customer Segmentation

![Python](https://img.shields.io/badge/Python-v3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-v3.0+-black?style=for-the-badge&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.2+-orange?style=for-the-badge&logo=scikit-learn)
![Chart.js](https://img.shields.io/badge/Chart.js-v4.0+-salmon?style=for-the-badge&logo=chartdotjs)

Aplikasi web dashboard ini merupakan implementasi sistematis dari teknik *Unsupervised Learning* menggunakan algoritma **K-Means Clustering** untuk melakukan segmentasi profil pelanggan mall. Proyek ini dirancang untuk menyelesaikan tantangan alokasi sumber daya operasional berbasis wawasan data numerik yang akurat.

---

## 🔎 Project Overview & Studi Kasus

### 1. Domain Proyek
Manajemen operasional mall dan strategi pengelolaan pengalaman pelanggan (*Customer Experience Management*).

### 2. Permasalahan (Problem Statements)
Pengelola mall menghadapi tantangan besar dalam mengalokasikan sumber daya secara efisien (seperti penentuan jenis tenant, pembagian zona hiburan, layanan fasilitas, hingga target promosi). Tanpa adanya pengelompokan karakteristik pengunjung yang sistematis, pihak manajemen tidak dapat menentukan area mana yang perlu ditingkatkan, jenis kemitraan retail apa yang perlu dihadirkan, atau bagaimana mengoptimalkan tata letak ruang demi mendongkrak keuntungan.

### 3. Tujuan & Manfaat Proyek
* **Segmentasi Terukur:** Mengidentifikasi kelompok-kelompok pengunjung mall berdasarkan kombinasi fitur Usia, Pendapatan Tahunan (*Annual Income*), dan Skor Pengeluaran (*Spending Score*).
* **Efisiensi Sumber Daya:** Membantu manajemen mall mengoptimalkan penempatan tenant, merancang zona pengalaman (*experience zone*) yang sesuai preferensi segmen dominan, dan menyusun anggaran operasional secara presisi.
* **Data-Driven Decision:** Mendukung proses negosiasi nilai kontrak dengan calon *tenant* menggunakan bukti data segmentasi pengunjung yang konkret di lapangan.

---

## ⚙️ Fitur Utama Aplikasi Web

Antarmuka dashboard web dikembangkan menggunakan Flask untuk kebutuhan presentasi bisnis interaktif, yang terbagi ke dalam **3 Tab Utama**:

* **🔍 Prediksi Cluster (Segmentasi Real-Time):** Menyediakan form input dinamis berbasis slider untuk parameter Gender, Usia, *Annual Income*, dan *Spending Score*. Sistem akan langsung melakukan normalisasi data dan memprediksi cluster pelanggan, lengkap dengan **nama segmen unik** serta **rekomendasi strategi manajemen mall** yang spesifik per segmen.
* **📊 Evaluasi Model (Analisis Distribusi):**
    Menampilkan metrik performa model final (Nilai **K=6** optimal & nilai **Silhouette Score**). Halaman ini menyajikan visualisasi grafik batang distribusi dari 200 data pengunjung asli, tabel parameter rata-rata setiap cluster, serta kartu detail deskripsi karakteristik 6 segmen.
* **⚙️ Optimasi Model (Riwayat Eksperimen):**
    Menampilkan visualisasi grafik interaktif kurva *Silhouette Score vs Jumlah K* menggunakan Chart.js. Halaman ini mendokumentasikan tabel komparasi dari **54 kombinasi parameter** hasil proses *hyperparameter tuning* dengan penanda (*highlight*) otomatis pada baris konfigurasi terbaik.

---

## 📂 Struktur Proyek & Alur Kerja (Workflow)

```text
mall_clustering/
├── app.py                     # Core Server Flask & Logika Bisnis API
├── templates/
│   └── index.html             # Antarmuka Dashboard (HTML5, CSS Grid, Chart.js)
├── models/
│   ├── kmeans_model.pkl       # Serialisasi Binary Model K-Means Optimal
│   └── scaler.pkl             # Serialisasi Binary StandardScaler Fitur
└── data/
    ├── customer_clustering_hasil.csv   # Dataset Hasil Prediksi Cluster Akhir
    ├── hasil_evaluasi.csv              # Statistik Profil Rata-Rata Cluster
    └── hasil_tuning.csv                # Rekam Jejak 54 Kombinasi Grid Search
```

---

## 📋 Pembagian Tahapan Kerja Notebook (Google Colab)

Riset pemodelan machine learning dalam proyek ini disusun ke dalam struktur sel notebook yang terorganisir sebagai berikut:

| Tahapan Utama | Cakupan Sel (Cells) | Penanggung Jawab | Deskripsi Detail Eksekusi Kode |
|----------|----------|----------|----------|
| **Persiapan Data** | Cell 1 – 9 | Ariska Putri & Haifa Zahirah Ramdhan | Import library, load raw dataset, pengecekan statistika deskriptif, penanganan missing values, pembuatan visualisasi korelasi (heatmap) dan pencilan (boxplot), normalisasi fitur via StandardScaler, serta penyimpanan data bersih ke format CSV. |
| **Pelatihan Model** | Cell 10 - 15 | Emeralda Iffatud Diana | Penjelasan konteks bisnis clustering, analisis penentuan jumlah klaster melalui grafik Elbow Method, proses fitting algoritma awal K-Means, penyusunan fungsi simpan model biner (.pkl), dan dokumentasi parameter awal. |
| **Evaluasi Model** | Cell 16 - 20 | Ellen Aplida Zalni & Muhammad Sufi Nadziffa Ridwan | Riset matematis metrik kedekatan klaster, kalkulasi nilai Silhouette Score komprehensif, perancangan grafik Silhouette Plot per klaster, perekaman metrik evaluasi ke CSV, serta penulisan analisis rekomendasi bisnis awal. |
| **Optimasi Model** | Cell 21 - 27 | Daffa Kamiliya Mufidah & Shabina Arvelista Rahman | Implementasi eksperimen otomatis (Grid Search) menguji kombinasi nilai K (2-10) × variasi metode inisialisasi (k-means++ & random) × parameter max_iter hingga total 54 kombinasi, penentuan best_config berbasis nilai Silhouette tertinggi, visualisasi komparatif grafik metrik, dan ekspor file model final. |

---

### 🚀 Panduan Instalasi & Setup Lingkungan (Jika ingin local run)

Panduan ini disusun untuk membantu pengguna baru menjalankan aplikasi dari nol secara lokal.

#### Langkah 1: Instalasi Core Python (Melalui Website Resmi)

1. Kunjungi halaman resmi: python.org/downloads
2. Unduh paket installer resmi (Direkomendasikan Python versi 3.10 atau yang lebih baru).
3. Saat menjalankan berkas instalasi .exe, WAJIB MENCENTANG KOTAK "Add python.exe to PATH" di bagian paling bawah jendela setup sebelum mengklik tombol Install Now.

#### Langkah 2: Setup Workspace & Direktori

Buka aplikasi PowerShell / Command Prompt (Windows) atau Terminal (macOS/Linux), lalu ketikkan perintah berikut untuk masuk ke folder proyek:

- Windows:
```bash
cd D:\PasundanPire-MallClustering
```
- macOS/Linux:
```bash
cd /path/ke/folder/PasundanPire-MallClustering
```

#### Langkah 3: Isolasi Lingkungan Virtual (.venv)

Pembuatan Virtual Environment sangat disarankan agar seluruh pustaka (library) dependencies proyek terisolasi dengan aman dari sistem global komputer Anda.

- Windows:
```bash
# 1. Membuat folder environment terisolasi
python -m venv .venv

# 2. bypass kebijakan eksekusi script jika aktivasi diblokir sistem Windows
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Aktivasi environment
.venv\Scripts\Activate.ps1
```
- macOS/Linux:
```bash
# 1. Membuat folder environment
python3 -m venv .venv

# 2. Aktivasi environment
source .venv/bin/activate
```

#### Langkah 4: Instalasi Dependencies Proyek

Setelah seluruh komponen pustaka pendukung berhasil terpasang, nyalakan server lokal Flask dengan mengeksekusi perintah berikut pada terminal Anda:

- Windows:
```bash
python app.py
```
- macOS/Linux:
```bash
python3 app.py
```

#### Cara Mengakses Dashboard:
1. Jalankan aplikasi web browser pilihan Anda (Google Chrome, Mozilla Firefox, Edge, atau Safari).

2. Salin dan kunjungi alamat URL lokal berikut pada address bar: http://127.0.0.1:5000

3. Anda dapat langsung menguji fungsionalitas simulasi prediksi pada Form Tab 1 serta memantau grafik analitik pada Tab 2 dan Tab 3.

---

### 🛠️ Troubleshooting (Penyelesaian Masalah)

- Masalah: Perintah python Tidak Dikenali (ObjectNotFound / CommandNotFoundException)
    - Solusi: Hal ini terjadi karena opsi PATH terlewat saat instalasi awal. Solusi tercepat adalah membuka kembali file installer .exe Python yang telah diunduh, klik menu Modify, centang opsi "Add Python to PATH", lalu klik simpan dan restart ulang aplikasi terminal Anda.

- Masalah: Error Eksekusi Script .ps1 Diblokir Sistem (UnauthorizedAccess)
    - Solusi: Ini adalah proteksi keamanan bawaan Windows PowerShell. Selesaikan masalah dengan mengetikkan perintah Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser, tekan enter, lalu jalankan kembali perintah aktivasi .venv.

- Masalah: Tampilan Grafik Tidak Muncul di Browser
    - Solusi: Dashboard web ini memuat pustaka visualisasi grafik secara daring. Pastikan perangkat komputer Anda terhubung ke jaringan internet yang stabil agar browser dapat memuat pustaka eksternal Chart.js dan font dari Google Fonts secara sempurna.