🛡️ Real-Time Face Mask Detection using MobileNetV2

Proyek ini adalah aplikasi web berbasis Streamlit yang mampu mendeteksi penggunaan masker wajah secara real-time maupun melalui unggahan gambar. Model dikembangkan menggunakan arsitektur MobileNetV2 dengan teknik Transfer Learning untuk mencapai akurasi yang tinggi namun tetap ringan dijalankan pada perangkat standar.

🚀 Fitur Utama

Dual Mode Detection: Mendukung deteksi via Live Webcam dan Upload Gambar (JPG/JPEG/PNG).

Tiga Kelas Klasifikasi:  

✅ Mask: Masker dipakai dengan benar.

⚠️ Mask Weared Incorrect: Masker dipakai tetapi tidak menutupi hidung/mulut.

❌ Not Mask: Tidak menggunakan masker.

Real-Time Performance: Optimasi menggunakan MobileNetV2 sehingga inferensi berjalan cepat (low latency).

Visual Feedback: Menampilkan bounding box berwarna (Hijau/Kuning/Merah) disertai persentase keyakinan (confidence score).


📊 Hasil Pelatihan Model

Model dilatih menggunakan TensorFlow/Keras dengan spesifikasi:

Base Model: MobileNetV2 (Pre-trained on ImageNet).

Akurasi Validasi: 97.21%.

Input Size: 224x224 piksel.

Optimizer: Adam.


🛠️ Teknologi yang Digunakan

Python (Bahasa pemrograman utama)

TensorFlow/Keras (Pengembangan model Deep Learning)

Streamlit (Framework antarmuka web)

OpenCV (Pengolahan citra & Haar Cascade face detection)

NumPy & Pandas (Pengolahan data)
