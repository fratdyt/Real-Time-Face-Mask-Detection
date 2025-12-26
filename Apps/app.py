import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image

# 1. LOAD MODEL & HAAR CASCADE
@st.cache_resource
def load_my_model():
    # Menggunakan path absolut
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'face_mask_detection.h5')
    
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di: {model_path}")
        st.stop()
        
    model = tf.keras.models.load_model(model_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    return model, face_cascade

model, face_cascade = load_my_model()

# 2. KONFIGURASI LABEL & WARNA (Format BGR untuk OpenCV)
# 0: Incorrect (Kuning), 1: Mask (Hijau), 2: Not Mask (Merah)
label_dict = {0: 'Mask Weared Incorrect', 1: 'Mask', 2: "Not Mask"} 
color_dict = {
    0: (0, 255, 255), # Kuning
    1: (0, 255, 0),   # Hijau
    2: (0, 0, 255)    # Merah
}

st.set_page_config(page_title='Mask Detection App', layout='wide')
st.title('🛡️ Real-Time Face Mask Detector')

option = st.sidebar.selectbox('Choose Mode:', ('Upload Image', 'Live Webcam'))

# 3. FUNGSI PROCESSING UTAMA
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah dengan parameter yang seimbang
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=6, 
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        # PADDING: Mengambil area sedikit lebih luas agar model melihat dahi/leher
        offset = 20
        y1, y2 = max(0, y - offset), min(frame.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(frame.shape[1], x + w + offset)
        
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            continue

        # PREPROCESSING: Konversi BGR ke RGB (Sangat Penting!)
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_img_rgb, (224, 224))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 224, 224, 3))

        # PREDIKSI
        result = model.predict(reshaped)[0]
        
        # Logika Tuning untuk Incorrect Mask (Index 0)
        # Jika probabilitas 'Incorrect' cukup signifikan (>30%), kita prioritaskan
        incorrect_prob = result[0]

        if incorrect_prob > 0.30:
            label = 0
            confidence = incorrect_prob
        else:
            label = np.argmax(result)
            confidence = result[label]

        # VISUALISASI: Gambar kotak & Teks
        color = color_dict[label]
        display_text = f'{label_dict[label]}: {confidence * 100:.2f}%'

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, display_text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame

# 4. LOGIKA ANTARMUKA (UPLOAD VS WEBCAM)
if option == 'Upload Image':
    uploaded_file = st.file_uploader('Choose images...', type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)

        # Konversi PIL (RGB) ke OpenCV (BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output_frame = process_frame(frame)

        # Tampilkan kembali ke Streamlit (Balikkan ke RGB)
        st.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), caption='Result Detection', use_container_width=True)

elif option == 'Live Webcam':
    st.info('Klik checkbox di bawah untuk mulai. Pastikan pencahayaan cukup.')

    run = st.checkbox('Turn On Camera')
    frame_window = st.image([])
    
    # Inisialisasi Kamera
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        
        if not ret:
            st.error('Gagal mengakses kamera.')
            break

        frame = process_frame(frame)
        
        # Update tampilan secara real-time
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        camera.release()
        st.write("Kamera dimatikan.")