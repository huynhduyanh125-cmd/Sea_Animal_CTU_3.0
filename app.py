import os
import gdown
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# 1. Điền ID file Google Drive của bạn vào đây
GDRIVE_FILE_ID = "1Wo0GKJkAMkc0gIIUgLupNYXN5aqr2lcq"
MODEL_PATH = "best_efficientnetb7_model.h5"


# 2. Hàm tự động tải mô hình từ Drive nếu chưa có trên server
@st.cache_resource
def load_my_model():
  if not os.path.exists(MODEL_PATH):
    with st.spinner(
        "⏳ Đang tải mô hình AI từ Google Drive (chỉ mất 1-2 phút lần đầu tiên)..."
    ):
      url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
      gdown.download(url, MODEL_PATH, quiet=False)
  return tf.keras.models.load_model(MODEL_PATH)


# Nạp mô hình
model = load_my_model()

# 3. Danh sách loài sinh vật biển
CLASS_NAMES = [
    'Clams',
    'Corals',
    'Crabs',
    'Dolphin',
    'Eel',
    'Jellyfish',
    'Lobster',
    'Nudibranchs',
    'Octopus',
    'Penguin',
    'Puffers',
    'Sea Rays',
    'Sea Urchins',
    'Seahorse',
    'Seal',
    'Sharks',
    'Shrimps',
    'Squid',
    'Starfish',
    'Whale',
]

# 4. Giao diện Web Streamlit
st.title('🌊 AI Phân Loại Sinh Vật Biển')
st.write(
    'Tải lên một bức ảnh sinh vật biển để mô hình EfficientNetB7 nhận diện!'
)

uploaded_file = st.file_uploader(
    'Chọn một tấm ảnh (JPG, PNG, JPEG)...', type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert('RGB')
  st.image(image, caption='Ảnh bạn đã tải lên', use_container_width=True)

  with st.spinner('AI đang phân tích bức ảnh...'):
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]

    st.subheader('📊 Kết quả dự đoán hàng đầu:')
    for idx in top_indices:
      class_name = CLASS_NAMES[idx]
      confidence = float(predictions[idx] * 100)
      st.write(f'**{class_name}**: {confidence:.2f}%')
      st.progress(int(confidence))
