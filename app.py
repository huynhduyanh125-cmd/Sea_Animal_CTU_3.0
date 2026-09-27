import os
import gdown
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# 1. Cấu hình giao diện trang web
st.set_page_config(
    page_title="AI Phân Loại Sinh Vật Biển", page_icon="🌊", layout="centered"
)

# 2. Điền ID file Google Drive chứa mô hình .h5 của bạn vào đây
GDRIVE_FILE_ID = "1Wo0GKJkAMkc0gIIUgLupNYXN5aqr2lcq"
MODEL_PATH = "best_efficientnetb7_model.h5"


# 3. Hàm tự động tải và nạp mô hình từ Drive
@st.cache_resource
def load_my_model():
  if not os.path.exists(MODEL_PATH):
    with st.spinner(
        "⏳ Đang tải mô hình AI từ Google Drive (chỉ tải lần đầu tiên)..."
    ):
      url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
      gdown.download(url, MODEL_PATH, quiet=False)
  return tf.keras.models.load_model(MODEL_PATH)


# Nạp mô hình
model = load_my_model()

# 4. Danh sách 20 loài sinh vật biển (Tiếng Việt)
CLASS_NAMES = [
    'Sò / Nghêu',  # Clams
    'San hô',  # Corals
    'Cua',  # Crabs
    'Cá heo',  # Dolphin
    'Cá chình',  # Eel
    'Sứa',  # Jellyfish
    'Tôm hùm',  # Lobster
    'Sên biển',  # Nudibranchs
    'Bạch tuộc',  # Octopus
    'Chim cánh cụt',  # Penguin
    'Cá nóc',  # Puffers
    'Cá đuối',  # Sea Rays
    'Cầu gai (Nhím biển)',  # Sea Urchins
    'Cá ngựa',  # Seahorse
    'Hải cẩu',  # Seal
    'Cá mập',  # Sharks
    'Tôm',  # Shrimps
    'Mực',  # Squid
    'Sao biển',  # Starfish
    'Cá voi',  # Whale
]

# 5. Giao diện chính của Web
st.title('🌊 AI Phân Loại Sinh Vật Biển')
st.write(
    'Tải lên một bức ảnh sinh vật biển để mô hình EfficientNetB7 nhận diện!'
)

uploaded_file = st.file_uploader(
    'Chọn một tấm ảnh (JPG, PNG, JPEG)...', type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
  # Hiển thị ảnh tải lên
  image = Image.open(uploaded_file).convert('RGB')
  st.image(image, caption='Ảnh bạn đã tải lên', use_container_width=True)

  with st.spinner('AI đang phân tích bức ảnh...'):
    # Tiền xử lý ảnh
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    # Dự đoán
    predictions = model.predict(img_array)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]  # Lấy Top 3 kết quả cao nhất

    # Hiển thị kết quả
    st.subheader('📊 Kết quả dự đoán hàng đầu:')
    for idx in top_indices:
      # Nếu chỉ số thuộc 20 loài đã khai báo thì hiện tên, ngược lại ghi "Loài chưa ghi nhận"
      if idx < len(CLASS_NAMES):
        class_name = CLASS_NAMES[idx]
      else:
        class_name = 'Loài chưa ghi nhận'

      confidence = float(predictions[idx] * 100)
      st.write(f'**{class_name}**: {confidence:.2f}%')
      st.progress(min(int(confidence), 100))
