import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# 1. Cấu hình giao diện trang web
st.set_page_config(
    page_title="AI Phân Loại Sinh Vật Biển", page_icon="🌊", layout="centered"
)


# 2. Nạp mô hình AI (dùng cache để web chạy siêu mượt không bị load lại)
@st.cache_resource
def load_my_model():
  return tf.keras.models.load_model('best_efficientnetb7_model.h5')


model = load_my_model()

# 3. Danh sách 20 loài sinh vật biển
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

# 4. Tiêu đề ứng dụng
st.title('🌊 AI Phân Loại Sinh Vật Biển')
st.write(
    'Tải lên một bức ảnh sinh vật biển để mô hình EfficientNetB7 nhận diện!'
)

# 5. Khung tải ảnh lên
uploaded_file = st.file_uploader(
    'Chọn một tấm ảnh (JPG, PNG, JPEG)...', type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
  # Hiển thị ảnh vừa chọn
  image = Image.open(uploaded_file).convert('RGB')
  st.image(image, caption='Ảnh bạn đã tải lên', use_container_width=True)

  with st.spinner('AI đang phân tích bức ảnh...'):
    # Preprocessing (Tiền xử lý ảnh)
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    # Dự đoán
    predictions = model.predict(img_array)[0]
    top_indices = np.argsort(predictions)[-3:][::-1]  # Lấy top 3 kết quả cao nhất

    # Hiển thị kết quả
    st.subheader('📊 Kết quả dự đoán hàng đầu:')
    for idx in top_indices:
      class_name = CLASS_NAMES[idx]
      confidence = float(predictions[idx] * 100)
      st.write(f'**{class_name}**: {confidence:.2f}%')
      st.progress(int(confidence))
