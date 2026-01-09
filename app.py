import streamlit as st
import os
import gdown  # Thư viện tải file từ Drive

# --- CẤU HÌNH TỰ ĐỘNG TẢI MODEL ---
# Hướng dẫn lấy ID: Link drive có dạng .../d/1A2b3C.../view -> ID là 1A2b3C...
CLS_MODEL_ID = '18ziycNqCuZQ7G9jkAe4nOM0004dBcNXE' 
SEG_MODEL_ID = '134Yb6dnyTa-d7UyrE--8MupQ2uMPIACV'

CLS_MODEL_PATH = 'cls_model.pth'
SEG_MODEL_PATH = 'seg_model.pth'

@st.cache_resource
def download_models():
    # Tải model chẩn đoán
    if not os.path.exists(CLS_MODEL_PATH):
        url = f'https://drive.google.com/uc?id={CLS_MODEL_ID}'
        gdown.download(url, CLS_MODEL_PATH, quiet=False)
    
    # Tải model phân đoạn
    if not os.path.exists(SEG_MODEL_PATH):
        url = f'https://drive.google.com/uc?id={SEG_MODEL_ID}'
        gdown.download(url, SEG_MODEL_PATH, quiet=False)

# Gọi hàm tải ngay khi App khởi động
download_models()

# ... (Phần code còn lại của app.py giữ nguyên) ...