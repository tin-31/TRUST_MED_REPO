import streamlit as st
import os
import gdown
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import pydicom  # <--- Thư viện mới để đọc DICOM

# ==========================================
# ⚙️ CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(
    page_title="TRUST-MED: AI Siêu âm Kháng Ảo giác",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem;}
    .risk-high {color: #d32f2f; font-weight: bold;}
    .risk-low {color: #388e3c; font-weight: bold;}
    .warning-box {background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; border-left: 5px solid #ffeeba;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 📥 TẢI MODEL TỪ DRIVE
# ==========================================
CLS_MODEL_ID = '18ziycNqCuZQ7G9jkAe4nOM0004dBcNXE' 
SEG_MODEL_ID = '134Yb6dnyTa-d7UyrE--8MupQ2uMPIACV'
CLS_MODEL_PATH = 'cls_model.pth'
SEG_MODEL_PATH = 'seg_model.pth'

@st.cache_resource
def download_models():
    if not os.path.exists(CLS_MODEL_PATH):
        with st.spinner("📥 Đang tải Model Chẩn đoán..."):
            gdown.download(f'https://drive.google.com/uc?id={CLS_MODEL_ID}', CLS_MODEL_PATH, quiet=False)
    if not os.path.exists(SEG_MODEL_PATH):
        with st.spinner("📥 Đang tải Model Phân đoạn..."):
            gdown.download(f'https://drive.google.com/uc?id={SEG_MODEL_ID}', SEG_MODEL_PATH, quiet=False)

download_models()

# ==========================================
# 🧠 ĐỊNH NGHĨA MODEL
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0): super().__init__(); self.alpha = alpha
    def forward(self, x): return GradientReversalFunction.apply(x, self.alpha)

class TrustMedNet(nn.Module):
    def __init__(self, num_domains=3):
        super().__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.n_features = self.backbone.num_features
        self.diagnosis_head = nn.Sequential(nn.Linear(self.n_features, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.grl = GRL()
        self.domain_head = nn.Sequential(nn.Linear(self.n_features, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_domains))
        self.trust_head = nn.Sequential(nn.Linear(self.n_features, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        feat = self.backbone(x)
        return self.diagnosis_head(feat), self.domain_head(self.grl(feat)), self.trust_head(feat), feat

@st.cache_resource
def load_loaded_models():
    cls_model = TrustMedNet(num_domains=3)
    try: cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
    except: cls_model = TrustMedNet(num_domains=5); cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
    cls_model.to(DEVICE).eval()
    
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
    try: seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
    except: pass
    seg_model.to(DEVICE).eval()
    return cls_model, seg_model

cls_model, seg_model = load_loaded_models()

# ==========================================
# 🛠️ CÁC HÀM XỬ LÝ ẢNH (Bao gồm DICOM)
# ==========================================
def read_dicom_image(file):
    """Đọc file DICOM và chuyển thành ảnh RGB"""
    try:
        dicom_data = pydicom.dcmread(file)
        img = dicom_data.pixel_array.astype(float)
        # Chuẩn hóa về 0-255
        img = (np.maximum(img, 0) / img.max()) * 255.0
        img = np.uint8(img)
        # Chuyển thành 3 kênh màu nếu là ảnh xám
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        return Image.fromarray(img)
    except Exception as e:
        return None

def analyze_tumor_geometry(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(h) / w if w > 0 else 0
    return {'area': area, 'perimeter': perimeter, 'aspect_ratio': aspect_ratio, 'contour': cnt, 'bbox': (x, y, w, h)}

# ==========================================
# 🖥️ GIAO DIỆN CHÍNH
# ==========================================
with st.sidebar:
    st.title("TRUST-MED AI")
    st.info("Phiên bản: **Pro v1.1 (DICOM Support)**")

st.markdown('<div class="main-header">🛡️ Hệ thống Chẩn đoán Siêu âm Kháng Ảo giác</div>', unsafe_allow_html=True)

col_input, col_result = st.columns([1, 2])

with col_input:
    st.subheader("1. Tải ảnh lên")
    uploaded_file = st.file_uploader("Chọn ảnh (PNG, JPG, DCM)", type=['png', 'jpg', 'jpeg', 'dcm'])
    
    img_pil = None
    if uploaded_file:
        # Xử lý file dựa trên đuôi mở rộng
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'dcm':
            img_pil = read_dicom_image(uploaded_file)
            if img_pil is None:
                st.error("Lỗi: Không thể đọc file DICOM này.")
        else:
            img_pil = Image.open(uploaded_file).convert('RGB')
            
        if img_pil:
            st.image(img_pil, caption=f"Ảnh đầu vào: {uploaded_file.name}", use_container_width=True)
            img_np = np.array(img_pil)
            analyze_btn = st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True)

with col_result:
    if uploaded_file and img_pil and analyze_btn:
        with st.spinner("🤖 AI đang phân tích và kiểm tra độ tin cậy..."):
            # 1. Preprocess
            aug_cls = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
            img_tensor = aug_cls(image=img_np)['image'].unsqueeze(0).to(DEVICE)
            
            aug_seg = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
            img_seg_tensor = aug_seg(image=img_np)['image'].unsqueeze(0).to(DEVICE)
            
            # 2. Inference
            with torch.no_grad():
                diag_logits, _, trust_logits, _ = cls_model(img_tensor)
                trust_score = torch.sigmoid(trust_logits).item() # Điểm tin cậy
                diag_prob = torch.sigmoid(diag_logits).item()
                
                mask_logits = seg_model(img_seg_tensor)
                mask_prob = torch.sigmoid(mask_logits).cpu().numpy()[0, 0]
                mask_bin = (mask_prob > 0.5).astype(np.uint8)

        # 3. HIỂN THỊ KẾT QUẢ
        # Ngưỡng tin cậy: Nếu < 0.6 thì cảnh báo (Anti-Hallucination hoạt động tại đây)
        if trust_score < 0.6:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.error("⛔ CẢNH BÁO: DỮ LIỆU BẤT THƯỜNG / KHÔNG HỢP LỆ")
            st.write(f"Độ tin cậy của AI: **{trust_score*100:.1f}%** (Dưới ngưỡng an toàn)")
            st.markdown("""
            **Lý do từ chối chẩn đoán:**
            - Hệ thống phát hiện đặc điểm hình ảnh **KHÔNG PHẢI SIÊU ÂM TUYẾN VÚ**.
            - Có thể đây là ảnh X-Quang (Mammography), MRI hoặc dữ liệu nhiễu.
            - AI từ chối đưa ra kết luận y khoa để tránh sai sót (Ảo giác).
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Nếu tin cậy cao -> Hiện kết quả chẩn đoán
            tab1, tab2 = st.tabs(["🩺 KẾT QUẢ", "📝 BÁO CÁO"])
            with tab1:
                c1, c2, c3 = st.columns(3)
                c1.metric("Độ tin cậy", f"{trust_score*100:.0f}%")
                c2.metric("Xác suất Ác tính", f"{diag_prob*100:.1f}%")
                c3.metric("Gợi ý BI-RADS", "4B/5" if diag_prob > 0.5 else "2/3")
                
                # Hiện ảnh phân đoạn
                mask_real = cv2.resize(mask_bin, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                metrics = analyze_tumor_geometry(mask_real)
                img_vis = img_np.copy()
                if metrics:
                    cv2.drawContours(img_vis, [metrics['contour']], -1, (0, 255, 0), 2)
                st.image(img_vis, caption="Vùng tổn thương (AI Segmented)", use_container_width=True)
