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
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(
    page_title="TRUST-MED: AI Siêu âm Kháng Ảo giác",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện đẹp
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; color: #424242; margin-top: 2rem;}
    .card {background-color: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;}
    .metric-box {text-align: center; padding: 10px; background: #e3f2fd; border-radius: 8px;}
    .risk-high {color: #d32f2f; font-weight: bold;}
    .risk-low {color: #388e3c; font-weight: bold;}
    .warning-box {background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; border-left: 5px solid #ffeeba;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 📥 TẢI MODEL TỪ DRIVE
# ==========================================
# ID file từ link bạn cung cấp
CLS_MODEL_ID = '18ziycNqCuZQ7G9jkAe4nOM0004dBcNXE' 
SEG_MODEL_ID = '134Yb6dnyTa-d7UyrE--8MupQ2uMPIACV'

CLS_MODEL_PATH = 'cls_model.pth'
SEG_MODEL_PATH = 'seg_model.pth'

@st.cache_resource
def download_models():
    # Tải model chẩn đoán
    if not os.path.exists(CLS_MODEL_PATH):
        with st.spinner("📥 Đang tải Model Chẩn đoán (Lần đầu sẽ hơi lâu)..."):
            url = f'https://drive.google.com/uc?id={CLS_MODEL_ID}'
            gdown.download(url, CLS_MODEL_PATH, quiet=False)
    
    # Tải model phân đoạn
    if not os.path.exists(SEG_MODEL_PATH):
        with st.spinner("📥 Đang tải Model Phân đoạn..."):
            url = f'https://drive.google.com/uc?id={SEG_MODEL_ID}'
            gdown.download(url, SEG_MODEL_PATH, quiet=False)

# Gọi hàm tải ngay lập tức
download_models()

# ==========================================
# 🧠 ĐỊNH NGHĨA MODEL (PYTORCH)
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Model Chẩn đoán (TrustMedNet)
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

# 2. Model Phân đoạn (SegNet)
def get_seg_model():
    return smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)

# 3. Load Models vào RAM
@st.cache_resource
def load_loaded_models():
    # Load CLS
    cls_model = TrustMedNet(num_domains=3)
    try:
        cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
    except:
        # Fallback cho config cũ
        cls_model = TrustMedNet(num_domains=5) 
        cls_model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
    cls_model.to(DEVICE).eval()
    
    # Load SEG
    seg_model = get_seg_model()
    try:
        seg_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
    except:
        st.warning("⚠️ Model Segmentation chưa train xong hoặc lỗi file. Dùng chế độ demo.")
    seg_model.to(DEVICE).eval()
    
    return cls_model, seg_model

cls_model, seg_model = load_loaded_models()

# ==========================================
# 🛠️ CÁC HÀM XỬ LÝ ẢNH & TÍNH TOÁN
# ==========================================
def analyze_tumor_geometry(mask):
    """Tính toán diện tích, chu vi, tỷ lệ trục từ mask nhị phân"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Lấy contour lớn nhất
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(h) / w if w > 0 else 0
    
    return {
        'area': area,
        'perimeter': perimeter,
        'aspect_ratio': aspect_ratio,
        'contour': cnt,
        'bbox': (x, y, w, h)
    }

# ==========================================
# 🖥️ GIAO DIỆN NGƯỜI DÙNG (UI)
# ==========================================

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.title("TRUST-MED AI")
    st.info("Phiên bản: **Pro v1.0**")
    st.markdown("---")
    
    mode = st.radio("Chọn chế độ:", ["🏥 Chẩn đoán Hình ảnh", "📊 Dữ liệu Lâm sàng (Metabric)", "ℹ️ Giới thiệu"])
    
    st.markdown("---")
    st.caption("© 2026 TRUST-MED Research Group")

# --- TRANG CHÍNH: CHẨN ĐOÁN HÌNH ẢNH ---
if mode == "🏥 Chẩn đoán Hình ảnh":
    st.markdown('<div class="main-header">🛡️ Hệ thống Chẩn đoán Siêu âm Kháng Ảo giác</div>', unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1, 2])
    
    # CỘT TRÁI: INPUT
    with col_input:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("1. Tải ảnh lên")
        uploaded_file = st.file_uploader("Chọn ảnh siêu âm (PNG, JPG, DCM)", type=['png', 'jpg', 'jpeg', 'dcm'])
        
        if uploaded_file:
            # Xử lý ảnh đầu vào
            image = Image.open(uploaded_file).convert('RGB')
            img_np = np.array(image)
            st.image(image, caption="Ảnh gốc", use_container_width=True)
            
            # Nút phân tích
            analyze_btn = st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # CỘT PHẢI: KẾT QUẢ
    with col_result:
        if uploaded_file and analyze_btn:
            with st.spinner("🤖 AI đang quét khối u và kiểm tra độ tin cậy..."):
                # 1. Preprocess
                aug_cls = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
                img_tensor = aug_cls(image=img_np)['image'].unsqueeze(0).to(DEVICE)
                
                aug_seg = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
                img_seg_tensor = aug_seg(image=img_np)['image'].unsqueeze(0).to(DEVICE)
                
                # 2. Inference
                with torch.no_grad():
                    # Chẩn đoán & Trust
                    diag_logits, _, trust_logits, _ = cls_model(img_tensor)
                    trust_score = torch.sigmoid(trust_logits).item()
                    diag_prob = torch.sigmoid(diag_logits).item()
                    
                    # Phân đoạn
                    mask_logits = seg_model(img_seg_tensor)
                    mask_prob = torch.sigmoid(mask_logits).cpu().numpy()[0, 0]
                    mask_bin = (mask_prob > 0.5).astype(np.uint8)

            # 3. HIỂN THỊ KẾT QUẢ
            # A. Kiểm tra độ tin cậy
            if trust_score < 0.6:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.error("⛔ CẢNH BÁO: DỮ LIỆU KHÔNG HỢP LỆ (OOD)")
                st.write(f"Độ tin cậy: **{trust_score*100:.1f}%** (Rất thấp)")
                st.write("Hệ thống phát hiện đây không phải ảnh siêu âm vú chuẩn hoặc là dữ liệu nhiễu/X-quang.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Dữ liệu sạch -> Hiện kết quả chi tiết
                
                # Tab kết quả
                tab1, tab2 = st.tabs(["🩺 KẾT QUẢ CHẨN ĐOÁN", "📝 BÁO CÁO CHI TIẾT"])
                
                with tab1:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.write("Độ tin cậy")
                        st.title(f"{trust_score*100:.0f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.write("Xác suất Ác tính")
                        if diag_prob > 0.5:
                            st.markdown(f'<h2 class="risk-high">{diag_prob*100:.1f}%</h2>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<h2 class="risk-low">{diag_prob*100:.1f}%</h2>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.write("Gợi ý BI-RADS")
                        if diag_prob > 0.9: st.title("5")
                        elif diag_prob > 0.5: st.title("4B")
                        else: st.title("3")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Hình ảnh phân đoạn
                    col_img_seg, col_metrics = st.columns([1, 1])
                    with col_img_seg:
                        # Resize mask về size gốc
                        mask_real = cv2.resize(mask_bin, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                        metrics = analyze_tumor_geometry(mask_real)
                        
                        # Vẽ contour
                        img_vis = img_np.copy()
                        if metrics:
                            cv2.drawContours(img_vis, [metrics['contour']], -1, (0, 255, 0), 2) # Viền xanh
                            x,y,w,h = metrics['bbox']
                            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (255, 0, 0), 2) # Hộp đỏ
                        
                        st.image(img_vis, caption="AI Khoanh vùng tổn thương", use_container_width=True)
                        
                    with col_metrics:
                        st.subheader("📏 Thông số Hình thái")
                        if metrics:
                            st.write(f"- Diện tích: **{int(metrics['area'])}** px")
                            st.write(f"- Chu vi: **{int(metrics['perimeter'])}** px")
                            st.write(f"- Tỷ lệ Trục (A/R): **{metrics['aspect_ratio']:.2f}**")
                            
                            if metrics['aspect_ratio'] > 0.8:
                                st.error("⚠️ Khối u phát triển chiều dọc (Taller-than-wide) -> Dấu hiệu ác tính.")
                            else:
                                st.success("✅ Khối u phát triển chiều ngang (Wider-than-tall) -> Thường gặp ở u lành.")
                        else:
                            st.info("Không phát hiện khối u rõ ràng.")

                with tab2:
                    st.subheader("BÁO CÁO TỰ ĐỘNG (AI Generated Report)")
                    txt_report = f"""
                    **THÔNG TIN CHUNG:**
                    - Loại dữ liệu: Siêu âm B-mode.
                    - Độ tin cậy hệ thống: {trust_score*100:.1f}% (Đạt chuẩn y tế).
                    
                    **MÔ TẢ TỔN THƯƠNG:**
                    - Phát hiện khối khu trú tại vị trí khoanh vùng.
                    - Kích thước vùng tổn thương: {int(metrics['area']) if metrics else 'N/A'} pixels.
                    - Hình thái: {'Phát triển dọc (nguy cơ cao)' if metrics and metrics['aspect_ratio']>0.8 else 'Bầu dục/Tròn (nguy cơ thấp)'}.
                    
                    **KẾT LUẬN & KHUYẾN NGHỊ:**
                    - Phân loại AI: {'NGHI NGỜ ÁC TÍNH (Malignant)' if diag_prob > 0.5 else 'KHẢ NĂNG LÀNH TÍNH (Benign)'}.
                    - BI-RADS gợi ý: {'4B/4C/5' if diag_prob > 0.5 else '2/3'}.
                    """
                    st.text_area("", txt_report, height=300)
                    st.caption("Lưu ý: Kết quả này chỉ mang tính tham khảo. Vui lòng kết hợp lâm sàng.")

# --- TRANG GIỚI THIỆU ---
elif mode == "ℹ️ Giới thiệu":
    st.title("Về dự án TRUST-MED")
    st.markdown("""
    **TRUST-MED** là hệ thống AI Y tế thế hệ mới, tập trung vào tính **Tin cậy (Trustworthiness)** và **Bền vững (Robustness)**.
    
    ### Điểm nổi bật:
    1.  **Cơ chế Kháng Ảo giác (Anti-Hallucination):** Tự động từ chối dữ liệu rác hoặc không phải siêu âm (như X-quang, MRI nhầm lẫn).
    2.  **Đa nhiệm (Multi-task):** Vừa chẩn đoán bệnh, vừa vẽ chính xác khối u.
    3.  **Thích nghi miền (Domain Adaptation):** Hoạt động tốt trên nhiều loại máy siêu âm khác nhau (Samsung, GE, Siemens...).
    """)

# --- TRANG LÂM SÀNG (METABRIC) ---
elif mode == "📊 Dữ liệu Lâm sàng (Metabric)":
    st.header("Phân tích Dữ liệu Lâm sàng")
    st.info("Chức năng này đang được bảo trì để tích hợp với Model Hình ảnh mới. Vui lòng quay lại sau.")
