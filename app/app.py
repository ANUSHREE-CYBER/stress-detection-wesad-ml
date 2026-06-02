import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# torch is optional - CNN features disabled on cloud
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── PATHS ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, "models")
PLOTS_PATH  = os.path.join(BASE_DIR, "data", "processed", "plots")
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed")
DEVICE      = torch.device("cpu") if TORCH_AVAILABLE else None

# ── CONSTANTS ─────────────────────────────────────────────────────────────
LABEL_NAMES  = {0: 'Baseline', 1: 'Stress', 2: 'Amusement'}
LABEL_COLORS = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}
LABEL_EMOJIS = {0: '😌', 1: '😰', 2: '😄'}
LABEL_DESC   = {
    0: 'Calm and relaxed state. No significant stress indicators detected.',
    1: 'Elevated stress detected. Physiological markers indicate stress response.',
    2: 'Positive emotional state. Indicators suggest amusement or happiness.'
}

# ── LOAD MODELS ───────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    xgb_model     = None
    feature_names = None
    fusion_config = None
    cnn           = None

    try:
        # Try loading the simple model first (cloud compatible)
        xgb_path = os.path.join(MODELS_PATH, "xgb_model_only.pkl")
        full_path = os.path.join(MODELS_PATH, "best_model_XGBoost.pkl")

        if os.path.exists(xgb_path):
            # Cloud version - no SMOTE pipeline
            xgb_raw       = joblib.load(xgb_path)
            scaler        = joblib.load(os.path.join(MODELS_PATH, "scaler_cloud.pkl"))
            # Wrap in a simple sklearn pipeline (no SMOTE)
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            xgb_model = Pipeline([('scaler', scaler), ('model', xgb_raw)])
        else:
            # Local version - full SMOTE pipeline
            xgb_model = joblib.load(full_path)

        feature_names = joblib.load(os.path.join(MODELS_PATH, "feature_names.pkl"))
        fusion_config = joblib.load(os.path.join(MODELS_PATH, "fusion_config.pkl"))

    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.stop()

    if TORCH_AVAILABLE:
        try:
            cnn = models.mobilenet_v2(weights=None)
            cnn.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(cnn.last_channel, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, 3)
            )
            cnn_path = os.path.join(MODELS_PATH, "best_cnn_model.pth")
            if os.path.exists(cnn_path):
                cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))
                cnn.eval()
        except Exception:
            cnn = None

    return xgb_model, feature_names, fusion_config, cnn

# ── UNPACK MODELS ─────────────────────────────────────────────────────────
xgb_model, feature_names, fusion_config, cnn_model = load_all_models()

# ── CNN INFERENCE FUNCTION ────────────────────────────────────────────────
def predict_face(image_input):
    if not TORCH_AVAILABLE or cnn_model is None:
        return np.array([0.33, 0.34, 0.33])

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if isinstance(image_input, np.ndarray):
        img = torch.tensor(image_input, dtype=torch.float32)
    else:
        img = image_input

    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    elif img.dim() == 3 and img.shape[0] == 1:
        img = img.repeat(3, 1, 1).unsqueeze(0)
    elif img.dim() == 3 and img.shape[0] == 3:
        img = img.unsqueeze(0)

    img = transform(img).to(DEVICE)
    with torch.no_grad():
        output = cnn_model(img)
        probs  = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs

def process_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((48, 48))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array, img

# ── SIDEBAR ───────────────────────────────────────────────────────────────
st.sidebar.title("🧠 Multimodal Stress Detection")
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Models Loaded:**
- ✅ XGBoost (Biosignal) — 97.14%
- {'✅' if TORCH_AVAILABLE else '⚠️'} MobileNetV2 (Face) — 63.30%
- ✅ Fusion Config loaded

**Device:** {'🖥️ GPU/CPU (PyTorch)' if TORCH_AVAILABLE else '💻 CPU only (no PyTorch)'}
""")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🔍 Biosignal Predict",
    "📸 Face Stress Detection",
    "🔀 Fusion Predict",
    "📊 Model Results",
    "📈 Visualizations"
])

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🧠 Multimodal Physiological Stress Detection")
    st.markdown("### Combining Biosignals + Facial Expressions using ML & Deep Learning")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📦 Bio Dataset",  "WESAD")
    col2.metric("😊 Face Dataset", "FER2013")
    col3.metric("👥 Subjects",     "15")
    col4.metric("🪟 Bio Windows",  "1,049")
    col5.metric("🖼️ Face Images", "35,887")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## 🔬 Biosignal Pipeline")
        for title, desc in [
            ("1️⃣ Data Loading",       "15 subjects, 60s sliding windows → 1,049 samples"),
            ("2️⃣ Signal Filtering",   "Bandpass ECG (0.5–40Hz), EDA, EMG, Respiration"),
            ("3️⃣ Feature Extraction", "34 features: HRV, EDA, Resp, EMG statistics"),
            ("4️⃣ Class Balancing",    "SMOTE inside each CV fold"),
            ("5️⃣ Model Training",     "SVM 95.8%, XGBoost 97.1%, MLP 93.0%"),
            ("6️⃣ Clustering",         "K-Means, PCA, t-SNE unsupervised analysis"),
        ]:
            with st.expander(title):
                st.write(desc)

    with col2:
        st.markdown("## 👁️ Vision Pipeline")
        for title, desc in [
            ("1️⃣ FER2013 Dataset",   "28,709 train + 7,178 test face images (48×48)"),
            ("2️⃣ Label Mapping",     "7 emotions → 3 stress classes"),
            ("3️⃣ Preprocessing",     "Grayscale → 3-channel, normalize, augment"),
            ("4️⃣ Transfer Learning", "MobileNetV2 pretrained on ImageNet"),
            ("5️⃣ Fine-tuning",       "Unfroze last 5 blocks, trained 35 epochs total"),
            ("6️⃣ Result",            "63.30% accuracy on held-out test set"),
        ]:
            with st.expander(title):
                st.write(desc)

    st.markdown("---")
    st.markdown("## 🔀 Fusion Architecture")
    st.code("""
ECG / EDA / EMG / Resp signals
        ↓
  34 features extracted
        ↓
   XGBoost Model ──────────────────────┐
   (97.14% accuracy)                   │  weight = 0.6
                                       ├──► Weighted Average ──► Final Stress Score
   Face Image (48×48)                  │  weight = 0.4
        ↓                              │
   MobileNetV2 CNN ────────────────────┘
   (63.30% accuracy)
    """, language="")

    st.markdown("---")
    st.markdown("## 📊 Dataset Distributions")
    col1, col2 = st.columns(2)
    labels = ['Baseline', 'Stress', 'Amusement']
    colors = ['#2196F3', '#F44336', '#4CAF50']

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, [570, 313, 166], color=colors, edgecolor='black')
        for bar, val in zip(bars, [570, 313, 166]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 4, str(val), ha='center', fontweight='bold')
        ax.set_title('WESAD — Biosignal Windows', fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, [9795, 8528, 10386], color=colors, edgecolor='black')
        for bar, val in zip(bars, [9795, 8528, 10386]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 50, str(val), ha='center', fontweight='bold')
        ax.set_title('FER2013 — Face Images (Train)', fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: BIOSIGNAL PREDICT
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Biosignal Predict":
    st.title("🔍 Biosignal Stress Prediction")
    st.markdown("Upload a CSV of extracted ECG/EDA/EMG/Respiration features.")
    st.markdown("---")

    if feature_names is not None:
        st.info(f"**Required columns ({len(feature_names)}):** " +
                ", ".join(feature_names[:6]) + " ... and 28 more")

    uploaded = st.file_uploader("Upload Features CSV", type=["csv"])

    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)
            st.success(f"Loaded: {df_input.shape[0]} rows")
            missing = [f for f in feature_names if f not in df_input.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                X_input = df_input[feature_names].values
                preds   = xgb_model.predict(X_input)
                probs   = xgb_model.predict_proba(X_input)

                st.markdown("### 🎯 Results")
                for i, (pred, prob) in enumerate(zip(preds, probs)):
                    col1, col2, col3 = st.columns([1, 2, 3])
                    col1.markdown(f"**Row {i+1}**")
                    col2.markdown(f"{LABEL_EMOJIS[pred]} **{LABEL_NAMES[pred]}**")
                    col3.progress(float(prob[pred]))

                st.markdown("### 📊 Summary")
                col1, col2, col3 = st.columns(3)
                for col, (lbl, name) in zip([col1, col2, col3], LABEL_NAMES.items()):
                    count = int(np.sum(preds == lbl))
                    col.metric(f"{LABEL_EMOJIS[lbl]} {name}", f"{count} windows")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.markdown("### 💡 No CSV? Use real sample data:")
        if st.button("🎲 Generate Sample from WESAD Data"):
            try:
                df_real  = pd.read_csv(os.path.join(DATA_PATH, "features.csv"))
                sample   = df_real.sample(1, random_state=np.random.randint(100))
                true_lbl = int(sample['label'].values[0])
                X_s      = sample[feature_names].values
                pred     = xgb_model.predict(X_s)[0]
                prob     = xgb_model.predict_proba(X_s)[0]

                st.markdown(f"## {LABEL_EMOJIS[pred]} Predicted: **{LABEL_NAMES[pred]}**")
                if true_lbl == pred:
                    st.success(f"✅ Correct! True label was also {LABEL_NAMES[true_lbl]}")
                else:
                    st.warning(f"True label: {LABEL_NAMES[true_lbl]}")

                st.markdown(f"*{LABEL_DESC[pred]}*")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(list(LABEL_NAMES.values()), prob,
                        color=[LABEL_COLORS[i] for i in range(3)])
                ax.set_xlabel('Probability')
                ax.set_title('Prediction Confidence')
                ax.set_xlim(0, 1)
                for i, v in enumerate(prob):
                    ax.text(v + 0.01, i, f'{v:.3f}', va='center')
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: FACE STRESS DETECTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "📸 Face Stress Detection":
    st.title("📸 Face-Based Stress Detection")
    st.markdown("Upload a face image — the CNN will predict your stress state.")
    st.markdown("---")

    if not TORCH_AVAILABLE:
        st.warning("CNN model not available on this deployment. Biosignal prediction and visualizations work fully. For CNN features, run the app locally.")

    st.info("""
    **How it works:**
    - Upload any face image (JPG, PNG)
    - It gets converted to 48×48 grayscale
    - MobileNetV2 CNN predicts stress class
    - Model trained on 28,709 FER2013 facial expression images
    """)

    uploaded_img = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        try:
            img_array, pil_img = process_uploaded_image(uploaded_img)
            probs = predict_face(img_array)
            pred  = int(np.argmax(probs))

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(pil_img.resize((200, 200)),
                         caption="Input (48×48 grayscale)", width=200)
            with col2:
                st.markdown(f"## {LABEL_EMOJIS[pred]} **{LABEL_NAMES[pred]}**")
                st.markdown(f"*{LABEL_DESC[pred]}*")
                st.markdown(f"**Confidence: {probs[pred]*100:.1f}%**")

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(list(LABEL_NAMES.values()), probs,
                        color=[LABEL_COLORS[i] for i in range(3)], edgecolor='black')
                ax.set_xlabel('Probability')
                ax.set_title('CNN Prediction Confidence', fontweight='bold')
                ax.set_xlim(0, 1)
                for i, v in enumerate(probs):
                    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown("---")
            st.markdown("### 📊 Emotion → Stress Mapping Used")
            mapping_df = pd.DataFrame({
                'Emotion' : ['angry','disgust','fear','happy','surprise','neutral','sad'],
                'Maps To' : ['Stress','Stress','Stress','Amusement','Amusement','Baseline','Baseline'],
                'Reason'  : ['Negative arousal','Negative arousal','High arousal stress',
                             'Positive affect','Positive arousal','Calm neutral','Low mood baseline']
            })
            st.dataframe(mapping_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.markdown("### 💡 Try with a sample face from FER2013:")
        if st.button("🎲 Load Random Test Face"):
            try:
                face_X_path = os.path.join(DATA_PATH, "face_X_test.npy")
                face_y_path = os.path.join(DATA_PATH, "face_y_test.npy")
                if not os.path.exists(face_X_path):
                    st.warning("Face test data not available on this deployment. Please upload an image instead.")
                else:
                    X_test    = np.load(face_X_path)
                    y_test    = np.load(face_y_path)
                    idx       = np.random.randint(len(X_test))
                    img_array = X_test[idx]
                    true_lbl  = int(y_test[idx])
                    probs     = predict_face(img_array)
                    pred      = int(np.argmax(probs))

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        pil_img = Image.fromarray((img_array * 255).astype(np.uint8))
                        st.image(pil_img.resize((200, 200)),
                                 caption=f"True: {LABEL_NAMES[true_lbl]}", width=200)
                    with col2:
                        st.markdown(f"## {LABEL_EMOJIS[pred]} **{LABEL_NAMES[pred]}**")
                        if pred == true_lbl:
                            st.success("✅ Correct prediction!")
                        else:
                            st.warning(f"True label: {LABEL_NAMES[true_lbl]}")
                        st.markdown(f"Confidence: **{probs[pred]*100:.1f}%**")
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.barh(list(LABEL_NAMES.values()), probs,
                                color=[LABEL_COLORS[i] for i in range(3)], edgecolor='black')
                        ax.set_xlabel('Probability')
                        ax.set_title('CNN Prediction', fontweight='bold')
                        ax.set_xlim(0, 1)
                        for i, v in enumerate(probs):
                            ax.text(v+0.01, i, f'{v:.3f}', va='center')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: FUSION PREDICT
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔀 Fusion Predict":
    st.title("🔀 Multimodal Fusion Prediction")
    st.markdown("Combine biosignal features + face image for the most accurate prediction.")
    st.markdown("---")

    if not TORCH_AVAILABLE:
        st.warning("CNN model not available on this deployment. Biosignal prediction and visualizations work fully. For CNN features, run the app locally.")

    xgb_w = fusion_config['xgb_weight'] if fusion_config else 0.6
    cnn_w = fusion_config['cnn_weight'] if fusion_config else 0.4
    st.info(f"**Fusion weights:** XGBoost × {xgb_w} + CNN × {cnn_w}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Step 1 — Upload Biosignal CSV")
        bio_file = st.file_uploader("Biosignal Features CSV", type=["csv"], key="fusion_bio")
    with col2:
        st.markdown("### 📸 Step 2 — Upload Face Image")
        face_file = st.file_uploader("Face Image", type=["jpg","jpeg","png"], key="fusion_face")

    st.markdown("---")
    use_sample = st.button("🎲 Use Sample Data (no upload needed)")

    if use_sample or (bio_file and face_file):
        try:
            if use_sample:
                df_real   = pd.read_csv(os.path.join(DATA_PATH, "features.csv"))
                sample    = df_real.sample(1, random_state=42)
                X_s       = sample[feature_names].values
                bio_probs = xgb_model.predict_proba(X_s)[0]
                bio_pred  = int(np.argmax(bio_probs))

                face_X_path = os.path.join(DATA_PATH, "face_X_test.npy")
                if os.path.exists(face_X_path):
                    X_test     = np.load(face_X_path)
                    img_array  = X_test[42]
                    face_probs = predict_face(img_array)
                else:
                    face_probs = np.array([0.33, 0.34, 0.33])
                    st.info("Face test data not on server — using neutral probabilities for demo.")
                face_pred = int(np.argmax(face_probs))
                st.success("Sample data loaded!")
            else:
                df_input  = pd.read_csv(bio_file)
                X_s       = df_input[feature_names].values[:1]
                bio_probs = xgb_model.predict_proba(X_s)[0]
                bio_pred  = int(np.argmax(bio_probs))
                img_array, _ = process_uploaded_image(face_file)
                face_probs   = predict_face(img_array)
                face_pred    = int(np.argmax(face_probs))

            fused_probs = xgb_w * bio_probs + cnn_w * face_probs
            final_pred  = int(np.argmax(fused_probs))

            st.markdown("## 🎯 Fusion Result")
            col1, col2, col3 = st.columns(3)
            for col, title, pred, probs, model_name in [
                (col1, "📊 Biosignal", bio_pred,   bio_probs,   "XGBoost"),
                (col2, "📸 Face",      face_pred,  face_probs,  "MobileNetV2"),
                (col3, "🔀 Fusion",    final_pred, fused_probs, "Fusion"),
            ]:
                with col:
                    st.markdown(f"### {title}")
                    st.markdown(f"**{LABEL_EMOJIS[pred]} {LABEL_NAMES[pred]}**")
                    st.caption(f"{model_name}: {probs[pred]*100:.1f}%")
                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    ax.barh(list(LABEL_NAMES.values()), probs,
                            color=[LABEL_COLORS[i] for i in range(3)])
                    ax.set_xlim(0, 1)
                    ax.set_title(model_name, fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            st.markdown("---")
            st.markdown(f"### Final Verdict: {LABEL_EMOJIS[final_pred]} **{LABEL_NAMES[final_pred]}**")
            st.markdown(f"*{LABEL_DESC[final_pred]}*")
            st.markdown(f"### 🎚️ Stress Score: **{fused_probs[1]*100:.1f}/100**")
            st.progress(float(fused_probs[1]))

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════
# PAGE 5: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Results":
    st.title("📊 Model Performance Results")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🥇 XGBoost",     "97.14%", "Biosignal")
    col2.metric("🥈 SVM",         "95.81%", "Biosignal")
    col3.metric("🥉 MLP",         "93.04%", "Biosignal")
    col4.metric("🤖 MobileNetV2", "63.30%", "Face CNN")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 Biosignal Models", "🖼️ CNN Model", "🔀 Fusion"])

    with tab1:
        reports = {
            'SVM':     {'Baseline':[0.97,0.97,0.97,570],'Stress':[0.96,0.98,0.97,313],'Amusement':[0.92,0.87,0.89,166]},
            'XGBoost': {'Baseline':[0.98,0.98,0.98,570],'Stress':[0.98,0.98,0.98,313],'Amusement':[0.94,0.92,0.93,166]},
            'MLP':     {'Baseline':[0.96,0.95,0.95,570],'Stress':[0.95,0.93,0.94,313],'Amusement':[0.81,0.86,0.83,166]},
        }
        selected  = st.selectbox("Select Model", list(reports.keys()))
        report_df = pd.DataFrame(reports[selected],
                                  index=['Precision','Recall','F1','Support']).T
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'),
                     use_container_width=True)
        for title, fname, width in [
            ("### Confusion Matrices", "06_confusion_matrices.png", 800),
            ("### Model Comparison",   "07_model_comparison.png",   700),
        ]:
            p = os.path.join(PLOTS_PATH, fname)
            if os.path.exists(p):
                st.markdown(title)
                st.image(p, width=width)

    with tab2:
        col1, col2, col3 = st.columns(3)
        col1.metric("Initial Accuracy",  "56.07%", "20 epochs")
        col2.metric("After Fine-tuning", "63.30%", "+7.23%")
        col3.metric("Trainable Params",  "77.3%",  "1.8M / 2.4M")
        cnn_report = pd.DataFrame({
            'Baseline'  :[0.60,0.60,0.60,2480],
            'Stress'    :[0.57,0.53,0.55,2093],
            'Amusement' :[0.70,0.74,0.71,2605],
        }, index=['Precision','Recall','F1','Support']).T
        st.dataframe(cnn_report, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            p = os.path.join(PLOTS_PATH, "13b_cnn_finetuning_curves.png")
            if os.path.exists(p):
                st.image(p, caption="Fine-tuning Curves")
        with col2:
            p = os.path.join(PLOTS_PATH, "14b_cnn_finetuned_confusion.png")
            if os.path.exists(p):
                st.image(p, caption="CNN Confusion Matrix")

    with tab3:
        col1, col2, col3 = st.columns(3)
        col1.metric("XGBoost Weight", f"{fusion_config['xgb_weight']}" if fusion_config else "0.6")
        col2.metric("CNN Weight",     f"{fusion_config['cnn_weight']}"  if fusion_config else "0.4")
        col3.metric("Architecture",   "Weighted Average")
        for fname, caption in [
            ("15_fusion_weights.png",        "Fusion Weight Analysis"),
            ("16_final_model_comparison.png", "Final Model Comparison"),
        ]:
            p = os.path.join(PLOTS_PATH, fname)
            if os.path.exists(p):
                st.image(p, caption=caption, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 6: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.title("📈 Data Visualizations")
    st.markdown("---")

    plot_files = {
        "01 — Class Distribution (Biosignal)"   : "01_label_distribution.png",
        "02 — Feature Boxplots"                 : "02_feature_boxplots.png",
        "03 — Correlation Heatmap"              : "03_correlation_heatmap.png",
        "04 — Violin Plots (HR & RMSSD)"        : "04_violin_plots.png",
        "05 — Feature Means per Class"          : "05_feature_means.png",
        "06 — Confusion Matrices (3 Models)"    : "06_confusion_matrices.png",
        "07 — Model Comparison (Acc & F1)"      : "07_model_comparison.png",
        "08 — K-Means Elbow Method"             : "08_elbow_method.png",
        "09 — PCA Projection"                   : "09_pca_visualization.png",
        "10 — t-SNE Projection"                 : "10_tsne_visualization.png",
        "11 — Sample Face Images per Class"     : "11_sample_faces.png",
        "12 — Face Dataset Distribution"        : "12_face_distribution.png",
        "13 — CNN Initial Training Curves"      : "13_cnn_training_curves.png",
        "13b — CNN Fine-tuning Curves"          : "13b_cnn_finetuning_curves.png",
        "14 — CNN Initial Confusion Matrix"     : "14_cnn_confusion_matrix.png",
        "14b — CNN Fine-tuned Confusion Matrix" : "14b_cnn_finetuned_confusion.png",
        "15 — Fusion Weight Analysis"           : "15_fusion_weights.png",
        "16 — Final Model Comparison"           : "16_final_model_comparison.png",
    }

    selected_plot = st.selectbox("Select Visualization (18 total)",
                                  list(plot_files.keys()))
    plot_path = os.path.join(PLOTS_PATH, plot_files[selected_plot])
    if os.path.exists(plot_path):
        st.image(plot_path, use_container_width=True)
    else:
        st.warning(f"Plot not found: {plot_files[selected_plot]}")

    st.markdown("---")
    st.markdown("### 🖼️ All Visualizations Gallery")
    cols = st.columns(3)
    for i, (name, fname) in enumerate(plot_files.items()):
        path = os.path.join(PLOTS_PATH, fname)
        if os.path.exists(path):
            with cols[i % 3]:
                st.image(path, caption=name, use_container_width=True)