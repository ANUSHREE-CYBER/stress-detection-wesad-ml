import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stress Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LOAD MODEL ────────────────────────────────────────────────────────────
MODELS_PATH = r"D:\STRESS DETECTION\models"
PLOTS_PATH  = r"D:\STRESS DETECTION\data\processed\plots"

@st.cache_resource
def load_model():
    model        = joblib.load(os.path.join(MODELS_PATH, "best_model_XGBoost.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_PATH, "feature_names.pkl"))
    return model, feature_names

model, feature_names = load_model()

LABEL_NAMES  = {0: 'Baseline', 1: 'Stress', 2: 'Amusement'}
LABEL_COLORS = {0: '#2196F3', 1: '#F44336', 2: '#4CAF50'}
LABEL_EMOJIS = {0: '😌', 1: '😰', 2: '😄'}

# ── SIDEBAR ───────────────────────────────────────────────────────────────
st.sidebar.title("🧠 Stress Detection")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project:** Physiological Stress Detection  
**Dataset:** WESAD (15 subjects)  
**Signals:** ECG, EDA, Respiration, EMG  
**Model:** XGBoost (97.14% accuracy)  
**Classes:** Baseline · Stress · Amusement
""")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate",
    ["🏠 Overview", "🔍 Predict Stress", "📊 Model Results", "📈 Visualizations"])

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🧠 Physiological Stress Detection System")
    st.markdown("### Using Machine Learning on ECG, EDA, Respiration & EMG Signals")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Dataset",    "WESAD")
    col2.metric("👥 Subjects",   "15")
    col3.metric("🪟 Windows",    "1,049")
    col4.metric("🎯 Best Model", "97.14% F1")

    st.markdown("---")
    st.markdown("## 🔬 Project Pipeline")

    steps = [
        ("1️⃣ Data Loading",        "Loaded 15 subject PKL files, applied 60s sliding windows with 50% overlap → 1,049 samples"),
        ("2️⃣ Signal Filtering",    "Bandpass filtered ECG (0.5–40Hz), segmented EDA, Respiration and EMG signals"),
        ("3️⃣ Feature Extraction",  "Extracted 34 features: 12 HRV, 8 EDA, 8 Respiration, 6 EMG statistics"),
        ("4️⃣ Class Balancing",     "Applied SMOTE oversampling inside each CV fold to handle class imbalance"),
        ("5️⃣ Model Training",      "Trained SVM, XGBoost, MLP with 5-fold stratified cross-validation"),
        ("6️⃣ Deployment",          "Streamlit web app with real-time prediction from uploaded feature CSV"),
    ]

    for title, desc in steps:
        with st.expander(title):
            st.write(desc)

    st.markdown("---")
    st.markdown("## 📊 Dataset Label Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    labels  = ['Baseline', 'Stress', 'Amusement']
    counts  = [570, 313, 166]
    colors  = ['#2196F3', '#F44336', '#4CAF50']
    bars    = ax.bar(labels, counts, color=colors, edgecolor='black')
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 4,
                str(val), ha='center', fontweight='bold')
    ax.set_ylabel("Number of Windows")
    ax.set_title("Class Distribution", fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDICT
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict Stress":
    st.title("🔍 Predict Stress Level")
    st.markdown("Upload a CSV file containing extracted features to get predictions.")
    st.markdown("---")

    st.info(f"**Expected columns ({len(feature_names)}):** " +
            ", ".join(feature_names[:6]) + " ... and more")

    uploaded = st.file_uploader("Upload Features CSV", type=["csv"])

    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)
            st.success(f"File loaded: {df_input.shape[0]} rows, {df_input.shape[1]} columns")

            # Check columns
            missing = [f for f in feature_names if f not in df_input.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                X_input = df_input[feature_names].values
                preds   = model.predict(X_input)

                df_results = df_input[feature_names].copy()
                df_results['Predicted Label']  = preds
                df_results['Predicted Class']  = [LABEL_NAMES[p]  for p in preds]
                df_results['Emoji']            = [LABEL_EMOJIS[p] for p in preds]

                st.markdown("### 🎯 Prediction Results")
                st.dataframe(df_results[['Predicted Class','Emoji']
                             + feature_names[:5]], use_container_width=True)

                # Summary
                st.markdown("### 📊 Prediction Summary")
                col1, col2, col3 = st.columns(3)
                for col, (lbl, name) in zip([col1,col2,col3], LABEL_NAMES.items()):
                    count = int(np.sum(preds == lbl))
                    pct   = count / len(preds) * 100
                    col.metric(f"{LABEL_EMOJIS[lbl]} {name}",
                               f"{count} windows", f"{pct:.1f}%")

                # Pie chart
                fig, ax = plt.subplots(figsize=(5, 5))
                unique, counts = np.unique(preds, return_counts=True)
                ax.pie(counts,
                       labels=[LABEL_NAMES[u] for u in unique],
                       colors=[LABEL_COLORS[u] for u in unique],
                       autopct='%1.1f%%', startangle=90)
                ax.set_title("Prediction Distribution", fontweight='bold')
                st.pyplot(fig)
                plt.close()

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.markdown("### 💡 Don't have a CSV? Use sample data below:")
        if st.button("🎲 Generate Sample Prediction"):
            from sklearn.preprocessing import StandardScaler
            np.random.seed(42)
            sample_data = {}
            for feat in feature_names:
                sample_data[feat] = [np.random.uniform(0.5, 1.5)]
            df_sample = pd.DataFrame(sample_data)
            pred = model.predict(df_sample[feature_names].values)[0]
            st.markdown(f"## Result: {LABEL_EMOJIS[pred]} **{LABEL_NAMES[pred]}**")
            st.markdown(f"The model predicted: **{LABEL_NAMES[pred]}**")
            st.dataframe(df_sample, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Results":
    st.title("📊 Model Performance Results")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    results = {
        'SVM':     {'accuracy': 95.81, 'f1': 95.77},
        'XGBoost': {'accuracy': 97.14, 'f1': 97.13},
        'MLP':     {'accuracy': 93.04, 'f1': 93.10},
    }
    for col, (name, res) in zip([col1,col2,col3], results.items()):
        col.metric(f"{'🏆' if name=='XGBoost' else '🤖'} {name}",
                   f"{res['accuracy']}%", f"F1: {res['f1']}%")

    st.markdown("---")
    st.markdown("### 🔢 Detailed Classification Reports")

    reports = {
        'SVM': {'Baseline':[0.97,0.97,0.97,570],
                'Stress'  :[0.96,0.98,0.97,313],
                'Amusement':[0.92,0.87,0.89,166]},
        'XGBoost':{'Baseline':[0.98,0.98,0.98,570],
                   'Stress'  :[0.98,0.98,0.98,313],
                   'Amusement':[0.94,0.92,0.93,166]},
        'MLP':{'Baseline':[0.96,0.95,0.95,570],
               'Stress'  :[0.95,0.93,0.94,313],
               'Amusement':[0.81,0.86,0.83,166]},
    }

    selected = st.selectbox("Select Model", list(reports.keys()))
    report_df = pd.DataFrame(reports[selected],
                             index=['Precision','Recall','F1','Support']).T
    st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'),
                 use_container_width=True)

    st.markdown("---")
    if os.path.exists(os.path.join(PLOTS_PATH, "06_confusion_matrices.png")):
        st.markdown("### 🔲 Confusion Matrices")
        st.image(os.path.join(PLOTS_PATH, "06_confusion_matrices.png"),
                 use_container_width=True)
    if os.path.exists(os.path.join(PLOTS_PATH, "07_model_comparison.png")):
        st.markdown("### 📊 Model Comparison")
        st.image(os.path.join(PLOTS_PATH, "07_model_comparison.png"),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.title("📈 Data Visualizations")
    st.markdown("---")

    plot_files = {
        "Class Distribution"        : "01_label_distribution.png",
        "Feature Boxplots"          : "02_feature_boxplots.png",
        "Correlation Heatmap"       : "03_correlation_heatmap.png",
        "Violin Plots (HR & RMSSD)" : "04_violin_plots.png",
        "Feature Means per Class"   : "05_feature_means.png",
    }

    selected_plot = st.selectbox("Select Visualization", list(plot_files.keys()))
    plot_path = os.path.join(PLOTS_PATH, plot_files[selected_plot])

    if os.path.exists(plot_path):
        st.image(plot_path, use_container_width=True)
    else:
        st.warning("Plot not found. Run the visualization script first.")