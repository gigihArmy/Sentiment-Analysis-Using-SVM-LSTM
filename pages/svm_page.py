import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ================= SESSION STATE =================
if "model_results" not in st.session_state:
    st.session_state.model_results = None

if "train_data" not in st.session_state:
    st.session_state.train_data = None

# ================= HEADER =================
st.title("Model Machine Learning SVM")
st.markdown(
    """
    Halaman ini digunakan untuk membangun dan mengevaluasi model **Support Vector Machine (SVM)** 
    pada data hasil **pelabelan sentimen e-wallet DANA** menggunakan pendekatan machine learning.
    """
)

st.markdown("---")

# ================= UPLOAD DATASET =================
st.subheader("Upload Dataset Hasil Pelabelan")
uploaded_file = st.file_uploader(
    "Upload file CSV hasil pelabelan",
    type=["csv"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        required_cols = {"clean", "label"}
        if not required_cols.issubset(df.columns):
            st.error(f"Dataset wajib memiliki kolom: {required_cols}")
        else:
            st.session_state.train_data = df
            st.success(f"Dataset berhasil dimuat! Total data: {len(df)}")
            
            st.write("**Distribusi Label Saat Ini:**")
            st.write(df['label'].value_counts())
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

# ================= MODEL TRAINING & EVALUATION =================
if st.session_state.train_data is not None:
    
    if st.button("Mulai Prediksi & Evaluasi"):
        df = st.session_state.train_data
        
        # Filter: Pastikan setiap kelas minimal punya 2 data agar bisa di-split & SMOTE
        class_counts = df['label'].value_counts()
        min_samples = class_counts.min()
        
        if min_samples < 2:
            st.error(f"Error: Ada kelas yang hanya memiliki {min_samples} sampel. "
                     "SMOTE memerlukan minimal 2 sampel per kelas. "
                     "Silakan tambah data atau hapus kelas yang sangat sedikit tersebut.")
        else:
            with st.spinner("Sedang memproses..."):
                X = df['clean'].astype(str)
                y = df['label']

                # Split Data 
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # TF-IDF
                tfidf = TfidfVectorizer()
                X_train_tfidf = tfidf.fit_transform(X_train_raw)
                X_test_tfidf = tfidf.transform(X_test_raw)

                # Hitung k_neighbors optimal untuk SMOTE
                # k_neighbors tidak boleh lebih besar dari (jumlah sampel kelas terkecil - 1)
                train_min_samples = y_train.value_counts().min()
                k_neigh = min(5, train_min_samples - 1)
                
                if k_neigh < 1:
                    k_neigh = 1

                kernels = [
                    {'name': 'Polynomial', 'kernel': 'poly', 'degree': 3},
                    {'name': 'RBF', 'kernel': 'rbf', 'degree': None},
                    {'name': 'Sigmoid', 'kernel': 'sigmoid', 'degree': None}
                ]

                results = {}
                for k in kernels:
                    # Pipeline dengan penyesuaian k_neighbors
                    pipeline = ImbPipeline([
                        ('smote', SMOTE(random_state=42, k_neighbors=k_neigh)),
                        ('svm', SVC(kernel=k['kernel'], degree=k['degree'] if k['degree'] else 3))
                    ])

                    pipeline.fit(X_train_tfidf, y_train)
                    y_pred = pipeline.predict(X_test_tfidf)

                    results[k['name']] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average="macro"),
                        'recall': recall_score(y_test, y_pred, average="macro"),
                        'report': classification_report(y_test, y_pred),
                        'cm': confusion_matrix(y_test, y_pred),
                        'labels': sorted(y.unique().tolist())
                    }

                st.session_state.model_results = results
                st.success(f"✅ Training selesai! (SMOTE k_neighbors disesuaikan ke: {k_neigh})")

# ================= DISPLAY RESULTS =================
if st.session_state.model_results is not None:
    st.markdown("---")
    st.subheader("Hasil Evaluasi Model")

    tab1, tab2, tab3 = st.tabs(["SVM Polynomial", "SVM RBF", "SVM Sigmoid"])
    
    tabs_config = [(tab1, "Polynomial"), (tab2, "RBF"), (tab3, "Sigmoid")]

    for tab, kernel_name in tabs_config:
        res = st.session_state.model_results[kernel_name]
        with tab:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Akurasi", f"{res['accuracy']*100:.2f}%")
            col_m2.metric("Presisi", f"{res['precision']*100:.2f}%")
            col_m3.metric("Recall", f"{res['recall']*100:.2f}%")

            st.write("**Classification Report:**")
            st.code(res['report'])

            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=res['labels'], yticklabels=res['labels'], annot_kws={'size': 7})
            plt.xlabel('Predicted')
            plt.ylabel('True')
            st.pyplot(fig, use_container_width=False)