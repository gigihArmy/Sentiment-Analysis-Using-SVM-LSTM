import streamlit as st

# ================= HEADER =================
st.title("Sistem Analisis Sentimen E-Wallet DANA")
st.markdown(
    "Aplikasi interaktif untuk menganalisis sentimen pengguna terhadap **E-Wallet DANA** "
    "berdasarkan ulasan dan komentar dari berbagai platform digital."
)

st.markdown("---")

# ================= DESKRIPSI =================
st.subheader("Deskripsi Sistem")
st.write(
    """
    Sistem ini dirancang untuk melakukan **analisis sentimen** terhadap opini pengguna 
    E-Wallet **DANA** yang dikumpulkan dari **Instagram** dan **Google Play Store**.  
    Dengan memanfaatkan pendekatan **Machine Learning** dan **Deep Learning**, sistem ini 
    mampu mengklasifikasikan sentimen pengguna ke dalam tiga kelas utama:
    **Positif**, **Netral**, dan **Negatif**.
    """
)

st.success(
    "Fokus utama sistem adalah memahami persepsi pengguna terhadap layanan, fitur, "
    "dan pengalaman penggunaan E-Wallet DANA."
)

st.markdown("---")

# ================= METRIC INTERAKTIF =================
st.subheader("Ringkasan Sistem")

col1, col2, col3 = st.columns(3)
col1.metric("Sumber Data", "2 Platform")
col2.metric("Metode Klasifikasi", "SVM & LSTM")
col3.metric("Kelas Sentimen", "3 Label")

st.markdown("---")

# ================= TAHAPAN SISTEM =================
st.header("Tahapan Sistem Analisis Sentimen")

st.markdown("Berikut adalah alur utama dari sistem analisis sentimen yang digunakan:")

# ---------- Crawling ----------
with st.expander("1️⃣ Pengumpulan Data (Crawling)"):
    st.markdown("""
    - Data dikumpulkan dari:
      - **Instagram** (komentar pada akun atau konten terkait DANA)
      - **Google Play Store** (ulasan aplikasi E-Wallet DANA)
    - Pengambilan data dilakukan berdasarkan kata kunci yang relevan.
    - Data disimpan dalam format **CSV** untuk proses lanjutan.
    """)

# ---------- Preprocessing ----------
with st.expander("2️⃣ Preprocessing Data Teks"):
    st.markdown("""
    Tahapan preprocessing bertujuan untuk membersihkan dan menyiapkan data teks:
    - **Case folding** (mengubah teks menjadi huruf kecil).
    - Menghapus URL, mention, hashtag, emoji, angka, dan simbol.
    - **Tokenisasi**, **stemming**, dan **normalisasi kata**.
    - Menghilangkan stopwords untuk meningkatkan kualitas fitur.
    """)

# ---------- Labeling ----------
with st.expander("3️⃣ Pelabelan Sentimen Otomatis"):
    st.markdown("""
    - Pelabelan awal dilakukan menggunakan pendekatan **lexicon-based**.
    - Sentimen dibagi menjadi tiga kelas:
      - **Positif**
      - **Netral**
      - **Negatif**
    - Label ini digunakan sebagai data latih untuk model klasifikasi.
    """)

# ---------- Model Selection ----------
with st.expander("4️⃣ Klasifikasi Menggunakan SVM & LSTM"):
    model = st.selectbox(
        "Pilih model klasifikasi yang digunakan:",
        ["Support Vector Machine (SVM)", "Long Short-Term Memory (LSTM)"]
    )

    if model == "Support Vector Machine (SVM)":
        st.markdown("""
        **SVM (Support Vector Machine)** digunakan sebagai model machine learning klasik:
        - Ekstraksi fitur menggunakan **TF-IDF**.
        - Pembagian data: **80% data latih** dan **20% data uji**.
        - Menggunakan kernel **Polynomial**, **RBF**, dan **Sigmoid**.
        - Evaluasi model:
          - Akurasi
          - Precision, Recall, F1-Score
          - Confusion Matrix
        """)

    else:
        st.markdown("""
        **LSTM (Long Short-Term Memory)** digunakan sebagai model deep learning:
        - Representasi teks menggunakan **Tokenization & Padding**.
        - Arsitektur LSTM mampu menangkap konteks urutan kata.
        - Cocok untuk analisis sentimen berbasis sekuens teks.
        - Evaluasi model menggunakan metrik klasifikasi yang sama dengan SVM.
        """)

# ---------- Evaluasi ----------
with st.expander("5️⃣ Evaluasi & Visualisasi"):
    st.markdown("""
    - Menampilkan hasil evaluasi model dalam bentuk:
      - Confusion Matrix
      - Classification Report
      - Visualisasi distribusi sentimen
    - Membandingkan performa **SVM vs LSTM** untuk menentukan model terbaik.
    """)

st.markdown("---")

# ================= INFO TAMBAHAN =================
st.info(
    "Gunakan **sidebar di sebelah kiri** untuk mengakses fitur lain seperti "
    "upload dataset, preprocessing, pelatihan model, dan visualisasi hasil."
)
