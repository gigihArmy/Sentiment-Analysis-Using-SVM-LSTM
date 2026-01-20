import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ================= SESSION STATE =================
if "concat_data" not in st.session_state:
    st.session_state.concat_data = None

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

# ================= HEADER =================
st.title("Text Preprocessing")
st.markdown(
    "Halaman ini digunakan untuk melakukan **preprocessing teks** pada dataset "
    "hasil crawling Instagram dan Google Playstore sebelum dilakukan analisis sentimen."
)

st.markdown("---")

# ================= UPLOAD FILE =================
st.subheader("Gabung Dataset Crawling")

st.markdown(
    """
    Jika datasetmu sudah tergabung, lewati tahapan ini.
    """
)

uploaded_files = st.file_uploader(
    label="Upload file CSV (Instagram & Playstore)",
    type=["csv"],
    accept_multiple_files=True
)

# ================= HELPER FUNCTIONS =================
def normalize_instagram(df):
    """
    Instagram:
    timestamp -> tanggal
    text -> text
    """
    df = df.rename(columns={
        "timestamp": "tanggal",
        "text": "text"
    })

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df[["tanggal", "text", "dataset"]]


def normalize_playstore(df):
    """
    Playstore:
    at -> tanggal
    content -> text
    """
    df = df.rename(columns={
        "at": "tanggal",
        "content": "text"
    })

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df[["tanggal", "text", "dataset"]]

# ================= PROCESS =================
if uploaded_files:
    normalized_dfs = []

    for file in uploaded_files:
        df = pd.read_csv(file)

        # Validasi kolom dataset
        if "dataset" not in df.columns:
            st.error(f"File `{file.name}` tidak memiliki kolom `dataset`.")
            continue

        # Deteksi tipe dataset
        if "timestamp" in df.columns and "text" in df.columns:
            df_norm = normalize_instagram(df)
            normalized_dfs.append(df_norm)

        elif "at" in df.columns and "content" in df.columns:
            df_norm = normalize_playstore(df)
            normalized_dfs.append(df_norm)

        else:
            st.error(
                f"Struktur kolom file `{file.name}` tidak dikenali.\n\n"
                "Pastikan file berasal dari Instagram atau Google Playstore."
            )

    if normalized_dfs:
        final_df = pd.concat(normalized_dfs, ignore_index=True)
        st.session_state.concat_data = final_df

        st.success(f"Berhasil Digabung Total data: {len(final_df)}")

if st.session_state.concat_data is not None:
    # ================= DOWNLOAD =================
    csv = st.session_state.concat_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Unduh Dataset",
        data=csv,
        file_name="dataset_dana_concat.csv",
        mime="text/csv"
    )

else:
    st.info("Upload File Dataset Jika ingin melakukan penggabungan dataset.")


# ================= PREPROCESSING FUNCTIONS =================
def case_folding(text):
    return text.lower() if isinstance(text, str) else text

def remove_username(text):
    return re.sub(r'@[^\s]+', '', text)

def remove_hashtag(text):
    return re.sub(r'#[^\s]+', '', text)

def remove_html(text):
    return re.sub(r'<.*?>', ' ', text) if isinstance(text, str) else text

def remove_numbers(text):
    return re.sub(r'\d+', '', text) if isinstance(text, str) else text

def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text) if isinstance(text, str) else text

def remove_symbols(text):
    return re.sub(r'[^a-zA-Z\s]', ' ', text) if isinstance(text, str) else text

def cleaning_pipeline(text):
    text = remove_username(text)
    text = remove_hashtag(text)
    text = remove_html(text)
    text = remove_url(text)
    text = remove_numbers(text)
    text = remove_symbols(text)
    return text.strip()

def tokenisasi(text):
    return word_tokenize(text)

# ===== Normalisasi =====
norm = pd.read_excel("kamus/kamuskatabaku_dana.xlsx")
kamus_norm = dict(zip(norm.iloc[:, 0], norm.iloc[:, 1]))

def normalisasi(tokens):
    return [kamus_norm[token] if token in kamus_norm else token for token in tokens]

# ===== Stopword =====
factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())

def stopword_removal(tokens):
    return [token for token in tokens if token not in stopwords]

# ===== Stemming =====
stemmer = StemmerFactory().create_stemmer()

def stemming(tokens):
    return [stemmer.stem(token) for token in tokens]

# ================= PROSES PREPROCESSING =================
st.markdown("---")
st.subheader("Proses Preprocessing")

# ================= UPLOAD DATASET =================
uploaded_file = st.file_uploader(
    "Upload file CSV hasil crawling & preprocessing awal",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("Dataset harus memiliki kolom `text`.")
    else:
        st.session_state.raw_df = df
        st.success(f"Dataset berhasil dimuat! Total data: {len(df)}")

# ================= TAMPILKAN DATA RAW =================
if st.session_state.raw_df is not None:
    st.subheader("Data Awal")
    st.dataframe(st.session_state.raw_df, use_container_width=True)

if st.button("Jalankan Preprocessing"):
    if st.session_state.raw_df is None:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        with st.spinner("Sedang melakukan preprocessing teks..."):
            df = st.session_state.raw_df.copy()

            df["casefolding"] = df["text"].apply(case_folding)
            df["cleaning"] = df["casefolding"].apply(cleaning_pipeline)
            df["tokenisasi"] = df["cleaning"].apply(tokenisasi)
            df["normalisasi"] = df["tokenisasi"].apply(normalisasi)
            df["stopwords"] = df["normalisasi"].apply(stopword_removal)
            df["stemming"] = df["stopwords"].apply(stemming)
            df["text_clean"] = df["stemming"].apply(lambda x: " ".join(x))

            st.session_state.processed_df = df

        st.success("✅ Preprocessing selesai!")

# ================= HASIL PREPROCESSING =================
if st.session_state.processed_df is not None:
    st.markdown("---")
    st.subheader("Hasil Preprocessing")

    st.dataframe(
        st.session_state.processed_df[
            [
                "text",
                "casefolding",
                "cleaning",
                "tokenisasi",
                "normalisasi",
                "stopwords",
                "stemming",
                "text_clean",
            ]
        ],
        use_container_width=True
    )

    # ================= DOWNLOAD =================
    csv = st.session_state.processed_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Unduh Hasil Preprocessing",
        data=csv,
        file_name="dataset_dana_text_preprocessing.csv",
        mime="text/csv"
    )