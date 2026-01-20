import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ================= SESSION STATE =================
if "labeling_df" not in st.session_state:
    st.session_state.labeling_df = None

if "raw_label_df" not in st.session_state:
    st.session_state.raw_label_df = None

# ================= HEADER =================
st.title("Pelabelan Lexicon")
st.markdown(
    "Halaman ini digunakan untuk melakukan **pelabelan sentimen otomatis** "
    "menggunakan pendekatan **lexicon-based** pada dataset E-Wallet **DANA**."
)

st.markdown("---")

# ================= UPLOAD DATASET =================
st.subheader("Upload Dataset Preprocessing")

uploaded_file = st.file_uploader(
    "Upload file CSV hasil preprocessing",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = {"tanggal", "text_clean", "dataset"}
    if not required_cols.issubset(df.columns):
        st.error(
            "Dataset tidak sesuai.\n\n"
            "Kolom wajib: `tanggal`, `text_clean`, `dataset`"
        )
    else:
        st.session_state.raw_label_df = df
        st.success(f"Dataset berhasil dimuat! Total data: {len(df)}")

# ================= FUNGSI LEXICON =================
@st.cache_data
def load_lexicon():
    return pd.read_csv("kamus/lexicon_based_dana.csv")

lexicon = load_lexicon()

def lexicon_label(text, lexicon):
    label = "Neutral"
    sentiment_score = 0.0

    for word in text.split():
        if word in lexicon["word"].values:
            weight = lexicon.loc[lexicon["word"] == word, "weight"].values[0]
            sentiment_score += weight

    if sentiment_score > 0:
        label = "Positive"
    elif sentiment_score < 0:
        label = "Negative"

    return label, sentiment_score

# ================= PROSES PELABELAN =================

if st.button("Proses Pelabelan"):
    if st.session_state.raw_label_df is None:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        with st.spinner("Sedang melakukan pelabelan sentimen..."):
            df = st.session_state.raw_label_df.copy()

            labels = []
            scores = []

            for text in df["text_clean"]:
                label, score = lexicon_label(str(text), lexicon)
                labels.append(label)
                scores.append(score)

            df["label"] = labels
            df["sentiment_score"] = scores

            # Ambil kolom sesuai permintaan
            df = df[
                ["tanggal", "text_clean", "label", "sentiment_score", "dataset"]
            ].rename(columns={"text_clean": "clean"})

            st.session_state.labeling_df = df

        st.success("✅ Pelabelan selesai!")

# ================= HASIL PELABELAN =================
if st.session_state.labeling_df is not None:
    st.markdown("---")
    st.subheader("Hasil Pelabelan Sentimen")

    st.dataframe(
        st.session_state.labeling_df,
        use_container_width=True
    )

    # ================= DISTRIBUSI LABEL =================
    st.markdown("---")
    st.subheader("Distribusi Sentimen")

    label_counts = st.session_state.labeling_df["label"].value_counts()

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(
        label_counts.index,
        label_counts.values,
        width=0.5
    )

    ax.set_title("Distribusi Sentimen", fontsize=11)
    ax.set_xlabel("Sentimen", fontsize=9)
    ax.set_ylabel("Jumlah", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    for i, v in enumerate(label_counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=8)

    st.pyplot(fig, use_container_width=False)


    # ================= DISTRIBUSI PER DATASET =================
    st.markdown("---")
    st.subheader("Distribusi Sentimen per Dataset")

    col1, col2 = st.columns(2)

    for i, source in enumerate(st.session_state.labeling_df["dataset"].unique()):
        source_df = st.session_state.labeling_df[
            st.session_state.labeling_df["dataset"] == source
        ]
        pie_counts = source_df["label"].value_counts()

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.pie(
            pie_counts,
            labels=pie_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 8}
        )
        ax.set_title(source, fontsize=10)
        ax.axis("equal")

        if i % 2 == 0:
            col1.pyplot(fig)
        else:
            col2.pyplot(fig)
    
    # ================= WORDCLOUD =================
    st.markdown("---")
    st.subheader("WordCloud Berdasarkan Sentimen & Dataset")
    col1, col2 = st.columns(2)

    with col1:
        sentiment_option = st.selectbox(
            "Filter Sentimen",
            ["All", "Positive", "Negative", "Neutral"]
        )

    with col2:
        dataset_option = st.selectbox(
            "Filter Dataset",
            ["All"] + list(st.session_state.labeling_df["dataset"].unique())
        )

    filtered_df = st.session_state.labeling_df.copy()

    if sentiment_option != "All":
        filtered_df = filtered_df[filtered_df["label"] == sentiment_option]

    if dataset_option != "All":
        filtered_df = filtered_df[filtered_df["dataset"] == dataset_option]
    
    if filtered_df.empty:
        st.warning("⚠️ Tidak ada data untuk kombinasi filter yang dipilih.")
    else:
        text_data = " ".join(filtered_df["clean"].astype(str))

        wordcloud = WordCloud(
            width=400,
            height=250,
            background_color="white",
            colormap="Blues",
            max_words=150
        ).generate(text_data)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig)


    # ================= DOWNLOAD =================
    csv = st.session_state.labeling_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Unduh Pelabelan Dataset ",
        data=csv,
        file_name="dataset_dana_lexicon_label.csv",
        mime="text/csv"
    )
