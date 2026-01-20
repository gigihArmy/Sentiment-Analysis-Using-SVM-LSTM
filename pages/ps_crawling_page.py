import streamlit as st
import pandas as pd
import asyncio
from google_play_scraper import reviews, Sort
import re

# ================= SESSION STATE =================
if "playstore_data" not in st.session_state:
    st.session_state.playstore_data = None

# ================= HEADER =================
st.title("Crawl Data Google Playstore")
st.markdown(
    "Halaman ini digunakan untuk melakukan **crawling ulasan Google Play Store** "
    "pada aplikasi **E-Wallet DANA**."
)

st.markdown("---")

# ================= INPUT FORM =================
st.subheader("Parameter Crawling")

with st.form("playstore_form"):
    playstore_url = st.text_input(
        label="URL Google Playstore",
        placeholder="https://play.google.com/store/apps/details?id=id.dana"
    )

    limit = st.number_input(
        label="Limit Data",
        min_value=10,
        max_value=20000,
        value=1000,
        step=100
    )

    submit = st.form_submit_button("Mulai Crawling")

# ================= HELPER =================
def extract_app_id(url):
    """
    Mengambil app_id dari URL Playstore
    """
    match = re.search(r"id=([a-zA-Z0-9._]+)", url)
    return match.group(1) if match else None

# ================= ASYNC FUNCTION =================
async def crawl_playstore(app_id, limit):
    result, _ = reviews(
        app_id,
        lang="id",
        country="id",
        sort=Sort.NEWEST,
        count=int(limit),
    )

    df = pd.DataFrame(result)
    df = df[["at", "content"]]

    # async dummy sleep (agar konsisten & non-blocking)
    await asyncio.sleep(0.1)

    return df

# ================= PROCESS =================
if submit:
    app_id = extract_app_id(playstore_url)

    if not playstore_url or not app_id:
        st.error("⚠️ URL Google Playstore tidak valid!")
    else:
        with st.spinner("Sedang melakukan crawling data..."):
            df = asyncio.run(crawl_playstore(app_id, limit))
            df['dataset'] = 'google playstore'
            st.session_state.playstore_data = df

        st.success(f"✅ Crawling selesai! Total data: {len(df)}")

# ================= DISPLAY DATA =================
if st.session_state.playstore_data is not None:
    st.markdown("---")
    st.subheader("Hasil Crawling Google Playstore")

    st.dataframe(
        st.session_state.playstore_data,
        use_container_width=True
    )

    # ================= DOWNLOAD =================
    csv = st.session_state.playstore_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Unduh Data (CSV)",
        data=csv,
        file_name="google_playstore_dana.csv",
        mime="text/csv"
    )

else:
    st.info("Data crawling akan ditampilkan setelah proses dijalankan.")
