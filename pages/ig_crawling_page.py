import streamlit as st
import pandas as pd
import asyncio
import time
from apify_client import ApifyClient
import re
from apify_client.errors import ApifyApiError


# ================= SESSION STATE =================
if "ig_data" not in st.session_state:
    st.session_state.ig_data = None

# ================= HEADER =================
st.title("Crawl Data Instagram")
st.markdown(
    "Halaman ini digunakan untuk melakukan **crawling komentar Instagram** "
    "yang berkaitan dengan **E-Wallet DANA** menggunakan layanan **Apify**."
)

st.markdown("---")

# ================= INPUT FORM =================
st.subheader("Parameter Crawling")

with st.form("crawl_form"):
    ig_url = st.text_input(
        label="URL Instagram",
        placeholder="https://www.instagram.com/p/xxxxx atau /reel/xxxxx"
    )

    apify_token = st.text_input(
        label="API Apify Client",
        type="password",
        placeholder="Masukkan API Token Apify"
    )

    limit = st.number_input(
        label="Limit Data",
        min_value=10,
        max_value=5000,
        value=500,
        step=50
    )

    submit = st.form_submit_button("Mulai Crawling")

# ================= ASYNC CRAWLING FUNCTION =================
async def crawl_instagram(url, token, limit):
    client = ApifyClient(token)

    run_input = {
        "directUrls": [url],
        "resultsLimit": int(limit),
        "isNewestComments": True,
        "includeNestedComments": False,
    }

    all_results = []
    seen_ids = set()

    run = client.actor("SbK00X0JYCPblD2wp").call(run_input=run_input)
    dataset_id = run["defaultDatasetId"]

    for item in client.dataset(dataset_id).iterate_items():
        comment_id = item.get("id")
        if comment_id not in seen_ids:
            seen_ids.add(comment_id)
            all_results.append({
                "timestamp": item.get("timestamp"),
                "text": item.get("text"),
            })

        # async sleep (non-blocking)
        await asyncio.sleep(0.01)

    return pd.DataFrame(all_results)

def is_valid_instagram_url(url):
    pattern = r"https?:\/\/(?:www\.)?instagram\.com\/(?:p|reel)\/[A-Za-z0-9_-]+"
    return re.match(pattern, url)

# ================= PROCESS =================
if submit:
    if not ig_url or not apify_token:
        st.error("⚠️ URL Instagram dan API Apify wajib diisi!")
    else:
        try:
            # Validasi URL dulu (UX friendly)
            if not is_valid_instagram_url(ig_url):
                st.warning(
                    "URL Instagram tidak valid.\n\n"
                    "Pastikan format URL seperti:\n"
                    "- https://www.instagram.com/p/xxxxx\n"
                    "- https://www.instagram.com/reel/xxxxx"
                )

            else:
                with st.spinner("Sedang melakukan crawling data Instagram..."):
                    df = asyncio.run(crawl_instagram(ig_url, apify_token, limit))

                if df.empty:
                    st.info("Tidak ada komentar yang berhasil diambil.")
                else:
                    df['dataset'] = 'instagram'
                    st.session_state.ig_data = df
                    st.success(f"✅ Crawling berhasil! Total data: {len(df)}")

        except ApifyApiError as e:
            error_message = str(e)

            # API TOKEN SALAH
            if "authentication token is not valid" in error_message.lower():
                st.error(
                    "API Apify tidak valid.\n\n"
                    "Silakan periksa kembali API Token Anda di dashboard Apify."
                )

            # URL SALAH (fallback jika lolos regex)
            elif "input.directurls" in error_message.lower():
                st.error(
                    "URL Instagram tidak sesuai format.\n\n"
                    "Gunakan URL postingan atau reel Instagram."
                )

            else:
                st.error(
                    "⚠️ Terjadi kesalahan saat proses crawling.\n\n"
                    "Silakan coba kembali atau periksa parameter input."
                )

        except Exception:
            st.error(
                "Sistem mengalami kendala teknis.\n\n"
                "Silakan coba kembali beberapa saat lagi."
            )

# ================= DISPLAY DATA =================
if st.session_state.ig_data is not None:
    st.markdown("---")
    st.subheader("Hasil Crawling")

    st.dataframe(
        st.session_state.ig_data,
        use_container_width=True
    )

    # ================= DOWNLOAD =================
    csv = st.session_state.ig_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Unduh Data (CSV)",
        data=csv,
        file_name="instagram_comments_dana.csv",
        mime="text/csv"
    )

else:
    st.info("Data crawling akan muncul di sini setelah proses dijalankan.")
