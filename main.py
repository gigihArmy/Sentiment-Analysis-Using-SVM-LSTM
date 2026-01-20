import streamlit as st

#multipage web
def run_streamlit():
    st.set_page_config(
        page_title="Analisis Sentimen Dana",
        layout="wide",
        page_icon=":material/edit:"
    )
    pg = st.navigation(
    {
        "Home": [st.Page("pages/homepage.py", title="Homepage", icon=":material/home:")],
        "Data Crawling": [
            st.Page("pages/ig_crawling_page.py", title="Instagram", icon=":material/download:"), 
            st.Page("pages/ps_crawling_page.py", title="Google Playstore", icon=":material/download:")
        ], 
        "Data Preprocessing": [
            st.Page("pages/preprocessing_page.py", title="Text Preprocessing", icon=":material/table_view:"),
            st.Page("pages/labelling_page.py", title="Lexicon Labelling", icon=":material/dictionary:")
        ], 
        "Model Train & Evaluation": [
            st.Page("pages/svm_page.py", title="SVM Model Train", icon=":material/model_training:"),
            st.Page("pages/lstm_model.py", title="LSTM Model Train", icon=":material/model_training:")
        ]
    })

    pg.run()

if __name__ == "__main__":
    run_streamlit()