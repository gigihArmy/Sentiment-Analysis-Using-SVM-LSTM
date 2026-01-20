import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Input, Dropout, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

# ================= SESSION STATE =================
if "lstm_results" not in st.session_state:
    st.session_state.lstm_results = None

if "lstm_history" not in st.session_state:
    st.session_state.lstm_history = None

# ================= HEADER =================
st.title("Model Deep Learning LSTM")
st.markdown(
    "Halaman ini digunakan untuk melatih model **Bidirectional LSTM** "
    "untuk analisis sentimen dataset E-Wallet DANA."
)

st.markdown("---")

# ================= UPLOAD DATASET =================
st.subheader("Upload Dataset Hasil Pelabelan")
uploaded_file = st.file_uploader(
    "Upload file CSV hasil pelabelan (harus memiliki kolom 'clean' dan 'label')",
    type=["csv"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        required_cols = {"clean", "label"}
        if not required_cols.issubset(df.columns):
            st.error(f"Dataset tidak sesuai. Pastikan memiliki kolom: {required_cols}")
        else:
            st.session_state.train_data_lstm = df
            st.success(f"Dataset berhasil dimuat! Total data: {len(df)}")
            st.write("**Distribusi Label Saat Ini:**")
            st.write(df['label'].value_counts())
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

# ================= PARAMETER MODEL =================
if "train_data_lstm" in st.session_state:
    st.markdown("---")
    st.subheader("Konfigurasi Parameter LSTM")
    
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Masukkan Jumlah Epoch", min_value=1, max_value=50, value=10)
        batch_size = st.selectbox("Batch Size", options=[8, 16, 32, 64], index=1)
    
    with col2:
        max_words = st.number_input("Max Words (Vocabulary)", value=5000)
        max_len = st.number_input("Max Sequence Length (Padding)", value=500)

    # ================= PROSES TRAINING =================
    if st.button("Mulai Prediksi dan Evaluasi"):
        with st.spinner("Sedang melatih model LSTM... Mohon tunggu (Proses ini memakan waktu)"):
            
            df = st.session_state.train_data_lstm.copy()
            
            # Label Encoding sesuai basic code
            label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            df['label'] = df['label'].map(label_map)
            
            # Jika ada label yang gagal di-map (NaN), hapus
            df = df.dropna(subset=['label'])
            
            X = df['clean'].astype(str)
            y = df['label'].values

            # Split Data (80:20)
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Tokenization & Padding
            tok = Tokenizer(num_words=max_words)
            tok.fit_on_texts(X_train_raw)

            X_train = pad_sequences(tok.texts_to_sequences(X_train_raw), maxlen=max_len)
            X_test = pad_sequences(tok.texts_to_sequences(X_test_raw), maxlen=max_len)

            # One Hot Encoding Target
            y_train = to_categorical(y_train_raw, num_classes=3)
            # y_test_raw tetap integer untuk memudahkan evaluasi sklearn

            # Build Model (Sesuai Basic Code Colab)
            model = Sequential([
                Input(shape=[max_len]),
                Embedding(max_words, 128, input_length=max_len),
                SpatialDropout1D(0.4),
                Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
                Dense(64, activation='relu'),
                Dropout(0.4),
                Dense(3, activation='softmax')
            ])

            model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

            # Callbacks
            callback = EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            )

            # Fit Model
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                callbacks=[callback],
                verbose=0
            )

            # Prediksi
            y_pred_prob = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred_prob, axis=1)

            # 8. Simpan ke Session State
            st.session_state.lstm_results = {
                'y_test': y_test_raw,
                'y_pred': y_pred_classes,
                'report': classification_report(y_test_raw, y_pred_classes, target_names=['Negative', 'Neutral', 'Positive']),
                'acc': accuracy_score(y_test_raw, y_pred_classes),
                'prec': precision_score(y_test_raw, y_pred_classes, average='macro'),
                'rec': recall_score(y_test_raw, y_pred_classes, average='macro'),
                'cm': confusion_matrix(y_test_raw, y_pred_classes)
            }
            st.session_state.lstm_history = history.history

            st.success("✅ Training dan Evaluasi Selesai!")

# ================= DISPLAY RESULTS =================
if st.session_state.lstm_results is not None:
    st.markdown("---")
    st.subheader("Hasil Evaluasi Model LSTM")

    # Metrics
    res = st.session_state.lstm_results
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Accuracy", f"{res['acc']*100:.2f}%")
    col_b.metric("Precision", f"{res['prec']*100:.2f}%")
    col_c.metric("Recall", f"{res['rec']*100:.2f}%")

    # Classification Report
    st.write("**Classification Report:**")
    st.code(res['report'])

    # Visualisasi Row
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.write("**Confusion Matrix:**")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            res['cm'], annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot(fig_cm)

    with col_plot2:
        st.write("**Training History (Accuracy):**")
        hist = st.session_state.lstm_history
        fig_acc, ax_acc = plt.subplots(figsize=(5, 4))
        ax_acc.plot(hist['accuracy'], label='Train Acc')
        ax_acc.plot(hist['val_accuracy'], label='Val Acc')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        st.pyplot(fig_acc)

    # Plot Loss (Bawah)
    st.markdown("---")
    st.write("**Training History (Loss):**")
    fig_loss, ax_loss = plt.subplots(figsize=(8, 3))
    ax_loss.plot(hist['loss'], label='Train Loss', color='orange')
    ax_loss.plot(hist['val_loss'], label='Val Loss', color='red')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    st.pyplot(fig_loss)