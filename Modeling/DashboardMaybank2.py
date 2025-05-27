from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from PIL import Image
import os, joblib

# Import tambahan untuk wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from io import BytesIO

# Sentiment model
import joblib
import numpy as np

BASE_DIR = os.path.dirname(__file__)  
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")

# Load sentiment stacked ensemble model
dt = joblib.load(MODEL_PATH)
pipe = dt['pipeline']
thresh_neg = dt['threshold']

# Page config
st.set_page_config(page_title='Maybank Sentiment Dashboard', layout='wide')
pio.templates.default = 'plotly_white'

# Navigasi rapat & terpusat styling
st.markdown(
    """
    <style>
    .nav-buttons-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .nav-buttons-container button {
        font-size: 1.1rem !important;
        padding: 0.5rem 1.2rem !important;
        border-radius: 0.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Plotly theme
pio.templates.default = 'plotly_white'

# Load data function
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower()
    for req in ['publish_date', 'text', 'source', 'sentiment']:
        if req not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {req}")
    if 'rating' in df.columns:
        df = df.drop(columns=['rating'])
    df = df.rename(columns={
        'publish_date': 'timestamp',
        'source': 'platform'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['day'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    return df

# Main data load
CSV_PATH   = os.path.join(BASE_DIR, "merged_all_real - FIX CSV BENER.csv")
df = load_data(CSV_PATH)

# Header with logo and title
col_logo, col_title = st.columns([1, 9])
with col_logo:
    logo = Image.open("logo.png")
    st.image(logo, width=200)
with col_title:
    st.markdown(
        '<h1 class="header-title">Maybank Social-Media Sentiment Dashboard</h1>',
        unsafe_allow_html=True
    )

# Tombol navigasi di bawah logo
tabs = ['General', 'Detail', 'Social Media', 'Sentiment']
icons = {'General': '📊', 'Detail': '📈', 'Social Media': '📱', 'Sentiment': '🤖'}
if 'page' not in st.session_state:
    st.session_state.page = 'General'
nav_cols = st.columns(len(tabs))
for i, t in enumerate(tabs):
    if nav_cols[i].button(f"{icons[t]} {t}"):
        st.session_state.page = t
page = st.session_state.page

# --- GENERAL ---
if page == 'General':
    st.subheader('General Overview')
    total = len(df)
    pos = df.sentiment.str.contains('pos', case=False).mean() * 100
    neg = df.sentiment.str.contains('neg', case=False).mean() * 100
    neu = df.sentiment.str.contains('neu', case=False).mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Posts', f"{total:,}")
    c2.metric('% Positif', f"{pos:.1f}%")
    c3.metric('% Negatif', f"{neg:.1f}%")
    c4.metric('% Netral', f"{neu:.1f}%")

    st.markdown('---')
    st.subheader('Distribusi Sosial Media')
    dplat = df.platform.value_counts().reset_index()
    dplat.columns = ['platform', 'count']
    col1, col2 = st.columns(2)
    fig_bar_plat = px.bar(dplat, x='platform', y='count', text='count', title='Bar Chart Platform')
    fig_pie_plat = px.pie(dplat, names='platform', values='count', title='Pie Chart Platform')
    col1.plotly_chart(fig_bar_plat, use_container_width=True)
    col2.plotly_chart(fig_pie_plat, use_container_width=True)

    st.subheader('Distribusi Sentiment')
    dsent = df.sentiment.value_counts().reset_index()
    dsent.columns = ['sentiment', 'count']
    e1, e2 = st.columns(2)
    fig_bar_sent = px.bar(dsent, x='sentiment', y='count', text='count', title='Bar Chart Sentiment')
    fig_pie_sent = px.pie(dsent, names='sentiment', values='count', title='Pie Chart Sentiment')
    e1.plotly_chart(fig_bar_sent, use_container_width=True)
    e2.plotly_chart(fig_pie_sent, use_container_width=True)

    st.subheader('Line Chart Sentiment per Hari')
    daily = df.groupby(['day', 'sentiment']).size().reset_index(name='count')
    fig_line = px.line(daily, x='day', y='count', color='sentiment', markers=True, title='Sentiment per Hari')
    st.plotly_chart(fig_line, use_container_width=True)

# --- DETAIL ---
elif page == 'Detail':
    st.subheader('Detail — Sentiment per Bulan')
    sel_month = st.selectbox('Pilih Bulan', sorted(df.month.unique()))
    dmon = df[df.month == sel_month]
    if dmon.empty:
        st.info('Tidak ada data untuk bulan ini.')
    else:
        st.markdown('---')
        st.subheader(f'Line Harian — {sel_month}')
        by_day = dmon.groupby(['day', 'sentiment']).size().reset_index(name='count')
        fig_d1 = px.line(by_day, x='day', y='count', color='sentiment', markers=True)
        st.plotly_chart(fig_d1, use_container_width=True)

        st.subheader(f'Bar & Pie Sentiment — {sel_month}')
        monthly = dmon.sentiment.value_counts().reset_index()
        monthly.columns = ['sentiment', 'count']
        mcol1, mcol2 = st.columns(2)
        mcol1.plotly_chart(px.bar(monthly, x='sentiment', y='count', text='count'), use_container_width=True)
        mcol2.plotly_chart(px.pie(monthly, names='sentiment', values='count'), use_container_width=True)

        st.subheader(f'3 Contoh Teks per Sentiment — {sel_month}')
        for sentiment_label in sorted(monthly.sentiment):
            with st.expander(sentiment_label):
                samples = dmon[dmon.sentiment == sentiment_label]['text'].head(3)
                for txt in samples:
                    st.write(f"- {txt}")

        # --- Wordcloud Section dengan stopwords khusus + slangwords normalisasi ---
        st.subheader(f'Wordcloud Kata untuk Bulan {sel_month}')
        text_for_wordcloud_raw = " ".join(dmon['text'].dropna().values)

        if len(text_for_wordcloud_raw.strip()) == 0:
            st.info('Tidak ada teks untuk membuat wordcloud.')
        else:
            slang_dict = {
                "gak": "tidak","yg" : "yang", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
                "aja": "saja", "banget": "sangat", "bgt": "sangat", "dgn": "dengan",
                "dr": "dari", "tp": "tapi", "tpi": "tapi", "krn": "karena",
                "krna": "karena", "jg": "juga", "kalo": "kalau", "kl": "kalau",
                "udh": "sudah", "sdh": "sudah", "blm": "belum", "tdk": "tidak",
                "td": "tadi", "skrg": "sekarang", "sm": "sama", "sy": "saya",
                "gw": "saya", "gue": "saya", "loe": "kamu", "lu": "kamu", "lo": "kamu",
                "trs": "terus", "ny": "nya", "bgt": "banget", "dl": "dulu",
                "dlu": "dulu", "bbrp": "beberapa", "lg": "lagi", "udh": "sudah",
                "udh": "sudah", "udah": "sudah", "tdk": "tidak", "dpt": "dapat",
                "bisa": "bisa", "gt": "gitu", "gitu": "seperti itu", "jg": "juga",
                "emg": "memang", "pdhl": "padahal", "kyk": "seperti", "trs": "terus",
                "sbnrnya": "sebenarnya", "cuma": "hanya", "doang": "saja", "aj": "saja",
                "btw": "ngomong-ngomong", "bikin": "membuat", "pake": "menggunakan",
                "abis": "habis", "makasih": "terima kasih", "makasi": "terima kasih",
                "trims": "terima kasih", "thanks": "terima kasih"
            }

            def normalize_slang(text: str) -> str:
                words = text.lower().split()
                normalized_words = [slang_dict.get(word, word) for word in words]
                return " ".join(normalized_words)

            text_normalized = normalize_slang(text_for_wordcloud_raw)
            custom_stopwords = set(STOPWORDS)
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=200,
                colormap='viridis',
                stopwords=custom_stopwords
            ).generate(text_normalized)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
            buf = BytesIO(); plt.savefig(buf,format='png'); buf.seek(0)
            st.image(buf); plt.close(fig)

# --- SOCIAL MEDIA ---
elif page == 'Social Media':
    st.subheader('Social Media Analysis')
    plat = st.selectbox('Pilih Platform', sorted(df.platform.unique()))
    sel_month_sm = st.selectbox('Pilih Bulan', sorted(df.month.unique()))
    sdf = df[df.platform == plat]
    st.markdown('---')
    st.subheader(f'{plat} — Sentiment per Bulan')
    by_mon = sdf.groupby(['month', 'sentiment']).size().reset_index(name='count')
    fig_sm1 = px.line(by_mon, x='month', y='count', color='sentiment', markers=True)
    fig_sm2 = px.bar(by_mon, x='month', y='count', color='sentiment', barmode='group')
    st.plotly_chart(fig_sm1, use_container_width=True)
    st.plotly_chart(fig_sm2, use_container_width=True)

    sel = sdf[sdf.month == sel_month_sm]
    if not sel.empty:
        st.subheader(f'{plat} — Pie Bulanan ({sel_month_sm})')
        viral = sel.sentiment.value_counts().reset_index()
        viral.columns = ['sentiment', 'count']
        st.plotly_chart(px.pie(viral, names='sentiment', values='count'), use_container_width=True)

    st.markdown('---')
    st.subheader(f'{plat} — Line & Bar Harian Keseluruhan')
    daily2 = sdf.groupby(['day', 'sentiment']).size().reset_index(name='count')
    fig_sm3 = px.line(daily2, x='day', y='count', color='sentiment', markers=True)
    fig_sm4 = px.bar(daily2, x='day', y='count', color='sentiment', barmode='group')
    st.plotly_chart(fig_sm3, use_container_width=True)
    st.plotly_chart(fig_sm4, use_container_width=True)

    st.subheader(f'{plat} — Pie Keseluruhan')
    overall = sdf.sentiment.value_counts().reset_index()
    overall.columns = ['sentiment', 'count']
    st.plotly_chart(px.pie(overall, names='sentiment', values='count'), use_container_width=True)

    st.subheader(f'3 Contoh Teks per Sentiment — {plat} ({sel_month_sm})')
    for sentiment_label in sorted(overall.sentiment):
        with st.expander(sentiment_label):
            samples = sel[sel.sentiment == sentiment_label]['text'].head(3)
            for txt in samples:
                st.write(f"- {txt}")

# --- SENTIMENT ---
else:  # page == 'Sentiment'
    st.subheader('Sentiment Prediction')
    st.write('Upload CSV berisi kolom `text` untuk prediksi sentimen.')
    upload = st.file_uploader('Pilih file CSV', type='csv')
    if upload:
        dfp = pd.read_csv(upload)
        if 'text' not in dfp.columns:
            st.error('CSV harus memiliki kolom `text`.')
        else:
            texts = dfp['text'].astype(str).tolist()
            proba = pipe.predict_proba(texts)
            idx   = list(pipe.classes_).index('negatif')
            preds = ['negatif' if p[idx]>=thresh_neg else pipe.classes_[np.argmax(p)] for p in proba]
            dfp['sentiment_pred'] = preds
            st.dataframe(dfp[['text','sentiment_pred']])
            csv_out = dfp.to_csv(index=False).encode('utf-8')
            st.download_button('Download CSV Hasil', csv_out, 'predictions.csv', 'text/csv')