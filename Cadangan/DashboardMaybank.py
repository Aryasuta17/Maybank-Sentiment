from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from PIL import Image

# Page config
st.set_page_config(page_title='Maybank Sentiment Dashboard', layout='wide')

st.markdown(
    """
    <style>
        div[data-baseweb="radio"] label {
            font-size: 1.5rem !important;
            padding: 0.75rem 1rem !important;
        }
        .css-1d391kg {
            font-size: 1.2em !important;
        }
        /* Header background and styling */
        .header-title {
            background-color: #FFEB3B;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Plotly theme
pio.templates.default = 'plotly_white'

# Load data function
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8')
    # normalize headers
    df.columns = df.columns.str.strip().str.lower()
    # required columns
    for req in ['publish_date', 'text', 'source', 'sentiment']:
        if req not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {req}")
    # drop rating if exists
    if 'rating' in df.columns:
        df = df.drop(columns=['rating'])
    # rename columns
    df = df.rename(columns={
        'publish_date': 'timestamp',
        'source': 'platform'
    })
    # parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    # extract day and month
    df['day'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    return df

# Main data load
df = load_data('merged_all_real - FIX CSV BENER.csv')

# Header with logo and title
col_logo, col_title = st.columns([1, 9])
with col_logo:
    logo = Image.open("logo.png")
    st.image(logo, width=200)  # increased logo size
with col_title:
    st.markdown(
        '<h1 class="header-title">Maybank Social-Media Sentiment Dashboard</h1>',
        unsafe_allow_html=True
    )

# Sidebar navigation with icons
tabs = ['General', 'Detail', 'Social Media']
icons = {'General': '📊', 'Detail': '📈', 'Social Media': '📱'}
page = st.sidebar.radio('', tabs, format_func=lambda t: f"{icons[t]} {t}")

# --- GENERAL ---
if page == 'General':
    st.subheader('General Overview')
    # metrics
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
    # distribution platform
    st.subheader('Distribusi Sosial Media')
    dplat = df.platform.value_counts().reset_index()
    dplat.columns = ['platform', 'count']
    col1, col2 = st.columns(2)
    fig_bar_plat = px.bar(dplat, x='platform', y='count', text='count', title='Bar Chart Platform')
    fig_pie_plat = px.pie(dplat, names='platform', values='count', title='Pie Chart Platform')
    col1.plotly_chart(fig_bar_plat, use_container_width=True)
    col2.plotly_chart(fig_pie_plat, use_container_width=True)

    # distribution sentiment
    st.subheader('Distribusi Sentiment')
    dsent = df.sentiment.value_counts().reset_index()
    dsent.columns = ['sentiment', 'count']
    e1, e2 = st.columns(2)
    fig_bar_sent = px.bar(dsent, x='sentiment', y='count', text='count', title='Bar Chart Sentiment')
    fig_pie_sent = px.pie(dsent, names='sentiment', values='count', title='Pie Chart Sentiment')
    e1.plotly_chart(fig_bar_sent, use_container_width=True)
    e2.plotly_chart(fig_pie_sent, use_container_width=True)

    # daily line
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
        # line per day
        st.subheader(f'Line Harian — {sel_month}')
        by_day = dmon.groupby(['day', 'sentiment']).size().reset_index(name='count')
        fig_d1 = px.line(by_day, x='day', y='count', color='sentiment', markers=True)
        st.plotly_chart(fig_d1, use_container_width=True)

        # bar & pie
        st.subheader(f'Bar & Pie Sentiment — {sel_month}')
        monthly = dmon.sentiment.value_counts().reset_index()
        monthly.columns = ['sentiment', 'count']
        mcol1, mcol2 = st.columns(2)
        mcol1.plotly_chart(px.bar(monthly, x='sentiment', y='count', text='count'), use_container_width=True)
        mcol2.plotly_chart(px.pie(monthly, names='sentiment', values='count'), use_container_width=True)

        # examples
        st.subheader(f'3 Contoh Teks per Sentiment — {sel_month}')
        for sentiment_label in sorted(monthly.sentiment):
            with st.expander(sentiment_label):
                samples = dmon[dmon.sentiment == sentiment_label]['text'].head(3)
                for txt in samples:
                    st.write(f"- {txt}")

# --- SOCIAL MEDIA ---
else:
    st.subheader('Social Media Analysis')
    plat = st.selectbox('Pilih Platform', sorted(df.platform.unique()))
    sel_month_sm = st.selectbox('Pilih Bulan', sorted(df.month.unique()))
    sdf = df[df.platform == plat]

    st.markdown('---')
    # line + bar per month
    st.subheader(f'{plat} — Sentiment per Bulan')
    by_mon = sdf.groupby(['month', 'sentiment']).size().reset_index(name='count')
    fig_sm1 = px.line(by_mon, x='month', y='count', color='sentiment', markers=True)
    fig_sm2 = px.bar(by_mon, x='month', y='count', color='sentiment', barmode='group')
    st.plotly_chart(fig_sm1, use_container_width=True)
    st.plotly_chart(fig_sm2, use_container_width=True)

    # pie for selected month
    sel = sdf[sdf.month == sel_month_sm]
    if not sel.empty:
        st.subheader(f'{plat} — Pie Bulanan ({sel_month_sm})')
        viral = sel.sentiment.value_counts().reset_index()
        viral.columns = ['sentiment', 'count']
        st.plotly_chart(px.pie(viral, names='sentiment', values='count'), use_container_width=True)

    st.markdown('---')
    # daily line + bar overall
    st.subheader(f'{plat} — Line & Bar Harian Keseluruhan')
    daily2 = sdf.groupby(['day', 'sentiment']).size().reset_index(name='count')
    fig_sm3 = px.line(daily2, x='day', y='count', color='sentiment', markers=True)
    fig_sm4 = px.bar(daily2, x='day', y='count', color='sentiment', barmode='group')
    st.plotly_chart(fig_sm3, use_container_width=True)
    st.plotly_chart(fig_sm4, use_container_width=True)

    # overall pie
    st.subheader(f'{plat} — Pie Keseluruhan')
    overall = sdf.sentiment.value_counts().reset_index()
    overall.columns = ['sentiment', 'count']
    st.plotly_chart(px.pie(overall, names='sentiment', values='count'), use_container_width=True)

    # examples social media
    st.subheader(f'3 Contoh Teks per Sentiment — {plat} ({sel_month_sm})')
    for sentiment_label in sorted(overall.sentiment):
        with st.expander(sentiment_label):
            samples = sel[sel.sentiment == sentiment_label]['text'].head(3)
            for txt in samples:
                st.write(f"- {txt}")