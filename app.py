import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from indonesian_number_normalizer import create_normalizer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve)
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Berita Ekonomi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Konfigurasi")
st.sidebar.markdown("---")

# Upload file
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV)", 
    type=['csv'],
    help="Upload file CSV dengan kolom: Title, Content, date, label, valas, faktor"
)

# Parameter preprocessing
st.sidebar.subheader("⚙️ Parameter Preprocessing")
use_stemming = st.sidebar.checkbox("Gunakan Stemming", value=True)
remove_ads = st.sidebar.checkbox("Hapus Konten Advertisement", value=True)
normalize_numbers = st.sidebar.checkbox("Normalisasi Angka", value=False)

# Parameter model - DIPERBAIKI untuk match notebook
st.sidebar.subheader("🤖 Parameter Model ComplementNB + TF-IDF")
use_notebook_params = st.sidebar.checkbox("🎯 Gunakan Parameter Notebook (untuk hasil identik)", value=False)

if use_notebook_params:
    st.sidebar.success("✅ Menggunakan parameter optimal dari notebook")
    max_features = 3500
    ngram_min = 1
    ngram_max = 2
    alpha_value = 0.5
    min_df = 3
    max_df = 0.92
    
    # Tampilkan parameter yang digunakan
    st.sidebar.write("**Parameter Notebook:**")
    st.sidebar.write(f"- Max Features: {max_features}")
    st.sidebar.write(f"- N-gram: ({ngram_min}, {ngram_max})")
    st.sidebar.write(f"- Alpha: {alpha_value}")
    st.sidebar.write(f"- Min DF: {min_df}")
    st.sidebar.write(f"- Max DF: {max_df}")
    
else:
    st.sidebar.warning("⚠️ Parameter custom - hasil mungkin berbeda dengan notebook")
    max_features = st.sidebar.slider("Max Features TF-IDF", 1000, 5000, 3500, 500)
    ngram_min = st.sidebar.selectbox("N-gram Min", [1, 2], index=0)
    ngram_max = st.sidebar.selectbox("N-gram Max", [1, 2, 3], index=1)
    alpha_value = st.sidebar.slider("Alpha ComplementNB", 0.1, 2.0, 0.5, 0.1)
    min_df = st.sidebar.slider("Min DF", 1, 10, 3)
    max_df = st.sidebar.slider("Max DF", 0.8, 0.99, 0.92, 0.01)

# Debug options
st.sidebar.markdown("---")
st.sidebar.subheader("🐛 Debug Options")
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
show_detailed_metrics = st.sidebar.checkbox("Show Detailed Metrics", value=False)

# Main content
st.markdown('<div class="main-header">📈 Analisis Sentimen Berita Ekonomi Indonesia</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Data berhasil dimuat: {len(df):,} berita")
        
        # Validasi kolom - DIPERBAIKI
        required_columns = ['Title', 'Content', 'date', 'label', 'valas']
        # Cek apakah sudah ada stemmed_text (sudah diproses)
        if 'stemmed_text' in df.columns:
            st.info("✅ Dataset sudah memiliki kolom 'stemmed_text' - akan menggunakan data yang sudah diproses")
            use_existing_stemmed = True
        else:
            st.info("ℹ️ Dataset tidak memiliki kolom 'stemmed_text' - akan melakukan preprocessing")
            use_existing_stemmed = False

        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Kolom yang hilang: {missing_columns}")
            st.stop()
        
        # Konversi tanggal
        df['tanggal'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        
        # Validasi dataset - TAMBAHAN BARU
        st.subheader("🔍 Validasi Dataset")
        
        # Check data distribution
        pos_count = (df['label'] == 'positif').sum()
        neg_count = (df['label'] == 'negatif').sum()
        total_count = len(df)
        pos_ratio = pos_count / total_count

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Berita", f"{total_count:,}")
        with col2:
            st.metric("Positif", f"{pos_count:,}")
        with col3:
            st.metric("Negatif", f"{neg_count:,}")  
        with col4:
            st.metric("Rasio Positif", f"{pos_ratio:.1%}")

        # Warning jika distribusi tidak seimbang
        if pos_ratio < 0.3 or pos_ratio > 0.7:
            st.warning(f"⚠️ Distribusi kelas tidak seimbang ({pos_ratio:.1%} positif). Ini mungkin mempengaruhi performa model.")

        # Check untuk data yang hilang
        missing_data = df[['Title', 'Content', 'label']].isnull().sum()
        if missing_data.sum() > 0:
            st.warning("⚠️ Ditemukan data yang hilang:")
            st.write(missing_data[missing_data > 0])

        # Expected dataset characteristics
        if use_existing_stemmed:
            st.success("✅ Dataset memiliki kolom 'stemmed_text' - menggunakan preprocessing yang sama dengan notebook")
        else:
            st.info("ℹ️ Dataset akan diproses ulang - hasil mungkin sedikit berbeda dengan notebook")
        
        # Cek apakah ada kolom faktor di dataset
        faktor_columns = [col for col in df.columns if 'faktor' in col.lower()]
        has_faktor_column = len(faktor_columns) > 0
        
        if has_faktor_column:
            st.info(f"✅ Kolom faktor ditemukan: {faktor_columns}")
        else:
            st.warning("⚠️ Kolom faktor tidak ditemukan, akan menggunakan klasifikasi otomatis")
        
        # Debug information
        if debug_mode:
            with st.expander("🐛 Debug Information"):
                st.write("**Dataset Info:**")
                st.write(f"- Shape: {df.shape}")
                st.write(f"- Columns: {list(df.columns)}")
                st.write(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                if 'stemmed_text' in df.columns:
                    avg_length = df['stemmed_text'].apply(len).mean()
                    st.write(f"- Average stemmed text length: {avg_length:.0f} characters")
                
                st.write("**Model Parameters:**")
                st.write(f"- Using existing stemmed: {use_existing_stemmed}")
                st.write(f"- Max features: {max_features}")
                st.write(f"- N-gram range: ({ngram_min}, {ngram_max})")
                st.write(f"- Alpha: {alpha_value}")
        
        # Tab utama
        if has_faktor_column:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview Data", 
                "🔧 Preprocessing", 
                "🏷️ Analisis Faktor", 
                "🤖 ComplementNB + TF-IDF", 
                "📈 Visualisasi & Korelasi"
            ])
        else:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview Data", 
                "🔧 Preprocessing", 
                "🏷️ Klasifikasi Faktor", 
                "🤖 ComplementNB + TF-IDF", 
                "📈 Visualisasi & Korelasi"
            ])
        
        with tab1:
            st.markdown('<div class="sub-header">📊 Overview Dataset</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Berita", f"{len(df):,}")
            with col2:
                st.metric("Rentang Waktu", f"{df['tanggal'].dt.year.min()} - {df['tanggal'].dt.year.max()}")
            with col3:
                pos_count = (df['label'] == 'positif').sum()
                st.metric("Berita Positif", f"{pos_count:,}")
            with col4:
                neg_count = (df['label'] == 'negatif').sum()
                st.metric("Berita Negatif", f"{neg_count:,}")
            
            # Distribusi label dengan Matplotlib
            st.subheader("Distribusi Label Sentimen")
            label_counts = df['label'].value_counts()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Pie chart
            ax1.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.set_title("Distribusi Sentimen Berita")
            
            # Bar chart
            bars = ax2.bar(label_counts.index, label_counts.values, color=['#ff7f0e', '#1f77b4'])
            ax2.set_title("Jumlah Berita per Label")
            ax2.set_ylabel("Jumlah Berita")
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                         f'{int(height)}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Timeline berita
            st.subheader("Timeline Jumlah Berita per Bulan")
            df_monthly = df.groupby(df['tanggal'].dt.to_period('M')).size().reset_index()
            df_monthly['tanggal'] = df_monthly['tanggal'].dt.to_timestamp()
            
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(df_monthly['tanggal'], df_monthly[0], marker='o', linewidth=2, markersize=4)
            ax.set_title("Jumlah Berita per Bulan")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Jumlah Berita")
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tampilkan info kolom dataset
            st.subheader("Informasi Dataset")
            st.write("**Kolom yang tersedia:**")
            col_info = pd.DataFrame({
                'Kolom': df.columns,
                'Tipe Data': df.dtypes,
                'Non-Null Count': df.count(),
                'Contoh Data': [str(df[col].iloc[0])[:50] + "..." if len(str(df[col].iloc[0])) > 50 else str(df[col].iloc[0]) for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        with tab2:
            st.markdown('<div class="sub-header">🔧 Preprocessing Teks</div>', unsafe_allow_html=True)
            
            # DIPERBAIKI - Conditional preprocessing
            if use_existing_stemmed:
                # Jika sudah ada stemmed_text, langsung gunakan
                st.success("✅ Dataset sudah memiliki kolom 'stemmed_text'")
                st.info("📊 Menggunakan data yang sudah diproses untuk hasil yang konsisten dengan notebook")
                
                # Tampilkan statistik data yang sudah ada
                st.subheader("📊 Statistik Data Preprocessing")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Dokumen", f"{len(df):,}")
                with col2:
                    avg_words = df['stemmed_text'].apply(lambda x: len(x.split())).mean()
                    st.metric("Rata-rata Kata", f"{avg_words:.1f}")
                with col3:
                    total_words = df['stemmed_text'].apply(lambda x: len(x.split())).sum()
                    st.metric("Total Kata", f"{total_words:,}")
                
                # Preview data
                st.subheader("Preview Data yang Sudah Diproses")
                sample_idx = st.selectbox("Pilih contoh berita:", range(min(10, len(df))))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Teks Asli:", df.iloc[sample_idx]['Content'][:500] + "...", height=200)
                with col2:
                    st.text_area("Stemmed Text (Siap untuk Model):", df.iloc[sample_idx]['stemmed_text'][:500] + "...", height=200)
                
                # Set flag bahwa preprocessing sudah selesai
                df['preprocessing_completed'] = True
                
            else:
                # Lakukan preprocessing seperti biasa
                # Initialize normalizer jika diperlukan
                if normalize_numbers:
                    try:
                        normalizer = create_normalizer()
                    except:
                        st.warning("⚠️ Indonesian number normalizer tidak tersedia, melanjutkan tanpa normalisasi angka")
                        normalize_numbers = False
                
                # Fungsi preprocessing
                def cleaning(teks):
                    if not isinstance(teks, str):
                        return ""
                    teks = re.sub(r'<.*?>', ' ', teks)
                    teks = re.sub(r'https?://\\S+|www\\.\\S+', ' ', teks)
                    if remove_ads:
                        teks = re.sub(r'ADVERTISEMENT.*?CONTENT', ' ', teks, flags=re.IGNORECASE | re.DOTALL)
                    teks = re.sub(r'[^\\w\\s\\d]', ' ', teks)
                    teks = re.sub(r'\\s+', ' ', teks).strip()
                    return teks
                
                def case_folding(teks):
                    return teks.lower()
                
                def normalisasi(teks):
                    if normalize_numbers:
                        try:
                            teks = normalizer.normalize_text(teks)
                        except:
                            pass
                    
                    slang_dict = {
                        'dgn': 'dengan', 'tdk': 'tidak', 'tsb': 'tersebut', 'utk': 'untuk',
                        'spy': 'supaya', 'krn': 'karena', 'jg': 'juga', 'bs': 'bisa',
                        'sdh': 'sudah', 'blm': 'belum', 'org': 'orang', 'yg': 'yang',
                        'sy': 'saya', 'dlm': 'dalam', 'pd': 'pada', 'dr': 'dari',
                        'kmrn': 'kemarin', 'skrg': 'sekarang', 'hrs': 'harus',
                        'msk': 'masuk', 'trs': 'terus', 'tp': 'tapi', 'kalo': 'kalau',
                        'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak', 'gk': 'tidak',
                        'thn': 'tahun', 'bln': 'bulan', 'sblm': 'sebelum', 'stlh': 'setelah',
                        'milyar': 'miliar', 'trilliun': 'triliun', 'jt': 'juta', 'rb': 'ribu',
                        '%': 'persen', 'usd': 'dolar amerika', 'rupiah': 'rupiah', 'rp': 'rupiah'
                    }
                    
                    pattern = r'\\b(' + '|'.join(slang_dict.keys()) + r')\\b'
                    def replace_match(match):
                        return slang_dict[match.group(0)]
                    return re.sub(pattern, replace_match, teks)
                
                def tokenzing(teks):
                    return word_tokenize(teks)
                
                # Download NLTK data jika belum ada
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    with st.spinner("Downloading NLTK data..."):
                        nltk.download('punkt')
                        nltk.download('stopwords')
                
                # Progress bar untuk preprocessing
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Melakukan preprocessing..."):
                    # Cleaning
                    status_text.text("🧹 Cleaning text...")
                    df['cleaned_text'] = df['Content'].fillna('').apply(cleaning)
                    progress_bar.progress(20)
                    
                    # Case folding
                    status_text.text("🔤 Case folding...")
                    df['case_folded_text'] = df['cleaned_text'].apply(case_folding)
                    progress_bar.progress(40)
                    
                    # Normalisasi
                    status_text.text("🔧 Normalisasi...")
                    df['normalized_text'] = df['case_folded_text'].apply(normalisasi)
                    progress_bar.progress(60)
                    
                    # Tokenisasi
                    status_text.text("✂️ Tokenisasi...")
                    df['tokens'] = df['normalized_text'].apply(tokenzing)
                    progress_bar.progress(70)
                    
                    # Stopwords removal
                    status_text.text("🚫 Menghapus stopwords...")
                    indo_stopwords = set(stopwords.words('indonesian'))
                    tambahan_stopwords = {
                        'ya', 'juga', 'dari', 'di', 'ke', 'pada', 'untuk', 'bagi', 'dan', 'atau', 
                        'tapi', 'namun', 'dengan', 'secara', 'oleh', 'karena', 'sehingga', 'agar',
                        'sebab', 'jika', 'bila', 'adalah', 'ini', 'itu', 'detik', 'kata', 'dalam',
                        'saat', 'akan', 'tidak', 'yang', 'belum', 'sudah', 'telah', 'bisa', 'dapat', 
                        'nya', 'pak', 'bu', 'hal', 'pun'
                    }
                    indo_stopwords.update(tambahan_stopwords)
                    
                    def remove_stopwords(tokens):
                        return [word for word in tokens if word.lower() not in indo_stopwords]
                    
                    df['filtered_tokens'] = df['tokens'].apply(remove_stopwords)
                    progress_bar.progress(80)
                    
                    # Stemming
                    if use_stemming:
                        status_text.text("🌱 Stemming...")
                        factory = StemmerFactory()
                        stemmer = factory.create_stemmer()
                        
                        def stemming(tokens):
                            return [stemmer.stem(word) for word in tokens]
                        
                        df['stemmed_tokens'] = df['filtered_tokens'].apply(stemming)
                        df['stemmed_text'] = df['stemmed_tokens'].apply(lambda x: ' '.join(x))
                    else:
                        df['stemmed_tokens'] = df['filtered_tokens']
                        df['stemmed_text'] = df['filtered_tokens'].apply(lambda x: ' '.join(x))
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Preprocessing selesai!")
                
                st.success("✅ Preprocessing berhasil!")
                
                # Tampilkan contoh hasil preprocessing
                st.subheader("Contoh Hasil Preprocessing")
                sample_idx = st.selectbox("Pilih contoh berita:", range(min(10, len(df))))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Teks Asli:", df.iloc[sample_idx]['Content'][:500] + "...", height=150)
                    st.text_area("Setelah Cleaning:", df.iloc[sample_idx]['cleaned_text'][:500] + "...", height=150)
                with col2:
                    st.text_area("Setelah Normalisasi:", df.iloc[sample_idx]['normalized_text'][:500] + "...", height=150)
                    st.text_area("Setelah Stemming:", df.iloc[sample_idx]['stemmed_text'][:500] + "...", height=150)
        
        with tab3:
            if has_faktor_column:
                st.markdown('<div class="sub-header">🏷️ Analisis Faktor Ekonomi (dari Dataset)</div>', unsafe_allow_html=True)
                
                # Baca faktor yang sudah ada di dataset
                faktor_col = faktor_columns[0]  # Ambil kolom faktor pertama
                st.info(f"📊 Menggunakan kolom faktor: **{faktor_col}**")
                
                # Analisis distribusi faktor
                if df[faktor_col].dtype == 'object':
                    # Jika faktor berupa string (contoh: "suku_bunga", "impor", dll)
                    faktor_counts = df[faktor_col].value_counts()
                    
                    st.subheader("Distribusi Faktor Ekonomi")
                    
                    col1, col2, col3 = st.columns(3)
                    for i, (faktor, count) in enumerate(faktor_counts.head(3).items()):
                        percentage = (count / len(df)) * 100
                        if i == 0:
                            with col1:
                                st.metric(f"{str(faktor).title()}", f"{count:,}", f"{percentage:.1f}%")
                        elif i == 1:
                            with col2:
                                st.metric(f"{str(faktor).title()}", f"{count:,}", f"{percentage:.1f}%")
                        else:
                            with col3:
                                st.metric(f"{str(faktor).title()}", f"{count:,}", f"{percentage:.1f}%")
                    
                    # Visualisasi
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Bar plot
                    bars = ax1.bar(range(len(faktor_counts)), faktor_counts.values, 
                                  color=['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd'][:len(faktor_counts)])
                    ax1.set_xticks(range(len(faktor_counts)))
                    ax1.set_xticklabels(faktor_counts.index, rotation=45, ha='right')
                    ax1.set_title("Distribusi Faktor Ekonomi")
                    ax1.set_ylabel("Jumlah Berita")
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                f'{int(height)}', ha='center', va='bottom')
                    
                    # Pie chart
                    ax2.pie(faktor_counts.values, labels=faktor_counts.index, autopct='%1.1f%%', startangle=90)
                    ax2.set_title("Proporsi Faktor Ekonomi")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Simpan informasi faktor untuk analisis selanjutnya
                    df['faktor_kategori'] = df[faktor_col]
                    
                else:
                    st.warning("⚠️ Format kolom faktor tidak dikenali. Menggunakan analisis umum.")
                
                # Analisis faktor vs sentimen
                st.subheader("Analisis Faktor vs Sentimen")
                
                if df[faktor_col].dtype == 'object':
                    # Cross-tabulation faktor vs sentimen
                    crosstab = pd.crosstab(df[faktor_col], df['label'], normalize='index') * 100
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    crosstab.plot(kind='bar', ax=ax, color=['#ff7f0e', '#1f77b4'])
                    ax.set_title("Distribusi Sentimen per Faktor Ekonomi (%)")
                    ax.set_ylabel("Persentase")
                    ax.set_xlabel("Faktor Ekonomi")
                    ax.legend(title="Sentimen")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Tabel statistik
                    st.write("**Tabel Distribusi Sentimen per Faktor:**")
                    crosstab_count = pd.crosstab(df[faktor_col], df['label'])
                    crosstab_count['Total'] = crosstab_count.sum(axis=1)
                    crosstab_count['% Positif'] = (crosstab_count['positif'] / crosstab_count['Total'] * 100).round(1)
                    st.dataframe(crosstab_count, use_container_width=True)
                
                # Contoh berita per faktor
                st.subheader("Contoh Berita per Faktor")
                
                if df[faktor_col].dtype == 'object':
                    unique_faktors = df[faktor_col].unique()
                    selected_faktor = st.selectbox("Pilih faktor:", unique_faktors)
                    
                    sample_berita = df[df[faktor_col] == selected_faktor].head(3)
                    for idx, row in sample_berita.iterrows():
                        with st.expander(f"📰 {row['Title'][:100]}..."):
                            st.write(f"**Tanggal:** {row['tanggal'].strftime('%d/%m/%Y')}")
                            st.write(f"**Label:** {row['label']}")
                            st.write(f"**Faktor:** {row[faktor_col]}")
                            st.write(f"**Konten:** {row['Content'][:300]}...")
            else:
                # Klasifikasi faktor otomatis (kode lama)
                st.markdown('<div class="sub-header">🏷️ Klasifikasi Faktor Ekonomi</div>', unsafe_allow_html=True)
                st.warning("⚠️ Kolom faktor tidak ditemukan, menggunakan klasifikasi otomatis...")
        with tab4:
            st.markdown('<div class="sub-header">🤖 Model ComplementNB + TF-IDF</div>', unsafe_allow_html=True)
            
            # Tampilkan parameter model yang digunakan
            st.subheader("⚙️ Parameter Model")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🔧 TF-IDF Parameters:**")
                st.write(f"- Max Features: {max_features}")
                st.write(f"- N-gram Range: ({ngram_min}, {ngram_max})")
                st.write(f"- Min DF: {min_df}")
                st.write(f"- Max DF: {max_df}")
            
            with col2:
                st.write("**🤖 ComplementNB Parameters:**")
                st.write(f"- Alpha: {alpha_value}")
                st.write(f"- Fit Prior: True")
                st.write(f"- Class Prior: None")
            
            with col3:
                st.write("**📊 Data Split:**")
                st.write(f"- Test Size: 20%")
                st.write(f"- Random State: 42")
                st.write(f"- Stratify: True")
            
            # Persiapan data untuk model - DIPERBAIKI untuk match notebook
            if use_existing_stemmed:
                st.info("🔄 Menggunakan preprocessing yang sama dengan notebook")
                # PERSIS seperti notebook
                df['lemmatized_text'] = df['stemmed_text']  # SAMA dengan notebook
                X_text = df['lemmatized_text']  # SAMA dengan notebook
            else:
                X_text = df['stemmed_text']

            y = df['label'].map({'positif': 1, 'negatif': 0})
            
            # Stopwords untuk TF-IDF
            stopword_factory = StopWordRemoverFactory()
            stopword_list = stopword_factory.get_stop_words()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_text, y, test_size=0.2, stratify=y, random_state=42
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Data", f"{len(df):,}")
            with col2:
                st.metric("Data Training", f"{len(X_train):,}")
            with col3:
                st.metric("Data Testing", f"{len(X_test):,}")
            with col4:
                pos_percentage = (y == 1).mean() * 100
                st.metric("% Positif", f"{pos_percentage:.1f}%")
            
            # Model training dengan ComplementNB + TF-IDF
            with st.spinner("Training ComplementNB + TF-IDF model..."):
                # Buat pipeline ComplementNB + TF-IDF
                model_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(
                        ngram_range=(ngram_min, ngram_max), 
                        max_features=max_features, 
                        min_df=min_df, 
                        max_df=max_df,
                        sublinear_tf=True, 
                        stop_words=stopword_list,
                        lowercase=True,
                        use_idf=True,
                        smooth_idf=True
                    )),
                    ('complementnb', ComplementNB(
                        alpha=alpha_value,
                        fit_prior=True,
                        class_prior=None
                    ))
                ])
                
                # Training
                model_pipeline.fit(X_train, y_train)
                
                # Prediksi
                y_pred = model_pipeline.predict(X_test)
                y_proba = model_pipeline.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_proba)
            
            st.success("✅ Model ComplementNB + TF-IDF berhasil dilatih!")
            
            # Tampilkan metrics
            st.subheader("📊 Performa Model ComplementNB + TF-IDF")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Akurasi", f"{accuracy:.3f}", f"{accuracy*100:.1f}%")
            with col2:
                st.metric("Precision", f"{precision:.3f}", f"{precision*100:.1f}%")
            with col3:
                st.metric("Recall", f"{recall:.3f}", f"{recall*100:.1f}%")
            with col4:
                st.metric("F1-Score", f"{f1_weighted:.3f}", f"{f1_weighted*100:.1f}%")
            with col5:
                st.metric("ROC AUC", f"{roc_auc:.3f}", f"{roc_auc*100:.1f}%")
            
            # PERBANDINGAN DENGAN NOTEBOOK - TAMBAHAN BARU
            st.subheader("🔍 Perbandingan dengan Hasil Notebook")

            expected_results = {
                'Accuracy': 0.7004,
                'Precision': 0.7021, 
                'Recall': 0.7004,
                'F1-Score': 0.6987,
                'ROC AUC': 0.7260  # Dari confusion matrix precision negatif
            }

            comparison_data = []
            actual_results = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall, 
                'F1-Score': f1_weighted,
                'ROC AUC': roc_auc
            }

            for metric in expected_results.keys():
                expected = expected_results[metric]
                actual = actual_results[metric]
                difference = abs(actual - expected)
                comparison_data.append({
                    'Metric': metric,
                    'Notebook': f"{expected:.4f}",
                    'Streamlit': f"{actual:.4f}",
                    'Difference': f"{difference:.4f}",
                    'Match': "✅" if difference < 0.01 else "⚠️" if difference < 0.05 else "❌"
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

            # Overall assessment
            overall_diff = sum([abs(actual_results[k] - expected_results[k]) for k in expected_results.keys()]) / len(expected_results)

            if overall_diff < 0.01:
                st.success("🎉 **EXCELLENT**: Hasil sangat mirip dengan notebook! (rata-rata selisih < 1%)")
            elif overall_diff < 0.03:
                st.info("✅ **GOOD**: Hasil cukup mirip dengan notebook (rata-rata selisih < 3%)")
            elif overall_diff < 0.05:
                st.warning("⚠️ **FAIR**: Ada perbedaan dengan notebook (rata-rata selisih < 5%)")
            else:
                st.error("❌ **POOR**: Perbedaan signifikan dengan notebook - periksa data dan parameter")

            # Tips untuk improvement
            if overall_diff > 0.01:
                st.info("""
                💡 **Tips untuk hasil yang lebih mirip:**
                - Pastikan menggunakan dataset yang **persis sama** dengan notebook
                - Aktifkan **"Gunakan Parameter Notebook"** di sidebar
                - Jika dataset memiliki kolom `stemmed_text`, pastikan digunakan
                - Periksa apakah ada perbedaan dalam preprocessing data
                """)
            
            # Interpretasi performa
            st.subheader("🎯 Interpretasi Performa")
            
            performance_interpretation = []
            if accuracy >= 0.80:
                performance_interpretation.append("🎉 **Akurasi Excellent** (≥80%): Model sangat baik dalam memprediksi sentimen")
            elif accuracy >= 0.70:
                performance_interpretation.append("✅ **Akurasi Good** (70-80%): Model cukup baik untuk aplikasi praktis")
            elif accuracy >= 0.60:
                performance_interpretation.append("⚠️ **Akurasi Fair** (60-70%): Model perlu perbaikan")
            else:
                performance_interpretation.append("❌ **Akurasi Poor** (<60%): Model memerlukan perbaikan signifikan")
            
            if roc_auc >= 0.80:
                performance_interpretation.append("🎯 **ROC AUC Excellent**: Kemampuan diskriminasi sangat baik")
            elif roc_auc >= 0.70:
                performance_interpretation.append("👍 **ROC AUC Good**: Kemampuan diskriminasi baik")
            else:
                performance_interpretation.append("⚠️ **ROC AUC Fair**: Kemampuan diskriminasi perlu diperbaiki")
            
            for interpretation in performance_interpretation:
                st.markdown(interpretation)
            
            # Confusion Matrix dan ROC Curve
            cm = confusion_matrix(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Negatif', 'Positif'], 
                           yticklabels=['Negatif', 'Positif'], ax=ax)
                ax.set_title('Confusion Matrix - ComplementNB + TF-IDF')
                ax.set_xlabel('Prediksi')
                ax.set_ylabel('Aktual')
                st.pyplot(fig)
                
                # Detail confusion matrix
                tn, fp, fn, tp = cm.ravel()
                st.write("**Detail Confusion Matrix:**")
                st.write(f"- True Negative: {tn}")
                st.write(f"- False Positive: {fp}")
                st.write(f"- False Negative: {fn}")
                st.write(f"- True Positive: {tp}")
            
            with col2:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ComplementNB + TF-IDF (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve - ComplementNB + TF-IDF')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Metrics per class
                precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
                precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
                recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
                recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                st.write("**Metrics per Kelas:**")
                st.write(f"**Positif:**")
                st.write(f"- Precision: {precision_pos:.3f}")
                st.write(f"- Recall: {recall_pos:.3f}")
                st.write(f"**Negatif:**")
                st.write(f"- Precision: {precision_neg:.3f}")
                st.write(f"- Recall: {recall_neg:.3f}")
            
            # Performance metrics bar chart
            st.subheader("📊 Ringkasan Performa Metrics")
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
            metrics_values = [accuracy, precision, recall, f1_weighted, roc_auc]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(metrics_names, metrics_values, 
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'], alpha=0.8)
            ax.set_ylim([0, 1])
            ax.set_title('Ringkasan Performa ComplementNB + TF-IDF', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score')
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # TF-IDF Feature Analysis
            if show_detailed_metrics:
                st.subheader("🔍 Analisis TF-IDF Features")
                
                try:
                    # Get feature names dari TF-IDF vectorizer
                    feature_names = model_pipeline.named_steps['tfidf'].get_feature_names_out()
                    
                    # Get feature log probabilities dari ComplementNB
                    feature_log_prob = model_pipeline.named_steps['complementnb'].feature_log_prob_
                    
                    # Top features untuk kelas positif dan negatif
                    pos_features_idx = np.argsort(feature_log_prob[1])[-20:][::-1]  # Top 20 positif
                    neg_features_idx = np.argsort(feature_log_prob[0])[-20:][::-1]  # Top 20 negatif
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🟢 Top 20 Features untuk Sentimen Positif:**")
                        pos_features = [(feature_names[i], feature_log_prob[1][i]) for i in pos_features_idx]
                        pos_df = pd.DataFrame(pos_features, columns=['Feature', 'Log_Probability'])
                        st.dataframe(pos_df.round(4), use_container_width=True)
                    
                    with col2:
                        st.write("**🔴 Top 20 Features untuk Sentimen Negatif:**")
                        neg_features = [(feature_names[i], feature_log_prob[0][i]) for i in neg_features_idx]
                        neg_df = pd.DataFrame(neg_features, columns=['Feature', 'Log_Probability'])
                        st.dataframe(neg_df.round(4), use_container_width=True)
                    
                    # Visualisasi top features
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
                    
                    # Top positive features
                    ax1.barh(range(10), [prob for _, prob in pos_features[:10]], color='green', alpha=0.7)
                    ax1.set_yticks(range(10))
                    ax1.set_yticklabels([feat for feat, _ in pos_features[:10]])
                    ax1.set_title('Top 10 Features - Sentimen Positif')
                    ax1.set_xlabel('Log Probability')
                    
                    # Top negative features
                    ax2.barh(range(10), [prob for _, prob in neg_features[:10]], color='red', alpha=0.7)
                    ax2.set_yticks(range(10))
                    ax2.set_yticklabels([feat for feat, _ in neg_features[:10]])
                    ax2.set_title('Top 10 Features - Sentimen Negatif')
                    ax2.set_xlabel('Log Probability')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.info("Analisis TF-IDF features tidak tersedia untuk model ini")
            
            # Prediksi untuk seluruh dataset
            with st.spinner("Melakukan prediksi untuk seluruh dataset..."):
                df['sentimen_pred'] = model_pipeline.predict(X_text)
                df['sentimen_proba'] = model_pipeline.predict_proba(X_text)[:, 1]
            
            st.success("✅ Prediksi ComplementNB + TF-IDF selesai!")
            
            # Classification Report
            st.subheader("📋 Classification Report Detail")
            report = classification_report(y_test, y_pred, target_names=['Negatif', 'Positif'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)
            
            # Model comparison dengan baseline
            st.subheader("🆚 Perbandingan dengan Baseline")
            
            # Baseline: Majority class classifier
            majority_class = y_train.mode()[0]
            baseline_pred = [majority_class] * len(y_test)
            baseline_accuracy = accuracy_score(y_test, baseline_pred)
            
            comparison_data = {
                'Model': ['Baseline (Majority Class)', 'ComplementNB + TF-IDF'],
                'Accuracy': [baseline_accuracy, accuracy],
                'Improvement': [0, accuracy - baseline_accuracy]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.round(4), use_container_width=True)
            
            improvement_pct = ((accuracy - baseline_accuracy) / baseline_accuracy) * 100
            st.success(f"🎯 **Improvement**: {improvement_pct:.1f}% lebih baik dari baseline!")
        
        with tab5:
            st.markdown('<div class="sub-header">📈 Visualisasi & Analisis Korelasi</div>', unsafe_allow_html=True)
            
            # Download data yfinance
            @st.cache_data
            def download_yfinance_data():
                try:
                    st.info("📥 Mengunduh data USD/IDR dari Yahoo Finance...")
                    yf_data = yf.download('IDR=X', start='2020-01-01', end='2024-12-31')
                    
                    if hasattr(yf_data.columns, 'droplevel'):
                        yf_data.columns = yf_data.columns.droplevel(1)
                    
                    df_yfinance = yf_data[['Close']].reset_index()
                    df_yfinance.rename(columns={'Date': 'tanggal', 'Close': 'kurs_yfinance'}, inplace=True)
                    df_yfinance['yf_change'] = df_yfinance['kurs_yfinance'].pct_change() * 100
                    df_yfinance = df_yfinance.dropna().reset_index(drop=True)
                    
                    return df_yfinance
                except Exception as e:
                    st.error(f"❌ Error mengunduh data yfinance: {e}")
                    return None
            
            df_yfinance = download_yfinance_data()
            
            if df_yfinance is not None:
                # Agregasi data harian
                df_daily_news = df.groupby('tanggal').agg({
                    'sentimen_pred': lambda x: (x == 1).mean(),
                    'sentimen_proba': 'mean',
                    'valas': 'first',
                    'Title': 'count'
                }).reset_index()
                
                df_daily_news.rename(columns={
                    'sentimen_pred': 'sentimen_score',
                    'sentimen_proba': 'sentimen_probability', 
                    'valas': 'kurs_dataset',
                    'Title': 'jumlah_berita'
                }, inplace=True)
                
                # Tambahkan informasi faktor jika ada
                if has_faktor_column:
                    # Agregasi faktor per hari
                    faktor_daily = df.groupby('tanggal')[faktor_columns[0]].apply(
                        lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
                    ).reset_index()
                    df_daily_news = pd.merge(df_daily_news, faktor_daily, on='tanggal', how='left')
                
                df_daily_news = df_daily_news.sort_values('tanggal').reset_index(drop=True)
                df_daily_news['dataset_change'] = df_daily_news['kurs_dataset'].pct_change() * 100
                
                # Merge data
                df_merged = pd.merge(df_daily_news, df_yfinance, on='tanggal', how='inner')
                
                # Data cleaning
                df_merged = df_merged[
                    (df_merged['jumlah_berita'] >= 1) &
                    (~pd.isna(df_merged['sentimen_score'])) &
                    (~pd.isna(df_merged['yf_change'])) &
                    (~pd.isna(df_merged['dataset_change'])) &
                    (~np.isinf(df_merged['yf_change'])) &
                    (~np.isinf(df_merged['dataset_change'])) &
                    (df_merged['sentimen_score'] >= 0) &
                    (df_merged['sentimen_score'] <= 1) &
                    (abs(df_merged['yf_change']) < 10) &
                    (abs(df_merged['dataset_change']) < 50)
                ].copy()
                
                st.success(f"✅ Data berhasil digabungkan: {len(df_merged):,} hari")
                
                 # Statistik deskriptif
                st.subheader("📊 Statistik Deskriptif")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rata-rata Sentimen", f"{df_merged['sentimen_score'].mean():.3f}")
                with col2:
                    st.metric("Std Dev Sentimen", f"{df_merged['sentimen_score'].std():.3f}")
                with col3:
                    st.metric("Rata-rata YF Change", f"{df_merged['yf_change'].mean():.3f}%")
                with col4:
                    st.metric("Std Dev YF Change", f"{df_merged['yf_change'].std():.3f}%")
                
                # Fungsi korelasi yang aman
                def safe_correlation(x, y, name):
                    try:
                        mask = ~(pd.isna(x) | pd.isna(y) | np.isinf(x) | np.isinf(y))
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        if len(x_clean) < 10:
                            return np.nan, np.nan, f"Data tidak cukup ({len(x_clean)} points)"
                        
                        if x_clean.var() == 0 or y_clean.var() == 0:
                            return np.nan, np.nan, "Variance nol"
                        
                        corr, p_val = pearsonr(x_clean, y_clean)
                        return corr, p_val, "OK"
                    
                    except Exception as e:
                        return np.nan, np.nan, f"Error: {str(e)}"
                
                # Hitung korelasi
                corr_sent_yf, p_sent_yf, status1 = safe_correlation(
                    df_merged['sentimen_score'], df_merged['yf_change'], "Sentimen vs yfinance"
                )
                
                corr_sent_dataset, p_sent_dataset, status2 = safe_correlation(
                    df_merged['sentimen_score'], df_merged['dataset_change'], "Sentimen vs Dataset"
                )
                
                corr_yf_dataset, p_yf_dataset, status3 = safe_correlation(
                    df_merged['kurs_yfinance'], df_merged['kurs_dataset'], "yfinance vs Dataset"
                )
                
                # Tampilkan hasil korelasi
                st.subheader("🔗 Analisis Korelasi ComplementNB + TF-IDF")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Sentimen vs yfinance", 
                        f"{corr_sent_yf:.4f}" if not np.isnan(corr_sent_yf) else "N/A",
                        f"p = {p_sent_yf:.4f}" if not np.isnan(p_sent_yf) else status1
                    )
                    if not np.isnan(p_sent_yf) and p_sent_yf < 0.05:
                        st.success("✅ Signifikan")
                    else:
                        st.warning("❌ Tidak Signifikan")
                
                with col2:
                    st.metric(
                        "Sentimen vs Dataset", 
                        f"{corr_sent_dataset:.4f}" if not np.isnan(corr_sent_dataset) else "N/A",
                        f"p = {p_sent_dataset:.4f}" if not np.isnan(p_sent_dataset) else status2
                    )
                    if not np.isnan(p_sent_dataset) and p_sent_dataset < 0.05:
                        st.success("✅ Signifikan")
                    else:
                        st.warning("❌ Tidak Signifikan")
                
                with col3:
                    st.metric(
                        "yfinance vs Dataset", 
                        f"{corr_yf_dataset:.4f}" if not np.isnan(corr_yf_dataset) else "N/A",
                        f"p = {p_yf_dataset:.4f}" if not np.isnan(p_yf_dataset) else status3
                    )
                    if not np.isnan(p_yf_dataset) and p_yf_dataset < 0.05:
                        st.success("✅ Signifikan")
                    else:
                        st.warning("❌ Tidak Signifikan")
                
                # Visualisasi Time Series dengan Matplotlib
                st.subheader("📈 Time Series Analysis")
                
                # Plot 1: Sentimen Time Series
                fig, ax = plt.subplots(figsize=(15, 6))
                ax.plot(df_merged['tanggal'], df_merged['sentimen_score'], 
                       color='blue', linewidth=2, alpha=0.8, label='Sentimen Score (ComplementNB)')
                ax.fill_between(df_merged['tanggal'], df_merged['sentimen_score'], 
                               alpha=0.3, color='blue')
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Netral (50%)')
                ax.set_title('Time Series Sentimen Berita (ComplementNB + TF-IDF)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Sentimen Score (0-1)')
                ax.set_xlabel('Tanggal')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Plot 2: Kurs Comparison
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
                
                # yfinance
                ax1.plot(df_merged['tanggal'], df_merged['kurs_yfinance'], 
                        color='red', linewidth=2, alpha=0.8, label='yfinance')
                ax1.set_title('Kurs USD/IDR - yfinance', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Kurs (IDR/USD)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Dataset
                ax2.plot(df_merged['tanggal'], df_merged['kurs_dataset'], 
                        color='orange', linewidth=2, alpha=0.8, label='Dataset')
                ax2.set_title('Kurs USD/IDR - Dataset', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Kurs (IDR/USD)')
                ax2.set_xlabel('Tanggal')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Format x-axis
                for ax in [ax1, ax2]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Scatter Plots
                st.subheader("🔍 Scatter Plot Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if not np.isnan(corr_sent_yf):
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(df_merged['sentimen_score'], df_merged['yf_change'], 
                                  alpha=0.6, s=50, c='red')
                        ax.set_xlabel('Sentimen Score (ComplementNB)')
                        ax.set_ylabel('yfinance Change (%)')
                        ax.set_title(f'Sentimen vs yfinance Change\nr = {corr_sent_yf:.3f}', 
                                    fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        # Add trendline
                        z = np.polyfit(df_merged['sentimen_score'], df_merged['yf_change'], 1)
                        p = np.poly1d(z)
                        ax.plot(df_merged['sentimen_score'], p(df_merged['sentimen_score']), 
                               "r--", alpha=0.8, linewidth=2)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Data tidak cukup untuk scatter plot sentimen vs yfinance")
                
                with col2:
                    if not np.isnan(corr_sent_dataset):
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(df_merged['sentimen_score'], df_merged['dataset_change'], 
                                  alpha=0.6, s=50, c='orange')
                        ax.set_xlabel('Sentimen Score (ComplementNB)')
                        ax.set_ylabel('Dataset Change (%)')
                        ax.set_title(f'Sentimen vs Dataset Change\nr = {corr_sent_dataset:.3f}', 
                                    fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        # Add trendline
                        z = np.polyfit(df_merged['sentimen_score'], df_merged['dataset_change'], 1)
                        p = np.poly1d(z)
                        ax.plot(df_merged['sentimen_score'], p(df_merged['sentimen_score']), 
                               "orange", linestyle="--", alpha=0.8, linewidth=2)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Data tidak cukup untuk scatter plot sentimen vs dataset")
                
                # Analisis per Faktor (jika ada)
                if has_faktor_column and faktor_columns[0] in df_merged.columns:
                    st.subheader("📊 Analisis Sentimen per Faktor Ekonomi")
                    
                    # Boxplot sentimen per faktor
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Sentimen per faktor
                    faktor_sentimen = []
                    faktors = df_merged[faktor_columns[0]].unique()
                    
                    for faktor in faktors:
                        faktor_data = df_merged[df_merged[faktor_columns[0]] == faktor]['sentimen_score']
                        faktor_sentimen.append(faktor_data)
                    
                    ax1.boxplot(faktor_sentimen, labels=faktors)
                    ax1.set_title('Distribusi Sentimen per Faktor')
                    ax1.set_ylabel('Sentimen Score')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.grid(True, alpha=0.3)
                    
                    # Rata-rata sentimen per faktor
                    avg_sentimen = df_merged.groupby(faktor_columns[0])['sentimen_score'].mean()
                    bars = ax2.bar(avg_sentimen.index, avg_sentimen.values, 
                                  color=['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728'][:len(avg_sentimen)])
                    ax2.set_title('Rata-rata Sentimen per Faktor')
                    ax2.set_ylabel('Rata-rata Sentimen Score')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, value in zip(bars, avg_sentimen.values):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Heatmap korelasi
                st.subheader("🔥 Correlation Heatmap")
                
                correlation_columns = ['sentimen_score', 'sentimen_probability', 'yf_change', 'dataset_change', 'jumlah_berita']
                correlation_data = df_merged[correlation_columns].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0,
                           square=True, ax=ax, cbar_kws={"shrink": .8})
                ax.set_title('Correlation Matrix - ComplementNB + TF-IDF', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary insights
                st.subheader("💡 Insights & Summary")
                
                insights = []
                
                if not np.isnan(corr_sent_yf):
                    if abs(corr_sent_yf) > 0.3:
                        insights.append(f"🔍 **Korelasi Sentimen vs yfinance**: {corr_sent_yf:.3f} - korelasi {'kuat' if abs(corr_sent_yf) > 0.5 else 'sedang'}")
                    else:
                        insights.append(f"🔍 **Korelasi Sentimen vs yfinance**: {corr_sent_yf:.3f} - korelasi lemah")
                
                if not np.isnan(corr_sent_dataset):
                    if abs(corr_sent_dataset) > 0.3:
                        insights.append(f"🔍 **Korelasi Sentimen vs Dataset**: {corr_sent_dataset:.3f} - korelasi {'kuat' if abs(corr_sent_dataset) > 0.5 else 'sedang'}")
                    else:
                        insights.append(f"🔍 **Korelasi Sentimen vs Dataset**: {corr_sent_dataset:.3f} - korelasi lemah")
                
                avg_sentiment = df_merged['sentimen_score'].mean()
                if avg_sentiment > 0.6:
                    insights.append(f"📈 **Sentimen rata-rata**: {avg_sentiment:.3f} - cenderung positif")
                elif avg_sentiment < 0.4:
                    insights.append(f"📉 **Sentimen rata-rata**: {avg_sentiment:.3f} - cenderung negatif")
                else:
                    insights.append(f"⚖️ **Sentimen rata-rata**: {avg_sentiment:.3f} - relatif netral")
                
                insights.append(f"📊 **Total data analisis**: {len(df_merged):,} hari")
                insights.append(f"📰 **Rata-rata berita per hari**: {df_merged['jumlah_berita'].mean():.1f}")
                insights.append(f"🤖 **Model yang digunakan**: ComplementNB + TF-IDF")
                insights.append(f"🎯 **Akurasi model**: {accuracy:.3f} ({accuracy*100:.1f}%)")
                
                for insight in insights:
                    st.markdown(insight)
                
                # Download hasil
                st.subheader("💾 Download Hasil")
                
                # Prepare download data
                download_data = df_merged[['tanggal', 'sentimen_score', 'sentimen_probability', 
                                         'kurs_yfinance', 'kurs_dataset', 'yf_change', 
                                         'dataset_change', 'jumlah_berita']]
                
                if has_faktor_column and faktor_columns[0] in df_merged.columns:
                    download_data[faktor_columns[0]] = df_merged[faktor_columns[0]]
                
                csv = download_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data Hasil Analisis (CSV)",
                    data=csv,
                    file_name=f"analisis_sentimen_complementnb_hasil_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.error("❌ Tidak dapat mengunduh data yfinance. Silakan coba lagi nanti.")
    
    except Exception as e:
        st.error(f"❌ Error saat memuat data: {str(e)}")
        st.error("Pastikan file CSV memiliki kolom: Title, Content, date, label, valas")
        st.error("Kolom faktor (opsional) akan digunakan jika tersedia")

else:
    st.info("👆 Silakan upload file CSV untuk memulai analisis")
    
    # Tampilkan instruksi
    st.markdown("""
    ## 📋 Instruksi Penggunaan
    
    1. **Upload Dataset**: Upload file CSV dengan kolom berikut:
       - `Title`: Judul berita
       - `Content`: Isi berita
       - `date`: Tanggal (format: DD/MM/YYYY)
       - `label`: Label sentimen (positif/negatif)
       - `valas`: Nilai tukar rupiah
       - `stemmed_text`: Teks yang sudah diproses (opsional - untuk hasil identik dengan notebook)
       - `faktor`: Faktor ekonomi (opsional - jika ada akan dibaca langsung)
    
    2. **Konfigurasi**: Sesuaikan parameter preprocessing dan model di sidebar
       - **RECOMMENDED**: Aktifkan "Gunakan Parameter Notebook" untuk hasil yang identik
    
    3. **Analisis**: Aplikasi akan melakukan:
       - Preprocessing teks (cleaning, normalisasi, stemming) atau menggunakan data yang sudah diproses
       - Analisis faktor ekonomi (dari dataset atau klasifikasi otomatis)
       - Training model **ComplementNB + TF-IDF** dengan parameter optimal
       - Analisis korelasi dengan data kurs
       - Visualisasi hasil
    
    ## 🔧 Fitur Utama
    
    - **Smart Preprocessing**: Deteksi otomatis data yang sudah diproses untuk konsistensi dengan notebook
    - **Parameter Notebook Mode**: Gunakan parameter yang sama dengan notebook untuk hasil identik
    - **Analisis Faktor**: Membaca faktor dari dataset atau klasifikasi otomatis
    - **Machine Learning**: **ComplementNB + TF-IDF** untuk prediksi sentimen
    - **Real-time Comparison**: Perbandingan hasil dengan notebook secara real-time
    - **Visualisasi**: Time series, scatter plots, correlation heatmap
    - **Analisis Korelasi**: Korelasi antara sentimen berita dan pergerakan kurs
    
    ## 🤖 Algoritma yang Digunakan
    
    - **ComplementNB (Complement Naive Bayes)**: Algoritma yang cocok untuk data imbalanced
    - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Feature extraction untuk teks
    - **Pipeline**: Integrasi seamless antara TF-IDF dan ComplementNB
    - **Parameter Optimal**: Menggunakan parameter yang telah dioptimasi dari notebook
    
    ## 📊 Expected Results
    
    - **Accuracy**: 70.04% (0.7004)
    - **Precision**: 70.21% (0.7021)
    - **Recall**: 70.04% (0.7004)
    - **F1-Score**: 69.87% (0.6987)
    - **ROC AUC**: 72.60% (0.7260)
    
    ## 💡 Tips untuk Hasil Optimal
    
    - **Gunakan dataset yang sama** dengan notebook untuk hasil identik
    - **Aktifkan "Gunakan Parameter Notebook"** di sidebar
    - **Jika dataset memiliki kolom `stemmed_text`**, aplikasi akan menggunakannya untuk konsistensi
    - **Enable Debug Mode** untuk troubleshooting jika diperlukan
    - **Pastikan format tanggal DD/MM/YYYY** untuk parsing yang benar
    
    ## 🚀 Kelebihan Aplikasi Ini
    
    ✅ **Automatic Detection**: Deteksi data yang sudah diproses  
    ✅ **Parameter Matching**: Mode parameter notebook untuk hasil identik  
    ✅ **Real-time Validation**: Perbandingan hasil dengan target notebook  
    ✅ **Smart Preprocessing**: Skip preprocessing jika data sudah siap  
    ✅ **Debug Mode**: Tools untuk troubleshooting dan optimization  
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📈 Aplikasi Analisis Sentimen Berita Ekonomi Indonesia<br>
    </div>
    """, 
    unsafe_allow_html=True
)