import os
import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================
# SAYFA AYARLARI
# =========================================
st.set_page_config(
    page_title="Kelimelerin Genetik Haritası",
    page_icon="🔤",
    layout="wide"
)

# =========================================
# DOSYA YOLLARI
# =========================================
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model")

DATASET_PATH = os.path.join(MODEL_DIR, "dataset.csv")
VECT_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLF_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
LBL_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# =========================================
# KÖKEN ADLARINI TÜRKÇELEŞTİRME
# =========================================
ORIGIN_TURKCE = {
    "Turkish": "Türkçe",
    "Arabic": "Arapça",
    "Persian": "Farsça",
    "English": "İngilizce",
    "French": "Fransızca",
    "Italian": "İtalyanca",
    "Greek": "Yunanca",
    "Latin": "Latince",
    "Chinese": "Çince",
    "Rumca": "Rumca",
    "Russian": "Rusça",
    "Unknown": "Bilinmiyor"
}

# =========================================
# KÖKEN RENKLERİ
# =========================================
ORIGIN_COLORS = {
    "Türkçe": "#315efb",
    "Arapça": "#16a34a",
    "Farsça": "#7c3aed",
    "İngilizce": "#0891b2",
    "Fransızca": "#dc2626",
    "İtalyanca": "#f59e0b",
    "Yunanca": "#8b5cf6",
    "Latince": "#64748b",
    "Çince": "#0f766e",
    "Rumca": "#9333ea",
    "Rusça": "#ea580c",
    "Bilinmiyor": "#94a3b8"
}

# =========================================
# ÖRNEK CÜMLELER
# =========================================
DEMO_SENTENCES = [
    "Okulda kitap okuyup pencerenin yanında elma yedim.",
    "Akşam evde kitap okuyup çay içtim.",
    "Dersten sonra kafede kahve içtik.",
    "Tren ile okula giderken telefonumdan müzik dinledim.",
    "Marketten meyve alıp eve geldim.",
    "Bahçede oturup arkadaşlarımla sohbet ettim.",
    "İnternetten video izledim.",
    "Okuldan sonra spora gidip yemek yedim.",
    "Ders çalıştıktan sonra biraz müzik dinledim."
]

# =========================================
# STİL
# =========================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f6f8fc 0%, #eef3f9 100%);
}

.block-container {
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}

/* Gereksiz üst boşlukları azalt */
[data-testid="stHeader"] {
    background: transparent;
}

.hero-box {
    background: white;
    border: 1px solid #e5ebf2;
    border-radius: 24px;
    box-shadow: 0 12px 30px rgba(24, 33, 47, 0.08);
    padding: 28px;
    margin-bottom: 18px;
}

.hero-wrap {
    display: flex;
    align-items: center;
    gap: 18px;
}

.hero-icon {
    width: 72px;
    height: 72px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #315efb, #6d8cff);
    color: white;
    font-size: 28px;
    font-weight: 800;
    flex-shrink: 0;
}

.hero-title {
    font-size: 2.35rem;
    font-weight: 800;
    color: #18212f;
    margin-bottom: 6px;
    line-height: 1.1;
}

.hero-subtitle {
    color: #5f6b7a;
    font-size: 1rem;
    line-height: 1.6;
}

.section-box {
    background: white;
    border: 1px solid #e5ebf2;
    border-radius: 24px;
    box-shadow: 0 12px 30px rgba(24, 33, 47, 0.08);
    padding: 24px;
    margin-bottom: 20px;
}

.small-help {
    color: #5f6b7a;
    font-size: 0.98rem;
    line-height: 1.6;
    margin-bottom: 8px;
}

.demo-label {
    color: #18212f;
    font-size: 1rem;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 10px;
}

.input-note {
    background: #eef4ff;
    color: #24407a;
    border: 1px solid #d6e4ff;
    border-radius: 14px;
    padding: 12px 14px;
    margin: 14px 0 14px 0;
    font-size: 0.98rem;
    line-height: 1.5;
    font-weight: 500;
}

.stat-card {
    background: #f8fbff;
    border: 1px solid #e5ebf2;
    border-radius: 18px;
    padding: 18px;
}

.stat-label {
    color: #5f6b7a;
    font-size: 0.92rem;
    margin-bottom: 8px;
}

.stat-value {
    color: #18212f;
    font-size: 1.8rem;
    font-weight: 800;
}

.word-card {
    background: #ffffff;
    border: 1px solid #e5ebf2;
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 14px;
}

.word-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
}

.word-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #18212f;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
    white-space: nowrap;
}

.meta-line {
    color: #5f6b7a;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 4px;
}

.meta-line strong {
    color: #18212f;
}

.legend-row {
    font-size: 0.95rem;
    margin-bottom: 8px;
    color: #18212f;
}

.legend-dot {
    font-weight: 900;
    margin-right: 6px;
}

.footer-note {
    color: #5f6b7a;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* TEXTAREA */
textarea {
    border: 2px solid #cfd8e6 !important;
    border-radius: 16px !important;
    background: #ffffff !important;
    font-size: 18px !important;
    line-height: 1.6 !important;
    padding: 14px !important;
}

textarea:focus {
    border: 2px solid #315efb !important;
    box-shadow: 0 0 0 5px rgba(49, 94, 251, 0.15) !important;
}

/* Genel butonlar */
.stButton > button {
    border-radius: 12px;
    min-height: 46px;
    font-weight: 700;
    transition: all 0.2s ease;
}

/* Demo butonları */
.demo-button button {
    border-radius: 16px !important;
    height: 52px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.12) !important;
}

.demo-button button:hover {
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 10px 24px rgba(0,0,0,0.18) !important;
    opacity: 0.95;
}

.btn1 button { background: linear-gradient(135deg, #4f46e5, #6366f1) !important; }
.btn2 button { background: linear-gradient(135deg, #16a34a, #22c55e) !important; }
.btn3 button { background: linear-gradient(135deg, #dc2626, #ef4444) !important; }
.btn4 button { background: linear-gradient(135deg, #ea580c, #f97316) !important; }
.btn5 button { background: linear-gradient(135deg, #0891b2, #06b6d4) !important; }
.btn6 button { background: linear-gradient(135deg, #7c3aed, #a855f7) !important; }
.btn7 button { background: linear-gradient(135deg, #0f766e, #14b8a6) !important; }
.btn8 button { background: linear-gradient(135deg, #b91c1c, #f43f5e) !important; }
.btn9 button { background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important; }
</style>
""", unsafe_allow_html=True)

# =========================================
# YÜKLEME FONKSİYONLARI
# =========================================
@st.cache_data
def load_lexicon(path=DATASET_PATH):
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path, sep=";")

    required_cols = {"word", "origin", "notes"}
    if not required_cols.issubset(df.columns):
        return {}

    df["word"] = df["word"].astype(str).str.strip().str.lower()
    df["origin"] = df["origin"].astype(str).str.strip()
    df["notes"] = df["notes"].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["word", "origin"])
    df = df[df["word"] != ""]
    df = df.drop_duplicates(subset=["word"], keep="first")

    lex = {}
    for _, row in df.iterrows():
        lex[row["word"]] = {
            "origin": row["origin"],
            "notes": row["notes"]
        }
    return lex

@st.cache_resource
def load_model():
    vect, clf, lbl = None, None, None
    try:
        if os.path.exists(VECT_PATH):
            vect = joblib.load(VECT_PATH)
        if os.path.exists(CLF_PATH):
            clf = joblib.load(CLF_PATH)
        if os.path.exists(LBL_PATH):
            lbl = joblib.load(LBL_PATH)
    except Exception:
        pass
    return vect, clf, lbl

# =========================================
# METİN İŞLEME
# =========================================
TOKEN_RE = re.compile(r"[A-Za-zÇÖÜĞİŞçöüğışİıÂâÊêÎîÔôÛû]+", re.UNICODE)

def tokenize(text):
    tokens = TOKEN_RE.findall(text)
    return [t for t in tokens if t.strip()]

def normalize_word(word):
    w = word.strip().lower()

    if not w:
        return w

    noun_suffixes = [
        "lardan", "lerden",
        "ların", "lerin",
        "lara", "lere",
        "larda", "lerde",
        "dan", "den", "tan", "ten",
        "lar", "ler",
        "da", "de", "ta", "te",
        "ya", "ye",
        "yı", "yi", "yu", "yü",
        "nın", "nin", "nun", "nün",
        "ın", "in", "un", "ün",
        "ımız", "imiz", "umuz", "ümüz",
        "mız", "miz", "muz", "müz",
        "sı", "si", "su", "sü",
        "ım", "im", "um", "üm",
        "ı", "i", "u", "ü",
        "a", "e",
        "m", "n"
    ]

    verb_suffixes = [
        "iyorum", "ıyorum", "uyorum", "üyorum",
        "iyoruz", "ıyoruz", "uyoruz", "üyoruz",
        "iyorsun", "ıyorsun", "uyorsun", "üyorsun",
        "iyor", "ıyor", "uyor", "üyor",
        "dım", "dim", "dum", "düm",
        "tım", "tim", "tum", "tüm",
        "dın", "din", "dun", "dün",
        "tı", "ti", "tu", "tü",
        "dı", "di", "du", "dü",
        "acak", "ecek",
        "malı", "meli",
        "mak", "mek"
    ]

    for suf in sorted(verb_suffixes, key=len, reverse=True):
        if w.endswith(suf) and len(w) > len(suf) + 1:
            candidate = w[:-len(suf)]
            if candidate:
                return candidate

    for suf in sorted(noun_suffixes, key=len, reverse=True):
        if w.endswith(suf) and len(w) > len(suf) + 1:
            candidate = w[:-len(suf)]
            if candidate:
                return candidate

    return w

def format_source(source):
    if source == "lexicon":
        return "TDK Sözlüğü"
    if source == "model":
        return "Makine Öğrenmesi Modeli"
    return "Bilinmiyor"

def confidence_label(confidence):
    if confidence >= 90:
        return "Yüksek"
    if confidence >= 60:
        return "Orta"
    return "Düşük"

def predict_origin_for_word(word, lexicon, vect, clf, lbl):
    original = word.strip()
    normalized = normalize_word(original)

    if original.lower() in lexicon:
        origin_en = lexicon[original.lower()]["origin"]
        notes = lexicon[original.lower()]["notes"]
        source = "lexicon"
        confidence = 1.0
    elif normalized in lexicon:
        origin_en = lexicon[normalized]["origin"]
        notes = lexicon[normalized]["notes"]
        source = "lexicon"
        confidence = 0.98
    elif vect is not None and clf is not None and lbl is not None:
        Xv = vect.transform([normalized])
        probs = clf.predict_proba(Xv)[0]
        idx = probs.argmax()
        origin_en = lbl.inverse_transform([idx])[0]
        notes = ""
        source = "model"
        confidence = float(probs[idx])
    else:
        origin_en = "Unknown"
        notes = ""
        source = "fallback"
        confidence = 0.0

    origin_tr = ORIGIN_TURKCE.get(origin_en, origin_en)
    confidence_pct = round(confidence * 100, 1)

    return {
        "text": original,
        "origin": origin_tr,
        "notes": notes if notes else "Açıklama bulunmuyor",
        "source": format_source(source),
        "confidence": confidence_pct,
        "confidence_label": confidence_label(confidence_pct)
    }

# =========================================
# GRAFİK
# =========================================
def draw_origin_chart(summary, origin_colors):
    if not summary:
        return None

    labels = [item["origin"] for item in summary]
    counts = [item["count"] for item in summary]
    colors = [origin_colors.get(label, "#94a3b8") for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.barh(labels, counts, color=colors, edgecolor="none", height=0.55)

    ax.set_xlabel("Kelime Sayısı", fontsize=11)
    ax.set_ylabel("Köken", fontsize=11)
    ax.set_title("Köken Dağılımı", fontsize=13, fontweight="bold", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    max_count = max(counts) if counts else 1
    ax.set_xlim(0, max_count + 1)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=10,
            fontweight="bold"
        )

    ax.invert_yaxis()
    fig.tight_layout()
    return fig

# =========================================
# STATE
# =========================================
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

def clear_text():
    st.session_state.input_text = ""

def set_demo_text(text):
    st.session_state.input_text = text

# =========================================
# VERİLERİ YÜKLE
# =========================================
LEXICON = load_lexicon()
VECT, CLF, LBL = load_model()

# =========================================
# BAŞLIK
# =========================================
st.markdown("""
<div class="hero-box">
    <div class="hero-wrap">
        <div class="hero-icon">Aa</div>
        <div>
            <div class="hero-title">Kelimelerin Genetik Haritası</div>
            <div class="hero-subtitle">
                Türkçedeki kelimelerin kökenlerini inceleyen yapay zekâ destekli etkileşimli analiz aracı
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================
# GİRİŞ BÖLÜMÜ
# =========================================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("Kelime veya kısa cümle giriniz")
st.markdown(
    '<div class="small-help">Sistem, girilen metni kelimelere ayırır ve her sözcüğün kökenini veri seti ve model yardımıyla tahmin eder.</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="demo-label">Hazır örnek cümleler</div>', unsafe_allow_html=True)

cols1 = st.columns(3)
cols2 = st.columns(3)
cols3 = st.columns(3)

with cols1[0]:
    st.markdown('<div class="demo-button btn1">', unsafe_allow_html=True)
    st.button("Örnek Cümle 1", use_container_width=True, on_click=set_demo_text, args=(DEMO_SENTENCES[0],))
    st.markdown('</div>', unsafe_allow_html=True)

with cols1[1]:
    st.markdown('<div class="demo-button btn2">', unsafe_allow_html=True)
    st.button("Örnek Cümle 2", use_container_width=True, on_click=set_demo_text, args=(DEMO_SENTENCES[1],))
    st.markdown('</div>', unsafe_allow_html=True)

with cols1[2]:
    st.markdown('<div class="demo-button btn3">', unsafe_allow_html=True)
    st.button("Örnek Cümle 3", use_container_width=True, on_click=set_demo_text, args=(DEMO_SENTENCES[2],))
    st.markdown('</div>', unsafe_allow_html=True)

with cols2[0]:
    st.markdown('<div class="demo-button btn4">', unsafe_allow_html=True)
    st.button("Örnek Cümle 4", use_container_width=True, on_click=set_demo_text, args=(DEMO_SENTENCES[3],))
    st.markdown('</div>', unsafe_allow_html=True)

with cols2[1]:
    st.markdown('<div class="demo-button btn5">', unsafe_allow_html=True)
    st.button("Örnek Cümle 5", use_container_width=True, on_click=set_demo_text, args=(DEMO_SENTENCES[4],))
    st.markdown('</div>', unsafe_allow_html=True)

with cols2[2]:
    st.markdown('<div class="demo-button btn6">', unsafe_allow_html=True)
    st.button("Örnek Cümle 6", use_container_width=True, on_click=set_demo_text, args=(DEMO_SENTENCES[5],))
    st.markdown('</div>', unsafe_allow_html=True)

with cols3[0]:
    st.markdown('<div class="demo-button btn7">', unsafe_allow_html=True)
    st.button("Örnek Cümle 7", use_container_width=True, on_click=set_demo_text, args=(DEMO_SENTENCES[6],))
    st.markdown('</div>', unsafe_allow_html=True)

with cols3[1]:
    st.markdown('<div class="demo-button btn8">', unsafe_allow_html=True)
    st.button("Örnek Cümle 8", use_container_width=True, on_click=set_demo_text, args=(DEMO_SENTENCES[7],))
    st.markdown('</div>', unsafe_allow_html=True)

with cols3[2]:
    st.markdown('<div class="demo-button btn9">', unsafe_allow_html=True)
    st.button("Örnek Cümle 9", use_container_width=True, on_click=set_demo_text, args=(DEMO_SENTENCES[8],))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="input-note">
    Örnek kelime: <strong>kitap</strong> &nbsp; | &nbsp;
    Örnek cümle: <strong>Akşam evde kitap okuyup çay içtim.</strong>
    </div>
    """,
    unsafe_allow_html=True
)

text = st.text_area(
    "Metin",
    key="input_text",
    height=170,
    label_visibility="collapsed",
    placeholder="Buraya bir kelime ya da kısa cümle yazın..."
)

col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    analyze = st.button("Analiz Et", use_container_width=True)
with col_btn2:
    st.button("Temizle", on_click=clear_text, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# ANALİZ
# =========================================
if analyze:
    tokens = tokenize(text)
    results = []
    counts = {}

    for token in tokens:
        result = predict_origin_for_word(token, LEXICON, VECT, CLF, LBL)
        results.append(result)
        counts[result["origin"]] = counts.get(result["origin"], 0) + 1

    summary = sorted(
        [{"origin": origin, "count": count} for origin, count in counts.items()],
        key=lambda x: x["count"],
        reverse=True
    )

    dominant_origin = summary[0]["origin"] if summary else "Yok"

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Analiz Sonuçları")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Toplam Sözcük</div>
            <div class="stat-value">{len(tokens)}</div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Köken Sayısı</div>
            <div class="stat-value">{len(counts)}</div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Baskın Köken</div>
            <div class="stat-value">{dominant_origin}</div>
        </div>
        """, unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("### Kelime Kartları")

        if not results:
            st.info("Analiz edilecek sözcük bulunamadı.")
        else:
            for item in results:
                color = ORIGIN_COLORS.get(item["origin"], "#94a3b8")

                st.markdown(f"""
                <div class="word-card">
                    <div class="word-header">
                        <div class="word-title">{item["text"]}</div>
                        <span class="badge" style="background:{color};">{item["origin"]}</span>
                    </div>
                    <div class="meta-line"><strong>Anlam / Not:</strong> {item["notes"]}</div>
                    <div class="meta-line"><strong>Sonuç Kaynağı:</strong> {item["source"]}</div>
                    <div class="meta-line"><strong>Güven Oranı:</strong> %{item["confidence"]} ({item["confidence_label"]})</div>
                </div>
                """, unsafe_allow_html=True)

    with right:
        st.markdown("### Köken Dağılımı")

        if summary:
            fig = draw_origin_chart(summary, ORIGIN_COLORS)
            st.pyplot(fig, use_container_width=True)

            st.markdown("#### Dağılım Özeti")
            for row in summary:
                color = ORIGIN_COLORS.get(row["origin"], "#94a3b8")
                st.markdown(
                    f"<div class='legend-row'><span class='legend-dot' style='color:{color}'>■</span>{row['origin']} — {row['count']}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("Gösterilecek dağılım verisi bulunmuyor.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# ALT BİLGİ
# =========================================
with st.expander("Uygulama hakkında"):
    st.markdown("""
<div class="footer-note">
Bu prototipte temel analiz birimi kelimedir.<br><br>
Kullanıcı tek kelime veya kısa cümle girebilir. Cümle girildiğinde sistem metni sözcüklere ayırır ve her kelimeyi ayrı değerlendirir.<br><br>
Bu sürüm özellikle isimler, kavramlar ve veri setinde yer alan temel sözcüklerde daha güçlü çalışır. Ekli kelimeler için basit bir sadeleştirme uygulanır.
</div>
""", unsafe_allow_html=True)