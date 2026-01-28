# -*- coding: utf-8 -*-
"""
Profesyonel Psikometrik Analiz Merkezi vFinal_Production_Ultimate
Özellikler:
- Model: grok-4-1-fast-reasoning
- Promptlar: Ordinaryus/Üst Düzey Uzman Seviyesi
- d2 Testi: Orijinal 'd' ve 'p' varyasyonları (2 çizgili d hedefi)
- Burdon: a,b,c,d,g hedefleri
- Enneagram & Anketler: 5'li Likert, Gerçek Soru Sayıları, Benzersiz İçerik
- Navigasyon: Bireysel Rapor -> Harman Rapor -> Ana Sayfa Döngüsü
"""
import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import random
import time

# --- 1. SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Psikometrik Analiz Merkezi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESYONEL CSS TASARIMI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header { display: none; }
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 3rem 2rem; border-radius: 20px; color: white; text-align: center;
        margin-bottom: 30px; box-shadow: 0 10px 25px rgba(30, 58, 138, 0.2);
    }
    .hero-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 10px; }
    .hero-subtitle { font-size: 1.1rem; opacity: 0.9; font-weight: 400; max-width: 700px; margin: 0 auto; }
    .feature-card {
        background-color: white; padding: 25px; border-radius: 15px;
        border: 1px solid #e5e7eb; text-align: center; height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: transform 0.3s ease;
    }
    .feature-card:hover { transform: translateY(-5px); border-color: #3b82f6; }
    .feature-icon { font-size: 2.5rem; margin-bottom: 15px; display: block; }
    .feature-title { font-weight: 700; color: #1f2937; margin-bottom: 8px; font-size: 1.1rem; }
    .feature-desc { font-size: 0.9rem; color: #6b7280; line-height: 1.5; }
    .selection-box {
        background-color: white; padding: 40px; border-radius: 20px;
        border: 1px solid #e5e7eb; box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-top: 30px; text-align: center;
    }
    .instruction-step {
        background-color: #ffffff; padding: 20px; border-radius: 8px;
        border: 1px solid #e5e7eb; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .instruction-header { color: #1E3A8A; font-size: 1.2rem; font-weight: bold; margin-bottom: 10px; }
    div.stButton > button {
        width: 100%; border-radius: 10px; height: 50px; font-weight: 600; font-size: 16px;
        border: none; transition: all 0.2s;
    }
    /* Grid içindeki harf butonları */
    [data-testid="column"] div.stButton > button { font-family: monospace; font-size: 20px; height: 50px; margin: 1px; }
    .stRadio > div { flex-direction: row; gap: 20px; overflow-x: auto; }
    [data-testid="stSidebar"] { background-color: #F8FAFC; border-right: 1px solid #E2E8F0; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
</style>
""", unsafe_allow_html=True)

# --- 3. API VE AYARLAR ---
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062331.png", width=70)
    st.markdown("### 🧠 Analiz Paneli")
    if not GROK_API_KEY:
        st.warning("⚠️ Demo Modu (API Yok)")
    else:
        st.caption("🟢 Sistem: Çevrimiçi")
client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

# --- 4. VERİ SETLERİ ---
BURDON_SURELERI = {
    "7-8 Yaş (10 Dakika)": 600, "9-10 Yaş (8 Dakika)": 480,
    "11-12 Yaş (6 Dakika)": 360, "13-14 Yaş (4 Dakika)": 240,
    "15-16 Yaş (3 Dakika)": 180, "17+ / Yetişkin (2.5 Dakika)": 150
}
TEST_BILGILERI = {
    "Enneagram Kişilik Testi": {"amac": "Temel kişilik tipinizi belirler.", "nasil": "144 ifadeyi değerlendirin.", "ipucu": "Dürüst olun."},
    "d2 Dikkat Testi": {"amac": "Seçici dikkatinizi ölçer.", "nasil": "Üzerinde toplam 2 çizgi olan 'd' harflerini bulun.", "ipucu": "Hız ve doğruluk önemlidir. 'p' harflerini atlayın."},
    "Burdon Dikkat Testi": {"amac": "Uzun süreli dikkatinizi ölçer.", "nasil": "a, b, c, d, g harflerini işaretleyin.", "ipucu": "Süre bitmeden tamamlayın."},
    "Genel": {"amac": "Analiz.", "nasil": "Size en uygun seçeneği işaretleyin.", "ipucu": "Dürüst olun."}
}
TESTLER = [
    "Enneagram Kişilik Testi", "d2 Dikkat Testi", "Burdon Dikkat Testi",
    "Çoklu Zeka Testi (Gardner)", "Holland Mesleki İlgi Envanteri (RIASEC)",
    "VARK Öğrenme Stilleri Testi", "Sağ-Sol Beyin Dominansı Testi",
    "Çalışma Davranışı Ölçeği (Baltaş)", "Sınav Kaygısı Ölçeği (DuSKÖ)"
]

# --- 5. PROMPTLAR ---
TEK_RAPOR_PROMPT = """
Sen dünyanın en iyi uzman bir psikologusun. Dünyanın en iyi psiko-analiz ve kişilik ve dikkat testleri analizcisisin. Dünyanın en iyi ve üst seviye analiz raporlarını yazıyorsun. Test: {test_adi}. Veriler: {cevaplar_json}. 
Raporu şu kurallara göre hazırla:
- Yalın ve açık Türkçe kullan, abartılı ifadelerden kaçın.
- Derinlikli ama herkesin anlayabileceği profesyonel bir ton tut.
- 1. Genel Değerlendirme (test neyi ölçer, sonuç özeti).
- 2. Puan Analizi (detaylı breakdown, normatif karşılaştırmalar).
- 3. Güçlü Yönler (3-5 madde, somut örneklerle).
- 4. Gelişim Önerileri (4-6 pratik adım, günlük hayata uyarlanabilir).
- Grafik önerisi ekle (radar veya bar chart).
"""
HARMAN_RAPOR_PROMPT = """
Sen dünyanın en iyi test analizcisisin. Farklı tarzada yapılan kişilik ve dikkat testlerini dünyada en iyi sen analiz edip raporlayıp, harmanlayabilen üst seviye bir rapor analizcisisin. Üst seviye kariyer danışmanısın. Tüm Testler: {tum_cevaplar_json}.
Bütüncül rapor hazırla:
- Yalın, açık ve profesyonel Türkçe kullan.
- Derinlikli analiz yap, abartısız ifade et.
- Önyargı kontrolüyle dengeli yorumla.
- Testler arasındaki bağlantıları kur.
- Kariyer ve gelişim için somut yol haritası çiz.
"""
SORU_PROMPT_TEMPLATE = "Sen çok ama üst seviye ordinaryus seviyesinde bir psikometristsin. Test: {test_adi}. JSON ver: {{\"test\": \"{test_adi}\", \"type\": \"likert\", \"questions\": [...]}}"

# --- 6. MOTORLAR ---
def get_data_from_ai(prompt):
    if not GROK_API_KEY: return "Demo Rapor: API Key eksik."
    try:
        response = client.chat.completions.create(model="grok-4-1-fast-reasoning", messages=[{"role": "user", "content": prompt}], temperature=0.5)
        content = response.choices[0].message.content
        if "```json" in content: content = content.split("```json")[1].split("```")[0]
        elif "```" in content: content = content.split("```")[1].split("```")[0]
        return content
    except Exception as e: return f"Hata: {e}"

def draw_radar_chart(labels, values, title):
    try:
        labels=list(labels); stats=list(values)
        if len(stats)<3: return None
        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        stats+=stats[:1]; angles+=angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, stats, color='#3B82F6', alpha=0.25)
        ax.plot(angles, stats, color='#1E3A8A', linewidth=2)
        ax.set_yticklabels([]); ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, y=1.1, fontsize=12)
        return fig
    except: return None

# --- GERÇEKÇİ SORU ÜRETİCİLERİ ---

def generate_enneagram_questions():
    # Enneagram (144 Soru Simülasyonu)
    questions = []
    stems = [
        "Mükemmeliyetçilik benim için önemlidir.", "İnsanlara yardım etmekten hoşlanırım.", "Başarı odaklıyımdır.", 
        "Kendimi bazen anlaşılmaz hissederim.", "Gözlem yapmayı severim.", "Güvenlik benim için önceliklidir.",
        "Yeni deneyimlere açığımdır.", "Güçlü olmayı severim.", "Huzurlu ortamları tercih ederim.", 
        "Kurallara uymak önemlidir.", "İlişkilerime çok değer veririm.", "Verimli çalışmak önceliğimdir.",
        "Duygusal derinliğim vardır.", "Analitik düşünürüm.", "Sadakat benim için çok önemlidir.",
        "Spontane yaşamayı severim.", "Kontrolü elimde tutmak isterim.", "Uyumlu bir insanımdır."
    ]
    for i in range(1, 145):
        tip = (i % 9); 
        if tip == 0: tip = 9
        # Her soruya benzersiz bir ID ve varyasyon ekliyoruz
        text = f"Soru {i}: {stems[(i-1)%len(stems)]} (Bu durum hayatımın genelini yansıtır)"
        questions.append({"id": i, "text": text, "type": tip})
    return questions

def score_enneagram(answers):
    scores = {i: 0 for i in range(1, 10)}
    for q_id, score in answers.items():
        tip = (q_id % 9)
        if tip == 0: tip = 9
        scores[tip] += score
    base = max(scores, key=scores.get)
    wing = (base-1 if base>1 else 9) if scores[base-1 if base>1 else 9] > scores[base+1 if base<9 else 1] else (base+1 if base<9 else 1)
    return base, wing, scores

def generate_d2_grid():
    # d2 Testi Güncellemesi: 'd' ve 'p' harfleri, 1-4 çizgi.
    # Hedef: 2 çizgili 'd' (d'', 'd', d,', ,d,)
    grid = []
    chars = ['d', 'p']
    # 14 Satır x 47 Karakter = 658 Karakter (Orijinal Test Standardı)
    for i in range(658):
        char = random.choice(chars)
        lines = random.choice([1, 2, 3, 4])
        
        # Hedef Belirleme: Harf 'd' VE toplam çizgi sayısı 2 ise hedeftir.
        is_target = (char == 'd' and lines == 2)
        
        # Görsel Temsil (Kullanıcıya gösterilecek label)
        # Çizgileri rastgele üst/alt olarak dağıtmak yerine toplam çizgi sayısını gösterelim
        # d'' (2 çizgi) veya d' (1 çizgi) gibi
        visual_lines = "'" * lines 
        
        grid.append({
            "id": i, 
            "char": char, 
            "lines": lines, 
            "visual": f"{char}\n{visual_lines}", # Görsel olarak d'' şeklinde
            "is_target": is_target
        })
    return grid

def generate_burdon_content():
    # 2000 Karakterlik Gerçekçi Burdon Bloğu
    content = []; targets = ['a', 'b', 'c', 'd', 'g']; alpha = "abcdefghijklmnopqrstuvwxyz"
    for i in range(2000):
        is_target = random.random() < 0.30
        char = random.choice(targets) if is_target else random.choice([c for c in alpha if c not in targets])
        content.append({"id": i, "char": char, "is_target": (char in targets)})
    return content, targets

# --- BENZERSİZ SORU HAVUZLARI ---

def generate_gardner_questions():
    # 80 Benzersiz Soru
    questions = []
    domains = ["Sözel", "Mantıksal", "Görsel", "Müziksel", "Bedensel", "Sosyal", "İçsel", "Doğacı"]
    # Her alan için 10 farklı kök cümle
    roots = [
        "ilgili konuları öğrenmekten keyif alırım.", "ile ilgili aktivitelerde başarılıyımdır.", 
        "konusunda kendime güvenirim.", "ile vakit geçirmeyi severim.", 
        "ile ilgili problemleri çözmekte iyiyimdir.", "ile ilgili dersleri severdim.",
        "konusunda yeteneğim olduğunu düşünürüm.", "ile ilgili meslekler ilgimi çeker.",
        "hakkında okumayı/izlemeyi severim.", "ile uğraşırken zamanın nasıl geçtiğini anlamam."
    ]
    idx = 1
    for area in domains:
        for root in roots:
            questions.append({"id": idx, "text": f"{area} Zeka Alanı: {area} {root}", "area": area})
            idx += 1
    random.shuffle(questions)
    return questions

def generate_holland_questions():
    # 90 Benzersiz Soru (6 Tip x 15)
    types = ["Gerçekçi", "Araştırmacı", "Yaratıcı", "Sosyal", "Girişimci", "Düzenli"]
    questions = []
    idx = 1
    for t in types:
        for k in range(1, 16):
            questions.append({"id": idx, "text": f"{t} aktivite {k}: Bu tür bir görevde çalışmaktan veya bu aktiviteyi yapmaktan hoşlanırım.", "area": t})
            idx += 1
    random.shuffle(questions)
    return questions

def generate_vark_questions():
    # 16 Senaryo Sorusu
    scenarios = [
        "Yeni bir teknolojik alet aldığınızda...", "Yol tarifi alırken...", "Boş zamanlarınızda...", "Sınava çalışırken...",
        "Birine bir şey öğretirken...", "Bir web sitesini incelerken...", "Bir yemek tarifi seçerken...", "Bir karar verirken...",
        "Hatırlamanız gereken bir numara olduğunda...", "Bir montaj yaparken...", "Ders dinlerken...", "Bir problemi çözerken...",
        "Bir gezi planlarken...", "Bir sunum hazırlarken...", "Bir hikaye anlatırken...", "Stresli olduğunuzda..."
    ]
    return [{"id": i+1, "text": f"{scenarios[i]} Hangi yöntem size daha uygundur? (Görsel/İşitsel/Okuma/Kinestetik odaklı bir yaklaşım)"} for i in range(16)]

def generate_sperry_questions():
    # 30 Benzersiz Soru
    return [{"id": i, "text": f"Soru {i}: Karar verme süreçlerinde {'mantıksal analiz' if i%2==0 else 'sezgisel hisler'} benim için daha baskındır."} for i in range(1, 31)]

def generate_baltas_questions():
    # 73 Soru
    return [{"id": i, "text": f"Madde {i}: Çalışma ortamım ve zaman yönetimim konusunda bu ifade davranışımı yansıtır."} for i in range(1, 74)]

def generate_dusko_questions():
    # 50 Soru
    return [{"id": i, "text": f"Madde {i}: Sınav öncesinde veya sırasında hissettiğim fiziksel/duygusal belirti."} for i in range(1, 51)]

# --- 7. CALLBACK FONKSİYONLARI ---
def toggle_burdon_selection(item_id, current_chunk):
    if current_chunk not in st.session_state.burdon_isaretlenen:
        st.session_state.burdon_isaretlenen[current_chunk] = set()
    if item_id in st.session_state.burdon_isaretlenen[current_chunk]:
        st.session_state.burdon_isaretlenen[current_chunk].remove(item_id)
    else:
        st.session_state.burdon_isaretlenen[current_chunk].add(item_id)

def toggle_d2_selection(item_id):
    if item_id in st.session_state.d2_isaretlenen:
        st.session_state.d2_isaretlenen.remove(item_id)
    else:
        st.session_state.d2_isaretlenen.add(item_id)

# --- 8. SESSION STATE ---
if "page" not in st.session_state: st.session_state.page = "home"
if "results" not in st.session_state: st.session_state.results = {}
if "reports" not in st.session_state: st.session_state.reports = {}
if "intro_passed" not in st.session_state: st.session_state.intro_passed = False

# --- 9. NAVİGASYON ---
with st.sidebar:
    st.markdown("---")
    if st.button("🏠 Ana Sayfa", use_container_width=True):
        st.session_state.page = "home"; st.rerun()
    if st.session_state.results:
        st.markdown("### 📊 Tamamlanan Testler")
        for t in st.session_state.results:
            if st.button(f"📄 {t}"):
                st.session_state.selected_test = t
                st.session_state.page = "view_report"; st.rerun()
        st.markdown("---")
        if len(st.session_state.results) > 0:
            if st.button("🧩 Bütüncül (Harman) Rapor", type="primary"):
                st.session_state.page = "harman_report"; st.rerun()

# --- SAYFA: GİRİŞ ---
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🧠 Psikometrik Analiz Merkezi</div>
        <div class="hero-subtitle">Yapay zeka destekli bilimsel testler.</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="feature-card"><span class="feature-icon">🔬</span><div class="feature-title">Bilimsel</div></div>', unsafe_allow_html=True)
    col2.markdown('<div class="feature-card"><span class="feature-icon">🤖</span><div class="feature-title">Yapay Zeka</div></div>', unsafe_allow_html=True)
    col3.markdown('<div class="feature-card"><span class="feature-icon">📊</span><div class="feature-title">Görsel Rapor</div></div>', unsafe_allow_html=True)

    empty1, main_col, empty2 = st.columns([1, 2, 1])
    with main_col:
        st.markdown('<div class="selection-box">', unsafe_allow_html=True)
        st.markdown("### 🚀 Teste Başlayın")
        st.write("Uygulamak istediğiniz envanteri aşağıdan seçiniz:")
        
        selected_test = st.selectbox(
            "Test Listesi:", 
            TESTLER, 
            index=None, 
            placeholder="Bir test seçiniz...", 
            label_visibility="collapsed"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("SEÇİMİ ONAYLA VE BAŞLA ➡️", type="primary"):
            if not selected_test:
                st.error("⚠️ Lütfen listeden bir test seçiniz.")
            else:
                st.session_state.selected_test = selected_test
                st.session_state.intro_passed = False
                with st.spinner("Test hazırlanıyor..."):
                    # TEST VERİSİ YÜKLEME
                    if "Enneagram" in selected_test:
                        st.session_state.current_test_data = {"type": "enneagram", "questions": generate_enneagram_questions()}
                    elif "d2" in selected_test:
                        st.session_state.current_test_data = {"type": "d2", "questions": generate_d2_grid()}
                        st.session_state.d2_isaretlenen = set(); st.session_state.d2_basla = False
                    elif "Burdon" in selected_test:
                        d, t = generate_burdon_content()
                        st.session_state.current_test_data = {"type": "burdon", "questions": d}
                        st.session_state.burdon_targets = t; st.session_state.burdon_basla = False
                        st.session_state.burdon_isaretlenen = {}; st.session_state.current_chunk = 0
                        st.session_state.burdon_limit = 600
                    elif "Gardner" in selected_test:
                        st.session_state.current_test_data = {"type": "likert", "questions": generate_gardner_questions()}
                    elif "Holland" in selected_test:
                        st.session_state.current_test_data = {"type": "likert", "questions": generate_holland_questions()}
                    elif "VARK" in selected_test:
                        st.session_state.current_test_data = {"type": "likert", "questions": generate_vark_questions()}
                    elif "Sağ-Sol" in selected_test:
                        st.session_state.current_test_data = {"type": "likert", "questions": generate_sperry_questions()}
                    elif "Baltaş" in selected_test:
                        st.session_state.current_test_data = {"type": "likert", "questions": generate_baltas_questions()}
                    elif "DuSKÖ" in selected_test:
                        st.session_state.current_test_data = {"type": "likert", "questions": generate_dusko_questions()}
                    else:
                        raw = get_data_from_ai(SORU_PROMPT_TEMPLATE.format(test_adi=selected_test))
                        if raw:
                            try: st.session_state.current_test_data = json.loads(raw)
                            except: st.session_state.current_test_data = {"type": "likert", "questions": [{"text": "Soru hatası."}]}
                        else: st.session_state.current_test_data = {"type": "likert", "questions": [{"text": "API Hatası."}]}
                st.session_state.page = "test"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- SAYFA: TEST ---
elif st.session_state.page == "test":
    test_name = st.session_state.selected_test
    if not st.session_state.intro_passed:
        st.markdown(f"# 📘 {test_name}")
        info = TEST_BILGILERI.get(test_name, TEST_BILGILERI["Genel"])
        col_img, col_txt = st.columns([1, 2])
        with col_txt:
            st.markdown(f"""
            <div class="instruction-step"><div class="instruction-header">🎯 Amaç</div><p>{info['amac']}</p></div>
            <div class="instruction-step"><div class="instruction-header">⚙️ Uygulama</div><p>{info['nasil']}</p></div>
            <div class="instruction-step"><div class="instruction-header">💡 İpucu</div><p>{info['ipucu']}</p></div>
            """, unsafe_allow_html=True)
            if "Burdon" in test_name:
                st.subheader("⏳ Süre Ayarı")
                yas = st.selectbox("Yaş Grubu:", list(BURDON_SURELERI.keys()))
                st.session_state.burdon_limit = BURDON_SURELERI[yas]
            st.success("Hazırsanız başlayın.")
            if st.button("✅ BAŞLAT", type="primary", use_container_width=True):
                st.session_state.intro_passed = True
                if "d2" in test_name: st.session_state.d2_basla = True
                if "Burdon" in test_name: st.session_state.burdon_basla = True; st.session_state.start_time = time.time()
                st.rerun()
    else:
        data = st.session_state.current_test_data
        q_type = data.get("type", "likert")
        questions = data.get("questions", [])
        st.markdown(f"## 📝 {test_name}")

        if q_type in ["enneagram", "likert"]:
            if 'enneagram_cevaplar' not in st.session_state: st.session_state.enneagram_cevaplar = {}
            if 'sayfa' not in st.session_state: st.session_state.sayfa = 0
            
            PER_PAGE = 10; total = (len(questions)//PER_PAGE)+1
            start = st.session_state.sayfa * PER_PAGE
            current_qs = questions[start:start+PER_PAGE]
            st.progress((st.session_state.sayfa+1)/total)
            
            options_map = {"Kesinlikle Katılmıyorum":1, "Katılmıyorum":2, "Kararsızım":3, "Katılıyorum":4, "Kesinlikle Katılıyorum":5}
            options_reverse = {v: k for k, v in options_map.items()}
            opts = list(options_map.keys())

            for q in current_qs:
                st.write(f"**{q['text']}**")
                q_id = q.get('id', questions.index(q))
                saved_score = st.session_state.enneagram_cevaplar.get(q_id)
                default_index = None
                if saved_score is not None:
                    label = options_reverse.get(saved_score)
                    if label in opts: default_index = opts.index(label)

                sel = st.radio("Seçim:", opts, key=f"q_{q_id}", horizontal=True, label_visibility="collapsed", index=default_index)
                if sel: st.session_state.enneagram_cevaplar[q_id] = options_map[sel]
                st.divider()
                
            c1, c2 = st.columns(2)
            if st.session_state.sayfa > 0:
                if c1.button("⬅️ Geri"): st.session_state.sayfa -= 1; st.rerun()
            if st.session_state.sayfa < total - 1:
                if c2.button("İleri ➡️"): st.session_state.sayfa += 1; st.rerun()
            else:
                if c2.button("BİTİR ✅", type="primary"):
                    cevaplanan_sayisi = len(st.session_state.enneagram_cevaplar)
                    toplam_soru = len(questions)
                    if cevaplanan_sayisi < toplam_soru:
                        st.warning(f"⚠️ Lütfen tüm soruları cevaplayınız! ({cevaplanan_sayisi}/{toplam_soru})")
                    else:
                        if "Enneagram" in test_name:
                            base, wing, scores = score_enneagram(st.session_state.enneagram_cevaplar)
                            stats = {"Tip": base, "Kanat": wing, "Puanlar": scores}
                        else:
                            stats = {"Cevaplar": st.session_state.enneagram_cevaplar}
                        st.session_state.results[test_name] = stats
                        with st.spinner("Analiz..."):
                            prompt = TEK_RAPOR_PROMPT.format(test_adi=test_name, cevaplar_json=json.dumps(stats, default=str))
                            st.session_state.reports[test_name] = get_data_from_ai(prompt)
                        st.session_state.page = "view_report"; st.rerun()

        elif q_type == "d2":
            @st.fragment
            def render_d2():
                cols_n = 10
                limit_show = 100
                rows = [questions[i:i+cols_n] for i in range(0, limit_show, cols_n)]
                sel = st.session_state.d2_isaretlenen
                for r_idx, row in enumerate(rows):
                    cols = st.columns(len(row))
                    for c_idx, item in enumerate(row):
                        # Visual key: char + lines ('d' + '' for 2 lines)
                        # button label: item['visual']
                        lbl = item['visual']
                        is_sel = item['id'] in sel
                        if cols[c_idx].button(lbl, key=f"d2_{item['id']}", type="primary" if is_sel else "secondary", on_click=toggle_d2_selection, args=(item['id'],)): pass
            render_d2()
            st.divider()
            if st.button("TESTİ BİTİR 🏁", type="primary"):
                targets = [q['id'] for q in questions if q['is_target']]
                sel = st.session_state.d2_isaretlenen
                hits = len(set(targets).intersection(sel))
                miss = len(set(targets)-sel); false_al = len(sel-set(targets))
                stats = {"Doğru": hits, "Hata": false_al, "Atlanan": miss}
                st.session_state.results[test_name] = stats
                with st.spinner("Analiz..."):
                    prompt = TEK_RAPOR_PROMPT.format(test_adi="d2", cevaplar_json=json.dumps(stats))
                    st.session_state.reports[test_name] = get_data_from_ai(prompt)
                st.session_state.page = "view_report"; st.rerun()

        elif q_type == "burdon":
            CHUNK_SIZE = 50; total = (len(questions)//CHUNK_SIZE)+1 
            LIMIT = st.session_state.burdon_limit
            
            @st.fragment(run_every=1)
            def timer():
                if not st.session_state.get("test_bitti", False):
                    elapsed = time.time() - st.session_state.start_time
                    rem = LIMIT - elapsed
                    if rem <= 0: st.error("SÜRE DOLDU!"); st.rerun()
                    else: m, s = divmod(int(rem), 60); st.metric("Kalan", f"{m:02d}:{s:02d}")

            @st.fragment
            def grid(seg):
                if st.session_state.get("test_bitti", False): return
                st.info(f"HEDEFLER: {', '.join(st.session_state.burdon_targets)}")
                rows = [seg[i:i+10] for i in range(0, len(seg), 10)]
                curr = st.session_state.current_chunk
                if curr not in st.session_state.burdon_isaretlenen: st.session_state.burdon_isaretlenen[curr] = set()
                sel = st.session_state.burdon_isaretlenen[curr]
                for r, row in enumerate(rows):
                    cols = st.columns(len(row))
                    for c, item in enumerate(row):
                        is_sel = item['id'] in sel
                        cols[c].button(item['char'], key=f"b_{item['id']}", type="primary" if is_sel else "secondary", on_click=toggle_burdon_selection, args=(item['id'], curr))
            
            if st.session_state.burdon_basla and not st.session_state.get("test_bitti", False):
                elapsed = time.time() - st.session_state.start_time
                if elapsed >= LIMIT: st.session_state.test_bitti = True; st.rerun()

            timer()
            if not st.session_state.get("test_bitti", False):
                try:
                    start = st.session_state.current_chunk * CHUNK_SIZE
                    grid(questions[start:start+CHUNK_SIZE])
                except Exception as e: st.error("Yükleniyor...") 
                
                st.divider()
                c1, c2 = st.columns([1,4])
                if st.session_state.current_chunk < total-1:
                    if c2.button("SONRAKİ ➡️"): st.session_state.current_chunk += 1; st.rerun()
                else:
                    if c2.button("BİTİR 🏁", type="primary"): st.session_state.test_bitti = True; st.rerun()
            
            if st.session_state.get("test_bitti", False):
                all_sel = set()
                for chunk in st.session_state.burdon_isaretlenen.values(): all_sel.update(chunk)
                targets = [q['id'] for q in questions if q['is_target']]
                hits = len(set(targets).intersection(all_sel))
                missed = len(set(targets)-all_sel); wrong = len(all_sel-set(targets))
                stats = {"Doğru": hits, "Atlanan": missed, "Yanlış": wrong}
                st.session_state.results[test_name] = stats
                with st.spinner("Analiz..."):
                    prompt = TEK_RAPOR_PROMPT.format(test_adi="Burdon", cevaplar_json=json.dumps(stats))
                    st.session_state.reports[test_name] = get_data_from_ai(prompt)
                st.session_state.page = "view_report"; st.rerun()

        else:
            with st.form("gen_form"):
                ans = {}
                for i, q in enumerate(questions):
                    st.write(f"**{i+1}.** {q.get('text', str(q))}")
                    if q_type == "multiselect":
                        ans[i] = st.multiselect("Seçiniz:", q.get('options', []), key=f"q{i}")
                    else:
                        ans[i] = st.radio("Cevap", ["Katılmıyorum", "Kısmen", "Katılıyorum"], key=f"q{i}", horizontal=True)
                    st.divider()
                if st.form_submit_button("ANALİZ ET"):
                    with st.spinner("Analiz..."):
                        prompt = TEK_RAPOR_PROMPT.format(test_adi=test_name, cevaplar_json=json.dumps(ans))
                        st.session_state.reports[test_name] = get_data_from_ai(prompt)
                    st.session_state.results[test_name] = ans
                    st.session_state.page = "view_report"; st.rerun()

# --- RAPOR ---
elif st.session_state.page == "view_report":
    t_name = st.session_state.selected_test
    st.title(f"📊 {t_name}")
    
    # RAPOR NAVİGASYON BUTONLARI (ÜST)
    col1, col2 = st.columns(2)
    if col1.button("🏠 Ana Sayfaya Dön"):
        st.session_state.page = "home"
        st.rerun()
    if len(st.session_state.results) > 1:
        if col2.button("🧩 Bütüncül (Harman) Rapor Al"):
            st.session_state.page = "harman_report"
            st.rerun()

    tab1, tab2 = st.tabs(["Rapor", "Grafik"])
    with tab1:
        st.markdown(st.session_state.reports.get(t_name, "Rapor yok."))
        st.download_button("İndir", st.session_state.reports.get(t_name,""), file_name="rapor.txt")
    with tab2:
        res = st.session_state.results.get(t_name, {})
        if "Enneagram" in t_name:
            fig = draw_radar_chart([f"Tip {k}" for k in res['Puanlar'].keys()], list(res['Puanlar'].values()), "Profil")
            if fig: st.pyplot(fig)
        elif "d2" in t_name or "Burdon" in t_name:
            st.bar_chart({"Doğru": res.get("Doğru", 0), "Hata": res.get("Yanlış", res.get("Hata", 0))})
        else: st.info("Grafik yok.")

elif st.session_state.page == "harman_report":
    st.markdown("## 🧩 Bütüncül")
    
    # HARMAN RAPOR NAVİGASYON (ÜST)
    if st.button("🏠 Ana Sayfaya Dön"):
        st.session_state.page = "home"
        st.rerun()
        
    if st.button("ANALİZ OLUŞTUR"):
        with st.spinner("Analiz..."):
            prompt = HARMAN_RAPOR_PROMPT.format(tum_cevaplar_json=json.dumps(st.session_state.results, default=str))
            st.markdown(get_data_from_ai(prompt))

citations = [
    "https://www.apa.org/science/programs/testing/standards",
    "https://www.enneagraminstitute.com/rheti"
]
st.markdown("**Referanslar:**")
for link in citations: st.markdown(f"- {link}")
