# -*- coding: utf-8 -*-
"""
Profesyonel Psikometrik Analiz Merkezi vFinal_Ultimate_Pro_CleanContent
"""
import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
import numpy as np
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
    "Enneagram Kişilik Testi": {"amac": "Temel kişilik tipinizi belirler.", "nasil": "İfadelerin size ne kadar uyduğunu işaretleyin.", "ipucu": "Dürüst olun."},
    "d2 Dikkat Testi": {"amac": "Seçici dikkatinizi ölçer.", "nasil": "2 çizgili d harflerini bulun.", "ipucu": "Hız ve doğruluk önemlidir."},
    "Burdon Dikkat Testi": {"amac": "Uzun süreli dikkatinizi ölçer.", "nasil": "a, b, c, d, g harflerini işaretleyin.", "ipucu": "Süre bitmeden tamamlayın."},
    "Genel": {"amac": "Kişisel analiz.", "nasil": "Size en uygun seçeneği işaretleyin.", "ipucu": "Dürüst olun."}
}

TESTLER = [
    "Enneagram Kişilik Testi", "d2 Dikkat Testi", "Burdon Dikkat Testi",
    "Çoklu Zeka Testi (Gardner)", "Holland Mesleki İlgi Envanteri (RIASEC)",
    "VARK Öğrenme Stilleri Testi", "Sağ-Sol Beyin Dominansı Testi",
    "Çalışma Davranışı Ölçeği (Baltaş)", "Sınav Kaygısı Ölçeği (DuSKÖ)"
]

# --- 5. PROMPTLAR ---
SORU_URETIM_PROMPT = """
Sen dünyanın en iyi Türk psikometrik test tasarımcısı ve çocuk/ergen psikolojisi uzmanısın.
GÖREV: Sadece belirtilen test için, orijinal testin soru sayısına ve yapısına TAM SADIK kalarak, tamamen yeni ve benzersiz sorular üret.
- Tüm sorular doğal, akıcı ve düzgün Türkçe olsun. ASLA devrik cümle kullanma.
- Her soru tek bir kısa, net ve sade cümle olsun.
- Sorular ortaokul ve lise öğrencisinin rahatça anlayabileceği kadar açık ve basit olsun.
- Hiçbir şekilde yönlendirme, manipülasyon, yargı, parantez içi açıklama, örnek veya ek bilgi ekleme.
- Sorular tamamen tarafsız ve objektif olsun, hiçbir duygu veya değer yargısı yükleme.
- Sorular psikolojik olarak derin ve kaliteli olsun; üst seviye analizlere olanak tanısın ama anlaşılırlığı asla feda etme.
- Tüm sorular 5'li Likert ölçeğine (Kesinlikle Katılmıyorum - Katılmıyorum - Kararsızım - Katılıyorum - Kesinlikle Katılıyorum) mükemmel uyumlu olsun.
- Aynı veya çok benzer ifadeler ASLA tekrarlanmasın.
- Çıktı SADECE ve SADECE geçerli JSON formatında olsun. Başka hiçbir metin, açıklama veya markdown yazma.
Testlere özgü zorunlu kurallar:
- Enneagram Kişilik Testi: Tam 144 soru üret. 9 tip için eşit dağılım (her tip tam 16 soru). RHETI tarzı kişisel ifadeler kullan ("Ben ...", "Benim için ... önemlidir" vb.).
- Çoklu Zeka Testi (Gardner): Tam 80 soru üret. 8 zeka alanı için tam 10'ar soru: Sözel, Mantıksal, Görsel, Müziksel, Bedensel, Sosyal, İçsel, Doğacı.
- Holland Mesleki İlgi Envanteri (RIASEC): Tam 90 soru üret. 6 tip için tam 15'er soru: Gerçekçi, Araştırmacı, Yaratıcı, Sosyal, Girişimci, Düzenli. Aktivite ve ilgi odaklı olsun.
- VARK Öğrenme Stilleri Testi: Tam 16 soru üret. Orijinal VARK senaryo tarzında günlük hayat durumları üzerinden tercih soruları.
- Sağ-Sol Beyin Dominansı Testi: Tam 30 soru üret. 15 sol beyin + 15 sağ beyin özelliği.
- Çalışma Davranışı Ölçeği (Baltaş): Tam 73 soru üret. Çalışma alışkanlıkları, motivasyon ve disiplin odaklı.
- Sınav Kaygısı Ölçeği (DuSKÖ): Tam 50 soru üret. Sınav kaygısı belirtileri odaklı.
JSON formatı kesin olarak şöyle olsun:
{
  "type": "likert",
  "questions": [
    {"id": 1, "text": "Soru metni burada"},
    ...
  ]
}
Enneagram için ekstra: {"id": 1, "text": "...", "type": 1} (type 1-9 integer)
Gardner için ekstra: {"id": 1, "text": "...", "area": "Sözel"}
Holland için ekstra: {"id": 1, "text": "...", "area": "Gerçekçi"}
Sadece istenen test için soru üret. Çıktıya kesinlikle başka hiçbir şey yazma.
Test adı: {test_adi}
"""

TEK_RAPOR_PROMPT = """
Sen dünyanın en iyi psikometrik test analizi uzmanısın.
GÖREV: Sadece verilen JSON verilerine dayanarak, test sonuçlarını nesnel ve veri odaklı şekilde analiz et.
Asla genel geçer bilgi verme, sadece kullanıcının puanları ve cevapları üzerinden yorum yap.
Rapor tamamen tarafsız olsun.
Test: {test_adi}
Veriler: {cevaplar_json}
Rapor Formatı (Tam olarak bu başlıkları kullan):
1. **Genel Değerlendirme:** Test sonuçlarının genel özeti.
2. **Puan Analizi:** Her alan/tip için alınan puanlar ve bu puanların anlamı (sayısal verilere dayanarak).
3. **Güçlü Yönler:** Yüksek puan alınan alanlardaki özellikler ve sonuçları.
4. **Gelişim Alanları:** Düşük puan alınan alanlardaki özellikler ve sonuçları.
5. **Öneriler:** Veri odaklı, uygulanabilir 4-5 somut tavsiye.
Dil: Sade, yalın ve profesyonel Türkçe. Tarafsız ve nesnel bir üslup kullan.
"""

HARMAN_RAPOR_PROMPT = """
Sen dünyanın en iyi psikometrik test sentez uzmanısın.
GÖREV: Verilen tüm test sonuçlarını (JSON) nesnel olarak birleştirerek bütüncül bir analiz üret.
Sadece verilen verilere dayan, dışarıdan bilgi ekleme.
Tüm Test Sonuçları: {tum_cevaplar_json}
Rapor Formatı (Tam olarak bu başlıkları kullan):
1. **Bütüncül Profil Özeti:** Testler arasındaki ilişkiler ve genel tablo.
2. **Ortak Güçlü Yönler:** Tüm testlerden çıkan yüksek puanlı özellikler.
3. **Kariyer Eğilimleri:** Profil bazında uygun meslek grupları ve nedenleri (veri odaklı).
4. **Öğrenme Stratejisi:** VARK, Gardner ve diğer testlere göre öğrenme özellikleri.
5. **Yol Haritası:**
   - Kısa vadeli (1-3 ay): Somut adımlar.
   - Orta vadeli (6-12 ay): Planlanabilir hedefler.
   - Uzun vadeli: Genel strateji.
Dil: Sade, yalın ve profesyonel Türkçe. Tamamen nesnel ve tarafsız üslup.
"""

# --- 6. MOTORLAR ---
def get_data_from_ai(prompt):
    if not GROK_API_KEY:
        return "Demo Rapor: API Key eksik."
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content
    except Exception as e:
        return f"Hata: {e}"

def draw_radar_chart(labels, values, title):
    try:
        labels = list(labels)
        stats = list(values)
        if len(stats) < 3:
            return None
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        stats += stats[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, stats, color='#3B82F6', alpha=0.25)
        ax.plot(angles, stats, color='#1E3A8A', linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, y=1.1, fontsize=12)
        return fig
    except:
        return None

# --- DİKKAT TESTLERİ ---
def generate_d2_grid():
    grid = []
    chars = ['d', 'p']
    for i in range(658):
        char = random.choice(chars)
        lines = random.choice([1, 2, 3, 4])
        is_target = (char == 'd' and lines == 2)
        visual_lines = "'" * lines
        grid.append({
            "id": i,
            "char": char,
            "lines": lines,
            "visual": f"{char}\n{visual_lines}",
            "is_target": is_target
        })
    return grid

def generate_burdon_content():
    content = []
    targets = ['a', 'b', 'c', 'd', 'g']
    alpha = "abcdefghijklmnopqrstuvwxyz"
    for i in range(2000):
        is_target = random.random() < 0.30
        char = random.choice(targets) if is_target else random.choice([c for c in alpha if c not in targets])
        content.append({"id": i, "char": char, "is_target": (char in targets)})
    return content, targets

# Enneagram puanlama
def score_enneagram(answers):
    scores = {i: 0 for i in range(1, 10)}
    questions = st.session_state.current_test_data["questions"]
    for q in questions:
        q_id = q["id"]
        score = answers.get(q_id)
        if score and "type" in q:
            scores[q["type"]] += score
    if sum(scores.values()) == 0:
        return None, None, scores
    base = max(scores, key=scores.get)
    left = base - 1 if base > 1 else 9
    right = base + 1 if base < 9 else 1
    wing = left if scores[left] > scores[right] else right
    return base, wing, scores

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
if "page" not in st.session_state:
    st.session_state.page = "home"
if "results" not in st.session_state:
    st.session_state.results = {}
if "reports" not in st.session_state:
    st.session_state.reports = {}
if "intro_passed" not in st.session_state:
    st.session_state.intro_passed = False

# --- 9. NAVİGASYON ---
with st.sidebar:
    st.markdown("---")
    if st.button("🏠 Ana Sayfa", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    if st.session_state.results:
        st.markdown("### 📊 Tamamlanan Testler")
        for t in st.session_state.results:
            if st.button(f"📄 {t}"):
                st.session_state.selected_test = t
                st.session_state.page = "view_report"
                st.rerun()
        st.markdown("---")
        if len(st.session_state.results) > 1:
            if st.button("🧩 Bütüncül (Harman) Rapor", type="primary"):
                st.session_state.page = "harman_report"
                st.rerun()

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
                with st.spinner("Sorular Grok API ile üretiliyor..."):
                    if "d2" in selected_test.lower():
                        st.session_state.current_test_data = {"type": "d2", "questions": generate_d2_grid()}
                        st.session_state.d2_isaretlenen = set()
                        st.session_state.d2_basla = False
                    elif "burdon" in selected_test.lower():
                        d, t = generate_burdon_content()
                        st.session_state.current_test_data = {"type": "burdon", "questions": d}
                        st.session_state.burdon_targets = t
                        st.session_state.burdon_basla = False
                        st.session_state.burdon_isaretlenen = {}
                        st.session_state.current_chunk = 0
                        st.session_state.burdon_limit = 600
                    else:
                        prompt = SORU_URETIM_PROMPT.format(test_adi=selected_test)
                        raw = get_data_from_ai(prompt)
                        try:
                            test_data = json.loads(raw)
                            if "Enneagram" in selected_test:
                                test_data["type"] = "enneagram"
                            else:
                                test_data["type"] = "likert"
                            st.session_state.current_test_data = test_data
                        except json.JSONDecodeError:
                            st.error("Soru üretimi başarısız. API yanıtı JSON değil.")
                            st.code(raw)
                            st.stop()
                st.session_state.page = "test"
                st.rerun()
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
                if "d2" in test_name:
                    st.session_state.d2_basla = True
                if "Burdon" in test_name:
                    st.session_state.burdon_basla = True
                    st.session_state.start_time = time.time()
                st.rerun()
    else:
        data = st.session_state.current_test_data
        q_type = data.get("type", "likert")
        questions = data.get("questions", [])
        st.markdown(f"## 📝 {test_name}")
       
        if q_type in ["enneagram", "likert"]:
            if 'cevaplar' not in st.session_state:
                st.session_state.cevaplar = {}
            if 'sayfa' not in st.session_state:
                st.session_state.sayfa = 0
           
            PER_PAGE = 10
            total = (len(questions) // PER_PAGE) + (1 if len(questions) % PER_PAGE else 0)
            start = st.session_state.sayfa * PER_PAGE
            current_qs = questions[start:start + PER_PAGE]
            st.progress((st.session_state.sayfa + 1) / total if total > 0 else 1)
           
            options_map = {"Kesinlikle Katılmıyorum": 1, "Katılmıyorum": 2, "Kararsızım": 3, "Katılıyorum": 4, "Kesinlikle Katılıyorum": 5}
            opts = list(options_map.keys())
            options_reverse = {v: k for k, v in options_map.items()}
           
            for q in current_qs:
                st.write(f"**{q['text']}**")
                q_id = q["id"]
                saved = st.session_state.cevaplar.get(q_id)
                default_index = opts.index(options_reverse[saved]) if saved in options_reverse else None
                sel = st.radio("Seçim:", opts, key=f"q_{q_id}", horizontal=True, label_visibility="collapsed", index=default_index)
                st.session_state.cevaplar[q_id] = options_map[sel]
                st.divider()
           
            c1, c2 = st.columns(2)
            if st.session_state.sayfa > 0:
                if c1.button("⬅️ Geri"):
                    st.session_state.sayfa -= 1
                    st.rerun()
            if st.session_state.sayfa < total - 1:
                if c2.button("İleri ➡️"):
                    st.session_state.sayfa += 1
                    st.rerun()
            else:
                if c2.button("BİTİR ✅", type="primary"):
                    if len(st.session_state.cevaplar) < len(questions):
                        st.warning(f"⚠️ Lütfen tüm soruları cevaplayınız! ({len(st.session_state.cevaplar)}/{len(questions)})")
                    else:
                        if "Enneagram" in test_name:
                            base, wing, scores = score_enneagram(st.session_state.cevaplar)
                            stats = {"Tip": base, "Kanat": wing, "Puanlar": scores}
                        else:
                            stats = {"Cevaplar": st.session_state.cevaplar}
                        st.session_state.results[test_name] = stats
                        with st.spinner("Analiz hazırlanıyor..."):
                            prompt = TEK_RAPOR_PROMPT.format(test_adi=test_name, cevaplar_json=json.dumps(stats, ensure_ascii=False))
                            st.session_state.reports[test_name] = get_data_from_ai(prompt)
                        st.session_state.page = "view_report"
                        st.rerun()
       
        elif q_type == "d2":
            @st.fragment
            def render_d2():
                cols_n = 10
                limit_show = 658
                rows = [questions[i:i+cols_n] for i in range(0, limit_show, cols_n)]
                sel = st.session_state.d2_isaretlenen
                for r_idx, row in enumerate(rows):
                    cols = st.columns(len(row))
                    for c_idx, item in enumerate(row):
                        lbl = item['visual']
                        is_sel = item['id'] in sel
                        cols[c_idx].button(lbl, key=f"d2_{item['id']}", type="primary" if is_sel else "secondary", on_click=toggle_d2_selection, args=(item['id'],))
            render_d2()
            st.divider()
            if st.button("TESTİ BİTİR 🏁", type="primary"):
                targets = [q['id'] for q in questions if q['is_target']]
                sel = st.session_state.d2_isaretlenen
                hits = len(set(targets).intersection(sel))
                false_al = len(sel - set(targets))
                miss = len(set(targets) - sel)
                stats = {"Doğru": hits, "Hata": false_al, "Atlanan": miss}
                st.session_state.results[test_name] = stats
                with st.spinner("Analiz..."):
                    prompt = TEK_RAPOR_PROMPT.format(test_adi="d2 Dikkat Testi", cevaplar_json=json.dumps(stats))
                    st.session_state.reports[test_name] = get_data_from_ai(prompt)
                st.session_state.page = "view_report"
                st.rerun()
       
        elif q_type == "burdon":
            CHUNK_SIZE = 50
            total = (len(questions) // CHUNK_SIZE) + 1
            LIMIT = st.session_state.burdon_limit
           
            @st.fragment(run_every=1)
            def timer():
                if not st.session_state.get("test_bitti", False):
                    elapsed = time.time() - st.session_state.start_time
                    rem = LIMIT - elapsed
                    if rem <= 0:
                        st.error("SÜRE DOLDU!")
                        st.session_state.test_bitti = True
                        st.rerun()
                    else:
                        m, s = divmod(int(rem), 60)
                        st.metric("Kalan Süre", f"{m:02d}:{s:02d}")
           
            @st.fragment
            def grid(seg):
                if st.session_state.get("test_bitti", False):
                    return
                st.info(f"HEDEFLER: {', '.join(st.session_state.burdon_targets)}")
                rows = [seg[i:i+10] for i in range(0, len(seg), 10)]
                curr = st.session_state.current_chunk
                if curr not in st.session_state.burdon_isaretlenen:
                    st.session_state.burdon_isaretlenen[curr] = set()
                sel = st.session_state.burdon_isaretlenen[curr]
                for r, row in enumerate(rows):
                    cols = st.columns(len(row))
                    for c, item in enumerate(row):
                        is_sel = item['id'] in sel
                        cols[c].button(item['char'], key=f"b_{item['id']}", type="primary" if is_sel else "secondary", on_click=toggle_burdon_selection, args=(item['id'], curr))
           
            timer()
            if not st.session_state.get("test_bitti", False):
                start = st.session_state.current_chunk * CHUNK_SIZE
                grid(questions[start:start + CHUNK_SIZE])
                st.divider()
                c1, c2 = st.columns([1, 4])
                if st.session_state.current_chunk < total - 1:
                    if c2.button("SONRAKİ ➡️"):
                        st.session_state.current_chunk += 1
                        st.rerun()
                else:
                    if c2.button("BİTİR 🏁", type="primary"):
                        st.session_state.test_bitti = True
                        st.rerun()
           
            if st.session_state.get("test_bitti", False):
                all_sel = set()
                for chunk in st.session_state.burdon_isaretlenen.values():
                    all_sel.update(chunk)
                targets = [q['id'] for q in questions if q['is_target']]
                hits = len(set(targets).intersection(all_sel))
                missed = len(set(targets) - all_sel)
                wrong = len(all_sel - set(targets))
                stats = {"Doğru": hits, "Atlanan": missed, "Yanlış": wrong}
                st.session_state.results[test_name] = stats
                with st.spinner("Analiz..."):
                    prompt = TEK_RAPOR_PROMPT.format(test_adi="Burdon Dikkat Testi", cevaplar_json=json.dumps(stats))
                    st.session_state.reports[test_name] = get_data_from_ai(prompt)
                st.session_state.page = "view_report"
                st.rerun()

# --- RAPOR ---
elif st.session_state.page == "view_report":
    t_name = st.session_state.selected_test
    st.title(f"📊 {t_name}")
   
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
        report = st.session_state.reports.get(t_name, "Rapor hazırlanamadı.")
        st.markdown(report)
        st.download_button("Raporu İndir", report, file_name=f"{t_name}_rapor.txt")
    with tab2:
        res = st.session_state.results.get(t_name, {})
        if "Enneagram" in t_name and "Puanlar" in res:
            fig = draw_radar_chart([f"Tip {k}" for k in res["Puanlar"].keys()], list(res["Puanlar"].values()), "Enneagram Profil")
            if fig:
                st.pyplot(fig)
        elif "d2" in t_name or "Burdon" in t_name:
            st.bar_chart({"Doğru": res.get("Doğru", 0), "Yanlış/Hata": res.get("Yanlış", res.get("Hata", 0)), "Atlanan": res.get("Atlanan", 0)})
        else:
            st.info("Bu test için grafik mevcut değil.")

# --- HARMAN RAPOR ---
elif st.session_state.page == "harman_report":
    st.markdown("## 🧩 Bütüncül Profil Raporu")
    if st.button("🏠 Ana Sayfaya Dön"):
        st.session_state.page = "home"
        st.rerun()
   
    if st.button("HARMAN RAPOR OLUŞTUR"):
        with st.spinner("Bütüncül analiz hazırlanıyor..."):
            prompt = HARMAN_RAPOR_PROMPT.format(tum_cevaplar_json=json.dumps(st.session_state.results, ensure_ascii=False))
            report = get_data_from_ai(prompt)
            st.markdown(report)

# --- REFERANSLAR ---
citations = [
    "https://www.apa.org/science/programs/testing/standards",
    "https://www.enneagraminstitute.com/rheti"
]
st.sidebar.markdown("**Referanslar:**")
for link in citations:
    st.sidebar.markdown(f"- {link}")
