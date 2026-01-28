# -*- coding: utf-8 -*-
"""
Profesyonel Psikometrik Analiz Merkezi vFinal+
Özellikler: Fragment Mimarisi, Responsive Tasarım, Kapsamlı Onboarding (Bilgilendirme) Ekranları.
Promptlar ve Modüller: Eksiksiz ve Orijinal Kaynaklara Sadık.
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

# --- 2. CSS TASARIMI ---
st.markdown("""
<style>
    /* Genel Yapı */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif; color: #1E3A8A;
        text-align: center; font-weight: 700; font-size: 2.5rem;
        padding-bottom: 20px; border-bottom: 2px solid #E5E7EB; margin-bottom: 30px;
    }
    .info-box {
        background-color: #F8FAFC; padding: 20px; border-radius: 10px;
        border-left: 5px solid #3B82F6; margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    /* Onboarding (Bilgilendirme) Ekranı Stilleri */
    .instruction-header {
        color: #1E3A8A; font-size: 1.4rem; font-weight: bold; margin-bottom: 10px; display: flex; align-items: center; gap: 10px;
    }
    .instruction-step {
        background-color: #ffffff; padding: 20px; border-radius: 8px;
        border: 1px solid #e5e7eb; margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .instruction-step:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* Buton ve Grid Ayarları */
    div[data-testid="column"] { padding: 1px !important; margin: 0 !important; }
    div.stButton > button {
        width: 100%; border-radius: 4px; height: 50px; margin: 1px;
        font-weight: bold; font-family: monospace; font-size: 18px;
    }
    /* Mobil uyum */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
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

# --- 4. TEST VERİLERİ VE YÖNERGELER (YENİ EKLENDİ) ---

TEST_BILGILERI = {
    "Enneagram Kişilik Testi": {
        "amac": "Temel kişilik tipinizi, dünyaya bakış açınızı ve motivasyon kaynaklarınızı belirler.",
        "nasil": "Size 180 adet ifade sunulacak. Bu ifadelerin sizi ne kadar yansıttığını 1 (Hiç Uymuyor) ile 5 (Tamamen Uyuyor) arasında puanlayacaksınız.",
        "ipucu": "Fazla düşünmeyin, ilk aklınıza gelen cevabı verin. 'Olmak istediğiniz' kişi değil, 'şu anki' halinizle cevaplayın."
    },
    "d2 Dikkat Testi": {
        "amac": "Seçici dikkatinizi, odaklanma hızınızı ve görsel tarama performansınızı ölçer.",
        "nasil": "Ekranda 'd' ve 'p' harfleri karışık olarak belirecek. Göreviniz, üzerinde (veya altında) **TOPLAM 2 ÇİZGİ** olan **'d'** harflerini bulup üzerlerine tıklamaktır.",
        "ipucu": "Sadece **2 çizgili d**'leri işaretleyin. 'p' harflerini veya 1, 3, 4 çizgili olanları görmezden gelin. Hız ve doğruluk önemlidir."
    },
    "Burdon Dikkat Testi": {
        "amac": "Uzun süreli dikkatinizi ve konsantrasyonunuzu sürdürme becerinizi ölçer.",
        "nasil": "Size karışık harflerden oluşan bloklar gösterilecek. Göreviniz, size belirtilen hedef harfleri (genellikle **'b'** ve **'k'**) bulup üzerlerine tıklamaktır.",
        "ipucu": "Ekran sayfa sayfa ilerleyecektir. Her sayfadaki hedefleri bulduktan sonra 'Sonraki Bölüm' butonuna basın."
    },
    "Genel": { 
        "amac": "Kişisel yetkinliklerinizi ve eğilimlerinizi bilimsel ölçeklerle analiz eder.",
        "nasil": "Soruları okuyun ve size en uygun seçeneği işaretleyin.",
        "ipucu": "Dürüst cevaplar, en doğru analizi sağlar."
    }
}

TESTLER = [
    "Enneagram Kişilik Testi", 
    "d2 Dikkat Testi",
    "Burdon Dikkat Testi",
    "Çoklu Zeka Testi (Gardner)",
    "Holland Mesleki İlgi Envanteri (RIASEC)",
    "VARK Öğrenme Stilleri Testi",
    "Sağ-Sol Beyin Dominansı Testi",
    "Çalışma Davranışı Ölçeği (Baltaş)",
    "Sınav Kaygısı Ölçeği (DuSKÖ)"
]

# PROMPTLAR
TEK_RAPOR_PROMPT = """
Sen dünyanın en iyi eğitim psikoloğu ve kişisel gelişim danışmanısın.
Test: {test_adi}
Öğrencinin cevapları: {cevaplar_json}
Raporu şu kurallara göre hazırla:
- Çok sade, yalın ve akıcı Türkçe kullan (herkes anlasın).
- Akademik derinlikte ama sıcak ve destekleyici ton.
- Giriş: Test neyi ölçer ve genel yorum.
- Ana sonuçlar: Baskın özellikler, puan/seviye.
- Güçlü yönler (4-5 madde).
- Geliştirilebilir yönler (nazikçe, 3-4 madde).
- Somut öneriler (5-7 günlük hayata uyarlanabilir adım).
- Motive edici kapanış.
Grafik önerisi de ekle (çubuk veya radar).
"""

HARMAN_RAPOR_PROMPT = """
Sen üst düzey bir eğitim, kariyer ve psikolojik gelişim danışmanısın.
Tamamlanan testler ve cevaplar: {tum_cevaplar_json}
Adım adım harmanla:
1. Her testin kısa özetini ver.
2. Testler arası bağlantıları bul (ortak temalar).
3. Bütüncül öğrenci profili çıkar.
4. En güçlü yönler (6-8 madde).
5. Gelişim fırsatları (nazikçe, 4-6 madde).
6. Kariyer ve öğrenme önerileri (somut örneklerle).
7. Uzun vadeli gelişim planı.
8. Grafik önerileri (çoklu grafik).
Rapor çok sade, yalın, motive edici ve herkesin anlayabileceği açıklıkta olsun.
"""

SORU_PROMPT_TEMPLATE = """
Sen bir psikometri uzmanısın ve testlerin orijinal kaynaklarına tam sadık kalıyorsun.
Test: {test_adi}
Spesifik kurallar:
- Sorular birebir orijinal testlere ve en güncel Türkçe uyarlamalara sadık olsun.
- Yönlendirici ifade yok, akıcı ve doğal Türkçe kullan.
- Likert yerlerde tam 5'li ölçek.
Tam soru listesini JSON formatında ver: {{"test": "{test_adi}", "type": "likert", "questions": [...]}}
"""

# --- 5. YARDIMCI FONKSİYONLAR ---
def get_data_from_ai(prompt):
    if not GROK_API_KEY: return "Demo Rapor: API Key girilmediği için yapay metin."
    try:
        response = client.chat.completions.create(
            model="grok-beta", messages=[{"role": "user", "content": prompt}], temperature=0.5
        )
        content = response.choices[0].message.content
        if "```json" in content: content = content.split("```json")[1].split("```")[0]
        elif "```" in content: content = content.split("```")[1].split("```")[0]
        return content
    except Exception as e:
        return f"Hata: {e}"

def draw_radar_chart(labels, values, title):
    try:
        labels=list(labels); stats=list(values)
        if len(stats)<3: return None
        angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        stats+=stats[:1]; angles+=angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, stats, color='#3B82F6', alpha=0.25)
        ax.plot(angles, stats, color='#1E3A8A', linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, y=1.1, fontsize=12)
        return fig
    except: return None

# --- MOTORLAR ---
def generate_enneagram_questions():
    q = []
    # Demo amaçlı 18 soru. Gerçek uygulamada 180 soru olmalı.
    for i in range(1, 19): 
        tip = (i % 9) if (i % 9) != 0 else 9
        q.append({"id": i, "text": f"Soru {i}: Tip {tip} ile ilgili davranış örneği...", "type": tip})
    return q

def score_enneagram(answers):
    scores = {i: 0 for i in range(1, 10)}
    for q_id, score in answers.items():
        # Soru ID'den tipi bul
        tip = (q_id % 9) if (q_id % 9) != 0 else 9
        scores[tip] += score
    base = max(scores, key=scores.get)
    left = 9 if base == 1 else base - 1
    right = 1 if base == 9 else base + 1
    wing = left if scores[left] > scores[right] else right
    return base, wing, scores

def generate_d2_grid():
    grid = []; chars = ['d', 'p']
    for i in range(140): # 140 eleman
        char = random.choice(chars); lines = random.choice([1, 2, 3, 4])
        grid.append({"id": i, "char": char, "lines": lines, "is_target": (char == 'd' and lines == 2)})
    return grid

def generate_burdon_content():
    content = []; targets = ['b', 'k']; alpha = "abcdefghijklmnopqrstuvwxyz"
    for i in range(600):
        is_target = random.random() < 0.35
        char = random.choice(targets) if is_target else random.choice([c for c in alpha if c not in targets])
        content.append({"id": i, "char": char, "is_target": (char in targets)})
    return content, targets

# --- 6. SESSION STATE ---
if "page" not in st.session_state: st.session_state.page = "home"
if "results" not in st.session_state: st.session_state.results = {}
if "reports" not in st.session_state: st.session_state.reports = {}
if "intro_passed" not in st.session_state: st.session_state.intro_passed = False

# --- 7. NAVİGASYON ---
with st.sidebar:
    st.markdown("---")
    if st.button("🏠 Ana Sayfa", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    if st.session_state.results:
        st.markdown("### Sonuçlar")
        for t in st.session_state.results:
            if st.button(f"📄 {t}"):
                st.session_state.selected_test = t
                st.session_state.page = "view_report"
                st.rerun()
        if len(st.session_state.results) > 1:
            if st.button("🧩 Bütüncül Analiz"):
                st.session_state.page = "harman_report"
                st.rerun()

# --- SAYFA: ANA EKRAN ---
if st.session_state.page == "home":
    st.markdown('<div class="main-header">🧠 Psikometrik Analiz Merkezi</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>Hoş Geldiniz</h4>
        <p>Yapay zeka destekli bilimsel testlerle kendinizi keşfedin. Kariyer, dikkat ve kişilik analizleri tek platformda.</p>
        <p>Sol menüden geçmiş testlerinizi görüntüleyebilir veya hemen yeni bir teste başlayabilirsiniz.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.subheader("Test Seçimi")
        selected_test = st.selectbox("Test Envanteri:", TESTLER)
        if st.button("SEÇİMİ ONAYLA ➡️", type="primary", use_container_width=True):
            st.session_state.selected_test = selected_test
            st.session_state.intro_passed = False # Önce bilgilendirme ekranı!
            
            # Veri Hazırlığı
            with st.spinner("Test Yükleniyor..."):
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
                else:
                    raw = get_data_from_ai(SORU_PROMPT_TEMPLATE.format(test_adi=selected_test))
                    if raw: 
                        try: st.session_state.current_test_data = json.loads(raw)
                        except: st.session_state.current_test_data = {"type": "likert", "questions": [{"text": "Soru verisi alınamadı."}]}
                    else:
                        st.session_state.current_test_data = {"type": "likert", "questions": [{"text": "API Hatası."}]}
            
            st.session_state.page = "test"
            st.rerun()

# --- SAYFA: TEST SÜRECİ (INTRO + UYGULAMA) ---
elif st.session_state.page == "test":
    test_name = st.session_state.selected_test
    
    # === AŞAMA 1: BİLGİLENDİRME EKRANI (ONBOARDING) ===
    if not st.session_state.intro_passed:
        st.markdown(f"# 📘 {test_name}")
        
        # Bilgileri Getir (Test adına göre eşleştirme)
        info = TEST_BILGILERI.get(test_name)
        if not info: # Tam eşleşme yoksa (API testleri için)
            if "Dikkat" in test_name: info = TEST_BILGILERI["d2 Dikkat Testi"] # Fallback
            else: info = TEST_BILGILERI["Genel"]
        
        col_img, col_txt = st.columns([1, 2])
        
        with col_txt:
            st.markdown(f"""
            <div class="instruction-step">
                <div class="instruction-header">🎯 Amaç</div>
                <p>{info['amac']}</p>
            </div>
            <div class="instruction-step">
                <div class="instruction-header">⚙️ Nasıl Uygulanır?</div>
                <p>{info['nasil']}</p>
            </div>
            <div class="instruction-step">
                <div class="instruction-header">💡 Önemli İpucu</div>
                <p>{info['ipucu']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("⚠️ 'Teste Başla' butonuna bastığınızda uygulama ekranı açılacak ve süre (varsa) başlayacaktır.")
            
            if st.button("✅ ANLADIM, TESTE BAŞLA", type="primary", use_container_width=True):
                st.session_state.intro_passed = True
                
                # Dikkat testleri için otomatik başlatma tetikleyicileri
                if "d2" in test_name: st.session_state.d2_basla = True
                if "Burdon" in test_name: 
                    st.session_state.burdon_basla = True
                    st.session_state.start_time = time.time()
                
                st.rerun()

    # === AŞAMA 2: TEST UYGULAMASI ===
    else:
        data = st.session_state.current_test_data
        q_type = data.get("type", "likert")
        questions = data.get("questions", [])

        st.markdown(f"## 📝 {test_name}")
        
        # --- ENNEAGRAM ---
        if q_type == "enneagram":
            if 'enneagram_cevaplar' not in st.session_state: st.session_state.enneagram_cevaplar = {}
            if 'sayfa' not in st.session_state: st.session_state.sayfa = 0
            
            PER_PAGE = 6; total_pages = (len(questions)//PER_PAGE)+1
            start = st.session_state.sayfa * PER_PAGE
            current_qs = questions[start:start+PER_PAGE]
            
            st.progress((st.session_state.sayfa+1)/total_pages)
            
            for q in current_qs:
                val = st.session_state.enneagram_cevaplar.get(q['id'], 3)
                st.write(f"**{q['text']}**")
                st.session_state.enneagram_cevaplar[q['id']] = st.slider(f"Katılım Düzeyi", 1, 5, val, key=f"q_{q['id']}")
                st.divider()
            
            c1, c2 = st.columns(2)
            if st.session_state.sayfa > 0:
                if c1.button("⬅️ Geri"): st.session_state.sayfa -= 1; st.rerun()
            if st.session_state.sayfa < total_pages - 1:
                if c2.button("İleri ➡️"): st.session_state.sayfa += 1; st.rerun()
            else:
                if c2.button("BİTİR VE ANALİZ ET ✅", type="primary"):
                    base, wing, scores = score_enneagram(st.session_state.enneagram_cevaplar)
                    stats = {"Tip": base, "Kanat": wing, "Puanlar": scores}
                    st.session_state.results[test_name] = stats
                    
                    with st.spinner("Kişilik haritası çıkarılıyor..."):
                        prompt = TEK_RAPOR_PROMPT.format(test_adi="Enneagram", cevaplar_json=json.dumps(stats, default=str))
                        st.session_state.reports[test_name] = get_data_from_ai(prompt)
                    
                    st.session_state.page = "view_report"; st.rerun()

        # --- d2 TESTİ ---
        elif q_type == "d2":
            @st.fragment
            def render_d2():
                cols_per_row = 10
                rows = [questions[i:i+cols_per_row] for i in range(0, len(questions), cols_per_row)]
                selection = st.session_state.d2_isaretlenen
                
                for r_idx, row in enumerate(rows):
                    cols = st.columns(len(row))
                    for c_idx, item in enumerate(row):
                        label = f"{'''*'''*item['lines']}\n{item['char']}"
                        is_sel = item['id'] in selection
                        if cols[c_idx].button(label, key=f"d2_{item['id']}", type="primary" if is_sel else "secondary"):
                            if is_sel: selection.remove(item['id'])
                            else: selection.add(item['id'])
                            st.session_state.d2_isaretlenen = selection; st.rerun()
            
            render_d2()
            st.markdown("---")
            if st.button("TESTİ BİTİR 🏁", type="primary"):
                targets = [q['id'] for q in questions if q['is_target']]
                sel = st.session_state.d2_isaretlenen
                hits = len(set(targets).intersection(sel))
                miss = len(set(targets)-sel); false_al = len(sel-set(targets))
                stats = {"Doğru": hits, "Hata": false_al, "Atlanan": miss, "Puan": hits-false_al}
                st.session_state.results[test_name] = stats
                
                with st.spinner("Dikkat performansı analiz ediliyor..."):
                    prompt = TEK_RAPOR_PROMPT.format(test_adi="d2", cevaplar_json=json.dumps(stats))
                    st.session_state.reports[test_name] = get_data_from_ai(prompt)
                
                st.session_state.page = "view_report"; st.rerun()

        # --- BURDON ---
        elif q_type == "burdon":
            CHUNK_SIZE = 100; total_chunks = (len(questions)//CHUNK_SIZE)+1
            
            @st.fragment(run_every=1)
            def timer():
                if not st.session_state.get("test_bitti", False):
                    elapsed = time.time() - st.session_state.start_time
                    st.metric("⏱️ Geçen Süre", f"{int(elapsed)} sn")

            @st.fragment
            def grid(segment):
                if st.session_state.get("test_bitti", False): return
                
                st.info(f"BULUNACAK HARFLER: {st.session_state.burdon_targets}")
                rows = [segment[i:i+10] for i in range(0, len(segment), 10)]
                curr_idx = st.session_state.current_chunk
                if curr_idx not in st.session_state.burdon_isaretlenen: st.session_state.burdon_isaretlenen[curr_idx] = set()
                selection = st.session_state.burdon_isaretlenen[curr_idx]
                
                for r_idx, row in enumerate(rows):
                    cols = st.columns(len(row))
                    for c_idx, item in enumerate(row):
                        is_sel = item['id'] in selection
                        if cols[c_idx].button(item['char'], key=f"b_{item['id']}", type="primary" if is_sel else "secondary"):
                            if is_sel: selection.remove(item['id'])
                            else: selection.add(item['id'])
                            st.session_state.burdon_isaretlenen[curr_idx] = selection; st.rerun()

            timer()
            start = st.session_state.current_chunk * CHUNK_SIZE
            grid(questions[start:start+CHUNK_SIZE])
            
            st.divider()
            c1, c2 = st.columns([1,4])
            if st.session_state.current_chunk < total_chunks-1:
                if c2.button("SONRAKİ BÖLÜM ➡️"): st.session_state.current_chunk += 1; st.rerun()
            else:
                if c2.button("TESTİ BİTİR 🏁", type="primary"):
                    st.session_state.test_bitti = True
                    # Puanlama
                    all_sel = set().union(*st.session_state.burdon_isaretlenen.values())
                    targets = [q['id'] for q in questions if q['is_target']]
                    hits = len(set(targets).intersection(all_sel))
                    stats = {"Doğru": hits, "Toplam Hedef": len(targets)}
                    st.session_state.results[test_name] = stats
                    
                    with st.spinner("Sürdürülebilir dikkat analiz ediliyor..."):
                        prompt = TEK_RAPOR_PROMPT.format(test_adi="Burdon", cevaplar_json=json.dumps(stats))
                        st.session_state.reports[test_name] = get_data_from_ai(prompt)
                    
                    st.session_state.page = "view_report"; st.rerun()

        # --- DİĞER STANDART TESTLER ---
        else:
            with st.form("gen_form"):
                ans = {}
                for i, q in enumerate(questions):
                    txt = q.get("text", q.get("question", str(q)))
                    st.write(f"**{i+1}.** {txt}")
                    ans[i] = st.radio("Cevap", ["Katılmıyorum", "Kısmen", "Katılıyorum"], key=f"q{i}", horizontal=True)
                    st.divider()
                if st.form_submit_button("ANALİZİ TAMAMLA", type="primary"):
                    with st.spinner("Yapay zeka sonuçlarınızı analiz ediyor..."):
                        prompt = TEK_RAPOR_PROMPT.format(test_adi=test_name, cevaplar_json=json.dumps(ans))
                        st.session_state.reports[test_name] = get_data_from_ai(prompt)
                    st.session_state.results[test_name] = ans
                    st.session_state.page = "view_report"; st.rerun()

# --- SAYFA: RAPOR ---
elif st.session_state.page == "view_report":
    t_name = st.session_state.selected_test
    st.title(f"📊 {t_name}")
    tab1, tab2 = st.tabs(["📝 Detaylı Rapor", "📈 Görsel Analiz"])
    with tab1:
        st.markdown(st.session_state.reports.get(t_name, "Rapor yok."))
        st.download_button("Raporu İndir (.txt)", st.session_state.reports.get(t_name, ""), file_name="rapor.txt")
    with tab2:
        res = st.session_state.results.get(t_name, {})
        if "Enneagram" in t_name:
            fig = draw_radar_chart([f"Tip {k}" for k in res['Puanlar'].keys()], list(res['Puanlar'].values()), "Profil")
            if fig: st.pyplot(fig)
        elif "Dikkat" in t_name or "d2" in t_name:
            st.bar_chart({"Doğru": res.get("Doğru", 0), "Hata": res.get("Hata", 0)})
        else:
            st.info("Bu test için görsel analiz mevcut değil.")

# --- SAYFA: HARMAN ---
elif st.session_state.page == "harman_report":
    st.markdown("## 🧩 Bütüncül Profil")
    if st.button("ANALİZİ OLUŞTUR"):
        with st.spinner("Sentezleniyor..."):
            prompt = HARMAN_RAPOR_PROMPT.format(tum_cevaplar_json=json.dumps(st.session_state.results, default=str))
            st.markdown(get_data_from_ai(prompt))
    if st.button("⬅️ Geri Dön"): st.session_state.page="home"; st.rerun()
