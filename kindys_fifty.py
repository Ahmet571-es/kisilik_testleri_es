# -*- coding: utf-8 -*-
"""
Bu dosya Streamlit uygulamasıdır.
Spyder üzerinden değil, Anaconda Prompt üzerinden şu komutla çalıştırılır:
streamlit run dosya_adi.py
"""

import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Psikometrik Test Merkezi", layout="wide")

# --- API AYARLARI ---
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")

if not GROK_API_KEY:
    # Eğer .env yoksa veya okuyamazsa geçici olarak buraya key yazılabilir (önerilmez)
    # GROK_API_KEY = "xai-..." 
    st.error("⚠️ GROK_API_KEY bulunamadı! .env dosyasını kontrol edin.")
    st.stop()

client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

# --- ORİJİNAL PROMPTLAR (DEĞİŞTİRİLMEDİ) ---

TESTLER = [
    "Çoklu Zeka Testi (Gardner)",
    "Çalışma Davranışı Ölçeği (Baltaş)",
    "Sınav Kaygısı Ölçeği (DuSKÖ)",
    "Burdon Dikkat Testi",
    "Holland Mesleki İlgi Envanteri (RIASEC)",
    "VARK Öğrenme Stilleri Testi",
    "Sağ-Sol Beyin Dominansı Testi"
]

SORU_PROMPT_TEMPLATE = """
Sen bir psikometri uzmanısın ve testlerin orijinal kaynaklarına tam sadık kalıyorsun.
Test: {test_adi}

Spesifik kurallar:
- Sorular birebir orijinal testlere ve en güncel Türkçe uyarlamalara sadık olsun (kaynaklar aşağıda).
- Yönlendirici ifade yok, akıcı ve doğal Türkçe kullan (devrik cümle kesinlikle yok).
- Likert yerlerde tam 5'li ölçek: Kesinlikle katılmıyorum, Pek katılmıyorum, Emin değilim, Biraz katılıyorum, Kesinlikle katılıyorum.
- Bias içermesin, kültürel olarak nötr olsun.

Test-spesifik kaynaklar ve talimatlar:
- Çoklu Zeka (Gardner): Howard Gardner 1983 teorisi + MIDAS Türkçe uyarlaması (International Journal of Human Sciences); 8 alan dengeli, ~79 madde Likert.
- Çalışma Davranışı (Baltaş): Acar Baltaş orijinal 73 madde Doğru/Yanlış.
- Sınav Kaygısı (DuSKÖ): Resmi DergiPark makalesi (2020'ler, 22 madde 5'li Likert, bio-psikososyal).
- Burdon Dikkat: Klasik a/b/d/g harf gridi, standart performans formatı.
- Holland RIASEC: John Holland modeli + Türkçe PGI-S uyarlaması (90 madde, Beğenirim/Beğenmem).
- VARK: Neil Fleming orijinal 16 madde çoklu seçim (eleştirel not: öğrenme çıktılarıyla ilişki sınırlı).
- Sağ-Sol Beyin: Popüler Sperry temelli 18 madde ikili seçim (not: nörobilimce mit).

Tam soru listesini JSON formatında ver: {{"test": "{test_adi}", "type": "likert/burdon/riaec/vark/binary", "questions": [...]}}
"""

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

# --- FONKSİYONLAR ---

def get_questions_api(test_name):
    """API'den soruları çeker ve JSON hatası varsa düzeltir"""
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": SORU_PROMPT_TEMPLATE.format(test_adi=test_name)}],
            temperature=0.5, # Biraz yaratıcılık için artırdık
            max_tokens=4000
        )
        content = response.choices[0].message.content
        # Markdown temizliği (API bazen ```json ... ``` döner)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content)
    except Exception as e:
        st.error(f"Hata: {e}")
        return None

def get_report_api(prompt):
    try:
        response = client.chat.completions.create(
            model="grok-4-1-fast-reasoning",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Rapor hatası: {e}"

# --- SESSION STATE (DURUM YÖNETİMİ) ---
if "page" not in st.session_state:
    st.session_state.page = "home"
if "results" not in st.session_state:
    st.session_state.results = {} # Tamamlanan testlerin cevapları burada tutulur
if "current_test_data" not in st.session_state:
    st.session_state.current_test_data = None

# --- NAVİGASYON ---
def go_home():
    st.session_state.page = "home"

# --- SAYFALAR ---

# 1. GİRİŞ SAYFASI
if st.session_state.page == "home":
    st.title("🧠 Kapsamlı Kişisel Gelişim Testleri")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("Tamamlanan Testler")
        if st.session_state.results:
            for t in st.session_state.results.keys():
                st.success(f"✅ {t}")
            
            if len(st.session_state.results) > 1:
                if st.button("Harmanlanmış Genel Rapor Al"):
                    st.session_state.page = "harman_report"
                    st.rerun()
        else:
            st.write("_Henüz test yapılmadı._")

    with col2:
        st.subheader("Yeni Test Başlat")
        selected_test = st.selectbox("Test Seçiniz:", TESTLER)
        
        if st.button("Testi Başlat", type="primary"):
            with st.spinner("Yapay zeka soruları orijinal kaynağa sadık kalarak hazırlıyor..."):
                data = get_questions_api(selected_test)
                if data:
                    st.session_state.current_test_data = data
                    st.session_state.selected_test = selected_test
                    st.session_state.page = "test"
                    st.rerun()

# 2. TEST SAYFASI (FORM YAPISI)

elif st.session_state.page == "test":
    data = st.session_state.current_test_data
    test_name = st.session_state.selected_test
    questions = data.get("questions", [])
    q_type = data.get("type", "likert")
    
    st.markdown(f"## 📝 {test_name}")
    
    # --- VARK BİLGİLENDİRMESİ ---
    if "VARK" in test_name:
        with st.expander("ℹ️ Teste Başlamadan Önce: V, A, R, K Nedir?", expanded=True):
            st.markdown("""
            Bu test öğrenme stilinizi belirler. Harflerin anlamları şöyledir:
            * **👀 V (Visual - Görsel):** Görerek öğrenenler. Grafik, harita ve şemaları severler.
            * **👂 A (Aural - İşitsel):** Duyarak öğrenenler. Dinlemeyi ve tartışmayı severler.
            * **📖 R (Read/Write - Okuma/Yazma):** Okuyup yazarak öğrenenler. Not tutmayı severler.
            * **✋ K (Kinesthetic - Kinestetik):** Dokunarak öğrenenler. Deney ve uygulamayı severler.
            """)
    
    # Form kullanarak sayfa yenilenmesini engelliyoruz
    with st.form(key="test_form"):
        user_answers = {}
        
        for i, q in enumerate(questions):
            # --- GÜVENLİ VERİ OKUMA (HATA DÜZELTİLDİ) ---
            if isinstance(q, dict):
                # AI bazen 'text' yerine 'question' diyebilir, önlem alıyoruz:
                q_text = q.get("text", q.get("question", str(q)))
            else:
                q_text = str(q)
            # -------------------------------------------

            st.markdown(f"**{i+1}. {q_text}**")
            
            # Soru tiplerine göre görselleştirme
            if q_type == "likert":
                user_answers[i] = st.radio(
                    "Cevabınız:",
                    ["Kesinlikle katılmıyorum", "Pek katılmıyorum", "Emin değilim", "Biraz katılıyorum", "Kesinlikle katılıyorum"],
                    key=f"q_{i}",
                    index=None, 
                    horizontal=True
                )
            
            elif q_type in ["binary", "riaec"]: 
                user_answers[i] = st.radio(
                    "Seçim:", 
                    ["Beğenmem / Hayır", "Beğenirim / Evet"], 
                    key=f"q_{i}",
                    index=None,
                    horizontal=True
                )
            
            elif q_type == "vark":
                opts = q.get("options", []) if isinstance(q, dict) else []
                # Eğer seçenekler gelmezse hata vermesin diye boş liste kontrolü
                if not opts: 
                    opts = ["Seçenek yüklenemedi", "Lütfen sayfayı yenileyin"]
                user_answers[i] = st.multiselect("Size uygun olanları seçin:", opts, key=f"q_{i}")
                
            elif q_type == "burdon":
                if isinstance(q, dict) and "grid" in q:
                    st.code(q["grid"])
                user_answers[i] = st.multiselect("İstenen harfleri işaretle (a,b,d,g):", ["a", "b", "d", "g"], key=f"q_{i}")
                
            st.markdown("---")
        
        submit_btn = st.form_submit_button("Testi Bitir ve Raporla")
        
    if submit_btn:
        if q_type == "likert" and any(v is None for v in user_answers.values()):
            st.warning("Lütfen tüm soruları cevaplayınız.")
        else:
            st.session_state.results[test_name] = user_answers
            st.session_state.page = "single_report"
            st.rerun()
            
    if st.button("❌ İptal Et"):
        go_home()
        st.rerun()

# 3. TEK TEST RAPORU
elif st.session_state.page == "single_report":
    test_name = st.session_state.selected_test
    answers = st.session_state.results[test_name]
    
    st.balloons()
    st.title("📊 Test Sonuç Raporu")
    
    with st.spinner("Uzman görüşü hazırlanıyor..."):
        prompt = TEK_RAPOR_PROMPT.format(test_adi=test_name, cevaplar_json=json.dumps(answers, ensure_ascii=False))
        report = get_report_api(prompt)
        
    st.markdown(report)
    
    if st.button("Ana Menüye Dön"):
        go_home()
        st.rerun()

# 4. HARMANLANMIŞ RAPOR
elif st.session_state.page == "harman_report":
    st.title("🧩 Bütüncül Kişilik ve Kariyer Analizi")
    st.info("Tamamladığınız tüm testler birleştirilerek analiz ediliyor...")
    
    with st.spinner("Büyük veri analizi yapılıyor..."):
        prompt = HARMAN_RAPOR_PROMPT.format(tum_cevaplar_json=json.dumps(st.session_state.results, ensure_ascii=False))
        harman_report = get_report_api(prompt)
        
    st.markdown(harman_report)
    
    # Basit bir grafik örneği
    st.write("---")
    st.subheader("Test Katılım İstatistiği")
    fig, ax = plt.subplots()
    ax.bar(list(st.session_state.results.keys()), [len(v) for v in st.session_state.results.values()], color="#00695c")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    
    if st.button("Ana Menüye Dön"):
        go_home()
        st.rerun()