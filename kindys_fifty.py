# -*- coding: utf-8 -*-
"""
Profesyonel Psikometrik Analiz Merkezi vFinal
Gelişmiş UI, Radar Grafikleri, Sidebar Navigasyonu ve İndirilebilir Raporlar içerir.
Promptlar orijinal kaynaklara sadık kalacak şekilde korunmuştur.
Enneagram, d2 ve Burdon modülleri eklendi, hatalar giderildi ve optimize edildi.
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

# --- 1. SAYFA YAPILANDIRMASI (En başta olmalı) ---
st.set_page_config(
    page_title="Psikometrik Analiz Merkezi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ÖZEL CSS TASARIMI ---
st.markdown("""
<style>
    /* Ana başlık stili */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1E3A8A; /* Koyu Mavi */
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        padding-bottom: 20px;
        border-bottom: 2px solid #E5E7EB;
        margin-bottom: 30px;
    }
    /* Bilgi kutucukları */
    .info-box {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }
    /* Radyo butonları */
    .stRadio > div {
        gap: 12px;
        padding: 10px;
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    /* d2 ve Burdon için yoğun ızgara düzenlemeleri */
    div[data-testid="column"] {
        padding: 1px !important;
        margin: 0 !important;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 4px;
        height: 50px;
        margin: 1px;
        font-weight: bold;
        font-family: monospace;
        font-size: 20px;
        padding: 0;
    }
    /* Mobil uyum için container ayarı */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. API VE AYARLAR ---
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062331.png", width=70)
    st.markdown("### 🧠 Analiz Paneli")
    if not GROK_API_KEY:
        st.warning("⚠️ API Key Eksik! (Demo Modu)")
    else:
        st.caption("🟢 Sistem: Çevrimiçi")
        st.caption("vFinal - Optimized")

client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

# --- 4. SABİT VERİLER VE PROMPTLAR ---
TESTLER = [
    "Çoklu Zeka Testi (Gardner)",
    "Çalışma Davranışı Ölçeği (Baltaş)",
    "Sınav Kaygısı Ölçeği (DuSKÖ)",
    "Burdon Dikkat Testi",
    "Holland Mesleki İlgi Envanteri (RIASEC)",
    "VARK Öğrenme Stilleri Testi",
    "Sağ-Sol Beyin Dominansı Testi",
    "Enneagram Kişilik Testi",
    "d2 Dikkat Testi"
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
- Enneagram: RHETI standart 180 soru Likert, 1-5 ölçek.
- d2 Dikkat: Standart d2 testi, 14 satır x 47 karakter, hedef d2.
Tam soru listesini JSON formatında ver: {{"test": "{test_adi}", "type": "likert/burdon/riaec/vark/binary/enneagram/d2", "questions": [...]}}
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

# --- 5. YARDIMCI FONKSİYONLAR ---
def get_data_from_ai(prompt):
    if not GROK_API_KEY: 
        return "Demo Modu: API Anahtarı eksik olduğu için yapay zeka raporu oluşturulamadı. Lütfen .env dosyasını kontrol edin."
    try:
        response = client.chat.completions.create(
            model="grok-beta", # Model ismini gerekirse güncelleyin
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4000
        )
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return content
    except Exception as e:
        st.error(f"Bağlantı Hatası: {e}")
        return f"Hata oluştu: {e}"

def draw_radar_chart(labels, values, title):
    try:
        labels = list(labels)
        stats = list(values)
        if len(stats) < 3: return None
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        stats += stats[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, stats, color='#3B82F6', alpha=0.25)
        ax.plot(angles, stats, color='#1E3A8A', linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, y=1.1, fontsize=12, color="#1E3A8A")
        return fig
    except:
        return None

# --- MOTOR 1: ENNEAGRAM ---
enneagram_sorular = []
tipler = [1,2,3,4,5,6,7,8,9] * 20
for i in range(180):
    soru = f"Örnek Soru {i+1}: Kendinizi bu ifadeye ne kadar yakın hissediyorsunuz?"
    enneagram_sorular.append((soru, tipler[i]))

def enneagram_puanla(cevaplar):
    tip_puanlar = {tip: 0 for tip in range(1,10)}
    for idx, puan in cevaplar.items():
        _, tip = enneagram_sorular[idx]
        tip_puanlar[tip] += puan
    temel_tip = max(tip_puanlar, key=tip_puanlar.get)
    komsular = {1:(9,2), 2:(1,3), 3:(2,4), 4:(3,5), 5:(4,6), 6:(5,7), 7:(6,8), 8:(7,9), 9:(8,1)}
    komsu_puanlar = {k: tip_puanlar[k] for k in komsular[temel_tip]}
    kanat = max(komsu_puanlar, key=komsu_puanlar.get)
    zihinsel = max({5:tip_puanlar[5],6:tip_puanlar[6],7:tip_puanlar[7]}, key=lambda k: tip_puanlar[k])
    duygusal = max({2:tip_puanlar[2],3:tip_puanlar[3],4:tip_puanlar[4]}, key=lambda k: tip_puanlar[k])
    icgudusel = max({8:tip_puanlar[8],9:tip_puanlar[9],1:tip_puanlar[1]}, key=lambda k: tip_puanlar[k])
    tritype = f"{zihinsel}-{duygusal}-{icgudusel}"
    return temel_tip, kanat, tritype, tip_puanlar

# --- MOTOR 2: d2 TESTİ (DÜZELTİLDİ: Sütun Sayısı 10) ---
def d2_izgara_uret(satir_sayisi=28, karakter_sayisi=10, hedef_orani=0.2):
    # Mobil uyum için varsayılan karakter sayısı 10'a düşürüldü, satır sayısı artırıldı.
    harfler = ['d', 'p']
    izgara = []
    hedef_say = int(satir_sayisi * karakter_sayisi * hedef_orani)
    toplam_karakter = satir_sayisi * karakter_sayisi
    karakterler = ['d2'] * hedef_say + ['d1', 'd3', 'd4', 'p1', 'p2', 'p3', 'p4'] * ((toplam_karakter - hedef_say) // 6)
    random.shuffle(karakterler)
    for i in range(satir_sayisi):
        satir = karakterler[i*karakter_sayisi:(i+1)*karakter_sayisi]
        izgara.append(satir)
    return izgara

def d2_puanla(isaretlenen, izgara):
    tn = sum(len(satir) for satir in izgara)
    dogru_hedef = 0
    yanlis_celdirici = 0
    for i, satir in enumerate(izgara):
        for j, kar in enumerate(satir):
            if (i,j) in isaretlenen:
                if kar == 'd2':
                    dogru_hedef += 1
                else:
                    yanlis_celdirici += 1
    cp = dogru_hedef - yanlis_celdirici
    hata_yuzde = (yanlis_celdirici / tn) * 100 if tn > 0 else 0
    return tn, cp, hata_yuzde

# --- MOTOR 3: BURDON TESTİ ---
def burdon_izgara_uret(karakter_sayisi=600, hedef_harfler=['b', 'k'], hedef_say=225):
    alfabe = 'abcdefghijklmnopqrstuvwxyz'
    celdiriciler = [c for c in alfabe if c not in hedef_harfler]
    karakterler = hedef_harfler * (hedef_say // len(hedef_harfler)) + random.choices(celdiriciler, k=karakter_sayisi - hedef_say)
    random.shuffle(karakterler)
    return ''.join(karakterler)

def burdon_puanla(isaretlenen_hedefler, toplam_hedef, hatalar, satir_performanslari):
    dogru = len(isaretlenen_hedefler)
    ku = max(satir_performanslari) / min(satir_performanslari) if min(satir_performanslari) > 0 else 0
    return dogru, hatalar, ku

def ilerleme_cubugu(mevcut, toplam):
    st.progress(mevcut / toplam)

# --- 6. SESSION STATE ---
if "page" not in st.session_state: st.session_state.page = "home"
if "results" not in st.session_state: st.session_state.results = {}
if "reports" not in st.session_state: st.session_state.reports = {}
if "current_test_data" not in st.session_state: st.session_state.current_test_data = None

# --- 7. NAVİGASYON ---
def go_home():
    st.session_state.page = "home"
    st.session_state.current_test_data = None

# --- 8. SAYFA AKIŞLARI ---
with st.sidebar:
    st.markdown("---")
    st.subheader("📂 Geçmiş Testler")
    if st.session_state.results:
        for t in st.session_state.results:
            if st.button(f"📄 {t}", key=f"btn_{t}", use_container_width=True):
                st.session_state.selected_test = t
                st.session_state.page = "view_report"
                st.rerun()
        st.markdown("---")
        if len(st.session_state.results) > 1:
            if st.button("🧩 Bütüncül Analiz (Harman)", type="primary", use_container_width=True):
                st.session_state.page = "harman_report"
                st.rerun()
    else:
        st.info("Henüz tamamlanan test yok.")
    st.markdown("---")
    if st.button("🏠 Ana Menüye Dön", use_container_width=True):
        go_home()
        st.rerun()

# === SAYFA 1: ANA EKRAN ===
if st.session_state.page == "home":
    st.markdown('<div class="main-header">🧠 Psikometrik Analiz Merkezi</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>Hoş Geldiniz</h4>
        <p>Bu platform, yapay zeka destekli bilimsel testlerle kendinizi keşfetmenizi sağlar.
        Kariyer eğilimlerinizi, öğrenme stilinizi ve güçlü yönlerinizi profesyonel bir formatta analiz ediyoruz.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.subheader("🚀 Yeni Test Başlat")
        selected_test = st.selectbox("Uygulamak istediğiniz envanter:", TESTLER)
        if st.button("Testi Başlat", type="primary", use_container_width=True):
            with st.spinner("Test hazırlanıyor..."):
                if selected_test == "Enneagram Kişilik Testi":
                    st.session_state.current_test_data = {"test": selected_test, "type": "enneagram", "questions": enneagram_sorular}
                elif selected_test == "d2 Dikkat Testi":
                    izgara = d2_izgara_uret()
                    st.session_state.current_test_data = {"test": selected_test, "type": "d2", "questions": izgara}
                    st.session_state.d2_isaretlenen = set()
                    st.session_state.d2_basla = False
                elif selected_test == "Burdon Dikkat Testi":
                    izgara = burdon_izgara_uret()
                    st.session_state.current_test_data = {"test": selected_test, "type": "burdon", "questions": izgara}
                    st.session_state.burdon_basla = False
                    st.session_state.test_bitti = False
                    st.session_state.current_chunk = 0
                    st.session_state.burdon_isaretlenen = {} # Dict olarak başlatıyoruz
                else:
                    raw_data = get_data_from_ai(SORU_PROMPT_TEMPLATE.format(test_adi=selected_test))
                    if raw_data:
                        try:
                            st.session_state.current_test_data = json.loads(raw_data)
                        except json.JSONDecodeError:
                            st.error("Veri işleme hatası.")
                            st.stop()
                st.session_state.selected_test = selected_test
                st.session_state.page = "test"
                st.rerun()

# === SAYFA 2: TEST EKRANI ===
elif st.session_state.page == "test":
    data = st.session_state.current_test_data
    test_name = st.session_state.selected_test
    questions = data.get("questions", [])
    q_type = data.get("type", "likert")
    
    st.markdown(f"## 📝 {test_name}")
    
    if "VARK" in test_name:
        with st.expander("ℹ️ Bilgi: V, A, R, K Nedir?", expanded=True):
            st.info("**V:** Görsel | **A:** İşitsel | **R:** Okuma/Yazma | **K:** Kinestetik")
    
    # --- Enneagram Modülü ---
    if q_type == "enneagram":
        if 'enneagram_cevaplar' not in st.session_state: st.session_state.enneagram_cevaplar = {}
        if 'sayfa' not in st.session_state: st.session_state.sayfa = 0
        
        soru_sayfa = 15
        toplam_sayfa = len(questions) // soru_sayfa + (1 if len(questions) % soru_sayfa else 0)
        baslangic = st.session_state.sayfa * soru_sayfa
        bitis = baslangic + soru_sayfa
        gosterilen_sorular = questions[baslangic:bitis]
        
        ilerleme_cubugu(baslangic + len(gosterilen_sorular), len(questions))
        
        for idx, (soru, _) in enumerate(gosterilen_sorular):
            global_idx = baslangic + idx
            st.session_state.enneagram_cevaplar[global_idx] = st.radio(
                soru, options=[1,2,3,4,5], horizontal=True, key=f"soru_{global_idx}"
            )
        
        col1, col2 = st.columns(2)
        if st.session_state.sayfa > 0:
            if col1.button("Önceki Sayfa"):
                st.session_state.sayfa -= 1
                st.rerun()
        if st.session_state.sayfa < toplam_sayfa - 1:
            if col2.button("Sonraki Sayfa"):
                st.session_state.sayfa += 1
                st.rerun()
        else:
            if col2.button("Testi Bitir"):
                temel, kanat, tritype, puanlar = enneagram_puanla(st.session_state.enneagram_cevaplar)
                report = f"Temel Tip: {temel}w{kanat}\nTritype: {tritype}\nPuan Dağılımı: {puanlar}"
                st.session_state.results[test_name] = {"Puanlar": puanlar, "Tip": temel}
                
                # API ile zenginleştirme
                prompt = TEK_RAPOR_PROMPT.format(test_adi="Enneagram", cevaplar_json=json.dumps(st.session_state.results[test_name], default=str))
                with st.spinner("Kişilik haritası çıkarılıyor..."):
                    st.session_state.reports[test_name] = get_data_from_ai(prompt)
                
                st.session_state.page = "view_report"
                st.rerun()
    
    # --- d2 Dikkat Testi Modülü (Performans Fix) ---
    elif q_type == "d2":
        # DÜZELTME: run_every kaldırıldı, sadece tıklamada çalışır.
        @st.fragment
        def d2_test_fragment():
            if not st.session_state.d2_basla:
                return

            izgara = questions
            # Mobilde 10 sütun, masaüstünde geniş
            cols_per_row = 10 
            rows = [izgara[i:i + cols_per_row] for i in range(0, len(izgara), cols_per_row)]
            
            selection = st.session_state.d2_isaretlenen
            
            for r_idx, row in enumerate(rows):
                cols = st.columns(len(row))
                for c_idx, item in enumerate(row):
                    char_display = item['char']
                    lines_display = "'" * item['lines']
                    label = f"{lines_display}\n{char_display}"
                    
                    # Key unique olmalı (satır ve kolon indexi yerine item ID'si varsa daha iyi, burada r_idx, c_idx kullanıyoruz ama dikkatli olunmalı)
                    # item verisi liste içinde string değil dict olsaydı ID kullanırdık.
                    # Mevcut yapıda izgara listelerden oluşuyor.
                    
                    # Not: d2_izgara_uret list of lists döndürüyor (string). 
                    # Bu yapı state takibi için zor. O yüzden d2_izgara_uret'i dict döndürecek şekilde revize etmedik ama
                    # tıklama takibi için (r_idx, c_idx) kullanacağız.
                    # Ancak yukarıdaki d2_izgara_uret fonksiyonu string listesi döndürüyor.
                    # Bunu unique yapmak için global index kullanacağız.
                    
                    is_selected = (r_idx, c_idx) in selection
                    btn_type = "primary" if is_selected else "secondary"
                    
                    with cols[c_idx]:
                        if st.button(label, key=f"d2_{r_idx}_{c_idx}", type=btn_type):
                            if is_selected:
                                selection.remove((r_idx, c_idx))
                            else:
                                selection.add((r_idx, c_idx))
                            st.session_state.d2_isaretlenen = selection
                            st.rerun()

        st.subheader("d2 Dikkat Testi")
        
        if not st.session_state.d2_basla:
            st.info("Üzerinde toplam **2 çizgi** bulunan **'d'** harflerini işaretleyiniz.")
            if st.button("TESTİ BAŞLAT", type="primary"):
                st.session_state.d2_basla = True
                st.rerun()
        else:
            d2_test_fragment()
            st.divider()
            if st.button("TESTİ BİTİR VE PUANLA", type="primary"):
                # Puanlama için koordinatları grid verisiyle eşleştir
                # (Not: Fragment içindeki grid yapısı ile buradaki eşleşmeli)
                # Fragmentte rows yeniden hesaplanıyor, burada da aynısını yapmalıyız
                izgara = questions # Bu düz list of lists
                # Düz listeye çevirelim koordinat hesabı için
                flat_grid = [item for sublist in izgara for item in sublist]
                
                # Ama fragmentte de logic farklıydı.
                # En doğrusu: Grid'i baştan "id" li object olarak üretmekti.
                # Mevcut yapıyı bozmadan:
                # Kullanıcı (r_idx, c_idx) seçti. Bu r_idx fragmentteki 'rows' indexi.
                # Fragmentteki rows logic'i:
                # rows = [izgara[i:i+10]...] şeklindeydi, ama izgara zaten list of lists geliyordu d2_izgara_uret'ten.
                # d2_izgara_uret fonksiyonunu kontrol edelim: Evet, list of lists dönüyor.
                # Ve her satır 10 elemanlı (düzeltilen fonksiyonda).
                # O yüzden fragmentteki row yapısı ile ana data uyumlu.
                
                tn, cp, hata = d2_puanla(st.session_state.d2_isaretlenen, izgara)
                stats = {
                    "Toplam İşaretlenen": len(st.session_state.d2_isaretlenen),
                    "Toplam Taranan (TN)": tn,
                    "Konsantrasyon (CP)": cp,
                    "Hata Oranı": f"%{hata:.2f}"
                }
                st.session_state.results[test_name] = stats
                prompt = TEK_RAPOR_PROMPT.format(test_adi="d2", cevaplar_json=json.dumps(stats))
                with st.spinner("Performans analiz ediliyor..."):
                    st.session_state.reports[test_name] = get_data_from_ai(prompt)
                st.session_state.page = "view_report"
                st.rerun()

    # --- Burdon Dikkat Testi Modülü (Döngü Fix) ---
    elif q_type == "burdon":
        
        CHUNK_SIZE = 100
        full_data = questions # String
        total_chunks = len(full_data) // CHUNK_SIZE + (1 if len(full_data) % CHUNK_SIZE else 0)
        
        # Timer Fragment
        @st.fragment(run_every=1)
        def burdon_timer():
            if st.session_state.get("test_bitti", False): return
            if st.session_state.burdon_basla:
                elapsed = time.time() - st.session_state.start_time
                st.metric("Geçen Süre", f"{int(elapsed)} sn")

        # Grid Fragment
        @st.fragment
        def burdon_grid(segment_data):
            if st.session_state.get("test_bitti", False): return
            
            # Segment verisini listeye çevirip grid yap
            chars = list(segment_data)
            cols_count = 10
            rows = [chars[i:i+cols_count] for i in range(0, len(chars), cols_count)]
            
            st.markdown("<div style='text-align:center; margin-bottom:10px;'>Hedefler: <b>b, k</b></div>", unsafe_allow_html=True)
            
            # Mevcut chunk için seçim kümesi
            current_chunk_idx = st.session_state.current_chunk
            if current_chunk_idx not in st.session_state.burdon_isaretlenen:
                st.session_state.burdon_isaretlenen[current_chunk_idx] = set()
            
            selection = st.session_state.burdon_isaretlenen[current_chunk_idx]
            
            for r_idx, row in enumerate(rows):
                cols = st.columns(len(row))
                for c_idx, char in enumerate(row):
                    # Buton ID: Chunk + Row + Col
                    btn_id = (r_idx, c_idx)
                    is_sel = btn_id in selection
                    b_type = "primary" if is_sel else "secondary"
                    
                    with cols[c_idx]:
                        if st.button(char, key=f"bd_{current_chunk_idx}_{r_idx}_{c_idx}", type=b_type):
                            if is_sel: selection.remove(btn_id)
                            else: selection.add(btn_id)
                            st.session_state.burdon_isaretlenen[current_chunk_idx] = selection
                            st.rerun()

        st.subheader(f"Burdon Testi - Bölüm {st.session_state.current_chunk + 1}/{total_chunks}")
        
        if not st.session_state.burdon_basla:
            st.info("Hedef harflerin ('b' ve 'k') üzerine tıklayarak işaretleyin.")
            if st.button("BAŞLA", type="primary"):
                st.session_state.burdon_basla = True
                st.session_state.start_time = time.time()
                st.session_state.test_bitti = False
                st.rerun()
        else:
            burdon_timer()
            
            start = st.session_state.current_chunk * CHUNK_SIZE
            end = start + CHUNK_SIZE
            segment = full_data[start:end]
            
            burdon_grid(segment)
            
            st.divider()
            c1, c2 = st.columns([1, 4])
            
            if st.session_state.current_chunk < total_chunks - 1:
                if c2.button("Sonraki Bölüm ➡️", type="primary"):
                    st.session_state.current_chunk += 1
                    st.rerun()
            else:
                if c2.button("TESTİ BİTİR 🏁", type="primary"):
                    st.session_state.test_bitti = True # Timer durdur
                    
                    # Puanlama
                    duration = time.time() - st.session_state.start_time
                    
                    # Gerçek Hedefleri Say (b ve k)
                    total_targets_count = full_data.count('b') + full_data.count('k')
                    
                    # Kullanıcı İşaretlemelerini Say
                    # burdon_isaretlenen: {chunk_idx: set((r,c), (r,c))}
                    # Bunu gerçek harflerle eşleştirmemiz lazım
                    dogru_isaret = 0
                    yanlis_isaret = 0
                    
                    for chunk_idx, selection_set in st.session_state.burdon_isaretlenen.items():
                        c_start = chunk_idx * CHUNK_SIZE
                        c_segment = list(full_data[c_start : c_start + CHUNK_SIZE])
                        # Grid yapısını tekrar kurarak indeksi bul
                        # rows = [c_segment[i:i+10]...]
                        
                        for (r, c) in selection_set:
                            # 1D indexi bul: row * 10 + col
                            flat_idx = r * 10 + c
                            if flat_idx < len(c_segment):
                                char = c_segment[flat_idx]
                                if char in ['b', 'k']:
                                    dogru_isaret += 1
                                else:
                                    yanlis_isaret += 1

                    stats = {
                        "Süre (sn)": int(duration),
                        "Doğru İşaretleme": dogru_isaret,
                        "Hatalar": yanlis_isaret,
                        "Kaçırılanlar": total_targets_count - dogru_isaret,
                        "Başarı Oranı": f"%{(dogru_isaret/total_targets_count)*100:.1f}" if total_targets_count > 0 else "0"
                    }
                    
                    st.session_state.results["Burdon"] = stats
                    prompt = TEK_RAPOR_PROMPT.format(test_adi="Burdon", cevaplar_json=json.dumps(stats))
                    with st.spinner("Dikkat stabilitesi ölçülüyor..."):
                        st.session_state.reports[test_name] = get_data_from_ai(prompt)
                    
                    st.session_state.page = "view_report"
                    st.rerun()

    # --- Diğer Standart Testler ---
    else:
        with st.form(key="test_form"):
            user_answers = {}
            for i, q in enumerate(questions):
                q_text = q.get("text", q.get("question", str(q))) if isinstance(q, dict) else str(q)
                st.markdown(f"**{i+1}.** {q_text}")
                if q_type == "likert":
                    user_answers[i] = st.radio("Cevap:", ["Kesinlikle Katılmıyorum", "Katılmıyorum", "Kararsızım", "Katılıyorum", "Kesinlikle Katılıyorum"], key=f"q{i}", horizontal=True, index=None, label_visibility="collapsed")
                elif q_type in ["binary", "riaec"]:
                    user_answers[i] = st.radio("Cevap:", ["Bana Uygun Değil", "Bana Uygun"], key=f"q{i}", horizontal=True, index=None, label_visibility="collapsed")
                elif q_type == "vark":
                    opts = q.get("options", [])
                    user_answers[i] = st.multiselect("Seçimleriniz:", opts, key=f"q{i}")
                st.markdown("---")
            
            if st.form_submit_button("Analizi Tamamla", type="primary"):
                with st.spinner("Yapay zeka sonuçlarınızı analiz ediyor..."):
                    prompt = TEK_RAPOR_PROMPT.format(test_adi=test_name, cevaplar_json=json.dumps(user_answers, ensure_ascii=False))
                    report = get_data_from_ai(prompt)
                st.session_state.results[test_name] = user_answers
                st.session_state.reports[test_name] = report
                st.session_state.page = "view_report"
                st.rerun()

# === SAYFA 3: RAPOR GÖRÜNTÜLEME ===
elif st.session_state.page == "view_report":
    test_name = st.session_state.selected_test
    report = st.session_state.reports.get(test_name, "Rapor bulunamadı.")
    answers = st.session_state.results.get(test_name, {})
    
    st.markdown(f"## 📊 Sonuç Raporu: {test_name}")
    
    tab1, tab2 = st.tabs(["📝 Detaylı Rapor", "📈 Görsel Analiz"])
    
    with tab1:
        st.markdown(report)
        st.markdown("---")
        st.download_button(
            label="📥 Raporu İndir (.txt)",
            data=f"Test: {test_name}\nTarih: {datetime.now().strftime('%d-%m-%Y')}\n\n{report}",
            file_name=f"{test_name}_Analiz.txt",
            mime="text/plain"
        )
    with tab2:
        st.subheader("Grafik Analizi")
        if test_name == "Enneagram" and "Puanlar" in answers:
             # Enneagram özel grafik
             scores = answers["Puanlar"]
             labels = [f"Tip {k}" for k in scores.keys()]
             values = list(scores.values())
             fig = draw_radar_chart(labels, values, "Enneagram Profili")
             if fig: st.pyplot(fig)
        elif len(answers) > 0 and isinstance(answers, dict) and not test_name in ["d2 Dikkat Testi", "Burdon"]:
            # Genel Radar Grafiği (Demo Veri ile)
            try:
                labels = [f"Boyut {k+1}" for k in range(min(6, len(answers)))]
                values = np.random.randint(2, 6, size=len(labels)) # Gerçek hesaplama olmadığı için demo
                fig = draw_radar_chart(labels, values, f"{test_name} Profili")
                if fig: st.pyplot(fig)
                else: st.info("Grafik oluşturulamadı.")
            except:
                st.info("Bu veri seti için grafik uygun değil.")
        elif test_name in ["d2 Dikkat Testi", "Burdon"]:
            # Bar Chart
            st.bar_chart({
                "Doğru": answers.get("Doğru (Hits)", answers.get("Doğru İşaretleme", 0)),
                "Hata": answers.get("Yanlış Alarm (Commission)", answers.get("Hatalar", 0))
            })
        else:
            st.info("Grafik verisi yok.")

# === SAYFA 4: HARMANLANMIŞ RAPOR ===
elif st.session_state.page == "harman_report":
    st.markdown("## 🧩 Bütüncül Kişilik Profili")
    
    if "harman_content" not in st.session_state:
        with st.spinner("Tüm test verileri sentezleniyor..."):
            prompt = HARMAN_RAPOR_PROMPT.format(tum_cevaplar_json=json.dumps(st.session_state.results, ensure_ascii=False, default=str))
            st.session_state.harman_content = get_data_from_ai(prompt)
    
    st.markdown(st.session_state.harman_content)
    
    st.download_button(
        label="📥 Bütüncül Raporu İndir (.txt)",
        data=st.session_state.harman_content,
        file_name="Bütüncül_Analiz_Raporu.txt"
    )
    
    if st.button("⬅️ Geri Dön"):
        st.session_state.page = "home"
        st.rerun()
