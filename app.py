# Çalıştırma: streamlit run app.py  (önce python tekno.py ile modeli kaydedin)
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Müşteri kaybı tahmini", page_icon=":material/person_alert:", layout="wide")

MODEL_YOLU = Path(__file__).with_name("churn_model.joblib")
KARAR_ESIGI = 0.5


@st.cache_resource
def modeli_yukle():
    return joblib.load(MODEL_YOLU)


if not MODEL_YOLU.exists():
    st.error("Model dosyası bulunamadı. Önce `python tekno.py` komutunu çalıştırın.", icon=":material/error:")
    st.stop()

paket = modeli_yukle()
model = paket["model"]
kolonlar = paket["kolonlar"]

# ---------------------------------------------------------------- varsayılanlar ve örnek profiller
VARSAYILAN = {
    "cinsiyet": "Female", "yasli": False, "partner": False, "bakmakla": False, "tenure": 12,
    "sozlesme": "Month-to-month", "odeme": "Electronic check", "e_fatura": True, "aylik_ucret": 70.0,
    "telefon": True, "coklu_hat": False, "internet": "Fiber optic",
    "guvenlik": False, "yedekleme": False, "cihaz_koruma": False, "teknik_destek": False, "tv": False, "film": False,
}
RISKLI_ORNEK = {
    **VARSAYILAN, "tenure": 3, "aylik_ucret": 85.0, "tv": True, "film": True,
}
SADIK_ORNEK = {
    **VARSAYILAN, "partner": True, "bakmakla": True, "tenure": 60, "sozlesme": "Two year",
    "odeme": "Credit card (automatic)", "e_fatura": False, "aylik_ucret": 55.0, "internet": "DSL",
    "guvenlik": True, "yedekleme": True, "cihaz_koruma": True, "teknik_destek": True,
}

for anahtar, deger in VARSAYILAN.items():
    st.session_state.setdefault(anahtar, deger)


def profil_uygula(degerler):
    st.session_state.update(degerler)


# ---------------------------------------------------------------- metin sözlükleri
CINSIYET = {"Female": "Kadın", "Male": "Erkek"}
SOZLESME = {"Month-to-month": "Aylık", "One year": "1 yıllık", "Two year": "2 yıllık"}
INTERNET = {"DSL": "DSL", "Fiber optic": "Fiber optik", "No": "Yok"}
ODEME = {
    "Electronic check": "Elektronik çek",
    "Mailed check": "Posta çeki",
    "Bank transfer (automatic)": "Banka havalesi (otomatik)",
    "Credit card (automatic)": "Kredi kartı (otomatik)",
}

# ---------------------------------------------------------------- başlık
st.title("Müşteri kaybı tahmini", icon=":material/person_alert:")
st.write(
    "Bir telekom müşterisinin bilgilerini girin. Model, müşterinin **aboneliğini bırakıp bırakmayacağını** "
    "(evet / hayır) ve bunun olasılığını tahmin eder."
)

tahmin_sekmesi, model_sekmesi = st.tabs([":material/query_stats: Tahmin", ":material/info: Model hakkında"])

with tahmin_sekmesi:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption("Hızlı başlangıç:")
        st.button("Riskli müşteri örneği", icon=":material/warning:", on_click=profil_uygula, args=(RISKLI_ORNEK,))
        st.button("Sadık müşteri örneği", icon=":material/verified:", on_click=profil_uygula, args=(SADIK_ORNEK,))
        st.button("Sıfırla", icon=":material/restart_alt:", on_click=profil_uygula, args=(VARSAYILAN,))

    girdi_kolonu, sonuc_kolonu = st.columns([3, 2], gap="large")

    # ------------------------------------------------------------ girdiler
    with girdi_kolonu:
        with st.container(border=True):
            st.subheader("Müşteri profili", icon=":material/person:")
            st.segmented_control("Cinsiyet", list(CINSIYET), format_func=CINSIYET.get, key="cinsiyet", required=True)
            sol, sag = st.columns(2)
            sol.toggle("65 yaş üstü", key="yasli")
            sag.toggle("Partneri var", key="partner")
            sol.toggle("Bakmakla yükümlü olduğu kişi var", key="bakmakla")
            st.slider("Müşteri süresi (ay)", 0, 72, key="tenure", help="Müşteri kaç aydır şirketin abonesi?")

        with st.container(border=True):
            st.subheader("Sözleşme ve ödeme", icon=":material/receipt_long:")
            st.segmented_control("Sözleşme türü", list(SOZLESME), format_func=SOZLESME.get, key="sozlesme", required=True)
            st.selectbox("Ödeme yöntemi", list(ODEME), format_func=ODEME.get, key="odeme")
            st.toggle("Kâğıtsız (e-fatura) faturalandırma", key="e_fatura")
            st.slider("Aylık ücret", 18.0, 120.0, step=0.5, key="aylik_ucret", format="%.1f")

        with st.container(border=True):
            st.subheader("Hizmetler", icon=":material/wifi:")
            st.segmented_control("İnternet hizmeti", list(INTERNET), format_func=INTERNET.get, key="internet", required=True)
            telefon_var = st.session_state["telefon"]
            internet_var = st.session_state["internet"] != "No"
            st.toggle("Telefon hizmeti", key="telefon")
            st.toggle("Birden fazla hat", key="coklu_hat", disabled=not telefon_var)
            if not internet_var:
                st.caption("İnternet hizmeti olmadığı için aşağıdaki ek hizmetler kapalı sayılır.")
            sol, sag = st.columns(2)
            sol.toggle("Online güvenlik", key="guvenlik", disabled=not internet_var)
            sag.toggle("Online yedekleme", key="yedekleme", disabled=not internet_var)
            sol.toggle("Cihaz koruma", key="cihaz_koruma", disabled=not internet_var)
            sag.toggle("Teknik destek", key="teknik_destek", disabled=not internet_var)
            sol.toggle("TV yayını", key="tv", disabled=not internet_var)
            sag.toggle("Film yayını", key="film", disabled=not internet_var)

    # ------------------------------------------------------------ model girdisi
    s = st.session_state
    telefon_var = s["telefon"]
    internet_var = s["internet"] != "No"

    def evet(anahtar, hizmet_var=True):
        # Hizmet yoksa ilgili özellik, eğitim verisindeki gibi "No" (0) sayılır
        return int(bool(s[anahtar]) and hizmet_var)

    satir = {
        "gender": int(s["cinsiyet"] == "Male"),
        "SeniorCitizen": evet("yasli"),
        "Partner": evet("partner"),
        "Dependents": evet("bakmakla"),
        "tenure": s["tenure"],
        "PhoneService": evet("telefon"),
        "MultipleLines": evet("coklu_hat", telefon_var),
        "OnlineSecurity": evet("guvenlik", internet_var),
        "OnlineBackup": evet("yedekleme", internet_var),
        "DeviceProtection": evet("cihaz_koruma", internet_var),
        "TechSupport": evet("teknik_destek", internet_var),
        "StreamingTV": evet("tv", internet_var),
        "StreamingMovies": evet("film", internet_var),
        "PaperlessBilling": evet("e_fatura"),
        "MonthlyCharges": s["aylik_ucret"],
        # Toplam ücret sorulmaz; süre x aylık ücret ile tahmin edilir
        "TotalCharges": s["tenure"] * s["aylik_ucret"],
        "InternetService_Fiber optic": int(s["internet"] == "Fiber optic"),
        "InternetService_No": int(s["internet"] == "No"),
        "Contract_One year": int(s["sozlesme"] == "One year"),
        "Contract_Two year": int(s["sozlesme"] == "Two year"),
        "PaymentMethod_Credit card (automatic)": int(s["odeme"] == "Credit card (automatic)"),
        "PaymentMethod_Electronic check": int(s["odeme"] == "Electronic check"),
        "PaymentMethod_Mailed check": int(s["odeme"] == "Mailed check"),
    }
    girdi = pd.DataFrame([satir])[kolonlar]
    olasilik = float(model.predict_proba(girdi)[0, 1])

    # ------------------------------------------------------------ sonuç
    with sonuc_kolonu:
        with st.container(border=True):
            st.subheader("Tahmin", icon=":material/target:")
            if olasilik >= KARAR_ESIGI:
                st.error(
                    "Model bu müşterinin aboneliği **bırakabileceğini** tahmin ediyor. "
                    "Elde tutma kampanyası değerlendirilebilir.",
                    title="Tahmin: evet, ayrılabilir",
                    icon=":material/warning:",
                )
            else:
                st.success(
                    "Model bu müşterinin **kalmasını** bekliyor.",
                    title="Tahmin: hayır, kalması beklenir",
                    icon=":material/check_circle:",
                )
            st.metric("Ayrılma olasılığı", f"%{olasilik * 100:.1f}")
            st.progress(olasilik)
            st.caption(f"Olasılık %{KARAR_ESIGI * 100:.0f} ve üzerindeyse tahmin \"evet\" olur.")

        # --------------------------------------------------------- neden bu tahmin?
        # Her özelliğin katkısı = katsayı x (müşterinin değeri - ortalama müşterinin değeri)
        on_isleyici, lr = model[0], model[-1]
        donusmus = on_isleyici.transform(girdi)[0]
        katki = lr.coef_[0] * (donusmus - np.array(paket["ortalama"]))
        katki_sozlugu = dict(zip((ad.split("__", 1)[1] for ad in on_isleyici.get_feature_names_out()), katki))

        gruplar = [
            ("Müşteri süresi", ["tenure", "TotalCharges"], f"{s['tenure']} ay"),
            ("Aylık ücret", ["MonthlyCharges"], f"{s['aylik_ucret']:.1f}"),
            ("Sözleşme", ["Contract_One year", "Contract_Two year"], SOZLESME[s["sozlesme"]]),
            ("İnternet", ["InternetService_Fiber optic", "InternetService_No"], INTERNET[s["internet"]]),
            ("Ödeme yöntemi", ["PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check",
                               "PaymentMethod_Mailed check"], ODEME[s["odeme"]]),
            ("Cinsiyet", ["gender"], CINSIYET[s["cinsiyet"]]),
        ]
        ikili = {
            "SeniorCitizen": "65 yaş üstü", "Partner": "Partner", "Dependents": "Bakmakla yükümlü",
            "PhoneService": "Telefon", "MultipleLines": "Çoklu hat", "OnlineSecurity": "Online güvenlik",
            "OnlineBackup": "Online yedekleme", "DeviceProtection": "Cihaz koruma", "TechSupport": "Teknik destek",
            "StreamingTV": "TV yayını", "StreamingMovies": "Film yayını", "PaperlessBilling": "E-fatura",
        }
        gruplar += [(ad, [kolon], "var" if satir[kolon] else "yok") for kolon, ad in ikili.items()]

        etkiler = pd.DataFrame(
            [
                {"Faktör": f"{ad}: {deger}", "Etki": sum(katki_sozlugu[k] for k in kolonlar_grubu)}
                for ad, kolonlar_grubu, deger in gruplar
            ]
        )
        etkiler["Yön"] = np.where(etkiler["Etki"] > 0, "Riski artırır", "Riski azaltır")
        etkiler = etkiler.reindex(etkiler["Etki"].abs().sort_values(ascending=False).index).head(7)

        with st.container(border=True):
            st.subheader("Neden bu tahmin?", icon=":material/lightbulb:")
            st.caption("Bu müşterinin değerleri, ortalama bir müşteriyle karşılaştırıldığında risk üzerindeki etkisi.")
            grafik = (
                alt.Chart(etkiler)
                .mark_bar()
                .encode(
                    x=alt.X("Etki:Q", title="← riski azaltır | riski artırır →"),
                    y=alt.Y("Faktör:N", sort=None, title=None),
                    color=alt.Color(
                        "Yön:N",
                        scale=alt.Scale(domain=["Riski artırır", "Riski azaltır"], range=["#D9534F", "#4C9F70"]),
                        legend=None,
                    ),
                    tooltip=["Faktör", alt.Tooltip("Etki:Q", format="+.2f"), "Yön"],
                )
            )
            st.altair_chart(
                grafik,
                width="stretch",
                alt="Yatay çubuk grafik: bu müşteri için ayrılma riskini en çok artıran ve azaltan faktörler",
            )

    st.caption(
        "Not: Toplam ücret sorulmaz, süre × aylık ücret olarak tahmin edilir. Model, ayrılacak müşterileri "
        "kaçırmamak için dengeli (balanced) eğitildi; bu yüzden bazı sadık müşterilere de yüksek risk verebilir."
    )

# ---------------------------------------------------------------- model sekmesi
with model_sekmesi:
    st.subheader("Kullanılan model")
    st.write(
        f"**{paket.get('model_adi', 'Logistic Regression')}**. Telco Customer Churn verisiyle eğitildi "
        "(7043 müşteri, %80 eğitim / %20 test). Aşağıdaki sayılar **test verisinde** ölçülmüştür; "
        "formda girdiğiniz müşteriye bağlı değildir."
    )

    metrikler = paket.get("metrikler")
    if metrikler:
        kolonlar_metrik = st.columns(len(metrikler))
        aciklamalar = {
            "Accuracy": "Tüm tahminlerin yüzde kaçı doğru",
            "Precision": "\"Ayrılacak\" denenlerin yüzde kaçı gerçekten ayrıldı",
            "Recall": "Gerçekte ayrılanların yüzde kaçını yakaladık",
            "F1": "Precision ve recall'un dengesi",
            "ROC-AUC": "Ayrılanı kalandan ayırma gücü (1'e yakın iyi)",
        }
        for kolon, (ad, deger) in zip(kolonlar_metrik, metrikler.items()):
            kolon.metric(ad, f"{deger:.2f}", help=aciklamalar.get(ad), border=True)

    cm = paket.get("confusion_matrix")
    if cm:
        st.subheader("Confusion matrix")
        cm_tablo = pd.DataFrame(
            cm,
            index=["Gerçek: kaldı", "Gerçek: ayrıldı"],
            columns=["Tahmin: kaldı", "Tahmin: ayrıldı"],
        )
        st.dataframe(cm_tablo, alt="Confusion matrix tablosu: gerçek ve tahmin edilen kaldı/ayrıldı sayıları")
        st.caption(
            f"Sol alt ({cm[1][0]}): yanlış \"hayır\", yani ayrılan müşteriyi kaçırdık. "
            f"Sağ üst ({cm[0][1]}): yanlış \"evet\", yani kalacak müşteriye gereksiz kampanya. "
            "Bu problemde yanlış \"hayır\" daha maliyetli olduğu için recall'u yüksek tutan model seçildi."
        )
