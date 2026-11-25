# Proje 1: Telco Customer Churn - Sınıflandırma
import os
import sys
import textwrap

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.stdout.reconfigure(encoding="utf-8")
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ============================================================
# 1. VERİYİ TANIMA
# ============================================================
df = pd.read_csv("IT_customer_churn.csv")

print("Satır, sütun sayısı:", df.shape)
print("\nİlk 5 satır:")
print(df.head())

print("\nSütun tipleri ve boş olmayan değer sayıları:")
df.info()

print("\nSayısal sütunların özeti:")
print(df.describe())

# Sütun açıklamaları
sutun_aciklamalari = {
    "gender": "Cinsiyet (Female/Male)",
    "SeniorCitizen": "65 yaş üstü mü? (1 = evet, 0 = hayır)",
    "Partner": "Eşi/partneri var mı?",
    "Dependents": "Bakmakla yükümlü olduğu kişi (çocuk vb.) var mı?",
    "tenure": "Müşterinin şirkette kaldığı ay sayısı",
    "PhoneService": "Telefon hizmeti var mı?",
    "MultipleLines": "Birden fazla hat var mı? (No phone service = telefon hizmeti yok)",
    "InternetService": "İnternet hizmeti türü (DSL / Fiber optic / No)",
    "OnlineSecurity": "Online güvenlik hizmeti var mı?",
    "OnlineBackup": "Online yedekleme hizmeti var mı?",
    "DeviceProtection": "Cihaz koruma hizmeti var mı?",
    "TechSupport": "Teknik destek hizmeti var mı?",
    "StreamingTV": "TV yayın hizmeti var mı?",
    "StreamingMovies": "Film yayın hizmeti var mı?",
    "Contract": "Sözleşme türü (Month-to-month / One year / Two year)",
    "PaperlessBilling": "Kâğıtsız (e-fatura) faturalandırma var mı?",
    "PaymentMethod": "Ödeme yöntemi (4 farklı yöntem)",
    "MonthlyCharges": "Aylık ücret",
    "TotalCharges": "Toplam ödenen ücret",
    "Churn": "HEDEF: Müşteri ayrıldı mı? (Yes = evet, No = hayır)",
}

print("\nSütun açıklamaları:")
ozet = pd.DataFrame(
    {
        "Tip": df.dtypes.astype(str),
        "Benzersiz değer": df.nunique(),
        "Açıklama": pd.Series(sutun_aciklamalari),
    }
)
print(ozet.to_string())

# Hedef değişken dağılımı: kaç "evet", kaç "hayır"?
sayilar = df["Churn"].value_counts()
yuzdeler = df["Churn"].value_counts(normalize=True) * 100

print("\nChurn dağılımı:")
print(pd.DataFrame({"Adet": sayilar, "Yüzde (%)": yuzdeler.round(2)}))

oran = sayilar["No"] / sayilar["Yes"]
print(f"\nHayır/Evet oranı: {oran:.2f} -> sınıflar dengesiz.")
print(
    "Yorum: Müşterilerin yaklaşık dörtte biri ayrılmış. Sınıflar dengesiz olduğu için "
    "her müşteriye 'kalır' diyen bir model bile yaklaşık %73 accuracy alır. "
    "Bu yüzden accuracy tek başına yanıltıcıdır; precision, recall ve "
    "confusion matrix de bakılmalıdır."
)

# Temizleme aşamasına hazırlık: gizli eksik değer kontrolü
print("\nTip kontrolü: TotalCharges sütunu sayısal olmalıydı, tipi:", df["TotalCharges"].dtype)
bos_total = df["TotalCharges"].astype(str).str.strip().eq("").sum()
print("TotalCharges içindeki boş (' ') değer sayısı:", bos_total)

# ============================================================
# 2. VERİ TEMİZLEME
# ============================================================
print("\n" + "=" * 60)
print("2. VERİ TEMİZLEME")
print("=" * 60)

# 2.1 TotalCharges: metinden sayıya çevir (boşluklar NaN olur)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\nEksik değerler (dönüşümden sonra):")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nTotalCharges'ı eksik olan satırlar:")
print(df.loc[df["TotalCharges"].isnull(), ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

# Hepsinin tenure değeri 0: henüz ilk faturasını ödememiş yeni müşteriler, toplam ücret 0'dır
assert (df.loc[df["TotalCharges"].isnull(), "tenure"] == 0).all()
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print("-> Hepsi tenure=0 olan yeni müşteri. TotalCharges 0 ile dolduruldu.")

# 2.2 Tekrar eden satırlar
tekrar = df.duplicated().sum()
print(f"\nTekrar eden satır sayısı: {tekrar}")
print("-> customerID olmadığı için aynı özelliklere sahip farklı müşteriler olabilir; silinmedi.")

# 2.3 Gereksiz sütunlar
print("\nGereksiz sütun: yok (ID sütunu yok, tüm sütunlar anlamlı).")

# 2.4 Anlamsal sadeleştirme: "No internet service" / "No phone service" aslında "No" demektir.
# Bu bilgi InternetService ve PhoneService sütunlarında zaten var.
hizmet_sutunlari = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
for sutun in hizmet_sutunlari:
    df[sutun] = df[sutun].replace({"No internet service": "No", "No phone service": "No"})

# Temiz ama okunabilir etiketli veri: görselleştirmede bu kullanılacak
df_temiz = df.copy()

# 2.5 Kategorik dönüşüm (modelde kullanılacak sayısal veri)
df_model = df.copy()

df_model["Churn"] = df_model["Churn"].map({"Yes": 1, "No": 0})
df_model["gender"] = df_model["gender"].map({"Female": 0, "Male": 1})

ikili_sutunlar = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"] + hizmet_sutunlari
for sutun in ikili_sutunlar:
    df_model[sutun] = df_model[sutun].map({"No": 0, "Yes": 1})

# 3'ten fazla sınıfı olan sütunlar: one-hot encoding (ilk sınıf atılır)
df_model = pd.get_dummies(
    df_model, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True, dtype=int
)

print("\nDönüşüm sonrası boyut:", df_model.shape)
print("Kalan eksik değer:", df_model.isnull().sum().sum())
print("Sayısal olmayan sütun:", list(df_model.select_dtypes(exclude="number").columns))
print("\nModel verisinin sütunları:")
print(list(df_model.columns))
print("\nİlk 5 satır:")
print(df_model.head())
print("\nChurn dağılımı (0/1):")
print(df_model["Churn"].value_counts())

# ============================================================
# 3. GÖRSELLEŞTİRME
# ============================================================
print("\n" + "=" * 60)
print("3. GÖRSELLEŞTİRME")
print("=" * 60)

os.makedirs("grafikler", exist_ok=True)
df_temiz["ChurnSayi"] = df_temiz["Churn"].map({"Yes": 1, "No": 0})
genel_oran = df_temiz["ChurnSayi"].mean() * 100
RENKLER = {"No": "#4C9F70", "Yes": "#D9534F"}


def kaydet(fig, dosya, yorum):
    """Yorumu grafiğin altına yazar, grafiği kaydeder ve yorumu terminale basar."""
    fig.text(0.5, 0.02, textwrap.fill("Yorum: " + yorum, 110), ha="center", va="bottom", fontsize=10, style="italic")
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(os.path.join("grafikler", dosya), dpi=120)
    print(f"\n[{dosya}] Yorum: {yorum}")


def churn_orani_cubugu(ax, sutun, baslik, xetiket=None):
    """Bir kategorinin her değeri için churn yüzdesini çizer."""
    oranlar = (df_temiz.groupby(sutun)["ChurnSayi"].mean() * 100).sort_values(ascending=False)
    sns.barplot(x=oranlar.index.astype(str), y=oranlar.values, ax=ax, color="#D9534F")
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=2)
    ax.axhline(genel_oran, color="gray", linestyle="--", label=f"Genel ortalama (%{genel_oran:.1f})")
    ax.set_title(baslik)
    ax.set_xlabel(xetiket or sutun)
    ax.set_ylabel("Churn oranı (%)")
    ax.set_ylim(0, oranlar.max() * 1.18)
    ax.legend()


# Grafik 1: Churn dağılımı
fig, ax = plt.subplots(figsize=(7, 6))
sns.countplot(data=df_temiz, x="Churn", hue="Churn", palette=RENKLER, order=["No", "Yes"], legend=False, ax=ax)
for kap, adet in zip(ax.containers[0], df_temiz["Churn"].value_counts()[["No", "Yes"]]):
    ax.text(kap.get_x() + kap.get_width() / 2, kap.get_height() + 40, f"{adet} (%{adet / len(df_temiz) * 100:.1f})", ha="center")
ax.set_title("Müşteri Ayrılma (Churn) Dağılımı")
ax.set_xlabel("Churn (No = kaldı, Yes = ayrıldı)")
ax.set_ylabel("Müşteri sayısı")
kaydet(fig, "1_churn_dagilimi.png",
       "Müşterilerin %73,5'i kalmış, %26,5'i ayrılmış. Sınıflar dengesiz; bu yüzden modelde sadece "
       "accuracy'ye değil recall ve precision'a da bakmalıyız.")

# Grafik 2: Sözleşme türüne göre churn
fig, ax = plt.subplots(figsize=(8, 6))
churn_orani_cubugu(ax, "Contract", "Sözleşme Türüne Göre Churn Oranı", "Sözleşme türü")
kaydet(fig, "2_sozlesme_churn.png",
       "Aylık sözleşmelilerin %42,7'si ayrılırken 1 yıllıklarda %11,3, 2 yıllıklarda %2,8 ayrılmış. "
       "Sözleşme süresi uzadıkça müşteri bağlılığı çok artıyor; bu, en güçlü özelliklerden biri.")

# Grafik 3: Müşteri süresi (tenure) dağılımı
fig, ax = plt.subplots(figsize=(9, 6))
sns.histplot(data=df_temiz, x="tenure", hue="Churn", palette=RENKLER, bins=36, kde=True, stat="density",
             common_norm=False, element="step", ax=ax)
ax.set_title("Müşteri Süresine (tenure) Göre Churn Dağılımı")
ax.set_xlabel("Şirkette kalınan ay sayısı")
ax.set_ylabel("Yoğunluk")
kaydet(fig, "3_tenure_dagilimi.png",
       "Ayrılan müşterilerin çoğu ilk aylarda ayrılıyor (ortanca 10 ay). Kalanların ortancası 38 ay. "
       "Yeni müşteriler en riskli grup; şirketi ne kadar uzun süre kullanırsa ayrılma ihtimali o kadar düşüyor.")

# Grafik 4: Aylık ücret
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(data=df_temiz, x="Churn", y="MonthlyCharges", hue="Churn", palette=RENKLER, order=["No", "Yes"],
            legend=False, ax=ax)
ax.set_title("Aylık Ücrete Göre Churn")
ax.set_xlabel("Churn (No = kaldı, Yes = ayrıldı)")
ax.set_ylabel("Aylık ücret")
kaydet(fig, "4_aylik_ucret_boxplot.png",
       "Ayrılan müşterilerin aylık ücret ortancası (~79,7) kalanlardan (~64,4) yüksek. "
       "Pahalı paket kullananlar daha kolay ayrılıyor; fiyat, ayrılma kararında etkili bir faktör olabilir.")

# Grafik 5: İnternet hizmeti türü
fig, ax = plt.subplots(figsize=(8, 6))
churn_orani_cubugu(ax, "InternetService", "İnternet Hizmeti Türüne Göre Churn Oranı", "İnternet hizmeti")
kaydet(fig, "5_internet_churn.png",
       "Fiber optik müşterilerinin %41,9'u ayrılmış; DSL'de %19, internet hizmeti olmayanlarda %7,4. "
       "Fiber optik pahalı olduğu veya hizmet kalitesinden memnun kalınmadığı için kayıp yüksek olabilir.")

# Grafik 6: Cinsiyet ve yaşlı müşteri
fig, eksenler = plt.subplots(1, 2, figsize=(11, 6))
churn_orani_cubugu(eksenler[0], "gender", "Cinsiyete Göre Churn Oranı", "Cinsiyet")
churn_orani_cubugu(eksenler[1], "SeniorCitizen", "Yaşlı Müşteri (1) / Diğer (0) Churn Oranı", "SeniorCitizen")
kaydet(fig, "6_cinsiyet_yasli_churn.png",
       "Kadın (%26,9) ve erkek (%26,2) neredeyse aynı; cinsiyet ayrılmayı etkilemiyor. "
       "Yaşlı müşterilerde ise oran %41,7 (diğerlerinde %23,6); yaş grubu belirgin bir etken.")

# Grafik 7: Ödeme yöntemi
fig, ax = plt.subplots(figsize=(10, 6))
churn_orani_cubugu(ax, "PaymentMethod", "Ödeme Yöntemine Göre Churn Oranı", "Ödeme yöntemi")
ax.tick_params(axis="x", rotation=15)
kaydet(fig, "7_odeme_yontemi_churn.png",
       "Elektronik çekle ödeyenlerin %45,3'ü ayrılmış; otomatik ödeme yapanlarda (banka %16,7, kredi kartı %15,2) "
       "çok düşük. Otomatik ödeme müşteriyi bağlıyor olabilir.")

# Grafik 8: Tüm özelliklerin Churn ile korelasyonu
korelasyon = df_model.corr()["Churn"].drop("Churn").sort_values()
fig, ax = plt.subplots(figsize=(9, 8))
sns.barplot(x=korelasyon.values, y=korelasyon.index, hue=korelasyon.values > 0,
            palette={True: "#D9534F", False: "#4C9F70"}, legend=False, ax=ax)
ax.set_title("Özelliklerin Churn ile Korelasyonu")
ax.set_xlabel("Korelasyon (pozitif: ayrılma ihtimalini artırır)")
ax.set_ylabel("")
kaydet(fig, "8_korelasyon.png",
       "Ayrılma ile en güçlü pozitif ilişki: fiber optik internet ve elektronik çek (~0,30). "
       "En güçlü negatif ilişki: tenure (-0,35), 2 yıllık sözleşme (-0,30) ve internet hizmetinin olmaması (-0,23); "
       "yani aylık sözleşmeli (referans sınıf) müşteriler daha riskli. Cinsiyet ve telefon hizmeti ise neredeyse etkisiz (~0).")

print("\nGrafikler 'grafikler' klasörüne kaydedildi.")

# ============================================================
# 4. MODEL
# ============================================================
print("\n" + "=" * 60)
print("4. MODEL")
print("=" * 60)

# 4.1 Train/test ayrımı: stratify ile her iki parçada da churn oranı aynı kalır
X = df_model.drop(columns="Churn")
y = df_model["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Eğitim: {X_train.shape[0]} satır, Test: {X_test.shape[0]} satır")
print(f"Eğitimde churn oranı: %{y_train.mean() * 100:.2f}, testte: %{y_test.mean() * 100:.2f}")

# 4.2 Ölçekleme sadece Logistic Regression için ve sadece eğitim verisine fit edilir (pipeline bunu garanti eder)
sayisal_sutunlar = ["tenure", "MonthlyCharges", "TotalCharges"]


def lr_modeli(**kwargs):
    olcekleyici = ColumnTransformer([("olcek", StandardScaler(), sayisal_sutunlar)], remainder="passthrough")
    return Pipeline([("olcek", olcekleyici), ("model", LogisticRegression(max_iter=1000, random_state=42, **kwargs))])


# 4.3 Decision Tree için max_depth seçimi: sadece eğitim verisi üzerinde 5 katlı çapraz doğrulama
print("\nDecision Tree derinlik seçimi (eğitim verisinde 5 katlı CV, F1 skoru):")
derinlik_sonuclari = {}
for derinlik in [2, 3, 4, 5, 6, 8, 10, None]:
    dt = DecisionTreeClassifier(max_depth=derinlik, class_weight="balanced", random_state=42)
    derinlik_sonuclari[derinlik] = cross_val_score(dt, X_train, y_train, cv=5, scoring="f1").mean()
    print(f"  max_depth={str(derinlik):>4}: F1 = {derinlik_sonuclari[derinlik]:.3f}")
en_iyi_derinlik = max(derinlik_sonuclari, key=derinlik_sonuclari.get)
print(f"-> Seçilen max_depth: {en_iyi_derinlik}")

# 4.4 Modelleri eğit
modeller = {
    "Logistic Regression": lr_modeli(),
    "Logistic Regression (balanced)": lr_modeli(class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=en_iyi_derinlik, random_state=42),
    "Decision Tree (balanced)": DecisionTreeClassifier(
        max_depth=en_iyi_derinlik, class_weight="balanced", random_state=42
    ),
}
for isim, model in modeller.items():
    model.fit(X_train, y_train)

# Aşırı öğrenme kontrolü için sınırsız derinlikte bir ağaç (karşılaştırma amaçlı, modeller sözlüğünde değil)
sinirsiz_agac = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

print("\nEğitim ve test accuracy (aşırı öğrenme kontrolü):")
for isim, model in {**modeller, "Decision Tree (sınırsız derinlik)": sinirsiz_agac}.items():
    print(f"  {isim:<36} eğitim: {model.score(X_train, y_train):.3f}   test: {model.score(X_test, y_test):.3f}")
print(
    "Yorum: Derinliği sınırlanmamış ağaç eğitim verisini ezberler (eğitim ~1.0) ama testte düşer. "
    "max_depth sınırlaması bu farkı kapatır."
)

# ============================================================
# 5. SONUÇ: DEĞERLENDİRME
# ============================================================
print("\n" + "=" * 60)
print("5. SONUÇ: DEĞERLENDİRME (test verisi)")
print("=" * 60)

sinif_isimleri = ["Kaldı (0)", "Ayrıldı (1)"]
tahminler = {isim: model.predict(X_test) for isim, model in modeller.items()}

sonuc_tablosu = pd.DataFrame(
    {
        isim: {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
        }
        for isim, y_pred in tahminler.items()
    }
).T
print("\nMetrik karşılaştırması (pozitif sınıf = Ayrıldı):")
print(sonuc_tablosu.round(3))

for isim, y_pred in tahminler.items():
    print(f"\n--- {isim} ---")
    print(classification_report(y_test, y_pred, target_names=sinif_isimleri, digits=3))

# Confusion matrix: her model için bir panel
fig, eksenler = plt.subplots(2, 2, figsize=(11, 10))
for ax, (isim, y_pred) in zip(eksenler.ravel(), tahminler.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Kaldı", "Ayrıldı"], yticklabels=["Kaldı", "Ayrıldı"])
    ax.set_title(isim)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
cm_sozlugu = {isim: confusion_matrix(y_test, y_pred) for isim, y_pred in tahminler.items()}
kaydet(fig, "9_confusion_matrix.png",
       "Varsayılan modeller (üst satır solu LR, alt satır solu DT) gerçekte ayrılan müşterilerin yaklaşık yarısını "
       "kaçırıyor (sol alt hücre: yanlış 'hayır'). 'balanced' modeller bu hatayı belirgin azaltıyor ama karşılığında "
       "daha fazla kalacak müşteriye yanlışlıkla 'ayrılacak' diyor (sağ üst hücre: yanlış 'evet').")

# Yanlış 'hayır' (FN) ve yanlış 'evet' (FP) sayıları
print("\nHata türleri (test verisi, 374 gerçek 'ayrıldı', 1035 gerçek 'kaldı'):")
hata_tablosu = pd.DataFrame(
    {isim: {"Yanlış 'hayır' (FN)": cm[1, 0], "Yanlış 'evet' (FP)": cm[0, 1]} for isim, cm in cm_sozlugu.items()}
).T
print(hata_tablosu)

# Maliyet karşılaştırması: oranlar VARSAYIMDIR (yeni müşteri kazanmak, elde tutmaktan ~5 kat pahalı kabul edildi)
MALIYET_FN = 5
MALIYET_FP = 1
hata_tablosu["Toplam maliyet"] = hata_tablosu["Yanlış 'hayır' (FN)"] * MALIYET_FN + hata_tablosu["Yanlış 'evet' (FP)"] * MALIYET_FP
print(f"\nVarsayım: bir FN maliyeti = {MALIYET_FN} birim, bir FP maliyeti = {MALIYET_FP} birim")
print(hata_tablosu["Toplam maliyet"].sort_values())

print(
    """
SORU: Bu problemde yanlış 'hayır' demek mi daha kötü, yanlış 'evet' demek mi? Neden?

CEVAP: Yanlış 'hayır' (False Negative) daha kötüdür.
- Yanlış 'hayır': Gerçekte ayrılacak müşteriye 'kalır' demişiz. Müşteriye hiçbir önlem alınmaz,
  müşteri kaybedilir. Yeni müşteri kazanmak, mevcut müşteriyi elde tutmaktan genellikle çok daha pahalıdır
  ve kaybedilen müşterinin gelecekteki tüm geliri de gider.
- Yanlış 'evet': Gerçekte kalacak müşteriye 'ayrılır' demişiz. Sonuç olarak bu müşteriye gereksiz bir
  indirim veya arama kampanyası yapılır. Maliyeti sınırlı ve telafi edilebilir.
Bu yüzden başarı ölçütü olarak recall'a (ayrılacak müşterilerin ne kadarını yakaladık) öncelik veriyoruz.
Ancak precision'ı tamamen göz ardı etmiyoruz: çok düşerse kampanya bütçesi boşa harcanır.
Sonuç: 'balanced' modeller recall'u ~%51-56'dan ~%78-79'a çıkardığı için bu problemde daha uygundur.
Accuracy'nin düşmesi (%80 -> %74) kabul edilebilir bir bedeldir.
"""
)

# ============================================================
# BONUS 1: 3 MODELİ KARŞILAŞTIRMA (Logistic Regression, Decision Tree, Random Forest)
# ============================================================
print("\n" + "=" * 60)
print("BONUS 1: MODEL KARŞILAŞTIRMASI")
print("=" * 60)

# Random Forest hiperparametreleri test verisine bakılmadan, makul değerlerle sabitlendi
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42
).fit(X_train, y_train)
tahminler["Random Forest (balanced)"] = rf.predict(X_test)
modeller["Random Forest (balanced)"] = rf

karsilastirilacaklar = ["Logistic Regression (balanced)", "Decision Tree (balanced)", "Random Forest (balanced)"]
karsilastirma = pd.DataFrame(
    {
        isim: {
            "Accuracy": accuracy_score(y_test, tahminler[isim]),
            "Precision": precision_score(y_test, tahminler[isim]),
            "Recall": recall_score(y_test, tahminler[isim]),
            "F1": f1_score(y_test, tahminler[isim]),
            "ROC-AUC": roc_auc_score(y_test, modeller[isim].predict_proba(X_test)[:, 1]),
        }
        for isim in karsilastirilacaklar
    }
).T
print(karsilastirma.round(3))

fig, ax = plt.subplots(figsize=(10, 6))
karsilastirma.drop(columns="ROC-AUC").plot(kind="bar", ax=ax, rot=0, colormap="viridis")
for kap in ax.containers:
    ax.bar_label(kap, fmt="%.2f", fontsize=8, padding=2)
ax.set_title("Modellerin Karşılaştırması (test verisi, class_weight='balanced')")
ax.set_ylim(0, 1)
ax.legend(loc="lower right")
kaydet(fig, "10_model_karsilastirma.png",
       "Üç model birbirine çok yakın: Random Forest hafif önde (recall 0,79, F1 0,63, ROC-AUC 0,846) ama fark küçük. "
       "Logistic Regression (ROC-AUC 0,842) çok daha basit ve yorumlanabilir olduğu için seçilen model olarak kullanıldı.")

# ============================================================
# BONUS 2: YANLIŞ TAHMİN ANALİZİ (Logistic Regression - balanced)
# ============================================================
print("\n" + "=" * 60)
print("BONUS 2: YANLIŞ TAHMİN ANALİZİ")
print("=" * 60)

secilen_model = "Logistic Regression (balanced)"
analiz = df_temiz.loc[
    X_test.index, ["tenure", "Contract", "InternetService", "PaymentMethod", "TechSupport", "MonthlyCharges", "SeniorCitizen"]
].copy()
analiz["Gercek"] = y_test
analiz["Tahmin"] = tahminler[secilen_model]
analiz["Ayrilma_olasiligi"] = modeller[secilen_model].predict_proba(X_test)[:, 1].round(2)

# Modelin en emin olduğu halde yanıldığı örnekler
fn_ornekleri = analiz[(analiz["Gercek"] == 1) & (analiz["Tahmin"] == 0)].sort_values("Ayrilma_olasiligi").head(3)
fp_ornekleri = analiz[(analiz["Gercek"] == 0) & (analiz["Tahmin"] == 1)].sort_values("Ayrilma_olasiligi", ascending=False).head(3)

print("\nYanlış 'hayır' örnekleri (gerçekte ayrıldı, model 'kalır' dedi):")
print(fn_ornekleri.to_string())
print("\nYanlış 'evet' örnekleri (gerçekte kaldı, model 'ayrılır' dedi):")
print(fp_ornekleri.to_string())

# Genel profil: yanlış 'hayır'lar doğru yakalanan ayrılanlardan ve kalanlardan nasıl ayrışıyor?
analiz["Grup"] = "Doğru: kaldı (TN)"
analiz.loc[(analiz["Gercek"] == 1) & (analiz["Tahmin"] == 1), "Grup"] = "Doğru: ayrıldı (TP)"
analiz.loc[(analiz["Gercek"] == 1) & (analiz["Tahmin"] == 0), "Grup"] = "Yanlış 'hayır' (FN)"
analiz.loc[(analiz["Gercek"] == 0) & (analiz["Tahmin"] == 1), "Grup"] = "Yanlış 'evet' (FP)"
profil = analiz.groupby("Grup").agg(
    adet=("tenure", "size"),
    ort_tenure=("tenure", "mean"),
    ort_aylik_ucret=("MonthlyCharges", "mean"),
    aylik_sozlesme_orani=("Contract", lambda s: (s == "Month-to-month").mean()),
    fiber_orani=("InternetService", lambda s: (s == "Fiber optic").mean()),
)
print("\nGrup profilleri:")
print(profil.round(2))

print(
    """
YORUM (hata analizi):
- Yanlış 'hayır' (FN): En emin yanılgılar uzun süreli (48-72 ay), 1-2 yıllık sözleşmeli, otomatik ödemeli ve
  teknik destek alan müşteriler. Bu özelliklerin hepsi grafiklerde 'kalma' belirtisiydi, model de bu yüzden
  ayrılma olasılığını %2-7 verdi. Bu müşteriler veride sadık görünüyor ama yine de ayrılmış; nedeni (taşınma,
  rakip teklifi, kişisel sebep vb.) veri setinde olmadığı için model bunu öğrenemez. Bu hata türü bu veriyle
  büyük ölçüde engellenemez. Genel profilde de FN grubu (ort. tenure 29 ay, %54 aylık sözleşme), doğru yakalanan
  ayrılanlardan (13 ay, %97 aylık sözleşme) çok farklı, yani 'tipik olmayan' ayrılanlar.
- Yanlış 'evet' (FP): Hepsi kısa süreli, aylık sözleşmeli, fiber optik, elektronik çekle ödeyen ve teknik
  desteği olmayan müşteriler; model %91-92 olasılık verdi. Bunlar en riskli profilin tam örneği, yani model
  mantıklı karar vermiş ama bu müşteriler bu sefer kalmış. FP grubu (ort. tenure 20 ay, %93 aylık sözleşme)
  TP grubuna çok benziyor; bu iki grubu ayıran bilgi veri setinde yok.
- Sonuç: Hatalar modelin hatasından çok verideki gürültüden (olmayan bilgiden) geliyor. FP'ler zaten 'riskli'
  müşteriler olduğu için onlara yapılan kampanya tamamen boşa gitmiş sayılmaz.
"""
)

# ============================================================
# BONUS 3: MODELİ KAYDET (Streamlit arayüzü app.py tarafından kullanılır)
# ============================================================
secilen_tahmin = tahminler[secilen_model]
joblib.dump(
    {
        "model": modeller[secilen_model],
        "kolonlar": list(X.columns),
        "model_adi": secilen_model,
        "metrikler": {
            "Accuracy": accuracy_score(y_test, secilen_tahmin),
            "Precision": precision_score(y_test, secilen_tahmin),
            "Recall": recall_score(y_test, secilen_tahmin),
            "F1": f1_score(y_test, secilen_tahmin),
            "ROC-AUC": roc_auc_score(y_test, modeller[secilen_model].predict_proba(X_test)[:, 1]),
        },
        "confusion_matrix": confusion_matrix(y_test, secilen_tahmin).tolist(),
        # Açıklama grafiği için: eğitim verisindeki ortalama müşterinin dönüştürülmüş özellik değerleri
        "ortalama": modeller[secilen_model][0].transform(X_train).mean(axis=0).tolist(),
    },
    "churn_model.joblib",
)
print("\nSeçilen model 'churn_model.joblib' dosyasına kaydedildi. Arayüz için: streamlit run app.py")

plt.show()
