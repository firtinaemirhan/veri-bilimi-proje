
import os
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Tüm rastgele işlemlerin aynı sonucu vermesi için sabit tohum değeri
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = "ciktilar"
os.makedirs(OUT_DIR, exist_ok=True)

def veri_yukle(dosya_yolu="gallstone.xlsx", sayfa="dataset"):
    """
    Gallstone veri setini Excel dosyasından okur ve özetler.

    Parametreler
    ------------
    dosya_yolu : str
        Excel dosyasının yolu.
    sayfa : str
        Okunacak sayfa adı.

    Döndürür
    --------
    pd.DataFrame
        Yüklenmiş veri çerçevesi.
    """
    df = pd.read_excel(dosya_yolu, sheet_name=sayfa)
    print(f"Veri seti yüklendi. Boyut: {df.shape}")
    print(f"Eksik değer sayısı: {df.isnull().sum().sum()}")
    print(f"Hedef değişken dağılımı:\n{df['Gallstone Status'].value_counts()}")
    return df

def on_isleme(df):
    """
    Hedef değişkeni ayırır ve eğitim/test bölmesi yapar.

    Not: UCI veri açıklamasında 0=pozitif, 1=negatif olarak verilmiştir.
    Klinik yorumla uyumlu olması için 0 sınıfını "safra taşı var" kabul
    ediyoruz; mevcut etiketleri olduğu gibi kullanıyoruz fakat raporlamada
    pozitif sınıfı 0 olarak ele alacağız (pos_label=0).
    """
    y = df["Gallstone Status"].values
    X = df.drop(columns=["Gallstone Status"]).values
    feature_names = df.drop(columns=["Gallstone Status"]).columns.tolist()

    # stratify=y: her bölümde sınıf oranları korunur
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    # Ölçekleme parametreleri yalnızca eğitim verisinden hesaplanır;
    # test verisine sadece transform uygulanır (veri sızıntısı önlenir)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Eğitim: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    return (X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled, feature_names)

def metrikleri_hesapla(y_true, y_pred, y_proba=None):
    """Tüm temel sınıflandırma metriklerini sözlük olarak döndürür."""
    # pos_label=0: UCI kodlamasında 0 = "safra taşı var" (klinik pozitif)
    sonuc = {
        "Doğruluk": accuracy_score(y_true, y_pred),
        "Kesinlik": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Duyarlılık": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }
    if y_proba is not None:
        # roc_auc_score yüksek skoru pozitif kabul eder;
        # etiketleri tersleyerek 0 sınıfını pozitif konuma taşıyoruz
        sonuc["ROC-AUC"] = roc_auc_score(1 - y_true, y_proba)
    return sonuc

def klasik_modelleri_egit(X_train, X_test, y_train, y_test,
                           X_train_s, X_test_s):
    """9 farklı klasik makine öğrenmesi modelini eğitir ve değerlendirir."""

    modeller = {
        "Rastgele Orman": (RandomForestClassifier(n_estimators=200, random_state=SEED),
                           "ham"),
        "Gradyan Artırma": (GradientBoostingClassifier(random_state=SEED), "ham"),

        "Karar Ağacı": (DecisionTreeClassifier(random_state=SEED), "ham"),

        "K-En Yakın Komşu": (KNeighborsClassifier(n_neighbors=7), "olcekli"),

        "Lojistik Regresyon": (LogisticRegression(max_iter=2000, random_state=SEED),
                               "olcekli"),

        "Naive Bayes": (GaussianNB(), "ham"),

        "Destek Vektör Makinesi": (SVC(kernel="rbf", probability=True,
                                       random_state=SEED), "olcekli"),

        "Yapay Sinir Ağı": (MLPClassifier(hidden_layer_sizes=(64, 32),
                                           max_iter=500, random_state=SEED),
                            "olcekli"),
    }

    sonuclar = {}
    egitilmis_modeller = {}

    for ad, (model, veri_tipi) in modeller.items():
        print(f"  -> {ad} eğitiliyor...")
        if veri_tipi == "olcekli":
            Xtr, Xte = X_train_s, X_test_s
        else:
            Xtr, Xte = X_train, X_test

        baslangic = time.time()
        model.fit(Xtr, y_train)
        sure = time.time() - baslangic

        y_pred = model.predict(Xte)
        try:
            # predict_proba sütun 0: P(safra taşı var), sütun 1: P(yok)
            y_proba = model.predict_proba(Xte)[:, 0]
        except AttributeError:
            y_proba = None

        sonuclar[ad] = metrikleri_hesapla(y_test, y_pred, y_proba)
        sonuclar[ad]["Süre (s)"] = round(sure, 3)
        egitilmis_modeller[ad] = (model, veri_tipi)

    return sonuclar, egitilmis_modeller

def esikli_lineer_regresyon(X_train_s, X_test_s, y_train, y_test, esik=0.5):
    """
    Lineer Regresyon modelini bir sınıflandırıcı gibi kullanır.
    Çıktıyı esik değerine göre 0/1'e çevirir.
    """
    print("  -> Eşikli Lineer Regresyon eğitiliyor...")
    model = LinearRegression()
    # Regresyon çıktısı "safra taşı olasılığı" anlamına gelmeli;
    # orijinal etiket 0=var olduğundan y'yi tersleyerek 1=var hale getiriyoruz
    model.fit(X_train_s, 1 - y_train)
    y_proba = model.predict(X_test_s)
    y_pred = (y_proba >= esik).astype(int)
    # Tahminleri orijinal etiket uzayına (0=var) geri döndürüyoruz
    y_pred_orig = 1 - y_pred
    sonuc = metrikleri_hesapla(y_test, y_pred_orig, y_proba)
    return sonuc, model

class CNN1D(nn.Module):
    """38 öznitelikli vektörü 1B sinyal gibi ele alan basit CNN."""
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

def cnn_egit(X_train_s, X_test_s, y_train, y_test, epoch=80, lr=1e-3):
    """1B CNN'i eğitir ve test sonuçlarını döndürür."""
    print("  -> CNN (1B) eğitiliyor...")

    # Conv1d (batch, kanal, uzunluk) bekler; unsqueeze(1) ile kanal boyutu eklenir
    Xtr = torch.tensor(X_train_s, dtype=torch.float32).unsqueeze(1)
    Xte = torch.tensor(X_test_s, dtype=torch.float32).unsqueeze(1)
    ytr = torch.tensor(y_train, dtype=torch.long)
    yte_np = y_test

    model = CNN1D(n_features=X_train_s.shape[1])
    kayip = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)

    model.train()
    for ep in range(epoch):
        toplam = 0.0
        for bx, by in loader:
            opt.zero_grad()  # önceki batch gradyanları birikmesin
            cik = model(bx)
            loss = kayip(cik, by)
            loss.backward()
            opt.step()
            toplam += loss.item()
        if (ep + 1) % 20 == 0:
            print(f"     epoch {ep+1:3d} | kayıp: {toplam/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        cik = model(Xte)
        olas = torch.softmax(cik, dim=1).numpy()
        y_pred = cik.argmax(dim=1).numpy()
        y_proba = olas[:, 0]  # sütun 0: P(safra taşı var)

    return metrikleri_hesapla(yte_np, y_pred, y_proba), model

def capraz_dogrulama(X, y, X_s):
    """
    Tüm klasik modeller için 5-katlı tabakalı çapraz doğrulama yapar.
    Sonuçları (ortalama, std) sözlüğü olarak döndürür.
    """
    print("\n5-Katlı Çapraz Doğrulama ")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_modelleri = {
        "Rastgele Orman": (RandomForestClassifier(n_estimators=200,
                                                   random_state=SEED), False),
        "Gradyan Artırma": (GradientBoostingClassifier(random_state=SEED), False),
        "Karar Ağacı": (DecisionTreeClassifier(random_state=SEED), False),
        "K-En Yakın Komşu": (KNeighborsClassifier(n_neighbors=7), True),
        "Lojistik Regresyon": (LogisticRegression(max_iter=2000,
                                                   random_state=SEED), True),
        "Naive Bayes": (GaussianNB(), False),
        "Destek Vektör Makinesi": (SVC(kernel="rbf", probability=True,
                                       random_state=SEED), True),
        "Yapay Sinir Ağı": (MLPClassifier(hidden_layer_sizes=(64, 32),
                                           max_iter=500, random_state=SEED), True),
    }

    cv_sonuclari = {}
    for ad, (model, olcek) in cv_modelleri.items():
        veri = X_s if olcek else X
        skorlar = cross_val_score(model, veri, y, cv=cv, scoring="accuracy",
                                  n_jobs=-1)
        cv_sonuclari[ad] = (skorlar.mean(), skorlar.std())
        print(f"  {ad:25s} : {skorlar.mean():.4f} ± {skorlar.std():.4f}")
    return cv_sonuclari

def hiperparametre_ara(X_train_s, y_train):
    """En başarılı 3 model için kısa bir GridSearchCV çalıştırır."""
    print("\nHiperparametre araması (Lojistik Regresyon, RF, MLP)")

    aramalar = {
        "Lojistik Regresyon": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}
        ),
        "Rastgele Orman": (
            RandomForestClassifier(random_state=SEED),
            {"n_estimators": [100, 200, 300],
             "max_depth": [None, 5, 10]}
        ),
        "Yapay Sinir Ağı": (
            MLPClassifier(max_iter=500, random_state=SEED),
            {"hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
             "alpha": [0.0001, 0.001]}
        ),
    }

    en_iyi = {}
    for ad, (mdl, param) in aramalar.items():
        gs = GridSearchCV(mdl, param, cv=5, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train_s, y_train)
        en_iyi[ad] = (gs.best_params_, gs.best_score_)
        print(f"  {ad:25s} -> {gs.best_score_:.4f}  |  {gs.best_params_}")
    return en_iyi

def ozellik_onem_grafigi(model, feature_names, n=12):
    """Rastgele Orman'dan Gini önemleri çizer ve dosyaya kaydeder."""
    onemler = model.feature_importances_
    sira = np.argsort(onemler)[::-1][:n]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(n), onemler[sira][::-1], color="#3C7DB8")
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in sira][::-1])
    ax.set_xlabel("Gini önem skoru")
    ax.set_title("Rastgele Orman - En Önemli 12 Özellik")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ozellik_onem.png", dpi=150)
    plt.close()

def shap_analizi(rf_model, X_train, X_test, feature_names):
    """SHAP analizi . Kütüphane yüklü değilse atlanır."""
    try:
        import shap
        print("\nSHAP analizi yapılıyor...")
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test[:50])
        # Ağaç modellerinde shap_values liste döner (her sınıf için bir dizi);
        # indeks 1 = "safra taşı yok" sınıfının açıklamaları (görselleştirme için)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        plt.figure()
        shap.summary_plot(shap_values, X_test[:50],
                          feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print("  SHAP grafiği kaydedildi: ciktilar/shap_summary.png")
    except ImportError:
        print("  [UYARI] SHAP kütüphanesi yok. Atlanıyor. "
              "Kurmak için: pip install shap")
    except Exception as e:
        print(f"  [UYARI] SHAP analizi başarısız: {e}")

def karisiklik_matrisi_ciz(model, X_test, y_test, ad="YSA"):
    """En başarılı modelin karışıklık matrisini çizer."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=["Yok", "Var"], yticklabels=["Yok", "Var"], ax=ax)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(f"Karışıklık Matrisi - {ad}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/karisiklik_matrisi.png", dpi=150)
    plt.close()

def model_karsilastirma_grafigi(sonuclar):
    """Tüm modellerin doğruluğunu yatay sütun grafiğinde gösterir."""
    df = pd.DataFrame(sonuclar).T.sort_values("Doğruluk", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df.index, df["Doğruluk"], color="#1976D2")
    ax.set_xlabel("Test Doğruluğu")
    ax.set_title("Tüm Modellerin Performans Karşılaştırması")
    for i, v in enumerate(df["Doğruluk"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_karsilastirma.png", dpi=150)
    plt.close()

def main():
    print("=" * 70)
    print(" SAFRA TAŞI HASTALIĞI TAHMİN ÇALIŞMASI - MAKİNE ÖĞRENMESİ")
    print("=" * 70)

    print("\nVeri yükleme")
    df = veri_yukle()

    print("\nÖn işleme (eğitim/test ayrımı + ölçekleme)")
    (X_train, X_test, y_train, y_test,
     X_train_s, X_test_s, feature_names) = on_isleme(df)

    print("\nKlasik makine öğrenmesi modelleri")
    sonuclar, egitilmis = klasik_modelleri_egit(
        X_train, X_test, y_train, y_test, X_train_s, X_test_s
    )

    sonuclar["Eşikli Lineer Regresyon"], _ = esikli_lineer_regresyon(
        X_train_s, X_test_s, y_train, y_test
    )

    sonuclar["CNN (1B)"], _ = cnn_egit(X_train_s, X_test_s, y_train, y_test)

    
    print(" TÜM MODELLERİN TEST KÜMESİ SONUÇLARI")
    
    df_sonuc = pd.DataFrame(sonuclar).T.sort_values("Doğruluk", ascending=False)
    print(df_sonuc.round(4).to_string())
    df_sonuc.to_csv(f"{OUT_DIR}/sonuclar.csv", encoding="utf-8-sig")

    # CV için eğitim ve test verisi birleştirilerek tüm veri seti kullanılır;
    # bölme CV'nin kendi içinde yeniden yapılır
    cv_sonuc = capraz_dogrulama(
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        np.vstack([X_train_s, X_test_s])
    )

    en_iyi = hiperparametre_ara(X_train_s, y_train)

    print("\nGörselleştirmeler oluşturuluyor")
    rf_model, _ = egitilmis["Rastgele Orman"]
    ysa_model, _ = egitilmis["Yapay Sinir Ağı"]

    ozellik_onem_grafigi(rf_model, feature_names)
    karisiklik_matrisi_ciz(ysa_model, X_test_s, y_test, ad="YSA")
    model_karsilastirma_grafigi(sonuclar)

    shap_analizi(rf_model, X_train, X_test, feature_names)

    
    print(" TAMAMLANDI - Tüm çıktılar 'ciktilar/' klasörüne kaydedildi")
    

if __name__ == "__main__":
    main()
