# Safra Taşı Hastalığının Makine Öğrenmesi ile Tahmini

Bu proje, **Gazi Üniversitesi Bilgisayar Mühendisliği** Makine Öğrenmesi dersi
kapsamında hazırlanmıştır. UCI Machine Learning Repository üzerinden
erişilebilen [Gallstone veri seti](https://archive.ics.uci.edu/dataset/1150/gallstone-1)
kullanılarak 10 farklı sınıflandırma algoritması karşılaştırmalı biçimde
uygulanmıştır.

## Yazarlar

| Ad Soyad | Numara | E-posta |
|---|---|---|
| Emirhan FIRTINA | 24181616030 | 24181616030@gazi.edu.tr |
| Mehmet Akif SOYSAL | 24181617005 | 24181617005@gazi.edu.tr |

## Proje Hakkında

Safra taşı hastalığı (kolelitiazis), klinik pratikte sık karşılaşılan,
hastaların bir bölümünde uzun süre belirti vermeyen ve tanısı çoğunlukla
ultrasonografi gibi görüntüleme yöntemlerine dayanan bir hepatobiliyer
problemdir. Bu çalışma, görüntüleme yapılmadan, yalnızca rutin biyokimya
ve biyoempedans verileriyle erken risk tahmininin mümkün olup olmadığını
araştırmaktadır.

## Kullanılan Algoritmalar

1. **Rastgele Orman** (Random Forest)
2. **Karar Ağacı** (Decision Tree)
3. **K-En Yakın Komşu** (KNN)
4. **Lojistik Regresyon** (Logistic Regression)
5. **Eşikli Lineer Regresyon** (Linear Regression with threshold)
6. **Naive Bayes** (Gaussian)
7. **Destek Vektör Makinesi** (SVM, RBF kernel)
8. **Gradyan Artırma** (Gradient Boosting)
9. **Yapay Sinir Ağı** (MLP - 64,32 nöron)
10. **1 Boyutlu CNN** (PyTorch)

## Ek Özellikler

Standart şablonun ötesinde aşağıdaki adımlar da uygulanmıştır:

- **5-katlı Stratified Cross Validation** ile model varyans analizi
- **GridSearchCV** ile hiperparametre araması (LR, RF, YSA için)
- **SHAP** kütüphanesi ile yorumlanabilirlik analizi
- **ROC eğrileri** ile çoklu model karşılaştırması
- Hem doğruluk hem de **klinik tarama odaklı duyarlılık** analizi

## Sonuçlar (Özet)

| Algoritma | Doğruluk | Duyarlılık | ROC-AUC |
|---|---|---|---|
| **Yapay Sinir Ağı** | **0.7812** | 0.7500 | 0.8750 |
| Lojistik Regresyon | 0.7656 | **0.8750** | **0.8770** |
| Eşikli Lineer Regresyon | 0.7500 | 0.8438 | 0.8398 |
| Gradyan Artırma | 0.7500 | 0.8125 | 0.8262 |
| Rastgele Orman | 0.7344 | 0.7500 | 0.8340 |

> **En etkili 3 değişken:** CRP, Vitamin D, AST

## Kurulum ve Çalıştırma

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Veri setini proje klasörüne yerleştir
#    (gallstone.xlsx olarak)

# 3. Çalıştır
python main.py
```

Çalıştırma sonunda tüm çıktılar `ciktilar/` klasörüne kaydedilir:

```
ciktilar/
├── sonuclar.csv              # Tüm modellerin metrikleri
├── model_karsilastirma.png   # Doğruluk karşılaştırma grafiği
├── ozellik_onem.png          # En önemli 12 özellik
├── karisiklik_matrisi.png    # YSA karışıklık matrisi
└── shap_summary.png          # SHAP analizi
```

## Proje Yapısı

```
.
├── main.py             # Ana kod (tüm akış)
├── grafikler.py        # Rapor için ek grafik üreticisi
├── requirements.txt    # Bağımlılıklar
├── README.md           # Bu dosya
├── rapor.docx          # Final rapor
├── sunum.pptx          # Sunum dosyası
└── gallstone.xlsx      # Veri seti (UCI'dan indirilmeli)
```

## Sistem Gereksinimleri

- Python 3.10+
- ~500 MB RAM
- Çalışma süresi: ~30 saniye (modern dizüstüde)

## Kaynaklar

Tam kaynakça raporun **KAYNAKÇA** bölümünde verilmiştir. Temel referans:
> Esen, I. ve ark. (2024). *Early prediction of gallstone disease with a
> machine learning-based method from bioimpedance and laboratory data.*
> Medicine.

## Lisans

Eğitim amaçlı, MIT Lisansı altında paylaşılmıştır.
