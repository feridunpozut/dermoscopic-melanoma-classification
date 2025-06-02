# 🧠 Dermoscopic Skin Lesion Classification

Bu proje, **ISIC (International Skin Imaging Collaboration)** veri setlerini kullanarak cilt lezyonlarının *benign* (iyi huylu) ve *malignant* (kötü huylu) olarak sınıflandırılması amacıyla geliştirilmiştir. Eğitim sücrecinde artefakt temizliği, renk normlaştırma, belirsizlik temelli öğrenme ve Grad-CAM gibi açıklanabilirlik teknikleri entegre edilmiştir.

## 📁 Proje Yapısı

```
.
🔹 app.py                         # Streamlit tabanlı arayüz (sınıflandırma + GradCAM görselleştirme)
🔹 artefact_removal.py           # Görüntülerden saç, vinyet ve düşük kontrast gibi artefaktları temizler
🔹 combine_dataset.py            # ISIC 2018, 2019 ve 2020 verilerini benzersiz biçimde birleştirir
🔹 gradcam_infer.py              # Belirtilen görüntü dizininde GradCAM ısı haritalarını üretir
🔹 hyperparameters_optimizer.py  # Optuna ile hiperparametre optimizasyonu yapar
🔹 isiic_downloader.py           # ISIC API üzerinden verilen ID'leri indirir
🔹 model.py                      # LiteSkinLesionClassifier modelini tanımlar
🔹 preprocessing.py              # Dataset sınıfı ve transform (ön işleme) adımları
🔹 test.py                       # Eğitilmiş modeli test verisi üzerinde değerlendirir
🔹 train_utils.py                # Seed ayarı ve entropy-weighted loss gibi yardımcı fonksiyonlar
🔹 train.py                      # K-Fold destekli model eğitimi
🔹 config/
└── config.yaml               # Tüm hiperparametre ve yol ayarlarını içerir
🔹 data/
└── splits/                   # Eğitim, validasyon ve test CSV'leri burada yer alır
```

## 🔬 Kullanılan Yöntemler

* **Artefakt Temizleme:** Görsellerdeki saç, vinyet etkisi ve kontrast düşüklüğü giderildi.
* **Veri Dengeleme:** Melanoma oranı %49.9 olacak şekilde ham veride örnekleme ile sağlandı.
* **Veri Artırma:** Döndürme, yatay-dikey çevirme gibi augmentasyonlar online olarak uygulandı.
* **Transfer Learning:** EfficientNet-b0 gibi önceden eğitilmiş hafif modeller kullanıldı.
* **Uncertainty-Aware Learning:** Entropiye dayalı ağırlıklı kayıpla, modelin kararsız olduğu örnekler öne çıkarıldı.
* **Grad-CAM:** Sınıflandırma kararlarının açıklanabilirliğini sağlamak için ısı haritaları oluşturuldu.
* **Optuna:** Hiperparametreler için optimize edilmiş öğrenme oranı, dropout ve batch size belirlendi.

## ⚙️ Kurulum

```bash
git clone https://github.com/kullanici/skin-lesion-classification.git
cd skin-lesion-classification
pip install -r requirements.txt
```

> Not: `config/config.yaml` dosyasındaki yolları kendi veri yapınıza göre düzenleyin.

## 🚀 Eğitim

```bash
python train.py
```

## 🧲 Test

```bash
python test.py
```

## 🔍 Grad-CAM ile Görselleştirme

```bash
python gradcam_infer.py
```

## 📊 Streamlit Arayüzü

```bash
streamlit run app.py
```

## 🔧 Hiperparametre Optimizasyonu

```bash
python hyperparameters_optimizer.py
```

## 🗒️ Lisans

Bu proje açık kaynak olup araştırma ve eğitim amaçlı kullanılabilir. Lütfen ISIC veri lisans koşullarını göz önünde bulundurunuz.

---

Hazırlayan: \[Feridun Pözüt]
