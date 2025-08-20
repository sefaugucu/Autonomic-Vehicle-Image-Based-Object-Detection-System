# Autonomic-Vehicle-Image-Based-Object-Detection-System
This project is a deep learning -based object detection system developed to help autonomous vehicles understand their environment. Yolo (You Only Look Once) can make real -time object detection by using architecture.
[My model training and outputs]
<img width="1576" height="620" alt="model eğitimi" src="https://github.com/user-attachments/assets/e9c9e76c-7a05-4cfd-b11b-29015552b9f3" />
<img width="1488" height="662" alt="model eğitimi 2" src="https://github.com/user-attachments/assets/26107af4-dcde-4e16-af5f-8fc2f1baa565" />
<img width="776" height="137" alt="Ekran görüntüsü 2025-08-19 023013" src="https://github.com/user-attachments/assets/3a9ccd83-a09a-47fa-90c2-1890018ba6ff" />
<img width="1790" height="950" alt="Ekran görüntüsü 2025-08-19 023803" src="https://github.com/user-attachments/assets/06f31f8e-ca61-4b84-a099-aca50e308a8f" />
<img width="1517" height="982" alt="Ekran görüntüsü 2025-08-19 022809" src="https://github.com/user-attachments/assets/04a205c2-9957-4266-b686-3833341b8f5a" />
<img width="1095" height="668" alt="Ekran görüntüsü 2025-08-19 022401" src="https://github.com/user-attachments/assets/972c70bf-7362-4650-a43e-470c8fc5ed80" />
<img width="989" height="974" alt="Ekran görüntüsü 2025-08-20 143240" src="https://github.com/user-attachments/assets/cca3f556-639e-45e6-9a83-4c6c76a1002b" />
## 📊 Results and Performance

### Loss Grafiği
Eğitim ve validation loss'un nasıl değiştiğini gösterir:
![Loss Grafiği](assets/training_loss_curve.png)

### Örnek Tespit
Modelimizin gerçek zamanlı bir tespitinin çıktısı:
![Örnek Tespit](assets/sample_detection.jpg)

### Model Metrikleri
| Metric     | Value |
|------------|-------|
| mAP@0.5    | 0.89  |
| Precision  | 0.92  |
| Recall     | 0.85  |
| FPS        | 45    |

# 🎯 Results

### Training Graphics
<p align="center">
  <img src="assets/loss_plot.png" width="45%" alt="Training Loss">
  <img src="assets/metrics_plot.png" width="45%" alt="Training Metrics">
</p>

### Confusion Matrix
<p align="center">
  <img src="assets/confusion_matrix.png" width="60%" alt="Confusion Matrix">
</p>

### Real-Time Detection Example
<p align="center">
  <img src="assets/demo.gif" alt="Real-Time Detection Demo">
</p>


# 🚗 Autonomous Vehicle Image-Based Object Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-brightgreen)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![YOLO](https://img.shields.io/badge/YOLO-v8-red)](https://github.com/ultralytics/ultralytics)

Otonom araçlar için geliştirilmiş, gerçek zamanlı çalışabilen ve YOLOv8 mimarisi ile güçlendirilmiş bir görüntü tabanlı nesne tespit sistemidir. Bu proje, araçların çevrelerini anlamaları ve güvenli bir şekilde navigasyon yapabilmeleri için kritik öneme sahip nesneleri (araç, yaya, trafik ışığı vb.) yüksek doğruluk ve hızla tespit eder.

---

## ✨ Öne Çıkan Özellikler

*   **🚀 Gerçek Zamanlı Performans:** YOLOv8 sayesinde yüksek FPS (Saniye Kare Sayısı) değerlerinde sorunsuz çalışır.
*   **🎯 Yüksek Doğruluk:** COCO veri seti üzerinde pre-trained bir model fine-tuning ile yüksek mAP değerlerine ulaşır.
*   **📷 Çoklu Girdi Desteği:** Web kamerası, video dosyaları ve görüntüler üzerinde çalışabilir.
*   **🛠️ Kolay Entegrasyon:** Modüler yapısı sayesinde farklı model ve veri setleriyle kolayca entegre edilebilir.
*   **📊 Kapsamlı Görselleştirme:** Eğitim metriği grafikleri, confusion matrix ve örnek çıktılarla performans raporlama.

---

## 📸 Örnek Çıktılar

<p align="center">
  <img src="assets/sample_detection.jpg" alt="Örnek Nesne Tespiti" width="800"/>
  <br>
  <em>Modelin gerçek zamanlı bir sahnede yaptığı tespit örneği.</em>
</p>

<p align="center">
  <img src="assets/training_curves.png" alt="Eğitim Grafikleri" width="800"/>
  <br>
  <em>Eğitim ve validation loss/metric grafikleri.</em>
</p>

---

## 🏗️ Mimari ve Teknolojiler

*   **Model Mimari:** [YOLOv8](https://github.com/ultralytics/ultralytics) (You Only Look Once)
*   **Programlama Dili:** Python 3.8+
*   **Ana Kütüphaneler:** OpenCV, PyTorch, Ultralytics, NumPy
*   **Veri Seti:** MS COCO (Common Objects in Context)
*   **Görselleştirme:** Matplotlib, Seaborn

---

## ⚡ Hızlı Başlangıç

### Gereksinimler ve Kurulum

1.  Depoyu klonlayın:
    ```bash
    git clone https://github.com/kullanici-adiniz/oproje.git
    cd oproje
    ```

2.  Gerekli kütüphaneleri yükleyin (Bir sanal ortam önerilir):
    ```bash
    pip install -r requirements.txt
    ```

### Kullanım

**Web kamerası ile gerçek zamanlı tespit:**
```bash
python src/detect.py --source 0
