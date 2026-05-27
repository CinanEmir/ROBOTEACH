# 🤖 RoboTeach: Videodan Görev Öğrenen İnsansı Robot Eğitim Sistemi (Learning from Observation)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics%20Engine-black)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environment-brightgreen)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Computer%20Vision-00A859)

**RoboTeach**, standart bir RGB videodaki insan hareketlerini bilgisayarlı görü algoritmalarıyla analiz ederek, elde edilen kinematik verileri Derin Pekiştirmeli Öğrenme (Deep RL) teknikleriyle fiziksel simülasyon ortamındaki insansı (humanoid) bir robota öğreten **uçtan uca (end-to-end)** bir robotik otonomi projesidir.

## 🚀 Proje Özeti ve Vizyon
Günümüzdeki endüstriyel insansı robotların (Tesla Optimus, Figure AI vb.) görev öğrenme süreçlerinden esinlenilerek geliştirilen bu proje, yüksek maliyetli Hareket Yakalama (MoCap) sistemlerine ihtiyaç duymadan, sadece bir kamera ve orta segment donanımlarla **"Gerçek İnsan Yürüyüşünün Taklidi"** görevini başarmıştır. 

Projenin nihai hedefi; simülasyonda (MuJoCo) eğitilen bu otonom kontrol politikalarının (policy), ilerleyen aşamalarda fiziksel robot donanımlarına aktarılabilir (*Sim-to-Real Transfer*) bir altyapıya sahip olmasıdır.

## ✨ Temel Özellikler (Key Features)

* **🎥 Görüntü İşleme & Kinematik Çıkarım:** MediaPipe kullanılarak videodan 33 farklı 3D anatomik nokta çıkarılır ve vektör matematiği ile robotun motor torklarının anlayacağı eklem açılarına (Joint Angles) dönüştürülür.
* **🧠 V10 Müfredatlı Öğrenme (Curriculum Learning):** PPO ajanının "Heykel Sendromu" (Statue Trap) gibi eylemsizlik krizlerine girmesini engellemek için, öğrenme süreci 3 dinamik faza bölünmüş ve robotun 100 milyon adım boyunca kademeli olarak yürümesi sağlanmıştır.
* **🛡️ Dinamik Postür Koruma Radarı (Anti-Hunch):** Robotun düşmemek için otonom olarak geliştirdiği kambur yürüme eğilimlerini engellemek için geliştirilen 3 katmanlı (Pitch, Abdomen, Height) dinamik ceza mekanizması.
* **⏱️ Sıkı Senkronizasyon (Strict Sync):** Referans animasyon ile simülasyon arasındaki faz kaymalarını (Phase Hacking) engelleyen matematiksel zaman kilidi.
* **💻 RoboTeach UI (Kullanıcı Arayüzü):** Ağır terminal komutlarını ortadan kaldıran; eğitim başlatmayı, model checkpoint'lerini yönetmeyi ve MuJoCo render'ını tek tıklamayla sunan karanlık tema odaklı kontrol paneli.

## 📂 Yazılım Mimarisi ve Dosya Yapısı

Sistem, modülerlik ve ölçeklenebilirlik prensiplerine uygun olarak tasarlanmıştır:

```text
RoboTeach_Project/
├── main.py                     # RoboTeach UI ve ana tetikleyici
├── humanoid_imitation_env.py   # Çevre, Kurallar ve Hibrit Ödül Motoru (Gymnasium)
├── humanoid.xml                # Biyomekanik modifikasyonlu MuJoCo robot modeli
├── train_hardcore.py           # PPO eğitim döngüsü ve hiperparametre yönetimi
├── visualize_model.py          # Eğitilmiş modellerin (Inference) görselleştirilmesi
├── database.py                 # SQLite veritabanı (UI konfigürasyonları için)
├── humanoid_ai.db              # Kalıcı veri ve log saklama dosyası
└── requirements.txt            # Gerekli kütüphaneler
