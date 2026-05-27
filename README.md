Markdown# 🤖 RoboTeach: Videodan Görev Öğrenen İnsansı Robot Eğitim Sistemi (Learning from Observation)

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

Proje, "Görüntü İşleme" ve "Pekiştirmeli Öğrenme" olmak üzere iki bağımsız modülden oluşan heterojen bir mimariye sahiptir:

### 1. Görüntü İşleme ve Veri Çıkarımı Modülü (Sistem I)
MediaPipe kullanılarak insan hareketlerinin dijitalleştirildiği ve kinematik hesaplamaların yapıldığı veri hazırlama katmanı:
```text
Vision_Module/
├── robo_env/                   # Python sanal ortam dizini
├── .env                        # Çevresel değişkenler ve konfigürasyonlar
├── .gitignore                  # Git takip dışı dosyalar
├── main.py                     # OpenCV ve MediaPipe pipeline'ını çalıştıran ana betik
├── plot_angles.py              # Çıkarılan eklem açılarını analiz eden ve görselleştiren araç
├── poses.jsonl                 # Tespit edilen 33 anatomik noktanın ham 3D koordinat çıktısı
├── angles.csv                  # Robot motorları için üretilen temizlenmiş eklem açı matrisi
└── requirements.txt            # MediaPipe, OpenCV, NumPy vb. bağımlılıklar
2. Simülasyon ve Derin Pekiştirmeli Öğrenme Modülü (Sistem II)Elde edilen angles.csv verisinin içeri aktarıldığı, otonom PPO eğitiminin yapıldığı ve RoboTeach arayüzünün bulunduğu karar katmanı:PlaintextRL_Simulation_Module/
├── main.py                     # RoboTeach UI ve sistemi başlatan ana tetikleyici
├── humanoid_imitation_env.py   # Çevre, Kurallar ve Hibrit Ödül Motoru (Gymnasium)
├── humanoid.xml                # Biyomekanik modifikasyonlu MuJoCo robot modeli
├── train_hardcore.py           # PPO eğitim döngüsü ve hiperparametre yönetimi
├── visualize_model.py          # Eğitilmiş modellerin (Inference) MuJoCo'da görselleştirilmesi
├── database.py                 # SQLite veritabanı (UI konfigürasyonları için)
├── humanoid_ai.db              # Kalıcı veri ve model kayıt yollarının tutulduğu dosya
└── requirements.txt            # Stable-Baselines3, PyTorch, Gymnasium vb. bağımlılıklar
🛠️ Kurulum (Installation)Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin:Repoyu bilgisayarınıza klonlayın:Bash   git clone [https://github.com/KULLANICI_ADINIZ/RoboTeach.git](https://github.com/KULLANICI_ADINIZ/RoboTeach.git)
   cd RoboTeach
Gerekli Python kütüphanelerini yükleyin (Python 3.10+ önerilir):Bash   # Sistem I (Görüntü İşleme) için:
   cd Vision_Module
   pip install -r requirements.txt

   # Sistem II (Simülasyon ve RL) için:
   cd ../RL_Simulation_Module
   pip install -r requirements.txt
🎮 Kullanım (Usage)Proje, geliştirici dostu RoboTeach UI üzerinden yönetilmektedir. Arayüzü başlatmak için terminale şu komutu girin:Bashcd RL_Simulation_Module
python main.py
Arayüz Üzerinden Yapılabilecekler:Yeni Eğitim: Vision_Module'den elde ettiğiniz angles.csv verisini seçerek sıfırdan PPO eğitimi başlatabilirsiniz.Model Yönetimi: Geçmiş eğitimlerinizi (.zip checkpoint'leri) görüntüleyebilir, kaldığınız yerden eğitime devam edebilir (Fine-Tuning) veya simülasyonu başlatabilirsiniz.Canlı Loglama: Eğitim sırasındaki TensorBoard çıktılarını ve donanım sıcaklık/kullanım değerlerini (örn. RTX 3050 Ti: 61°C) anlık takip edebilirsiniz.📊 Testler ve SonuçlarRobot, MuJoCo fizik motorunda toplam 100 milyon simülasyon adımı boyunca eğitilmiştir. Eğitim süreci boyunca 230. adımlardaki kronik düşüşler Müfredatlı Öğrenme stratejisiyle aşılmış ve ajan sürekli (marathon) yürüyüş yeteneği kazanmıştır.Erken Aşama (0-1M Adım)Orta Aşama (1-5M Adım)Son Aşama (Nihai Model)Denge arayışı ve sık düşüşlerAdımlama ve ritim yakalamaPürüzsüz ve otonom yürüyüş👉 Eğitim Süreci ve Yürüme Testi Analiz Videosunu YouTube'da İzleyin👥 Ekip ve KatkılarBu proje, Ondokuz Mayıs Üniversitesi Bilgisayar Mühendisliği lisans bitirme projesi kapsamında geliştirilmiştir.RL, Fizik Simülasyonu ve Arayüz (Backend/Frontend): Emir Cinan, Semih, Muhammet KöseGörüntü İşleme, Kinematik ve Biyomekanik Veri Çıkarımı: Ertuğrul Han ŞenProje Danışmanı: Doç. Dr. İsmail İşeri📝 LisansBu proje MIT Lisansı altında lisanslanmıştır.
