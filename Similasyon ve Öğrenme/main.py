import os
import sys
import GPUtil
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFrame, 
                             QStackedWidget, QFileDialog, QListWidget, QInputDialog, QMessageBox, QPlainTextEdit)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor
import qtawesome as qta
from humanoid_imitation_env import HumanoidWalkingEnv
from stable_baselines3 import PPO
import string

import os
import string
import shutil


def get_next_robot_name(base_path="./models"):
    """Klasörleri alfabetik olarak tarar ve bir sonraki ismi belirler."""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return "robotA"
    
    existing_folders = [f for f in os.listdir(base_path) if f.startswith("robot")]
    if not existing_folders:
        return "robotA"
    
    # En son klasörü bul (Alfabetik sıralama ile)
    last_folder = sorted(existing_folders)[-1]
    last_letter = last_folder.replace("robot", "")
    
    alphabet = list(string.ascii_uppercase)
    try:
        current_index = alphabet.index(last_letter)
        next_letter = alphabet[current_index + 1]
    except (ValueError, IndexError):
        # Z'den sonrasını sayısal olarak devam ettir
        next_letter = str(len(existing_folders) + 1)
        
    return f"robot{next_letter}"

def prepare_training_folder(csv_source_path):
    """Yeni robot için gerekli klasör yapısını hazırlar."""
    robot_name = get_next_robot_name()
    robot_path = os.path.join("./models", robot_name)
    
    # Gerekli alt klasörler
    os.makedirs(os.path.join(robot_path, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(robot_path, "savepoints"), exist_ok=True)
    
    # Kullanılan CSV'nin bir kopyasını klasörün içine atalım (Arşiv için)
    dest_csv = os.path.join(robot_path, "motion_data.csv")
    shutil.copy(csv_source_path, dest_csv)
    
    print(f"📁 Yeni eğitim dizini hazır: {robot_path}")
    return robot_name, dest_csv

def prepare_training_folder(csv_source_path):
    """Yeni robot için gerekli klasör yapısını MUTLAK YOLLARLA hazırlar."""
    try:
        # 1. Scriptin bulunduğu ana dizini bul (En sağlam yöntem)
        # Bu satır, projenin C:\Projeler\Humanoid_Project gibi gerçek konumunu verir
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Yeni robot ismini al
        robot_name = get_next_robot_name(os.path.join(base_dir, "models"))
        
        # 3. Yolları mutlak hale getir
        robot_path = os.path.join(base_dir, "models", robot_name)
        checkpoint_path = os.path.join(robot_path, "checkpoint")
        savepoints_path = os.path.join(robot_path, "savepoints")
        
        # 4. Klasörleri oluştur (Alt klasörlerle birlikte)
        # exist_ok=True: Klasör zaten varsa hata vermez
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(savepoints_path, exist_ok=True)
        
        # 5. CSV'yi kopyala
        dest_csv = os.path.join(robot_path, "motion_data.csv")
        shutil.copy(csv_source_path, dest_csv)
        
        # DEBUG: Gerçekten nereye oluşturduğunu terminalde gör
        print(f"✅ Klasor Yapisi Basariyla Kuruldu:")
        print(f"   -> Ana: {os.path.abspath(robot_path)}")
        print(f"   -> Checkpoint: {os.path.abspath(checkpoint_path)}")
        print(f"   -> Savepoints: {os.path.abspath(savepoints_path)}")
        
        return robot_name, dest_csv

    except Exception as e:
        print(f"❌ Klasor Hazirlama Hatasi: {e}")
        return None, None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Klasör yoksa oluştur (Mühendislik tedbiri)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
from PyQt6.QtCore import QThread, pyqtSignal
import time
from stable_baselines3.common.callbacks import CheckpointCallback

import subprocess

class TrainingWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, task_name, model_path, is_new=True):
        super().__init__()
        self.task_name = task_name
        self.model_path = model_path
        self.process = None  # Süreci burada saklayacağız
        self.is_running = True

    def run(self):
        script_path = os.path.join(os.path.dirname(__file__), "train_hardcore.py")
        
        # Subprocess'i başlatırken referansı self.process'e eşitliyoruz
        self.process = subprocess.Popen(
            [sys.executable, script_path, self.task_name, self.model_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )

        try:
            # Süreç canlı olduğu ve biz durdurmadığımız sürece logları oku
            while self.is_running and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self.log_signal.emit(line.strip())
                elif self.process.poll() is not None:
                    break # Süreç bittiyse döngüden çık
        except Exception as e:
            self.log_signal.emit(f"⚠️ Log Okuma Hatası: {str(e)}")
        finally:
            self.stop_process() # Her durumda temizlik yap
            self.finished_signal.emit()

    def stop_process(self):
        """Eğitimi dışarıdan durdurmak için kullanılan acil durum fonksiyonu."""
        self.is_running = False
        if self.process:
            # Önce nazikçe (terminate), sonra zorla (kill) kapatıyoruz
            self.process.terminate() 
            try:
                self.process.wait(timeout=2) # 2 saniye bekle
            except subprocess.TimeoutExpired:
                self.process.kill() # Kapanmadıysa zorla öldür
            self.process = None

class ModernUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Humanoid Command Center v2.5")
        self.resize(1250, 850)
        self.setStyleSheet("background-color: #1a1b26; color: #cfc9c2;")

        # --- ANA LAYOUT ---
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.current_path = MODELS_DIR
        

        # --- 1. SIDEBAR (YAN MENÜ) ---
        sidebar = QFrame()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("background-color: #16161e; border-right: 1px solid #24283b;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 40, 20, 40)
        sidebar_layout.setSpacing(15)

        logo = QLabel("ROBOTEACH")
        logo.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        logo.setStyleSheet("color: #7aa2f7; margin-bottom: 40px;")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(logo)
        

        # Navigasyon Butonları
        self.btn_dash = self.create_nav_btn("fa5s.rocket", " Yeni Eğitim")
        self.btn_tasks = self.create_nav_btn("fa5s.robot", " Robot Modelleri")
        self.btn_monitor = self.create_nav_btn("fa5s.terminal", " Canlı Analiz/Log")
        
        sidebar_layout.addWidget(self.btn_dash)
        sidebar_layout.addWidget(self.btn_tasks)
        sidebar_layout.addWidget(self.btn_monitor)
        sidebar_layout.addStretch()
        
        main_layout.addWidget(sidebar)

        # --- 2. CONTENT AREA (İÇERİK ALANI) ---
        content_layout = QVBoxLayout()
        
        # --- 2. CONTENT AREA ---
        content_layout = QVBoxLayout()

        # Header (Üst Panel)
        header = QFrame()
        header.setFixedHeight(80)
        header.setStyleSheet("background-color: #1a1b26; border-bottom: 1px solid #24283b;")
        
        # Layout'u oluştururken ebeveyn (header) vermiyoruz, sonra set ediyoruz
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(30, 0, 30, 0)
        header.setLayout(header_layout) # Layout'u frame'e burada bağladık (En güvenli yol)

        # Durum Etiketi
        self.status_label = QLabel("Sistem Hazır")
        self.status_label.setStyleSheet("color: #9ece6a; font-weight: bold; font-size: 15px;")
        
        # Durdurma Butonu
        self.btn_stop_training = QPushButton(qta.icon('fa5s.stop-circle', color='#f7768e'), " Eğitimi Durdur")
        self.btn_stop_training.setEnabled(False) 
        self.btn_stop_training.setStyleSheet("""
            QPushButton { 
                background-color: #24283b; color: #f7768e; border: 1px solid #f7768e; 
                padding: 8px 18px; border-radius: 8px; font-weight: bold;
            }
            QPushButton:hover { background-color: #f7768e; color: white; }
            QPushButton:disabled { border-color: #414868; color: #414868; }
        """)
        self.btn_stop_training.clicked.connect(self.stop_active_training)

        # GPU Etiketi
        self.gpu_label = QLabel("GPU: Taranıyor...")
        self.gpu_label.setStyleSheet("font-family: 'Consolas'; color: #7aa2f7; font-size: 14px;")

        # Widget'ları Layout'a Ekle (Soldan Sağa Sıralama)
        header_layout.addWidget(self.status_label)
        header_layout.addStretch() # Aradaki boşluğu açar
        header_layout.addWidget(self.btn_stop_training)
        header_layout.addSpacing(20) # Buton ve GPU arasına minik boşluk
        header_layout.addWidget(self.gpu_label)

        content_layout.addWidget(header)

        # Sayfa Değiştirici (Stacked Widget)
        self.pages = QStackedWidget()
        self.init_dashboard()    # Sayfa 0
        self.init_task_manager() # Sayfa 1
        self.init_log_view()     # Sayfa 2
        content_layout.addWidget(self.pages)
        
        main_layout.addLayout(content_layout)

        # Sayfa Geçiş Sinyalleri
        self.btn_dash.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        self.btn_tasks.clicked.connect(self.sync_and_show_tasks) # Sekmeye tıklandığında klasörü tara
        self.btn_monitor.clicked.connect(lambda: self.pages.setCurrentIndex(2))
        
        # GPU ve Sistem Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gpu_stats)
        self.timer.start(2000)

    def create_nav_btn(self, icon, text):
        btn = QPushButton(qta.icon(icon, color='#7aa2f7'), text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton {
                background-color: transparent; border: none; color: #a9b1d6;
                font-size: 17px; padding: 18px; text-align: left; border-radius: 12px;
            }
            QPushButton:hover { background-color: #24283b; color: white; }
        """)
        return btn
    def stop_active_training(self):
        """Kullanıcı butona bastığında eğitimi sonlandırır."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            # Kullanıcıya bir onay soralım (Yanlışlıkla basmalara karşı)
            confirm = QMessageBox.question(self, 'Eğitimi Durdur', 
                "Eğitimi sonlandırmak istediğine emin misin? Mevcut checkpointler saklanacaktır.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if confirm == QMessageBox.StandardButton.Yes:
                self.worker.stop_process() # Süreci öldüren fonksiyonu çağır
                self.log_message("🛑 Eğitim kullanıcı tarafından durduruldu.")
                self.status_label.setText("Durum: Sistem Hazır (Eğitim Durduruldu)")
                self.btn_stop_training.setEnabled(False)

    def update_gpu_stats(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self.gpu_label.setText(f"RTX 3050 Ti: {int(gpu.temperature)}°C | %{int(gpu.load*100)}")
        except: pass

    # --- DASHBOARD (YENİ EĞİTİM BAŞLATMA) ---
    def init_dashboard(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(50, 50, 50, 50)
        
        card = QFrame()
        card.setStyleSheet("background-color: #24283b; border-radius: 30px; border: 2px dashed #414868;")
        card_layout = QVBoxLayout(card)
        
        btn = QPushButton(qta.icon('fa5s.plus-square', color='#7aa2f7'), " SIFIRDAN ROBOT EĞİTİMİ BAŞLAT")
        btn.setStyleSheet("background: transparent; border: none; color: #7aa2f7; font-size: 22px; font-weight: bold; padding: 50px;")
        btn.clicked.connect(self.start_new_robot_training)

        desc = QLabel("Mac'ten gelen koordinat (.csv) verisini seçerek yeni bir robot kimliği oluşturun.")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #565f89; font-size: 16px; border: none;")

        card_layout.addStretch()
        card_layout.addWidget(btn)
        card_layout.addWidget(desc)
        card_layout.addStretch()
        
        layout.addWidget(card)
        self.pages.addWidget(page)

    def init_task_manager(self):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(30, 30, 30, 30)

            # 1. ÖNCE NESNEYİ OLUŞTUR (Sıralama hatasını düzelttik)
            self.task_list = QListWidget()
            self.task_list.setStyleSheet("""
                QListWidget { background-color: #1a1b26; border: none; outline: none; margin-top: 10px;}
                QListWidget::item { background-color: #24283b; margin-bottom: 12px; padding: 20px; border-radius: 12px; color: #cfc9c2; font-weight: bold;}
                QListWidget::item:selected { border: 2px solid #7aa2f7; background-color: #24283b; color: white; }
            """)

            # Üst Bar ve Refresh
            top_bar = QHBoxLayout()
            title = QLabel("Mevcut Modeller ve Checkpointler")
            title.setStyleSheet("font-size: 20px; font-weight: bold; color: #7aa2f7;")

            self.back_btn = QPushButton(qta.icon('fa5s.arrow-left', color='#7aa2f7'), " Geri")
            self.back_btn.setStyleSheet("background: #24283b; padding: 5px 15px; border-radius: 5px;")
            self.back_btn.clicked.connect(self.go_back)
            self.back_btn.hide() # Ana klasördeyken gizli
            
            # Mevcut navigasyon yolunu tutan değişken
            self.current_path = MODELS_DIR
            
            # 2. ŞİMDİ SİNYALLERİ BAĞLA (Nesne artık tanımlı olduğu için hata vermez)
            self.task_list.itemDoubleClicked.connect(self.on_item_double_clicked)
            self.task_list.itemClicked.connect(self.toggle_task_panel)
            
            self.refresh_btn = QPushButton(qta.icon('fa5s.sync-alt', color='#7aa2f7'), "")
            self.refresh_btn.setFixedSize(40, 40)
            self.refresh_btn.clicked.connect(self.sync_and_show_tasks)
            
            # Layout'a elemanları ekle (back_btn eklendi)
            top_bar.addWidget(title)
            top_bar.addStretch()
            top_bar.addWidget(self.back_btn) 
            top_bar.addWidget(self.refresh_btn)
            top_bar.addWidget(self.back_btn) # Butonu ekrana basar
            top_bar.addWidget(self.refresh_btn)
            layout.addLayout(top_bar)
            layout.addLayout(top_bar)
            
            layout.addWidget(self.task_list)

            # Kontrol Paneli (Toggle Panel)
            self.detail_panel = QFrame()
            self.detail_panel.setStyleSheet("background-color: #16161e; border-radius: 20px; border: 1px solid #414868; padding: 15px;")
            panel_layout = QVBoxLayout(self.detail_panel)
            
            self.panel_title = QLabel("Robot Kontrol")
            self.panel_title.setStyleSheet("font-weight: bold; font-size: 18px; color: #7aa2f7; border: none; margin-bottom: 10px;")
            panel_layout.addWidget(self.panel_title)
            
            self.btn_cont_old = QPushButton(qta.icon('fa5s.play'), " Mevcut Beyinle Devam Et (Checkpoint)")
            self.btn_cont_new = QPushButton(qta.icon('fa5s.plus-circle'), " Yeni Veri Enjekte Et (Fine-Tuning)")
            
            for btn in [self.btn_cont_old, self.btn_cont_new]:
                btn.setStyleSheet("background: #24283b; padding: 15px; border-radius: 10px; text-align: left; border: none; color: white;")
                btn.clicked.connect(self.on_training_button_click)
                panel_layout.addWidget(btn)
            
            self.detail_panel.hide()
            layout.addWidget(self.detail_panel)
            
            self.pages.addWidget(page)

    # --- LOG VE ANALİZ SAYFASI ---
    def init_log_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 30, 30, 30)
        
        title = QLabel("Sistem Logları")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #7aa2f7; margin-bottom: 10px;")
        
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #16161e; color: #9ece6a; font-family: 'Consolas'; font-size: 13px; border-radius: 15px; padding: 15px; border: 1px solid #24283b;")
        
        layout.addWidget(title)
        layout.addWidget(self.log_console)
        self.pages.addWidget(page)

    # --- YARDIMCI FONKSİYONLAR ---
    def log_message(self, message):
        self.log_console.appendPlainText(f">>> {message}")
        self.log_console.moveCursor(QTextCursor.MoveOperation.End)

    # ModernUI sınıfının __init__ kısmına bunu ekle:
# self.current_path = MODELS_DIR

    def sync_and_show_tasks(self, target_path=None):
        if target_path:
            self.current_path = target_path
        
        self.pages.setCurrentIndex(1)
        self.task_list.clear()
        self.detail_panel.hide()

        if not os.path.exists(self.current_path): return

        # Klasörleri listele
        for item in os.listdir(self.current_path):
            full_path = os.path.join(self.current_path, item)
            if os.path.isdir(full_path):
                files_inside = len([z for z in os.listdir(full_path) if z.endswith('.zip')])
                sub_folders = len([d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))])
                info_text = f"({files_inside} Models)" if files_inside > 0 else f"({sub_folders} Folders)"
                
                # KRİTİK: Formatımız "📂 [KLASÖR] isim (ek_bilgi)"
                self.task_list.addItem(f"📂 [KLASÖR] {item} {info_text}")

        # Modelleri listele
        for item in os.listdir(self.current_path):
            if item.endswith('.zip'):
                self.task_list.addItem(f"🧠 [MODEL] {item}")
        is_at_root = os.path.abspath(self.current_path) == os.path.abspath(MODELS_DIR)
        self.back_btn.setVisible(not is_at_root)
    def on_item_double_clicked(self, item):
        """Çift tıklandığında listedeki süslü yazıları temizler ve klasöre girer."""
        raw_text = item.text()
        
        # SÜPER TEMİZLİK: Listeye ne eklediysek onu siliyoruz
        clean_name = raw_text.replace("📂 [KLASÖR] ", "")
        clean_name = clean_name.replace("🧠 [MODEL] ", "")
        
        # Eğer klasörün yanında parantezli bir sayı varsa onu da at (örn: (5 Models))
        clean_name = clean_name.split(" (")[0].strip()
        
        # Tam yolu oluştur
        new_path = os.path.abspath(os.path.join(self.current_path, clean_name))

        # KONTROL: Eğer bu bir klasörse içeri gir
        if os.path.isdir(new_path):
            self.sync_and_show_tasks(new_path)
            self.log_message(f"📂 Girilen Dizin: {clean_name}")
            
        # Eğer bu bir .zip ise simülasyonda aç
        elif clean_name.lower().endswith(".zip"):
            self.log_message(f"👁️ Model İzleniyor: {clean_name}")
            vis_script = os.path.join(BASE_DIR, "visualize_model.py")
            subprocess.Popen([sys.executable, vis_script, new_path])
    def go_back(self):
        """Bir üst klasöre dön."""
        parent = os.path.dirname(self.current_path)
        if os.path.abspath(MODELS_DIR) in os.path.abspath(parent) or \
           os.path.abspath(MODELS_DIR) == os.path.abspath(parent):
            self.sync_and_show_tasks(parent)

    def toggle_task_panel(self, item):
        if self.detail_panel.isVisible():
            self.detail_panel.hide()
        else:
            self.panel_title.setText(f"Kontrol: {item.text()}")
            self.detail_panel.show()

    def start_new_robot_training(self):
        """Arayüzdeki butona basıldığında sıfırdan eğitimi başlatan ana süreç."""
        # 1. Kullanıcıya CSV seçtir
        csv_file, _ = QFileDialog.getOpenFileName(self, "Eğitim İçin CSV Seç", "", "CSV Files (*.csv)")
        
        if not csv_file:
            self.log_message("⚠️ İşlem iptal edildi: CSV seçilmedi.")
            return

        # 2. Klasörleri ve ismi hazırla
        robot_name, dest_csv = prepare_training_folder(csv_file)
        
        if robot_name and dest_csv:
            # 3. Görsel geçişleri yap
            self.pages.setCurrentIndex(2) # Log sayfasını aç
            self.log_message(f"✅ Yeni Proje: {robot_name}")
            self.log_message(f"📁 Klasörler ve CSV hazırlandı.")
            
            # 4. Eğitimi sıfırdan başlat (is_new=True)
            self.status_label.setText(f"Durum: {robot_name} Sıfırdan Eğitiliyor...")
            self.worker = TrainingWorker(robot_name, dest_csv, is_new=True)
            self.worker.log_signal.connect(self.log_message)
            self.worker.finished_signal.connect(lambda: self.status_label.setText("Durum: Tamamlandı"))
            # Eğitimi başlatan satırların oraya ekle:
            self.btn_stop_training.setEnabled(True) 
            self.status_label.setText("Durum: Eğitim Devam Ediyor...")
            self.worker.start()
        else:
            QMessageBox.critical(self, "Hata", "Klasör yapısı oluşturulurken bir sorun çıktı!")

    def on_training_button_click(self):
        selected_item = self.task_list.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Hata", "Lütfen bir model seçin!")
            return

        raw_text = selected_item.text()
        # Robot adını temizle (Örn: "robotA")
        clean_name = raw_text.split(" (")[0].replace("📂 [KLASÖR] ", "").replace("🧠 [MODEL] ", "").strip()
        if clean_name.lower().endswith(".zip"): clean_name = clean_name[:-4]

        # --- YENİ KLASÖR YAPISI ---
        robot_folder = os.path.join(MODELS_DIR, clean_name)
        # Senin istediğin: Giriş verisi buraya bakacak
        checkpoint_dir = os.path.join(robot_folder, "checkpoint") 
        
        target_model_path = None

        # 1. Adım: 'checkpoint' klasörünün içine bak
        if os.path.exists(checkpoint_dir):
            zips = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
            if zips:
                # En son eklenen beyin dosyasını bul
                target_model_path = max(zips, key=os.path.getmtime)
                self.log_message(f"✅ Giriş modeli 'checkpoint' klasöründe bulundu: {os.path.basename(target_model_path)}")

        # 2. Adım: Eğer 'checkpoint' boşsa hata ver (Çünkü artık oraya bakıyoruz)
        if not target_model_path:
            QMessageBox.critical(self, "Hata", f"Eğitimin başlaması için '{checkpoint_dir}' klasöründe en az bir .zip dosyası olmalı!")
            return

        # --- EĞİTİMİ BAŞLAT ---
        self.pages.setCurrentIndex(2)
        self.log_message(f"🚀 {clean_name} için eğitim başlatılıyor...")
        
        self.worker = TrainingWorker(clean_name, target_model_path)
        self.worker.log_signal.connect(self.log_message)
        self.worker.finished_signal.connect(lambda: self.status_label.setText("Durum: Tamamlandı"))
        # Eğitimi başlatan satırların oraya ekle:
        self.btn_stop_training.setEnabled(True) 
        self.status_label.setText("Durum: Eğitim Devam Ediyor...")
        self.worker.start()
        
        self.status_label.setText(f"Durum: {clean_name} Eğitiliyor...")
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernUI()
    window.show()
    sys.exit(app.exec())