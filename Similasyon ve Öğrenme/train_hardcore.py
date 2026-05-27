import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from humanoid_imitation_env import HumanoidWalkingEnv

# 1. MUTLAK YOL AYARI
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. make_env (Her çekirdeğe Monitor ekledik)
def make_env(env_csv_path):
    def _init():
        env = HumanoidWalkingEnv(csv_path=env_csv_path)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    # ARGÜMANLARI AL
    if len(sys.argv) < 3:
        print("Hata: Eksik argüman!")
        sys.exit(1)

    task_name = sys.argv[1]
    input_path = sys.argv[2] # CSV veya ZIP yolu
    
    clean_name = task_name.replace(".zip", "").replace(".ZIP", "")

    # 3. CSV YOLUNU BELİRLE
    csv_path = os.path.join(SCRIPT_DIR, "models", clean_name, "motion_data.csv")

    # 4. ORTAM KURULUMU (6 Çekirdek + Monitor)
    num_envs = 6
    env = SubprocVecEnv([make_env(csv_path) for _ in range(num_envs)])
    env = VecMonitor(env)

    # 5. SIFIRDAN EĞİTİM Mİ YOKSA DEVAM MI?
    if input_path.lower().endswith(".csv"):
        model = PPO("MlpPolicy", env, verbose=1, device="cpu", learning_rate=5e-5)
        print(f"Sifirdan egitim baslatiliyor: {clean_name}")
    else:
        model = PPO.load(
        input_path,
        env=env,
        device="cpu",
        custom_objects={
            "learning_rate": 1e-5,
            "clip_range": 0.1        # ← buraya ekle
        }
        )
        
    

    # 6. KAYIT YOLLARI
    base_models_dir = os.path.join(SCRIPT_DIR, "models")
    save_dir = os.path.join(base_models_dir, clean_name, "savepoints")
    os.makedirs(save_dir, exist_ok=True)

    # Ara kayıt sıklığı (86666 x 6 çekirdek = yaklaşık her 520k adımda bir kayıt)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500000, 1), 
        save_path=save_dir, 
        name_prefix=f"humanoid_{clean_name}"
    )   
    
    print(f"Egitim '{clean_name}' icin basladi.")
    print(f"Referans Veri: {csv_path}")

    # 7. EĞİTİMİ BAŞLAT (2 Milyon Adım)
    model.learn(
        total_timesteps=30000000,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )

    # 8. FİNAL KAYDI (UI İÇİN CHECKPOINT KLASÖRÜNE FİX)
    final_checkpoint_dir = os.path.join(base_models_dir, clean_name, "checkpoint")
    os.makedirs(final_checkpoint_dir, exist_ok=True) 

    # Modeli 'checkpoint' klasörünün tam içine yönlendirdik
    final_model_path = os.path.join(final_checkpoint_dir, f"{clean_name}_final")
    
    model.save(final_model_path)
    print(f"Egitim bitti. Final model 'checkpoint' klasörüne kaydedildi: {os.path.abspath(final_model_path)}")