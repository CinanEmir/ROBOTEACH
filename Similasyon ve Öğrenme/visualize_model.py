import sys
import os
import time
from stable_baselines3 import PPO
from humanoid_imitation_env import HumanoidWalkingEnv

def launch(model_path):
    # Model yolundan robotun ana klasörünü bulup CSV'yi çekelim
    # models/robotA/savepoints/model.zip -> models/robotA/motion_data.csv
    robot_folder = os.path.dirname(os.path.dirname(model_path))
    csv_path = os.path.join(robot_folder, "motion_data.csv")

    if not os.path.exists(csv_path):
        csv_path = None # CSV yoksa sinüs moduna düşer

    # Kendi hazırladığımız 12 eklemli ortamı kullanıyoruz
    env = HumanoidWalkingEnv(render_mode="human", csv_path=csv_path)
    
    print(f"--- Simülasyon Başladı ---")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Referans: {csv_path if csv_path else 'Sinüs Modu'}")

    model = PPO.load(model_path, device="cpu")
    obs, _ = env.reset()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # 60 FPS civarı bir izleme deneyimi
            time.sleep(1/60) 

            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("İzleme kullanıcı tarafından kapatıldı.")
    finally:
        env.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        launch(sys.argv[1])
    else:
        print("Hata: Model yolu belirtilmedi!")