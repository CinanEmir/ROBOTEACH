import gymnasium as gym
import numpy as np
import pandas as pd
import os

class HumanoidWalkingEnv(gym.Env):
    """
    ROBOTEACH V9 - Kambur Yürüyüş Önleme + Yüksek Sadakatli Taklit
    """

    def __init__(self, render_mode=None, csv_path="perfect_walking_gait.csv"):
        self.env = gym.make("Humanoid-v5", render_mode=render_mode)
        self.action_space      = self.env.action_space
        self.observation_space = self.env.observation_space

        self.global_step_count  = 3_639_948
        self.current_phase_idx  = 0
        self._prev_action       = None
        self._episode_steps     = 0    

        self.reference_data = None
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            self.reference_data = self.df.values
            print(f"ROBOTEACH V9 HAZIR: Kambur Yürüyüş Önleme aktif.")

        self.mapping_indices = [
            (1,  2, 150.0),  
            (11, 4, 80.0),   (5,  5, 80.0),
            (12, 6, 100.0),  (6,  7, 100.0),  (13, 8, 30.0),   (7,  9, 30.0),
        ]
        self._total_weight = sum(w for _, _, w in self.mapping_indices)

    def step(self, action):
        obs, _env_reward, terminated, truncated, info = self.env.step(action)
        self.global_step_count += 1
        self._episode_steps    += 1

        torso_height = float(obs[0])
        joint_angles = obs[7:28]
        torso_quat   = obs[3:7]

        # 1. Faz Geçişi
        s_start = 250_000
        s_end   = 416_666
        t = self.global_step_count
        alpha = float(np.clip((t - s_start) / (s_end - s_start), 0.0, 1.0))

        # 2. Taklit
        imitation_r = 0.0
        if self.reference_data is not None:
            best_error = float('inf')
            best_idx   = self.current_phase_idx
            
            for i in range(-1, 3):
                cand = (self.current_phase_idx + i) % len(self.reference_data)
                row  = self.reference_data[cand]
                err  = 0.0
                for j_idx, c_idx, w in self.mapping_indices:
                    diff = joint_angles[j_idx] - np.radians(row[c_idx])
                    err += w * diff * diff
                if err < best_error:
                    best_error = err
                    best_idx   = cand
            self.current_phase_idx = best_idx
            imitation_r = float(np.exp(-20.0 * (best_error / self._total_weight)))

        # 3. Denge (genel)
        upright_r = float(np.exp(-10.0 * (1.0 - float(torso_quat[0])) ** 2))

        # 4. YENİ — Pitch cezası (öne eğilme / kamburluk)
        qw, qx, qy, qz = torso_quat
        sinp    = 2.0 * (qw * qy - qz * qx)
        pitch   = float(np.arcsin(np.clip(sinp, -1.0, 1.0)))
        posture_r = float(np.exp(-8.0 * pitch ** 2))

        # 5. YENİ — Abdomen direkt ceza (bel bükülmesi)
        abdomen_angle = float(joint_angles[1])   # abdomen_y
        abdomen_r = float(np.exp(-15.0 * abdomen_angle ** 2))

        # 6. YENİ — Torso yükseklik cezası (eğilince alçalır)
        height_r = float(np.exp(-10.0 * (torso_height - 1.28) ** 2))

        # 7. İleri Hız
        fwd_vel   = float(info.get('x_velocity', 0.0))
        forward_r = float(np.exp(-2.0 * (fwd_vel - 1.0) ** 2))

        # 8. Pürüzsüzlük
        if self._prev_action is not None:
            jerk     = action - self._prev_action
            smooth_r = float(np.exp(-2.0 * float(np.mean(jerk ** 2))))
        else:
            smooth_r = 1.0
        self._prev_action = action.copy()

        # 9. Dikey Hız Filtresi
        z_vel = float(obs[2]) if len(obs) > 2 else 0.0
        penalty_val  = np.maximum(0, np.abs(z_vel) - 0.3)
        jump_penalty = float(np.exp(-10.0 * (penalty_val ** 2)))

        # 10. Yan Açılma
        hip_x_r  = float(np.abs(joint_angles[4]))
        hip_x_l  = float(np.abs(joint_angles[10]))
        spread_r = float(np.exp(-5.0 * (hip_x_r + hip_x_l)))

        # 11. Diz Çekme
        knee_left_angle  = np.abs(joint_angles[12])
        knee_right_angle = np.abs(joint_angles[6])
        knee_penalty_val = (np.maximum(0, knee_left_angle - 1.5) +
                            np.maximum(0, knee_right_angle - 1.5))
        knee_penalty = float(np.exp(-5.0 * knee_penalty_val))

        # 12. Enerji
        energy_r = float(np.exp(-0.05 * float(np.sum(action ** 2))))

        # ── ÖDÜL EKONOMİSİ ───────────────────────────────────────────────
        # Upright ağırlığı 3'e bölündü, yeni 3 postür terimi eklendi
        # Toplam max (alpha=1): 1.00
        w_survival  = 0.05
        w_upright   = 0.05   # 0.15'ten düşürüldü — pitch+abdomen+height aldı
        w_posture   = 0.05   # YENİ: pitch cezası
        w_abdomen   = 0.05   # YENİ: bel bükülmesi
        w_height    = 0.05   # YENİ: torso yüksekliği
        w_imitation = alpha * 0.50
        w_forward   = alpha * 0.15
        w_smooth    = 0.05
        w_jump_fix  = 0.03
        w_spread    = 0.02
        w_knee_fix  = 0.02
        w_energy    = 0.03

        reward = (
            w_survival  * 1.0         +
            w_upright   * upright_r   +
            w_posture   * posture_r   +
            w_abdomen   * abdomen_r   +
            w_height    * height_r    +
            w_imitation * imitation_r +
            w_forward   * forward_r   +
            w_smooth    * smooth_r    +
            w_jump_fix  * jump_penalty +
            w_spread    * spread_r    +
            w_knee_fix  * knee_penalty +
            w_energy    * energy_r
        )

        reward = float(np.clip(reward, 0.0, 1.5))

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self._episode_steps = 0
        self._prev_action   = None
        return obs, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()