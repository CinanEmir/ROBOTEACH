import cv2
import json
import math
import os
import re
from pathlib import Path

import mediapipe as mp
from dotenv import load_dotenv

# .env dosyasındaki değişkenleri sisteme yükle
load_dotenv()

# --------------------------------------------------
# AYARLAR
# --------------------------------------------------
# Değerleri önce .env dosyasından çekmeye çalışır, 
# eğer bulamazsa (veya .env yoksa) virgülden sonraki değeri kullanır.
VIDEO_PATH = os.getenv("VIDEO_PATH", "çekim8.mov")
OUT_JSONL = os.getenv("OUT_JSONL", "poses.jsonl")
OUT_ANGLES = os.getenv("OUT_ANGLES", "angles.csv")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# --------------------------------------------------
# YARDIMCI FONKSİYONLAR
# --------------------------------------------------
def angle_3p(a, b, c):
    """
    3 nokta ile açı hesaplar. b merkez noktadır.
    a, b, c = (x, y, z)
    """
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c

    ab = (ax - bx, ay - by, az - bz)
    cb = (cx - bx, cy - by, cz - bz)

    dot = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]
    nab = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2) + 1e-9
    ncb = math.sqrt(cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2) + 1e-9

    cosv = max(-1.0, min(1.0, dot / (nab * ncb)))
    return math.degrees(math.acos(cosv))


def distance_3d(p1, p2):
    """İki nokta arasındaki 3D Öklid mesafesini hesaplar (Gripper için)"""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)


def clamp(value, min_val, max_val):
    if value is None:
        return None
    return max(min(value, max_val), min_val)


def fmt(value):
    return "" if value is None else f"{value:.3f}"


def lm_to_xyz(lm):
    return (float(lm.x), float(lm.y), float(lm.z))


def midpoint(a, b):
    return (
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        (a[2] + b[2]) / 2.0,
    )


def next_video_name(prefix="video", ext=".mp4"):
    files = os.listdir(".")
    nums = []

    pattern = re.compile(rf"^{prefix}(\d+){re.escape(ext)}$")

    for f in files:
        m = pattern.match(f)
        if m:
            nums.append(int(m.group(1)))

    next_num = max(nums) + 1 if nums else 1
    return f"{prefix}{next_num}{ext}"


def signed_angle_2d(v1, v2):
    """
    2D signed angle (derece)
    v1, v2 = (x, y)
    """
    x1, y1 = v1
    x2, y2 = v2
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    ang = math.degrees(math.atan2(det, dot))
    return ang


def torso_pitch_from_midpoints(mid_shoulder, mid_hip):
    """
    Gövde pitch proxy:
    Mid-hip -> mid-shoulder vektörünü dikey eksenle karşılaştırır.
    UPRIGHT ≈ 0°, öne/arkaya eğilmede artar.
    """
    vx = mid_shoulder[0] - mid_hip[0]
    vy = mid_shoulder[1] - mid_hip[1]

    # görüntüde yukarı yön yaklaşık (0, -1)
    up = (0.0, -1.0)

    norm_v = math.sqrt(vx * vx + vy * vy) + 1e-9
    v = (vx / norm_v, vy / norm_v)

    dot = v[0] * up[0] + v[1] * up[1]
    dot = max(-1.0, min(1.0, dot))

    return math.degrees(math.acos(dot))


def torso_roll_from_shoulders(left_shoulder, right_shoulder):
    """
    Gövde roll proxy:
    Omuz çizgisinin yatay eksene göre eğimi.
    """
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]

    # yatay eksen
    horizontal = (1.0, 0.0)

    ang = signed_angle_2d(horizontal, (dx, dy))
    return abs(ang)


# --------------------------------------------------
# ANA FONKSİYON
# --------------------------------------------------
def main():
    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Video yok: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_name = next_video_name()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, open(OUT_JSONL, "w", encoding="utf-8") as fjson, open(OUT_ANGLES, "w", encoding="utf-8") as fang:

        # Robot mantığına daha yakın tam teşekküllü CSV başlığı
        fang.write(
            "frame,time_sec,"
            "head_pitch_deg,"
            "abdomen_y_deg,abdomen_x_deg,"
            "hip_y_left_deg,hip_y_right_deg,"
            "knee_left_deg,knee_right_deg,"
            "ankle_y_left_deg,ankle_y_right_deg,"
            "shoulder1_left_deg,shoulder1_right_deg,"
            "shoulder2_left_deg,shoulder2_right_deg,"
            "elbow_left_deg,elbow_right_deg,"
            "wrist_left_deg,wrist_right_deg,"
            "gripper_left_dist,gripper_right_dist\n"
        )

        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            record = {
                "frame": frame_idx,
                "time_sec": frame_idx / fps,
                "landmarks": None
            }

            abdomen_y = None   # torso_pitch proxy
            abdomen_x = None   # torso_roll proxy

            hip_y_left = None
            hip_y_right = None

            knee_left = None
            knee_right = None

            ankle_y_left = None
            ankle_y_right = None

            shoulder1_left = None
            shoulder1_right = None

            elbow_left = None
            elbow_right = None
            
            # --- YENİ EKLENEN DEĞİŞKENLER (Hiçbirleştirilmedi) ---
            head_pitch = None
            
            shoulder2_left = None
            shoulder2_right = None
            
            wrist_left = None
            wrist_right = None
            
            gripper_left = None
            gripper_right = None

            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark

                record["landmarks"] = [
                    {
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "v": float(lm.visibility)
                    }
                    for lm in lms
                ]

                # -------------------------
                # LANDMARKLAR
                # -------------------------
                NOSE = lm_to_xyz(lms[mp_pose.PoseLandmark.NOSE])

                LS = lm_to_xyz(lms[mp_pose.PoseLandmark.LEFT_SHOULDER])
                RS = lm_to_xyz(lms[mp_pose.PoseLandmark.RIGHT_SHOULDER])

                LE = lm_to_xyz(lms[mp_pose.PoseLandmark.LEFT_ELBOW])
                RE = lm_to_xyz(lms[mp_pose.PoseLandmark.RIGHT_ELBOW])

                LW = lm_to_xyz(lms[mp_pose.PoseLandmark.LEFT_WRIST])
                RW = lm_to_xyz(lms[mp_pose.PoseLandmark.RIGHT_WRIST])
                
                L_INDEX = lm_to_xyz(lms[mp_pose.PoseLandmark.LEFT_INDEX])
                R_INDEX = lm_to_xyz(lms[mp_pose.PoseLandmark.RIGHT_INDEX])
                L_THUMB = lm_to_xyz(lms[mp_pose.PoseLandmark.LEFT_THUMB])
                R_THUMB = lm_to_xyz(lms[mp_pose.PoseLandmark.RIGHT_THUMB])

                LH = lm_to_xyz(lms[mp_pose.PoseLandmark.LEFT_HIP])
                RH = lm_to_xyz(lms[mp_pose.PoseLandmark.RIGHT_HIP])

                LK = lm_to_xyz(lms[mp_pose.PoseLandmark.LEFT_KNEE])
                RK = lm_to_xyz(lms[mp_pose.PoseLandmark.RIGHT_KNEE])

                LA = lm_to_xyz(lms[mp_pose.PoseLandmark.LEFT_ANKLE])
                RA = lm_to_xyz(lms[mp_pose.PoseLandmark.RIGHT_ANKLE])

                LFI = lm_to_xyz(lms[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])
                RFI = lm_to_xyz(lms[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])

                mid_shoulder = midpoint(LS, RS)
                mid_hip = midpoint(LH, RH)

                # -------------------------
                # GÖVDE VE KAFA
                # -------------------------
                head_pitch = clamp(angle_3p(NOSE, mid_shoulder, mid_hip), 0, 180)
                abdomen_y = clamp(torso_pitch_from_midpoints(mid_shoulder, mid_hip), 0, 90)
                abdomen_x = clamp(torso_roll_from_shoulders(LS, RS), 0, 60)

                # -------------------------
                # KALÇA (hip_y proxy)
                # -------------------------
                hip_y_left = clamp(angle_3p(LS, LH, LK), 0, 180)
                hip_y_right = clamp(angle_3p(RS, RH, RK), 0, 180)

                # -------------------------
                # DİZ
                # -------------------------
                knee_left = clamp(angle_3p(LH, LK, LA), 0, 160)
                knee_right = clamp(angle_3p(RH, RK, RA), 0, 160)

                # -------------------------
                # AYAK BİLEĞİ
                # -------------------------
                ankle_y_left = clamp(angle_3p(LK, LA, LFI), 0, 180)
                ankle_y_right = clamp(angle_3p(RK, RA, RFI), 0, 180)

                # -------------------------
                # OMUZ 1 (Pitch - Öne/Arkaya)
                # -------------------------
                shoulder1_left = clamp(angle_3p(LE, LS, LH), 0, 180)
                shoulder1_right = clamp(angle_3p(RE, RS, RH), 0, 180)

                # -------------------------
                # OMUZ 2 (Roll - Yana Açma)
                # -------------------------
                shoulder2_left = clamp(angle_3p(RS, LS, LE), 0, 180)
                shoulder2_right = clamp(angle_3p(LS, RS, RE), 0, 180)

                # -------------------------
                # DİRSEK
                # -------------------------
                elbow_left = clamp(angle_3p(LS, LE, LW), 0, 150)
                elbow_right = clamp(angle_3p(RS, RE, RW), 0, 150)

                # -------------------------
                # EL BİLEĞİ (Bükülme)
                # -------------------------
                wrist_left = clamp(angle_3p(LE, LW, L_INDEX), 0, 180)
                wrist_right = clamp(angle_3p(RE, RW, R_INDEX), 0, 180)

                # -------------------------
                # GRIPPER (Kıskaç Mesafesi)
                # -------------------------
                gripper_left = distance_3d(L_INDEX, L_THUMB)
                gripper_right = distance_3d(R_INDEX, R_THUMB)

                # İskelet çizimi
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # -------------------------
                # DEBUG YAZILARI (Orijinal yapı, alt alta)
                # -------------------------
                texts = [
                    f"abdomen_y:    {abdomen_y:.1f}" if abdomen_y is not None else "abdomen_y: N/A",
                    f"abdomen_x:    {abdomen_x:.1f}" if abdomen_x is not None else "abdomen_x: N/A",
                    f"hip_y_left:   {hip_y_left:.1f}" if hip_y_left is not None else "hip_y_left: N/A",
                    f"hip_y_right:  {hip_y_right:.1f}" if hip_y_right is not None else "hip_y_right: N/A",
                    f"knee_left:    {knee_left:.1f}" if knee_left is not None else "knee_left: N/A",
                    f"knee_right:   {knee_right:.1f}" if knee_right is not None else "knee_right: N/A",
                    f"ankle_y_left: {ankle_y_left:.1f}" if ankle_y_left is not None else "ankle_y_left: N/A",
                    f"ankle_y_right:{ankle_y_right:.1f}" if ankle_y_right is not None else "ankle_y_right: N/A",
                    f"shoulder1_l:  {shoulder1_left:.1f}" if shoulder1_left is not None else "shoulder1_l: N/A",
                    f"shoulder1_r:  {shoulder1_right:.1f}" if shoulder1_right is not None else "shoulder1_r: N/A",
                    f"elbow_left:   {elbow_left:.1f}" if elbow_left is not None else "elbow_left: N/A",
                    f"elbow_right:  {elbow_right:.1f}" if elbow_right is not None else "elbow_right: N/A",
                    # --- YENİ EKLENEN EKRAN YAZILARI ---
                    f"head_pitch:   {head_pitch:.1f}" if head_pitch is not None else "head_pitch: N/A",
                    f"shoulder2_l:  {shoulder2_left:.1f}" if shoulder2_left is not None else "shoulder2_l: N/A",
                    f"shoulder2_r:  {shoulder2_right:.1f}" if shoulder2_right is not None else "shoulder2_r: N/A",
                    f"wrist_left:   {wrist_left:.1f}" if wrist_left is not None else "wrist_left: N/A",
                    f"wrist_right:  {wrist_right:.1f}" if wrist_right is not None else "wrist_right: N/A",
                    f"gripper_left: {gripper_left:.2f}" if gripper_left is not None else "gripper_left: N/A",
                    f"gripper_right:{gripper_right:.2f}" if gripper_right is not None else "gripper_right: N/A",
                ]

                start_x = 20
                start_y = 30
                line_gap = 24

                for i, text in enumerate(texts):
                    y = start_y + i * line_gap
                    cv2.putText(
                        frame,
                        text,
                        (start_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 0),
                        2
                    )

            # JSONL yaz
            fjson.write(json.dumps(record, ensure_ascii=False) + "\n")

            # CSV yaz
            fang.write(
                f"{frame_idx},{frame_idx / fps:.4f},"
                f"{fmt(head_pitch)},"
                f"{fmt(abdomen_y)},"
                f"{fmt(abdomen_x)},"
                f"{fmt(hip_y_left)},"
                f"{fmt(hip_y_right)},"
                f"{fmt(knee_left)},"
                f"{fmt(knee_right)},"
                f"{fmt(ankle_y_left)},"
                f"{fmt(ankle_y_right)},"
                f"{fmt(shoulder1_left)},"
                f"{fmt(shoulder1_right)},"
                f"{fmt(shoulder2_left)},"
                f"{fmt(shoulder2_right)},"
                f"{fmt(elbow_left)},"
                f"{fmt(elbow_right)},"
                f"{fmt(wrist_left)},"
                f"{fmt(wrist_right)},"
                f"{fmt(gripper_left)},"
                f"{fmt(gripper_right)}\n"
            )

            video_writer.write(frame)
            frame_idx += 1

    cap.release()
    video_writer.release()

    print("Bitti ✅")
    print("Pose çıktı:", OUT_JSONL)
    print("Açı çıktı :", OUT_ANGLES)
    print("Video çıktı:", output_name)


if __name__ == "__main__":
    main()