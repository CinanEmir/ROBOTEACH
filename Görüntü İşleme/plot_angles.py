import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "angles.csv"


def main():
    df = pd.read_csv(CSV_FILE)

    time = df["time_sec"]

    # 1) Gövde grafiği
    plt.figure(figsize=(12, 6))
    plt.plot(time, df["abdomen_y_deg"], label="Abdomen Y (Torso Pitch)", linewidth=2)
    plt.plot(time, df["abdomen_x_deg"], label="Abdomen X (Torso Roll)", linewidth=2)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Angle (degrees)")
    plt.title("Torso Angle Variation Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Bacak grafiği
    plt.figure(figsize=(12, 6))
    plt.plot(time, df["hip_y_left_deg"], label="Left Hip", linewidth=2)
    plt.plot(time, df["hip_y_right_deg"], label="Right Hip", linewidth=2)
    plt.plot(time, df["knee_left_deg"], label="Left Knee", linewidth=2)
    plt.plot(time, df["knee_right_deg"], label="Right Knee", linewidth=2)
    plt.plot(time, df["ankle_y_left_deg"], label="Left Ankle", linewidth=2)
    plt.plot(time, df["ankle_y_right_deg"], label="Right Ankle", linewidth=2)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Angle (degrees)")
    plt.title("Leg Joint Angle Variation Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Kol grafiği
    plt.figure(figsize=(12, 6))
    plt.plot(time, df["shoulder1_left_deg"], label="Left Shoulder", linewidth=2)
    plt.plot(time, df["shoulder1_right_deg"], label="Right Shoulder", linewidth=2)
    plt.plot(time, df["elbow_left_deg"], label="Left Elbow", linewidth=2)
    plt.plot(time, df["elbow_right_deg"], label="Right Elbow", linewidth=2)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Angle (degrees)")
    plt.title("Arm Joint Angle Variation Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()