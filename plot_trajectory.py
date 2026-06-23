import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

csv_file = 'latency_data_stable.csv'

if not os.path.exists(csv_file):
    print(f"❌ Không tìm thấy file {csv_file}. Bác chạy robot trước để sinh file nhé!")
    sys.exit()

# 1. ĐỌC DỮ LIỆU
df = pd.read_csv(csv_file)

# Lọc bỏ những dòng đầu tiên lúc vừa khởi động để đồ thị sạch
df = df[df['Time'] > 0.5]

time_data = df['Time'].values
master_pos = df['Master_J1_Pos'].values
ur3_pos = df['UR3_J1_Pos'].values

# 2. TÍNH ĐỘ TRỄ BẰNG TƯƠNG QUAN CHÉO (CROSS-CORRELATION)
# Lùi lại mean để tín hiệu xoay quanh mốc 0
master_centered = master_pos - np.mean(master_pos)
ur3_centered = ur3_pos - np.mean(ur3_pos)

correlation = np.correlate(master_centered, ur3_centered, mode='full')
lag_index = np.argmax(correlation) - (len(ur3_centered) - 1)

# dt trung bình của hệ thống
dt = np.mean(np.diff(time_data))
latency_seconds = abs(lag_index * dt)
latency_ms = latency_seconds * 1000

# 3. VẼ ĐỒ THỊ
plt.figure(figsize=(10, 5))
plt.plot(time_data, master_pos, label='Feetech Master (Target)', color='blue', linewidth=2, linestyle='--')
plt.plot(time_data, ur3_pos, label='UR3 Slave (Actual)', color='red', linewidth=2)

plt.title(f'Đánh giá quỹ đạo và Độ trễ (Khớp Base)\nĐộ trễ tính toán: {latency_ms:.1f} ms', fontsize=14, fontweight='bold')
plt.xlabel('Thời gian (s)', fontsize=12)
plt.ylabel('Vị trí Góc (rad)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right', fontsize=12)

# Tính Sai số bám RMSE
rmse = np.sqrt(np.mean((master_pos - ur3_pos)**2))
plt.text(0.02, 0.05, f'Sai số bám RMSE: {rmse:.4f} rad', 
         transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()
plt.show()
